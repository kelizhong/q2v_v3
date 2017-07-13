# coding=utf-8
"""
Change the hardcoded host urls below with your own hosts.
Run like this:
pc-01$ CUDA_VISIBLE_DEVICES='' python train.py --job_type="ps" --task_index=0 --data_stream_port=5558 --gpu=0 --ps_hosts='localhost:2221' --worker_hosts='localhost:2222,localhost:2223,localhost:2224'
pc-02$ CUDA_VISIBLE_DEVICES=1 python train.py --job_type="worker" --task_index=0 --data_stream_port=5558 --gpu=1 --ps_hosts='localhost:2221' --worker_hosts='localhost:2222,localhost:2223,localhost:2224'
pc-03$ CUDA_VISIBLE_DEVICES=2 python train.py --job_type="worker" --task_index=1 --data_stream_port=5558 --gpu=2 --ps_hosts='localhost:2221' --worker_hosts='localhost:2222,localhost:2223,localhost:2224'
pc-04$ CUDA_VISIBLE_DEVICES=3 python train.py --job_type="worker" --task_index=2 --data_stream_port=5558 --gpu=3 --ps_hosts='localhost:2221' --worker_hosts='localhost:2222,localhost:2223,localhost:2224'

# single machine with zmq stream:
CUDA_VISIBLE_DEVICES=0 python train.py --gpu 0 --data_stream_port 5558

# single machine with local file stream:
CUDA_VISIBLE_DEVICES=0 python train.py --gpu 0


"""

import time

import logbook as logging
import tensorflow as tf
import os

import numpy as np
import math
from collections import defaultdict
from config.config import FLAGS
from data_io.distribute_stream.aksis_data_receiver import AksisDataReceiver
from data_io.single_stream.aksis_data_stream import AksisDataStream
from helper.model_helper import create_model
from utils.decorator_util import memoized
from utils.log_util import setup_logger
from utils.data_util import prepare_train_pair_batch


class Trainer(object):
    """query2vec model trainer"""

    def __init__(self, ps_hosts='', worker_hosts='', model_dir='./data/models', job_type='worker', model_name='q2v', task_index=0,
                 gpu=None, is_sync=False, raw_data_path=None, batch_size=128, display_freq=10, source_maxlen=None,
                 target_maxlen=None, top_words=None, vocabulary_data_dir=None, data_stream_port=None):
        """
        Parameters
        ----------
        job_type: job type, One of 'ps', 'worker', 'single
            1) single type is for single machine training, not need zmq stream
            2) ps: parameter server
        ps_hosts: Comma-separated list of hostname:port pairs for parameter server, like '127.0.0.1:2223,127.0.0.1:2224'
        worker_hosts: Comma-separated list of hostname:port pairs for worker, like '127.0.0.1:2223,127.0.0.1:2224'
        task_index: int
            Index of task within the job
        gpu: int
            specify the gpu to use. Set it to None to use cpu to train
        model_dir: trained model directory
        is_sync: bool
            whether to synchronize, aggregate gradients
        raw_data_path: raw corpus data path for local data stream to product train data
        batch_size: int
            batch size for local data stream
        display_freq: int
            Display training status every this iteration
        source_maxlen: int
            max number of words/tokens in each source sequence.
        target_maxlen: int
            max number of words/tokens in each target sequence
        top_words: the vocabulary size
        vocabulary_data_dir:
            the directory to store/load vocabulary data(vocabulary with most `top_words` common words and word counter)
        data_stream_port: int
            port for data zmq stream
        model_name: model name, model will be stored in os.path.join(model_dir, model_name)
        """
        self.job_type = job_type
        self.ps_hosts = ps_hosts.split(",")
        self.worker_hosts = worker_hosts.split(",")
        self.task_index = task_index
        self.gpu = gpu
        self.model_dir = os.path.join(model_dir, model_name)
        self.is_sync = is_sync
        self.raw_data_path = raw_data_path
        self.display_freq = display_freq
        # batch_size only for data_local_stream
        self.batch_size = batch_size
        self.source_maxlen = source_maxlen
        self.target_maxlen = target_maxlen
        self.top_words = top_words
        self.vocabulary_data_dir = vocabulary_data_dir
        self.data_stream_port = data_stream_port

    @property
    @memoized
    def master(self):
        """define execution engine"""
        if self.job_type == "single":
            master = ""
        else:
            master = self.server.target
        return master

    @property
    @memoized
    def server(self):
        """define an in-process TensorFlow server, for use in distributed training"""
        assert self.job_type != 'single', "Not support cluster for single machine training"
        server = tf.train.Server(self.cluster, job_name=self.job_type, task_index=self.task_index)
        return server

    @property
    @memoized
    def cluster(self):
        """represents the set of processes that participate in a distributed TensorFlow computation"""
        assert self.job_type != 'single', "Not support cluster for single machine training"
        cluster = tf.train.ClusterSpec({"ps": self.ps_hosts, "worker": self.worker_hosts})
        return cluster

    @property
    @memoized
    def core_str(self):
        core_str = "cpu:0" if (self.gpu is None or self.gpu == "") else "gpu:%d" % int(self.gpu)
        return core_str

    @property
    @memoized
    def device(self):
        if self.job_type == "worker":
            device = tf.train.replica_device_setter(cluster=self.cluster,
                                                    worker_device='job:worker/task:%d/%s' % (
                                                        self.task_index, self.core_str),
                                                    ps_device='job:ps/task:%d/%s' % (self.task_index, self.core_str))
        else:
            device = "/" + self.core_str

        return device

    @property
    def data_zmq_stream(self):
        """product data with zmq stream """
        if self.data_stream_port is None:
            raise Exception("port is not defined for zmq stream")
        data_stream = AksisDataReceiver(port=self.data_stream_port)
        return data_stream

    @property
    def data_local_stream(self):
        """product data by parsing the local file"""
        data_stream = AksisDataStream(self.vocabulary_data_dir, top_words=self.top_words,
                                      batch_size=self.batch_size,
                                      raw_data_path=self.raw_data_path).generate_batch_data()
        return data_stream

    @property
    def data_stream(self):
        """data stream to produce train data"""
        if self.data_stream_port:
            stream = self.data_zmq_stream
        else:
            stream = self.data_local_stream
        return stream

    @staticmethod
    def _log_variable_info():
        tensor_memory = defaultdict(int)
        for item in tf.global_variables():
            logging.info("variable:{}, device:{}", item.name, item.device)
        # TODO int32 and float32, dtype?
        for item in tf.trainable_variables():
            tensor_memory[item.device] += int(np.prod(item.shape))
        for key, value in tensor_memory.items():
            logging.info("device:{}, memory:{}", key, value)

    def train(self):
        if self.job_type == "ps":
            logging.info("starting parameter:{}", self.task_index)
            self.server.join()
        else:
            with tf.device(self.device):
                with tf.Session(target=self.master,
                                config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                                      log_device_placement=FLAGS.log_device_placement, )) as sess:
                    logging.info("creating model for  worker:{}", self.task_index)
                    model = create_model(sess, FLAGS, model_name=FLAGS.model_name)
                self._log_variable_info()
                init_op = tf.global_variables_initializer()
            sv = tf.train.Supervisor(is_chief=(self.task_index == 0),
                                     logdir=self.model_dir,
                                     init_op=init_op,
                                     saver=model.saver,
                                     global_step=model.global_step,
                                     save_model_secs=60)
            gpu_options = tf.GPUOptions(allow_growth=True)
            session_config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                            log_device_placement=FLAGS.log_device_placement,
                                            gpu_options=gpu_options,
                                            intra_op_parallelism_threads=16)
            with sv.prepare_or_wait_for_session(master=self.master, config=session_config) as sess:
                # TODO add tensorboard support
                if self.task_index == 0 and self.is_sync and self.job_type == 'worker':
                    # TODO and synchronize optimizer
                    sv.start_queue_runners(sess, [model.chief_queue_runner])
                    sess.run(model.init_token_op)

                step_time, loss = 0.0, 0.0
                words_done, sents_done = 0, 0
                data_stream = self.data_stream
                for sources, targets in data_stream:
                    start_time = time.time()
                    sources, source_lens, targets, target_lens = prepare_train_pair_batch(sources, targets,
                                                                                          source_maxlen=self.source_maxlen,
                                                                                          target_maxlen=self.target_maxlen)
                    # Get a batch from training parallel data
                    if sources is None or targets is None:
                        logging.warn('No samples under source_max_seq_length {} or target_max_seq_length {}',
                                     self.source_maxlen, self.target_maxlen)
                        continue

                    # Execute a single training step
                    step_loss = model.train(sess, encoder_inputs=sources, encoder_inputs_length=source_lens,
                                            decoder_inputs=targets, decoder_inputs_length=target_lens)
                    model.adjust_lr_rate(sess, step_loss)
                    time_elapsed = time.time() - start_time
                    step_time = time_elapsed / self.display_freq
                    loss += step_loss / self.display_freq
                    words_done += float(np.sum(source_lens + target_lens))
                    sents_done += float(sources.shape[0])  # batch_size

                    # Once in a while, print statistics, and run evals.
                    # Increase the epoch index of the model
                    model.global_epoch_step_op.eval()
                    if model.global_step.eval() % self.display_freq == 0:
                        avg_perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                        words_per_sec = words_done / time_elapsed
                        sents_per_sec = sents_done / time_elapsed
                        logging.info(
                            "global step %d, learning rate %.4f, step-time:%.2f, step-loss:%.4f, loss:%.4f, perplexity:%.4f, %.4f sents/s, %.4f words/s" %
                            (model.global_step.eval(), model.learning_rate.eval(), step_time, step_loss, loss, avg_perplexity,
                             sents_per_sec, words_per_sec))
                        # set zero timer and loss.
                        words_done, sents_done, loss = 0.0, 0.0, 0.0

                sv.stop()


def main(_):
    setup_logger(FLAGS.log_file_name)
    if FLAGS.debug:
        # https://github.com/tensorflow/tensorflow/commit/ec1403e7dc2b919531e527d36d28659f60621c9e
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu if FLAGS.gpu else ''

    trainer = Trainer(raw_data_path=FLAGS.raw_data_path, vocabulary_data_dir=FLAGS.vocabulary_data_dir,
                      data_stream_port=FLAGS.data_stream_port, top_words=FLAGS.vocabulary_size,
                      source_maxlen=FLAGS.source_maxlen, target_maxlen=FLAGS.target_maxlen, job_type=FLAGS.job_type,
                      ps_hosts=FLAGS.ps_hosts, worker_hosts=FLAGS.worker_hosts, task_index=FLAGS.task_index,
                      gpu=FLAGS.gpu, model_dir=FLAGS.model_dir, is_sync=FLAGS.is_sync, batch_size=FLAGS.batch_size,
                      display_freq=FLAGS.display_freq, model_name=FLAGS.model_name)
    trainer.train()


if __name__ == "__main__":
    tf.app.run()
