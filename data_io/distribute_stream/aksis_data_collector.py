# coding=utf-8
# pylint: disable=no-member, invalid-name, ungrouped-imports, too-many-arguments
"""aksis data collector that collect the data from parser worker"""
import logbook as logging
import pickle
from multiprocessing import Process

from zmq.eventloop import ioloop
from zmq.eventloop.zmqstream import ZMQStream

from utils.appmetric_util import AppMetric
from zmq.decorators import socket
import zmq


class AksisDataCollector(Process):
    """Collector that collect the data from parser worker and add to the
    bucket queue and send the data to trainer

    Parameters
    ----------
        ip : str
            The ip address string without the port to pass to ``Socket.bind()``.
        buckets: tuple list
            The buckets for seq2seq model, a list with (encoder length, decoder length)
        batch_size: int
            Batch size for each databatch
        frontend_port: int
            Port for the incoming traffic
        backend_port: int
            Port for the outbound traffic
        metric_interval: int
            Report the metric for every metric_interval second
        name: str
            Collector process name
    """

    def __init__(self, ip, batch_size, frontend_port=5557, backend_port=5558,
                 metric_interval=10, name="AksisDataCollectorProcess"):
        Process.__init__(self)
        self.ip = ip
        self.batch_size = batch_size
        self.frontend_port = frontend_port
        self.backend_port = backend_port
        self.metric_interval = metric_interval
        self.name = name
        self._sources, self._source_lens, self._targets, self._target_lens, self._labels = [], [], [], [], []

    # pylint: disable=arguments-differ
    @socket(zmq.PULL)
    @socket(zmq.PUSH)
    def run(self, receiver, sender):
        receiver.bind("tcp://{}:{}".format(self.ip, self.frontend_port))
        sender.bind("tcp://{}:{}".format(self.ip, self.backend_port))
        # set up bucket queue
        metric = AppMetric(name=self.name, interval=self.metric_interval)
        # pylint: disable=line-too-long
        logging.info("start collector {}, ip:{}, frontend port:{}, backend port:{}", self.name, self.ip,
                     self.frontend_port,
                     self.backend_port)
        ioloop.install()
        loop = ioloop.IOLoop.instance()
        pull_stream = ZMQStream(receiver, loop)


        def _on_recv(msg):
            # accept the msg and add to the bucket queue and send the batch data to trainer
            # encoder_sentence_id for query in aksis data
            # decoder_sentence_id for title in aksis data
            source_tokens, source_len, target_tokens, target_len, label_id = pickle.loads(msg[0])
            # add the data from parser worker, and get data from the batch queue
            self.insert_data_object(source_tokens, source_len, target_tokens, target_len, label_id)
            if self.data_object_length == self.batch_size:
                sender.send_pyobj(self.data_object)
                metric.notify(self.batch_size)
                self.clear_data_object()

        pull_stream.on_recv(_on_recv)

        loop.start()

    @property
    def data_object_length(self):
        return len(self._sources)

    def insert_data_object(self, source_tokens, source_len, target_tokens, target_len, label_id):
        self._sources.append(source_tokens)
        self._source_lens.append(source_len)
        self._targets.append(target_tokens)
        self._target_lens.append(target_len)
        self._labels.append(label_id)

    def clear_data_object(self):
        del self._sources[:]
        del self._source_lens[:]
        del self._targets[:]
        del self._target_lens[:]
        del self._labels[:]

    @property
    def data_object(self):
        return self._sources, self._source_lens, self._targets, self._target_lens, self._labels