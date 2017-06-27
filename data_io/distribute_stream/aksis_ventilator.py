# coding=utf-8
# pylint: disable=too-many-instance-attributes, too-many-arguments
"""ventilitor that read/produce the corpus data"""
import os
from multiprocessing import Process
import fnmatch
import logbook as logging
import zmq
from zmq.decorators import socket
from utils.appmetric_util import AppMetric
from utils.data_util import negative_sampling_train_data_generator
from exception.file_exception import FileNotFoundError


class AksisDataVentilatorProcess(Process):
    """Process to read the corpus data
    Parameters
    ----------
        data_dir : str
            Data_dir for the aksis corpus data
        file_pattern: tuple
            File pattern use to distinguish different corpus, every file pattern will start
            a ventilitor process.
            File pattern is tuple type(file pattern, dropout). Dropout is the probability
            to ignore the data.
            If dropout < 0, all the data will be accepted to be trained
        ip : str
            The ip address string without the port to pass to ``Socket.bind()``.
        port: int
            Port to produce the raw data
        num_epoch: int
            end epoch of producing the data
        name: str
            process name
    """

    def __init__(self, file_pattern, data_dir,
                 num_epoch=65535, dropout=-1, ip='127.0.0.1', port='5555',
                 metric_interval=30, neg_number=5, name='VentilatorProcess'):
        Process.__init__(self)
        self.file_pattern = file_pattern
        self.data_dir = data_dir
        self.num_epoch = num_epoch
        self.dropout = float(dropout)
        # pylint: disable=invalid-name
        self.ip = ip
        self.port = port
        self.metric_interval = metric_interval
        self.neg_number = neg_number
        self.name = name

    # pylint: disable=arguments-differ, no-member
    @socket(zmq.PUSH)
    def run(self, sender):
        sender.connect("tcp://{}:{}".format(self.ip, self.port))
        logging.info("process {} connect {}:{} and start produce data",
                     self.name, self.ip, self.port)
        metric = AppMetric(name=self.name, interval=self.metric_interval)
        try:
            data_stream = self.get_data_stream()
        except FileNotFoundError, e:
            return
        for i in xrange(self.num_epoch):
            for data in data_stream:
                sender.send_pyobj(data)
                metric.notify(1)
            data_stream = self.get_data_stream()
            logging.info("process {} finish {} epoch", self.name, i)

    def get_data_stream(self):
        """data stream generate the query, title data"""
        data_files = fnmatch.filter(os.listdir(self.data_dir), self.file_pattern)

        if len(data_files) <= 0:
            raise FileNotFoundError("no files are found for file pattern {} in {}".format(self.file_pattern,
                                                                                                  self.data_dir))
        action_files = [os.path.join(self.data_dir, filename) for filename in data_files]

        for source, target, label in negative_sampling_train_data_generator(action_files, self.neg_number, self.dropout):
            yield source, target, label
