import abc
import six
from utils.data_util import trigram_encoding


@six.add_metaclass(abc.ABCMeta)
class BatchDataHandler(object):
    """handler to parse and generate data
    Parameters
    ----------
        vocabulary: vocabulary object
            vocabulary from AKSIS corpus data or custom string
        batch_size: int
            Batch size for each data batch
        source_maxlen: int
            max number of words in each source sequence.
        target_maxlen: int
            max number of words in each target sequence.
    """

    def __init__(self, vocabulary, source_maxlen, target_maxlen, batch_size):
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self.source_maxlen = source_maxlen
        self.target_maxlen = target_maxlen
        self._sources, self._source_tokens, self._targets, self._target_tokens, self._labels = [], [], [], [], []

    @property
    def data_object_length(self):
        return len(self._sources)

    @abc.abstractmethod
    def parse_and_insert_data_object(self, source, target, label=1):
        """parse data using trigram parser, insert it to data_object to generate batch data"""
        raise NotImplementedError

    def insert_data_object(self, source, source_tokens, target, target_tokens, label_id):
        """insert parsed data and return data_object, clean data_object when size reach batch size"""
        if self.data_object_length == self.batch_size:
            self.clear_data_object()
        if len(source_tokens): # for decode/inference, target_tokens is None
            self._sources.append(source)
            self._source_tokens.append(source_tokens)
            self._targets.append(target)
            self._target_tokens.append(target_tokens)
            self._labels.append(label_id)
        return self.data_object

    def clear_data_object(self):
        """clean data_object"""
        del self._sources[:]
        del self._source_tokens[:]
        del self._targets[:]
        del self._target_tokens[:]
        del self._labels[:]

    @property
    def data_object(self):
        return self._sources, self._source_tokens, self._targets, self._target_tokens, self._labels


class BatchDataTrigramHandler(BatchDataHandler):
    """handler to parse with trigram parser and generate data
    Parameters
    ----------
        vocabulary: vocabulary object
            vocabulary from AKSIS corpus data or custom string
        batch_size: int
            Batch size for each data batch
        source_maxlen: int
            max number of words in each source sequence.
        target_maxlen: int
            max number of words in each target sequence.
    """

    def __init__(self, vocabulary, source_maxlen, target_maxlen, batch_size):
        super().__init__(vocabulary, source_maxlen, target_maxlen, batch_size)

    def parse_and_insert_data_object(self, source, target, label=1):
        """parse data using trigram parser, insert it to data_object to generate batch data"""
        source, source_tokens = trigram_encoding(source, self.vocabulary, self.source_maxlen)
        target, target_tokens = trigram_encoding(target, self.vocabulary, self.target_maxlen)
        data_object = self.insert_data_object(source, source_tokens, target, target_tokens, label)
        return data_object
