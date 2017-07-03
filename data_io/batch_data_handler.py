from utils.data_util import trigram_encoding


class BatchDataHandler(object):
    def __init__(self, vocabulary, source_max_seq_length, target_max_seq_length, batch_size):
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self.source_max_seq_length = source_max_seq_length
        self.target_max_seq_length = target_max_seq_length
        self._sources, self._source_lens, self._targets, self._target_lens, self._labels = [], [], [], [], []

    @property
    def data_object_length(self):
        return len(self._sources)

    def parse_and_insert_data_object(self, source, target, label=1):
        source_tokens = trigram_encoding(source, self.vocabulary, self.source_max_seq_length)
        target_tokens = trigram_encoding(target, self.vocabulary, self.target_max_seq_length)
        data_object = self.insert_data_object(source_tokens, target_tokens, label)
        return data_object

    def insert_data_object(self, source_tokens, target_tokens, label_id):
        if self.data_object_length == self.batch_size:
            self.clear_data_object()
        if len(source_tokens) and len(target_tokens):
            self._sources.append(source_tokens)
            self._targets.append(target_tokens)
            self._labels.append(label_id)
        return self.data_object

    def clear_data_object(self):
        del self._sources[:]
        del self._source_lens[:]
        del self._targets[:]
        del self._target_lens[:]
        del self._labels[:]

    @property
    def data_object(self):
        return self._sources, self._targets