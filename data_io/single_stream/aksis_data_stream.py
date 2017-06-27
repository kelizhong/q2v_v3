from utils.data_util import query_title_score_generator_from_aksis_data, sentence_to_padding_tokens, trigram_encoding
from vocabulary.vocab import VocabularyFromCustomStringTrigram


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

    def parse_and_insert_data_object(self, source, target, label):
        source_tokens = trigram_encoding(source, self.vocabulary)
        target_tokens = trigram_encoding(target, self.vocabulary)
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


class AksisDataStream(object):
    def __init__(self, vocabulary_data_dir, top_words, special_words, source_max_seq_length, target_max_seq_length, batch_size, raw_data_path=None, stop_freq=-1):
        self.stop_freq = stop_freq
        self.batch_size = batch_size
        self.raw_data_path = raw_data_path
        vocabulary = self._init_vocabulary(vocabulary_data_dir, top_words)
        self.batch_data = BatchDataHandler(vocabulary, source_max_seq_length, target_max_seq_length, batch_size)

    def generate_batch_data(self):
        for num, (source, target) in enumerate(query_title_score_generator_from_aksis_data(self.raw_data_path)):
            if num % 1000 == 0:
                print("  reading data line %d" % num)

            if num > 50000:
                break
            self.batch_data.parse_and_insert_data_object(source, target, 1)
            if self.batch_data.data_object_length == self.batch_size:
                yield self.batch_data.data_object

    def _init_vocabulary(self, vocabulary_data_dir, top_words):
        # Load vocabulary
        vocab = VocabularyFromCustomStringTrigram(vocabulary_data_dir,
                                                  top_words).build_vocabulary_from_pickle()
        return vocab

