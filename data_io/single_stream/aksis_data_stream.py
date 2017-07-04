from utils.data_util import query_title_score_generator_from_aksis_data
from utils.decorator_util import memoized
from vocabulary.vocab import VocabularyFromCustomStringTrigram
from ..batch_data_handler import BatchDataTrigramHandler


class AksisDataStream(object):
    """data stream for single machine, just for test without start the zmq ventilator

    Parameters
    ----------
        vocabulary_data_dir: str
            Path for vocabulary from AKSIS corpus data or custom string
        top_words: int
            Only use the top_words in vocabulary
        batch_size: int
            Batch size for each data batch
        source_maxlen: int
            max number of words in each source sequence.
        target_maxlen: int
            max number of words in each target sequence.
    """

    def __init__(self, vocabulary_data_dir, top_words, source_maxlen, target_maxlen, batch_size, raw_data_path=None):
        self.vocabulary_data_dir = vocabulary_data_dir
        self.top_words = top_words
        self.batch_size = batch_size
        self.raw_data_path = raw_data_path
        self.batch_data = BatchDataTrigramHandler(self.vocabulary, source_maxlen, target_maxlen, batch_size)

    def generate_batch_data(self):
        for num, (source, target) in enumerate(query_title_score_generator_from_aksis_data(self.raw_data_path)):
            if num % 1000 == 0:
                print("reading data line %d" % num)

            self.batch_data.parse_and_insert_data_object(source, target)
            if self.batch_data.data_object_length == self.batch_size:
                source_tokens, target_tokens, _ = self.batch_data.data_object
                yield source_tokens, target_tokens

    @property
    @memoized
    def vocabulary(self):
        """load vocabulary"""
        vocab = VocabularyFromCustomStringTrigram(self.vocabulary_data_dir,
                                                  top_words=self.top_words).build_vocabulary_from_pickle()
        return vocab
