import os
import sys
import tensorflow as tf
from helper.model_helper import create_model
from config.config import FLAGS
from data_io.batch_data_handler import BatchDataTrigramHandler
from utils.data_util import prepare_decode_batch
from vocabulary.vocab import VocabularyFromCustomStringTrigram
from utils.decorator_util import memoized
from utils.math_util import cos_distance
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector


class Inference(object):

    def __init__(self, source_maxlen=30, batch_size=65536, min_words=-1):
        self.sess = tf.Session()
        self.model = self._init_model()
        self.batch_data = BatchDataTrigramHandler(self.vocabulary, batch_size=sys.maxsize, min_words=min_words)
        self.batch_size = batch_size
        self.source_maxlen = source_maxlen

    def _init_model(self):
        model = create_model(self.sess, FLAGS, mode='encode')
        return model

    def encode(self, inputs):
        sources = None
        source_tokens = []
        self.batch_data.clear_data_object()
        for each in inputs:
            sources, source_tokens, _, _ = self.batch_data.parse_and_insert_data_object(each, None)
        result = None
        if len(sources) > 0:
            source_tokens, source_lens = prepare_decode_batch(source_tokens, maxlen=self.source_maxlen)
            result = self.model.encode(self.sess, source_tokens, source_lens)
        return sources, result

    def batch_encode(self, file):
        model_path = os.path.join(FLAGS.model_dir, "embedding")
        writer = tf.summary.FileWriter(model_path, self.sess.graph)
        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = 'item_embedding'
        embed.metadata_path = os.path.join(model_path, 'metadata.csv')
        projector.visualize_embeddings(writer, config)
        sources = set()
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                sources.add(line)
        metadata_path = embed.metadata_path

        sources_list = []
        embedding_list = []
        count = 0
        # the left over elements that would be truncated by zip
        for each in zip(*[iter(sources)]*self.batch_size):
            print(each)
            batch_sources, result = self.encode(each)
            print(batch_sources)
            if result is not None:
                sources_list.extend(batch_sources)
                embedding_list.append(result)
                count += self.batch_size
                print("Finished: %d" % count)

        with open(metadata_path, 'w+') as item_file:
            item_file.write('id\tchar\n')
            for i, each in enumerate(sources_list):
                item_file.write('{}\t{}\n'.format(i, each))
            print('metadata file created')

        concat = np.concatenate(embedding_list, axis=0)
        item_size, unit_size = concat.shape
        item_embedding = tf.get_variable(embed.tensor_name, [item_size, unit_size])
        assign_op = item_embedding.assign(concat)
        self.sess.run(assign_op)
        saver = tf.train.Saver([item_embedding])
        saver.save(self.sess, model_path, global_step=self.model.global_step)

    @property
    @memoized
    def vocabulary(self):
        """load vocabulary"""
        vocab = VocabularyFromCustomStringTrigram(FLAGS.vocabulary_data_dir).build_vocabulary_from_pickle()
        return vocab

i = Inference(source_maxlen=FLAGS.source_maxlen, batch_size=2)
i.batch_encode('source')
#v = i.encode(["nike shoe men", "apple mac mini"])
#print(cos_distance(v[0], v[1]))