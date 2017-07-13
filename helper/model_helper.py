import os
from collections import OrderedDict

import logbook as logging
import tensorflow as tf

from model.seq2seq import Seq2SeqModel


def create_model(session, flags_config, mode='train', model_name='q2v'):
    """Create query2vec model and initialize or load parameters in session."""
    logging.info("Creating {} layers of {} units.", flags_config.num_layers, flags_config.hidden_units)
    config = OrderedDict(sorted(flags_config.__flags.items()))
    model = Seq2SeqModel(config, mode)

    ckpt = tf.train.get_checkpoint_state(os.path.join(flags_config.model_dir, model_name))
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logging.info("Reloading model parameters from {}", ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)

    else:
        assert mode == 'train', "Can not find existed model, please double check your model path"
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())

    return model
