import logbook as logging
import tensorflow as tf

from config.config import FLAGS
from collections import OrderedDict
from seq2seq import Seq2SeqModel


def create_model(session, flags_config):
    """Create query2vec model and initialize or load parameters in session."""
    logging.info("Creating {} layers of {} units.", FLAGS.num_layers, FLAGS.hidden_units)
    config = OrderedDict(sorted(flags_config.__flags.items()))
    model = Seq2SeqModel(config, 'train')

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logging.info("Reloading model parameters from {}", ckpt.model_checkpoint_path)
        model.restore(session, ckpt.model_checkpoint_path)

    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())

    return model