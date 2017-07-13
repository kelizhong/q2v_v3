import os
import tensorflow as tf

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))

# Extra vocabulary symbols
_GO = '_GO'
EOS = '_EOS' # also function as PAD

extra_tokens = [_GO, EOS]

start_token = extra_tokens.index(_GO)	# start_token = 0
end_token = extra_tokens.index(EOS)	# end_token = 1

special_words = {_GO: start_token, EOS: end_token}
# Run time variables
tf.app.flags.DEFINE_string("gpu", None, "specify the gpu to use")

tf.app.flags.DEFINE_string("log_file_name", os.path.join(project_dir, 'data/logs', 'q2v.log'), "Log data file name")
tf.app.flags.DEFINE_string("raw_data_path", os.path.join(project_dir, 'data/rawdata/test.add'), "port for data zmq stream")
tf.app.flags.DEFINE_string("vocabulary_data_dir", os.path.join(project_dir, 'data/vocabulary'), "port for data zmq stream")

# For distributed
# cluster specification
tf.app.flags.DEFINE_string("ps_hosts", "0.0.0.0:2221",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "0.0.0.0:2222",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_type", "single", "One of 'ps', 'worker', 'single")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_boolean("is_sync", True, "whether to synchronize, aggregate gradients")

# Network parameters
tf.app.flags.DEFINE_boolean('bidirectional', True, 'Enable bidirectional encoder')
tf.app.flags.DEFINE_string('cell_type', 'lstm', 'RNN cell for encoder and decoder, default: lstm')
tf.app.flags.DEFINE_string('attention_type', 'bahdanau', 'Attention mechanism: (bahdanau, luong), default: bahdanau')
tf.app.flags.DEFINE_integer('hidden_units', 128, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('num_layers', 3, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 64, 'Embedding dimensions of encoder and decoder inputs')
tf.app.flags.DEFINE_integer("vocabulary_size", 64005, "Sequence vocabulary size in the mapping task.")

tf.app.flags.DEFINE_boolean('use_residual', True, 'Use residual connection between layers')
tf.app.flags.DEFINE_boolean('attn_input_feeding', False, 'Use input feeding method in attentional decoder')
tf.app.flags.DEFINE_boolean('use_dropout', True, 'Use dropout in each rnn cell')
tf.app.flags.DEFINE_float('dropout_rate', 0.3, 'Dropout probability for input/output/state units (0.0: no dropout)')

# Training parameters
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
tf.app.flags.DEFINE_float('min_learning_rate', 0.00002, 'minimum Learning rate')
tf.app.flags.DEFINE_integer('decay_steps', 3000, 'how many steps to update the learning rate.')
tf.app.flags.DEFINE_float("lr_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float('max_gradient_norm', 5.0, 'Clip gradients to this norm')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_integer('display_freq', 1, 'Display training status every this iteration')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop, cocob)')
tf.app.flags.DEFINE_string("model_dir", os.path.join(project_dir, 'data/models'), "Trained model directory.")
tf.app.flags.DEFINE_string('model_name', 'q2v', 'File name used for model checkpoints')
tf.app.flags.DEFINE_boolean('shuffle_each_epoch', True, 'Shuffle training dataset for each epoch')
tf.app.flags.DEFINE_boolean('sort_by_length', True, 'Sort pre-fetched minibatches by their target sequence lengths')
tf.app.flags.DEFINE_boolean('use_fp16', False, 'Use half precision float16 instead of float32 as dtype')

# Runtime parameters
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

tf.app.flags.DEFINE_integer("data_stream_port", None, "port for data zmq stream")
tf.app.flags.DEFINE_integer("source_maxlen", 30, "max number of words/tokens in each source sequence.")
tf.app.flags.DEFINE_integer("target_maxlen", 80, "max number of words/tokens in each target sequence.")

tf.app.flags.DEFINE_integer("beam_width", 0, "beam width")
tf.app.flags.DEFINE_integer("max_decode_step", 3, "max_decode_step")
tf.app.flags.DEFINE_boolean('debug', False, 'Enable debug')

FLAGS = tf.app.flags.FLAGS
