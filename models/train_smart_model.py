from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import sys

ON_SERVER = False
if not ON_SERVER:
    sys.path.append("/mnt/disk2/git/repository/smart_play_set")
else:
    sys.path.append("/nfs/home6/gdemo059/smart_play_set")

from preprocessing.smart_data_utils import create_train_test_set, get_class_freq
from models.lstm_simpel import SimpleLSTM
import tensorflow as tf
import numpy as np


LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 1000
MAX_EPOCHS_DEFAULT = 300
EVAL_FREQ_DEFAULT = BATCH_SIZE_DEFAULT * 15
CHECKPOINT_FREQ_DEFAULT = 15000
PRINT_FREQ_DEFAULT = BATCH_SIZE_DEFAULT * 15
OPTIMIZER_DEFAULT = 'adam'
WEIGHT_INITIALIZATION_DEFAULT = 'normal'
WEIGHT_INITIALIZATION_SCALE_DEFAULT = 1e-3
DEFAULT_REG_TERM = 0.1
DFLT_DROPOUT_RT = 0.
LSTM_NUM_HIDDEN = 32
DEFAULT_DATA_LABEL = "20hz_1axis_high03hz_6377_64_False"
DEFAULT_MODEL = "smart_lstm"

if not ON_SERVER:
    DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
    LOG_DIR_DEFAULT = '/mnt/disk2/git/repository/smart_play_set/models/logs'
    CHECKPOINT_DIR_DEFAULT = '/mnt/disk2/git/repository/smart_play_set/models/checkpoints'
else:
    DATA_DIR_DEFAULT = '/nfs/home6/gdemo059/smart_play_set/data'
    LOG_DIR_DEFAULT = '/nfs/home6/gdemo059/smart_play_set/models/logs/cifar10'
    CHECKPOINT_DIR_DEFAULT = '/nfs/home6/gdemo059/smart_play_set/models/checkpoints'


CHECKPT_FILE = 'lstm_model.ckpt'

OPTIMIZER_DICT = {'sgd': tf.train.GradientDescentOptimizer,     # Gradient Descent
                  'adadelta': tf.train.AdadeltaOptimizer,       # Adadelta
                  'adagrad': tf.train.AdagradOptimizer,         # Adagrad
                  'adam': tf.train.AdamOptimizer,               # Adam
                  'rmsprop': tf.train.RMSPropOptimizer          # RMSprop
                  }

WEIGHT_INITIALIZATION_DICT = {
        'xavier': tf.contrib.layers.xavier_initializer(), # Xavier initialisation
        'normal': tf.random_normal_initializer(stddev=WEIGHT_INITIALIZATION_SCALE_DEFAULT),
        'uniform': tf.random_uniform_initializer(minval=-WEIGHT_INITIALIZATION_SCALE_DEFAULT,
                                                 maxval=WEIGHT_INITIALIZATION_SCALE_DEFAULT)
    }

# Those are separate normalized input features for the neural network
INPUT_SIGNAL_TYPES = [
        "raw_cube_acc_x_",
        "raw_cube_acc_y_",
        "raw_cube_acc_z_",
        "linear_cube_acc_x_",
        "linear_cube_acc_y_",
        "linear_cube_acc_z_",
        "game_cos_sim"
    ]

# Classes to discriminate
LABELS = [
        "Normal",
        "Deficiency 1",
        "Deficiency 2"
    ]


class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self, dataset):
        # Input data
        # time_steps per series, num of sliding windows (batch_size, steps=# of measurements inside one sliding window)
        self.n_steps = dataset.train.data.shape[1]
        self.num_of_channels = dataset.train.data.shape[2]
        # Training
        self.learning_rate = FLAGS.learning_rate
        self.lambda_reg = DEFAULT_REG_TERM  # not in use
        self.training_steps = FLAGS.max_epochs * dataset.train.data.shape[0]
        self.batch_size = FLAGS.batch_size
        self.dropout_perc = FLAGS.drop_out
        self.num_of_classes = dataset.train.num_of_classes


def train_step(loss, get_grads=False):
    """
    Defines the ops to conduct an optimization step. You can set a learning
    rate scheduler or pick your favorite optimizer here. This set of operations
    should be applicable to both ConvNet() and Siamese() objects.

    Args:
        loss: scalar float Tensor, full loss = cross_entropy + reg_loss
        get_grads: log gradients, but only for a subset of the variables
    Returns:
        train_op: Ops for optimization.
    """

    t_vars = tf.trainable_variables()
    # all_vars = [var.name for var in t_vars]
    # print(">>>>>>>>>> all trainable variables", all_vars)

    if not get_grads:
        train_vars = t_vars
    else:
        train_vars = t_vars
        log_grads  = [var for var in t_vars if ('weights' in var.name or 'Matrix' in var.name)]
        log_var_names = [var.name for var in log_grads]

    optimizer = OPTIMIZER_DICT[OPTIMIZER_DEFAULT](FLAGS.learning_rate)
    # train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
    #                                     "scope/prefix/for/first/vars")
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

    if get_grads:
        # Compute the gradients for a list of variables.
        grads_and_vars = optimizer.compute_gradients(loss, train_vars)
        # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
        # need to the 'gradient' part, for example cap them, etc.
        capped_grads_and_vars = grads_and_vars
        # for var in capped_grads_and_vars:
        #    print("Trainable variable ", var[1].name)
        # Ask the optimizer to apply the capped gradients.
        train_op = optimizer.apply_gradients(capped_grads_and_vars, global_step=global_step)

        for var in capped_grads_and_vars:
            if var[1].name in log_var_names:
                print("**** Log gradient *** %s" % var[1].name)
                tf.histogram_summary(var[1].name + '/gradient', var[0])
    else:
        train_op = optimizer.minimize(loss, var_list=train_vars, global_step=global_step)

    return train_op


def train():
    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    # the original data contains 8 columns
    # 1-3: 3-axial acc data, butterworth filter/low 8Hz
    # 4: norm of column 1-3
    # 5-7: 3-axial acc data without gravity
    # 8: cosine similarity measure
    smart_play = create_train_test_set(e_date='20161206', data_label=FLAGS.data_label,
                                       split_perc=0.25, binary=False, one_hot=True, inc_game_state=True,
                                       inc_linear_acc=True, use_level_labels=True)
    config = Config(smart_play)
    print("---- Mean/stddev of data dimensions ----")
    print(smart_play.test.data.shape, smart_play.test.labels.shape,
          np.mean(smart_play.test.data, axis=(0, 1)), np.std(smart_play.test.data, axis=(0, 1)))

    def create_feed_dict(train=True, dropout_prob=config.dropout_perc):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train:
            xs, ys = smart_play.train.next_batch(config.batch_size)

        else:
            xs, ys = smart_play.test.data, smart_play.test.labels

        return {x_data: xs, y_data: ys, drop_prob: dropout_prob, is_training: train}

    x_data = tf.placeholder(tf.float32, shape=(None, config.n_steps, config.num_of_channels))
    y_data = tf.placeholder(tf.float32, shape=(None, config.num_of_classes))
    drop_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)

    model = SimpleLSTM(config.num_of_channels, config.num_of_classes, num_of_time_steps=config.n_steps,
                       num_hidden_units=FLAGS.num_hidden, is_training=is_training,
                       weight_initializer=WEIGHT_INITIALIZATION_DICT[FLAGS.weight_init],
                       weight_regularizer=tf.contrib.layers.regularizers.l2_regularizer(config.lambda_reg),
                       dropout_rate=drop_prob, batch_normalization=FLAGS.use_bn)
    pred_y = model.inference(x_data)
    # calculate probabilities for predictions
    probs_y = tf.nn.softmax(pred_y)
    # compute loss
    loss = model.loss_with_softmax(pred_y, y_data)
    # calculate accuracy
    acc = model.accuracy(pred_y, y_data)
    train_op = train_step(loss, get_grads=True)
    # Build the summary tensor based on the TF collection of summaries in order to use Tensorboard
    merge_summary = tf.merge_all_summaries()

    # Add the variable initializer Op.
    init = tf.initialize_all_variables()
    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    # Create a session for running Ops on the Graph.
    sess = tf.Session()
    # Instantiate a SummaryWriter to output summaries and the Graph.
    train_writer = tf.train.SummaryWriter(FLAGS.log_dir + "/train", sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test')

    # Run the Op to initialize the variables.
    tf.get_default_graph().finalize()
    sess.run(init)
    chkpoint_file = os.path.join(FLAGS.checkpoint_dir, CHECKPT_FILE)

    best_accuracy = 0.0
    # Start training for each batch and loop epochs
    step = 1
    while step * config.batch_size <= config.training_steps:
        start_time = time.time()
        train_feed_dict = create_feed_dict(train=True, dropout_prob=config.dropout_perc)
        total_steps = step * config.batch_size
        if total_steps % FLAGS.print_freq == 0 or (total_steps + 1) == config.training_steps:
            # train & evaluate the model
            loss_value, _, summary, acc_train = sess.run([loss, train_op, merge_summary, acc],
                                                         feed_dict=train_feed_dict)
            train_writer.add_summary(summary, step)
            duration = time.time() - start_time
            print('Train eval - step %d: loss = %.4f (%.3f sec) - train acc %.3f' %
                  (total_steps, loss_value, duration, acc_train))
        else:
            # just train and compute loss
            loss_value, _ = sess.run([loss, train_op], feed_dict=train_feed_dict)

        # Evaluate on test set
        if total_steps % FLAGS.eval_freq == 0 or (total_steps + 1) == config.training_steps:
            # BE AWARE, always use dropout 0 when evaluating
            test_feed_dict = create_feed_dict(train=False, dropout_prob=0.)
            summary, loss_test, acc_test = sess.run([merge_summary, loss, acc], feed_dict=test_feed_dict)
            print('Test eval: --- test loss: %.4f ----- test accuracy: %0.04f' % (loss_test, acc_test))
            best_accuracy = max(best_accuracy, acc_test)
            test_writer.add_summary(summary, step)

        # save a checkpoint of the model
        if step % FLAGS.checkpoint_freq == 0 or (total_steps + 1) == config.training_steps:
            saver.save(sess, chkpoint_file)  # , global_step=step)

        step += 1

    print("Optimization Finished!")
    print("")
    print("best epoch's test accuracy: {0:.3f}".format(best_accuracy))
    print("")


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def initialize_folders():
    """
    Initializes all folders in FLAGS variable.
    """

    FLAGS.log_dir = os.path.join(FLAGS.log_dir, FLAGS.model_name)
    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    FLAGS.checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name)
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)


def main(_):
    print_flags()

    initialize_folders()
    train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_label', type=str, default=DEFAULT_DATA_LABEL,
                        help='Data label which identifies the data to load from file')
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL,
                        help='Default model name, used to create log-dirs and checkpoints dir')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--drop_out', type=float, default=DFLT_DROPOUT_RT,
                        help='Dropout rate')
    parser.add_argument('--max_epochs', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--print_freq', type=int, default=PRINT_FREQ_DEFAULT,
                        help='Frequency of evaluation on the train set')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--num_hidden', type=int, default=LSTM_NUM_HIDDEN,
                        help='Number of hidden units of LSTM.')
    parser.add_argument('--log_dir', type=str, default=LOG_DIR_DEFAULT,
                        help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR_DEFAULT,
                        help='Checkpoint directory')
    parser.add_argument('--checkpoint_freq', type=int, default=CHECKPOINT_FREQ_DEFAULT,
                        help='Frequency with which the model state is saved.')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER_DEFAULT,
                        help='Optimizer to use [sgd, adadelta, adagrad, adam, rmsprop].')
    parser.add_argument('--use_bn', action='store_true',
                        help='Indicating whether convnet should use batch-norm between conv-blocks')
    parser.add_argument('--weight_init', type=str, default=WEIGHT_INITIALIZATION_DEFAULT,
                        help='Weight initialization type [xavier, normal, uniform].')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()

# python models/train_smart_model.py --max_epochs=300 --data_label=20hz_1axis_high03hz_6377_64_False --model_name=smart_lstm1

# 20hz_1axis_low8hz_12812_32_False  1.5 second window
# 20hz_1axis_low8hz_6377_64_False     3 second window
# 20hz_1axis_low8hz_3135_128_False    6 seconds
# 20hz_1axis_low8hz_1540_256_False   12.5 second window

