from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

ON_SERVER = False
if not ON_SERVER:
    sys.path.append("/mnt/disk2/git/repository/smart_play_set")
else:
    sys.path.append("/nfs/home6/gdemo059/smart_play_set")

from preprocessing.smart_data_utils import one_hot_encoding
from models.lstm_simpel import SimpleLSTM
import tensorflow as tf
import numpy as np


LEARNING_RATE_DEFAULT = 0.0025
BATCH_SIZE_DEFAULT = 1500
MAX_STEPS_DEFAULT = 1000
EVAL_FREQ_DEFAULT = 100
CHECKPOINT_FREQ_DEFAULT = 1000
PRINT_FREQ_DEFAULT = 100
OPTIMIZER_DEFAULT = 'adam'
WEIGHT_INITIALIZATION_DEFAULT = 'normal'
WEIGHT_INITIALIZATION_SCALE_DEFAULT = 1e-3
DFLT_DROPOUT_RT = 0.
LSTM_NUM_HIDDEN = 9
DEFAULT_DATA_LABEL = "20hz_1axis_low8hz_6377_64_False"
DEFAULT_MODEL = "slstm"

if not ON_SERVER:
    DATA_DIR_DEFAULT = '/mnt/disk2/git/repository/smart_play_set/data'
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

# different input channels for RNN, accelerometer, gyroscope, accelerometer without gravity butterworth filter
INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_"
    ]

# Output classes to learn how to classify
LABELS = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    ]


def extract_batch_size(_train, step, batch_size):
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index]

    return batch_s


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

    DATA_PATH = "data/"
    DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"
    TRAIN = "train/"
    TEST = "test/"

    # Load "X" (the neural network's training and testing inputs)
    def load_X(X_signals_paths):
        X_signals = []

        for signal_type_path in X_signals_paths:
            file = open(signal_type_path, 'rb')
            # Read dataset from disk, dealing with text files' syntax
            X_signals.append(
                [np.array(serie, dtype=np.float32) for serie in [
                    row.replace('  ', ' ').strip().split(' ') for row in file
                    ]]
            )
            file.close()

        return np.transpose(np.array(X_signals), (1, 2, 0))

    X_train_signals_paths = [
        DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
        ]
    X_test_signals_paths = [
        DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
        ]

    X_train = load_X(X_train_signals_paths)
    X_test = load_X(X_test_signals_paths)

    # Load data
    def load_y(y_path):
        file = open(y_path, 'rb')
        y_ = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in file
                ]],
            dtype=np.int32
        )
        file.close()

        # Subtract 1 in order to have class labels starting with 0
        return y_ - 1

    y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
    y_test_path = DATASET_PATH + TEST + "y_test.txt"

    y_train = load_y(y_train_path)
    y_test = load_y(y_test_path)

    # Input Data
    training_data_count = len(X_train)
    n_steps = len(X_train[0])  # 128 timesteps per series
    num_of_channels = len(X_train[0][0])  # 9 input parameters per timestep

    # LSTM Neural Network's internal structure
    n_classes = 6

    # Training
    lambda_loss_amount = 0.0015
    training_iters = training_data_count * 300  # Loop 300 times on the dataset
    batch_size = 1500
    display_iter = 15000  # To show test set accuracy during training

    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("(X shape, y shape, every X's mean, every X's standard deviation)")
    print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
    print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

    x_data = tf.placeholder(tf.float32, shape=(None, n_steps, num_of_channels))
    y_data = tf.placeholder(tf.float32, shape=(None, n_classes))

    model = SimpleLSTM(num_of_channels, n_classes, num_of_time_steps=n_steps,
                       num_hidden_units=FLAGS.num_hidden,
                       weight_initializer=WEIGHT_INITIALIZATION_DICT[FLAGS.weight_init],
                       weight_regularizer=tf.contrib.layers.regularizers.l2_regularizer(lambda_loss_amount),
                       dropout_rate=0., batch_normalization=True)

    pred = model.inference(x_data)
    # calculate probabilities for predictions
    probs_y = tf.nn.softmax(pred)
    # compute loss
    cost = model.loss_with_softmax(pred, y_data)
    # calculate accuracy
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_data, 1))
    accuracy = model.accuracy(pred, y_data)
    train_op = train_step(cost, get_grads=True)
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

    # Perform Training steps with "batch_size" iterations at each loop
    step = 1
    while step * batch_size <= training_iters:
        batch_xs = extract_batch_size(X_train, step, batch_size)
        batch_ys = one_hot_encoding(extract_batch_size(y_train, step, batch_size))

        # Fit training using batch data
        _, loss, acc = sess.run([train_op, cost, accuracy], feed_dict={x_data: batch_xs, y_data: batch_ys})

        # Evaluate network only at some steps for faster training:
        if (step * batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
            # To not spam console, show training accuracy/loss in this "if"
            print("Training iter #" + str(step * batch_size)  + ":   Batch Loss = " + "{:.6f}".format(loss) +
                  ", Accuracy = {}".format(acc))
            summary = sess.run(merge_summary, feed_dict={x_data: batch_xs, y_data: batch_ys})
            train_writer.add_summary(summary, step)
            # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
            loss, acc, summary = sess.run([cost, accuracy, merge_summary], feed_dict={x_data: X_test,
                                                                                      y_data: one_hot_encoding(y_test)})
            print("PERFORMANCE ON TEST SET: " + "Batch Loss = {}".format(loss) + ", Accuracy = {}".format(acc))
            test_writer.add_summary(summary, step)

        step += 1

    print("Optimization Finished!")

    # Accuracy for test data

    one_hot_predictions, accuracy, final_loss = sess.run(
        [pred, accuracy, cost],
        feed_dict={x_data: X_test, y_data: one_hot_encoding(y_test)})

    print("FINAL RESULT: " + "Batch Loss = {}".format(final_loss) + ", Accuracy = {}".format(accuracy))


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
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--v', type=int, default=MAX_STEPS_DEFAULT,
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

# python models/lstm_simpel.py --max_steps=500 --data_label=20hz_1axis_low8hz_6377_64_False --model_name=slstm4
