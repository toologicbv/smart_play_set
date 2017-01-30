from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
import numpy as np

ON_SERVER=False
MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
UPDATE_OPS_COLLECTION = 'convnet_update_ops'


class SimpleLSTM(object):

    """model a LSTM Network,
      it stacks 2 LSTM layers, each layer has n_hidden=32 cells
       and 1 output layer, it is a full connet layer
      argument:
        feature_mat: numpy array feature matrix, shape=[batch_size,time_steps,n_inputs]
        config: class containing config of network
      return:
              : matrix  output shape [batch_size,n_classes]
    """
    def __init__(self, num_of_channels, num_of_classes, num_of_time_steps, dropout_rate=0.,
                 num_hidden_units=32,
                 weight_initializer=tf.random_normal_initializer(stddev=0.001),
                 weight_regularizer=None,
                 batch_normalization=False,
                 is_training=True):
        # weight_regularizer=tf.contrib.layers.regularizers.l2_regularizer(0.001)
        # LSTM structure
        self.n_inputs = num_of_channels  # Features count is of 9: three 3D sensors features over time
        self.n_hidden = num_hidden_units  # nb of neurons inside the neural network
        self.n_classes = num_of_classes  # Final output classes
        self.dropout_rate = dropout_rate
        self.weight_initializer = weight_initializer
        self.weight_regularizer = weight_regularizer
        # added extra class parameter indicating whether or not to use batch normalization
        self.batch_normalization = batch_normalization
        self.num_of_steps = num_of_time_steps
        self.is_training = is_training

    def _batch_norm_wrapper(self, inputs):

        x_shape = inputs.get_shape()
        params_shape = x_shape[-1:]
        axis = list(range(len(x_shape) - 1))

        beta = tf.get_variable(name='beta',
                               shape=params_shape,
                               initializer=tf.zeros_initializer)
        if ON_SERVER:
            gamma = tf.get_variable(name='gamma',
                                    shape=params_shape,
                                    initializer=tf.ones_initializer)
        else:
            gamma = tf.get_variable(name='gamma',
                                    shape=params_shape,
                                    initializer=tf.ones_initializer)

        moving_mean = tf.get_variable(name='moving_mean',
                                      shape=params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        if ON_SERVER:
            moving_variance = tf.get_variable(name='moving_variance',
                                              shape=params_shape,
                                              initializer=tf.ones_initializer,
                                              trainable=False)
        else:
            moving_variance = tf.get_variable(name='moving_variance',
                                              shape=params_shape,
                                              initializer=tf.ones_initializer,
                                              trainable=False)
        mean, variance = tf.nn.moments(inputs, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                                   mean, BN_DECAY)
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, BN_DECAY)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
        if_cond = tf.equal(True, self.is_training)
        mean, variance = control_flow_ops.cond(
            if_cond, lambda: (mean, variance),
            lambda: (moving_mean, moving_variance))

        inputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, BN_EPSILON)
        return inputs

    def inference(self, X):
        """

        :param X: is a 3D-tensor dim0=batch_size, dim1=num_time_steps, dim2=input-feature-dim
        :return:
        """

        # Exchange dim 1 and dim 0
        feature_mat = tf.transpose(X, [1, 0, 2])
        # New feature_mat's shape: [time_steps, batch_size, n_inputs]

        # Temporarily crush the feature_mat's dimensions
        feature_mat = tf.reshape(feature_mat, [-1, self.n_inputs])
        # New feature_mat's shape: [time_steps*batch_size, n_inputs]

        # Linear activation, reshaping inputs to the LSTM's number of hidden:
        with tf.variable_scope("input") as scope:
            weights = tf.get_variable('weights', shape=[self.n_inputs, self.n_hidden],
                                      initializer=self.weight_initializer,
                                      regularizer=self.weight_regularizer)
            # No regularizer on bias...
            biases = tf.get_variable('biases', shape=[self.n_hidden], initializer=tf.constant_initializer(0.0))

            feature_mat = tf.matmul(feature_mat, weights) + biases
            if_cond = tf.equal(True, self.is_training)
            linear_out = tf.cond(if_cond, lambda: tf.nn.dropout(feature_mat, (1. - self.dropout_rate)),
                                lambda: feature_mat)
            tf.histogram_summary(weights.name, weights)
            # New feature_mat's shape: [time_steps*batch_size, n_hidden]

        # Split the series because the rnn cell needs time_steps features, each of shape:
        linear_out = tf.split(0, self.num_of_steps, linear_out)
        # linear_out is a tensor of shape [batch_size, n_hidden]

        # Define LSTM cell of first hidden layer:
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0)

        # Stack two LSTM layers, both layers has the same shape
        lstm_layers = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2)
        # lstm_layers = tf.nn.rnn_cell.MultiRNNCell([lstm_cell])

        # Get LSTM outputs
        outputs, _ = tf.nn.rnn(lstm_layers, linear_out, dtype=tf.float32, scope="LSTM")
        # outputs' shape [batch_size, n_classes]
        if self.batch_normalization:
            encoded_state = self._batch_norm_wrapper(outputs[-1])
        else:
            encoded_state = outputs[-1]
        # Linear activation
        # Get the last output tensor of shape [batch_size, n_classes]
        with tf.variable_scope("output") as scope:
            weights = tf.get_variable('weights', shape=[self.n_hidden, self.n_classes],
                                      initializer=self.weight_initializer,
                                      regularizer=self.weight_regularizer)
            # No regularizer on bias...
            biases = tf.get_variable('biases', shape=[self.n_classes], initializer=tf.constant_initializer(0.0))
            tf.histogram_summary(weights.name, weights)
            a_out = tf.matmul(encoded_state, weights) + biases
            if_cond = tf.equal(True, self.is_training)
            final_out = tf.cond(if_cond, lambda: tf.nn.dropout(a_out, (1. - self.dropout_rate)),
                                 lambda: a_out)

        return final_out

    def loss_with_softmax(self, logits, labels):
        # cross entropy
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf.to_float(labels), name='cross-entropy')
        # get the regularization terms
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.reduce_mean(cross_entropy, name='entropy_mean') + np.sum(reg_losses)
        tf.scalar_summary("loss", loss)
        return loss

    def accuracy(self, logits, labels):

        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.scalar_summary("accuracy", accuracy)

        return accuracy


