import tensorflow as tf
import numpy as np
import csv
import math
import sys
from tensorflow.contrib import rnn


class DeepLSTM:

    def __init__(self, input_dim, output_dim,
                 seq_size, hidden_dim, layer, learning_rate, dropout):

        # Hyperparameters
        self.input_dim = input_dim      # input dim for each step
        self.output_dim = output_dim  # output dim for last step, that is class number
        self.seq_size = seq_size    # step number, that is, object number
        self.hidden_dim = hidden_dim    # hidden dim in each cell, input gate, forget gate, output gate, hidden state, output, all weights and biases
        self.layer = layer              # deep of lstm
        self.learning_rate = learning_rate
        self.dropout = dropout

        # Weight variables and placeholders
        self.W_out = tf.Variable(tf.random_normal([hidden_dim, output_dim]), name='W_out')
        self.b_out = tf.Variable(tf.random_normal([output_dim]), name='b_out')

        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])  # input data
        self.y = tf.placeholder(tf.float32, [None, output_dim])  # ground truth class, one hot representation
        self.keep_prob = tf.placeholder(tf.float32)

        self.y_hat = self.model()       # output class score, before softmax
        self.softmax = tf.nn.softmax(self.y_hat)

        # Cost optimizer
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_hat)
        self.loss = tf.reduce_mean(cross_entropy)

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        # assess
        correct_pred = tf.equal(tf.argmax(self.softmax, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # saver
        self.saver = tf.train.Saver()

    def get_a_cell(self):

        if self.dropout:
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)   # only dropout input between layers, no dropout between memory
            return lstm_cell
        else:
            return rnn.BasicLSTMCell(self.hidden_dim)

    def model(self):
        """
        :param x: inputs of size [N, seq_size, input_size]
        :param W_out: matrix of fully-connected output layer weights
        :param b_out: vector of fully-connected output layer biases
        """
        # stack cell
        cell = rnn.MultiRNNCell([self.get_a_cell() for _ in range(self.layer)])

        # initial state with 0s
        batch_size = tf.shape(self.x)[0]
        h0 = cell.zero_state(batch_size, tf.float32)

        # outputs:  all outputs of the last layer and in all time steps , [batch_size, seq_size, hidden_dim]
        # states: hidden states in the last time step,
        #         layer * LSTMStateTuple(hidden_state, output), both are [batch_size, hidden_dim]
        if self.dropout:
            self.x_drop = tf.nn.dropout(self.x, self.keep_prob)
            outputs, states = tf.nn.dynamic_rnn(cell, self.x_drop, dtype=tf.float32, initial_state=h0)
        else:
            outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32, initial_state=h0)
        last_output = outputs[:, -1, :]

        out = tf.matmul(last_output, self.W_out) + self.b_out

        return out

    def get_batch(self, input, output, batch_size, mode):

        if mode == 'train':

            # random select index in np.arange(len(X))
            # length = batch_size
            # replace indicate whether choosing repeatedly, false means can not
            index = np.random.choice(len(input), batch_size, replace=False)
            return input[index], output[index]

        elif mode == 'test':
            return input[:batch_size], output[:batch_size]
        else:
            sys.exit()

    def train_test(self, training_input, training_output, training_name,
                         test_input, test_output, test_name,
                         batch_size_train, batch_size_test,
                         epoch):

        iteration_train = int(len(training_input) / batch_size_train)
        iteration_test = int(len(test_input) / batch_size_test)

        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()   # share variable between time steps
            sess.run(tf.global_variables_initializer())

            for i in range(epoch):

                # training
                for j in range(iteration_train):

                    input_batch, output_batch = self.get_batch(training_input, training_output, batch_size_train,
                                                               mode='train')
                    _, loss = sess.run([self.optimizer, self.loss], feed_dict={self.x: input_batch, self.y: output_batch})

                if i % 100 == 0:
                    print('epoch: {0}, training loss = {1}'.format(i, loss))

                    # test
                    accuracy = 0.
                    count = 0
                    for j in range(iteration_test):

                        input_batch, output_batch = self.get_batch(test_input, test_output, batch_size_test,
                                                                   mode='test')

                        count += 1
                        predictions = sess.run(self.softmax, feed_dict={self.x: input_batch, self.keep_prob: 1.0})
                        accuracy += self.average_test(predictions, output_batch)

                    print('test accuracy =', accuracy/count)
                    print('\n')

    def train_test_dropout(self, training_input, training_output, training_name,
                                 test_input, test_output, test_name,
                                 batch_size_train, batch_size_test,
                                 epoch, keep_prob):

        iteration_train = int(len(training_input) / batch_size_train)
        iteration_test = int(len(test_input) / batch_size_test)

        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()  # share variable between time steps
            sess.run(tf.global_variables_initializer())

            for i in range(epoch):

                # training
                for j in range(iteration_train):
                    input_batch, output_batch = self.get_batch(training_input, training_output, batch_size_train,
                                                               mode='train')
                    _, loss = sess.run([self.optimizer, self.loss],
                                       feed_dict={self.x: input_batch, self.y: output_batch, self.keep_prob: keep_prob})

                if i % 100 == 0:
                    print('epoch: {0}, training loss = {1}'.format(i, loss))

                    # test
                    accuracy = 0.
                    count = 0
                    for j in range(iteration_test):
                        input_batch, output_batch = self.get_batch(test_input, test_output, batch_size_test,
                                                                   mode='test')

                        count += 1
                        predictions = sess.run(self.softmax, feed_dict={self.x: input_batch, self.keep_prob: 1.0})
                        accuracy += self.average_test(predictions, output_batch)

                    print('test accuracy =', accuracy / count)
                    print('\n')

    def average_test(self, predictions, ground_truth):

        accuracy = 0.

        # predictions: [None, class_number], confidence
        # ground_truth: [None, class_number], one-hot
        for i in np.arange(0, len(predictions), 2):

            average_predictions = (predictions[i] + predictions[i+1]) / 2.0
            if np.argmax(average_predictions) == np.argmax(ground_truth[i]):
                accuracy += 1.0

        return accuracy / (len(predictions) / 2.0)




# # test
# model = DeepLSTM(input_dim=8, output_dim=3,
#                  seq_size=20,
#                  hidden_dim=10, layer=2,
#                  learning_rate=0.01, dropout=True)






