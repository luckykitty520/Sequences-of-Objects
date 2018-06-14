import tensorflow as tf
import numpy as np
import csv
import math
import sys

from models.deepLSTM import DeepLSTM


def input_object_sequence(file):

    sequence = []

    with open(file, 'rt') as f:
        reader = csv.reader(f)
        for row in reader:
            sequence.append(row)

    return np.array(sequence).astype(np.float32)


def format_object_sequence(original_object_sequence, object_number):

    x = []
    original_object_number = len(original_object_sequence)
    image_width = original_object_sequence[0][0]
    image_height = original_object_sequence[0][1]

    for i in np.arange(object_number):

        if i < original_object_number:
            temp = original_object_sequence[i]

        else:
            temp = []
            temp.append(image_width)
            temp.append(image_height)
            temp.extend([0., 0., 0., 0., 0., 0.])  # padding 0s

        x.append(temp)

    return np.array(x).astype(np.float32)


"""
    hyper parameters
"""
start_label = 0     # composition class
end_label = 2

object_number = 10     # for painting with insufficient objects, padding 0s
hidden_dim = 20
layer = 2

learning_rate = 0.01
dropout = True
keep_prob = 0.8  # useful when dropout = True


"""
    input the sequence of objects of demo painting & format it
"""
# input sequence of objects
sequence_path = './demo/' + str(sys.argv[1]).split('/')[-1][:-4] + '.csv'
object_sequence = input_object_sequence(sequence_path)

# format
x = format_object_sequence(object_sequence, object_number)
# print(x)
x = np.expand_dims(x, axis=0)
# print(np.shape(x))


"""
    new graph, restore parameters, predict 
"""
# new graph
model = DeepLSTM(input_dim=8, output_dim=end_label - start_label + 1,
                 seq_size=object_number,
                 hidden_dim=hidden_dim, layer=layer,
                 learning_rate=learning_rate, dropout=dropout)

# restore parameters & predict
saver = tf.train.Saver()
model_path = './checkpoints/SO.ckpt'

with tf.Session() as sess:
    tf.get_variable_scope().reuse_variables()  # share variable between time steps
    sess.run(tf.global_variables_initializer())

    # restore parameters
    saver.restore(sess, model_path)

    # predict
    prob = sess.run(model.softmax, feed_dict={model.x: x, model.keep_prob: 1.0})[0]

print('\n\n\n')
print('probability of lofty and remote:', prob[0])
print('probability of wide and remote:', prob[2])
print('probability of deep and remote:', prob[1])


