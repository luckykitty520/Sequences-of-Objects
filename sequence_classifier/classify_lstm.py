import tensorflow as tf
import numpy as np
import csv
import math


from models.deepLSTM import DeepLSTM


def input_data(file):

    data = dict()

    with open(file, 'rt') as f:
        reader = csv.reader(f)
        for row in reader:
            key = row[0].split('/')[-1]
            value = row[1:]
            data[key] = value

    return data


def one_hot_representation(class_label, start_label, end_label):

    vector = []
    for i in np.arange(end_label - start_label + 1):
        if i == class_label - start_label:
            vector.append(1.0)
        else:
            vector.append(0.0)

    return vector


def construct_objects(image_width, image_height, object_sequence, object_number):

    objects = []

    real_object_number = int(len(object_sequence) / 6.0)

    for i in np.arange(object_number):

        temp = []

        if i < real_object_number:
            temp.append(image_width)
            temp.append(image_height)
            temp.extend(object_sequence[i * 6: i * 6 + 6])

        else:
            temp.append(image_width)
            temp.append(image_height)
            temp.extend([0., 0., 0., 0., 0., 0.])   # padding 0s

        objects.append(temp)

    return objects


def construct_format_data(data, start_label, end_label, object_number):

    input_data = []
    output_data = []
    image_name = []

    for image in data:

        # image_name
        image_name.append(image)

        # output, one hot representation
        image_class = int(data[image][0])
        image_one_hot = one_hot_representation(image_class, start_label, end_label)
        output_data.append(image_one_hot)

        # input
        image_width = data[image][1]
        image_height = data[image][2]
        # [object_number, a_object]
        objects = construct_objects(image_width, image_height, data[image][3:], object_number)
        input_data.append(objects)

    return np.array(input_data).astype(np.float32), np.array(output_data).astype(np.float32), image_name


def shuffle(input, output, name):

    # re-order np.arrange(len(name))
    permutation = np.random.permutation(len(name))

    shuffle_input = []
    shuffle_output = []
    shuffle_name = []

    for i in permutation:

        shuffle_input.append(input[i])
        shuffle_output.append(output[i])
        shuffle_name.append(name[i])

    return np.array(shuffle_input), np.array(shuffle_output), shuffle_name


"""
    hyper parameters
"""
fold = 7  # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (10-fold cross validation)
training_file = './datasets/train/' + str(fold) + '.csv'
test_file = './datasets/test/' + str(fold) + '.csv'

start_label = 0     # composition class
end_label = 2

object_number = 10     # for painting with insufficient objects, padding 0s
hidden_dim = 20
layer = 2

learning_rate = 0.01
epoch = 5000
batch_size_train = 256
batch_size_test = 52

dropout = True
keep_prob = 0.8  # useful when dropout = True

"""
    input training & inference data
    format to lstm input & output shape
"""
##########################################
# input training & inference data
# dict, key = image name, value = feature
# feature: [composition class, width, height, objects...]
# object: [object class,
#          top left corner_x, top left corner_y,
#          bottom right corner_x, bottom right corner_y,
#          confidence]
training_data = input_data(training_file)
print(len(training_data))
test_data = input_data(test_file)
print(len(test_data))
print('\n\n')


##########################################
# construct input & output
# training_input: [None, object_number, a_object]
# a_object: [painting_width, painting_height,
#            object class,
#            top left corner_x, top left corner_y,
#            bottom right corner_x, bottom right corner_y,
#            confidence]
#
# training_output: one-hot representation
#
# training_name: image name
training_input, training_output, training_name = construct_format_data(training_data,
                                                                       start_label, end_label,
                                                                       object_number)
print(np.shape(training_input))
# print(training_input[0])
print(np.shape(training_output))
# print(training_output[0])
print(np.shape(training_name))
# print(training_name[0])
print('\n')


# test data
test_input, test_output, test_name = construct_format_data(test_data,
                                                           start_label, end_label,
                                                           object_number)
print(np.shape(test_input))
# print(test_input[0])
print(np.shape(test_output))
# print(test_output[0])
print(np.shape(test_name))
# print(test_name[0])
print('\n')


"""
    training & test
"""
# shuffle training data
training_input, training_output, training_name = shuffle(training_input, training_output, training_name)
print(np.shape(training_input))
# print(training_input[0])
print(np.shape(training_output))
# print(training_output[0])
print(np.shape(training_name))
# print(training_name[0])
print('\n')


# new model
model = DeepLSTM(input_dim=8, output_dim=end_label - start_label + 1,
                 seq_size=object_number,
                 hidden_dim=hidden_dim, layer=layer,
                 learning_rate=learning_rate, dropout=dropout)

# train & test
if not dropout:
    model.train_test(training_input, training_output, training_name,
                     test_input, test_output, test_name,
                     batch_size_train=batch_size_train, batch_size_test=batch_size_test,
                     epoch=epoch)

else:
    model.train_test_dropout(training_input, training_output, training_name,
                             test_input, test_output, test_name,
                             batch_size_train=batch_size_train, batch_size_test=batch_size_test,
                             epoch=epoch, keep_prob=keep_prob)




