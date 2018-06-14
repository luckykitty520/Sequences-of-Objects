import numpy as np
import csv


def input_data(path, label_start, label_end):
    data = []

    label_number = label_end - label_start + 1
    for i in np.arange(label_number):
        data.append([])

    with open(path, 'rt') as f:
        reader = csv.reader(f)
        for row in reader:
            temp = './data/paint_augmentation/' + str(row[0]) + '/' + row[1]
            data[ int(row[0]) - label_start ].append(temp)

    return data


def construct_cross_validation(data, fold):

    train_data = []
    test_data = []

    for i in np.arange(len(data)):

        # divide train & test
        test_number = int(len(data[i]) / 10)
        temp = data[i]

        train_temp = temp[0:fold * test_number]
        train_temp = train_temp + temp[(fold + 1) * test_number:]

        test_temp = temp[fold * test_number:(fold + 1) * test_number]

        # flip horizontally
        train_temp_flip = []
        for item in train_temp:

            train_temp_flip.append(item)

            flip_item = item[:-4] + "_f.jpg"
            train_temp_flip.append(flip_item)

        test_temp_flip = []
        for item in test_temp:

            test_temp_flip.append(item)

            flip_item = item[:-4] + "_f.jpg"
            test_temp_flip.append(flip_item)

        train_data.append(train_temp_flip)
        test_data.append(test_temp_flip)

    return train_data, test_data


if __name__ == '__main__':

    label_start = 0
    label_end = 2

    fold = 0    # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (10-fold cross validation)

    ###########################
    # input data [label, path]
    data = input_data('./data/title_label.csv', label_start, label_end)
    print(len(data))
    for i in np.arange(len(data)):
        print(len(data[i]))
        print(data[i][0])
    print('\n\n')

    # 10-fold cross validation
    # select int( count[i] / 10 ) in each class
    # train data, test data [label, path]
    # every image is flipped horizontally
    train_data, test_data = construct_cross_validation(data, fold)
    print(len(train_data))
    for i in np.arange(len(train_data)):
        print(len(train_data[i]))
        print(train_data[i][0])
        print(train_data[i][1])
    print('\n')
    print(len(test_data))
    for i in np.arange(len(test_data)):
        print(len(test_data[i]))
        print(test_data[i][0])
        print(test_data[i][1])
    print('\n\n')

    ###########################
    # output
    path = './data/train/' + str(fold) + '.csv'
    with open(path, 'w', newline='') as out:
        writer = csv.writer(out)
        for i in np.arange(len(train_data)):
            for j in np.arange(len(train_data[i])):
                results = []
                results.append(train_data[i][j])
                results.append(i)
                writer.writerow(results)

    path = './data/test/' + str(fold) + '.csv'
    with open(path, 'w', newline='') as out:
        writer = csv.writer(out)
        for i in np.arange(len(test_data)):
            for j in np.arange(len(test_data[i])):
                results = []
                results.append(test_data[i][j])
                results.append(i)
                writer.writerow(results)













