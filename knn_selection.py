import time
from copy import deepcopy
import numpy as np
import plotly.express as px


def get_accuracy(feature_label, data):
    correct = 0
    for i in range(0, len(data)):
        cur_feature_arr = np.tile(data[i], (len(data), 1))
        matrix_pow = np.power(cur_feature_arr - data, 2)
        distance_sum = np.sum(matrix_pow, axis=1)  #.reshape(len(data), 1)
        every_euclidean_distance = np.sqrt(distance_sum)
        nearest_neighbor_location = every_euclidean_distance.argsort()[1]
        nearest_neighbor_label = feature_label[nearest_neighbor_location]
        if feature_label[i] == nearest_neighbor_label:
            correct += 1
    accuracy = correct / len(data) * 100
    return round(accuracy, 1)


def forward_selection(feature_type, feature_data, feature_cnt):
    print('Beginning search.')
    idx_list = list(range(1, feature_cnt + 1))
    max_accuracy = list()
    max_accuracy_idx = -1
    high_accuracy_idxes = [list()]
    for i in range(1, feature_cnt + 1):
        cur_max_accuracy = 0
        cur_high_accuracy_idxes = None

        for v in idx_list:
            labels_set = [v] + high_accuracy_idxes[-1]  # if len(high_accuracy_idxes) > 0 else [])
            labels_set.sort()
            labels_idx = [i - 1 for i in labels_set]
            acc = get_accuracy(feature_type, feature_data[:, labels_idx])
            if acc > cur_max_accuracy:
                cur_max_accuracy = acc
                cur_high_accuracy_idxes = labels_set
            print('Using feature(s) {} accuracy is {}%'.format(labels_set, acc))
        print('Feature set {} was best, accuracy is {}%'.format(cur_high_accuracy_idxes, cur_max_accuracy))
        max_accuracy_idx = i - 1 if len(max_accuracy) == 0 or cur_max_accuracy > max(max_accuracy) else max_accuracy_idx
        max_accuracy.append(cur_max_accuracy)
        high_accuracy_idxes.append(cur_high_accuracy_idxes)

        for v in cur_high_accuracy_idxes:
            if v in idx_list:
                idx_list.remove(v)
    print('Finished search!! The best feature subset is {}, which has an accuracy of {}%'.format(high_accuracy_idxes[max_accuracy_idx + 1], max(max_accuracy)))
    return max_accuracy, high_accuracy_idxes[1:]


def backward_elimination(feature_type, feature_data, feature_cnt):
    print('Beginning search.')
    idx_list = list(range(1, feature_cnt + 1))
    max_accuracy = list()
    max_accuracy_idx = -1
    high_accuracy_idxes = list()
    q = [idx_list]
    while len(q) > 0:
        cur_max_accuracy = 0
        cur_high_accuracy_idxes = None
        for labels_set in q:
            labels_idx = [i - 1 for i in labels_set]
            acc = get_accuracy(feature_type, feature_data[:, labels_idx])
            if acc > cur_max_accuracy:
                cur_max_accuracy = acc
                cur_high_accuracy_idxes = labels_set
            print('Using feature(s) {} accuracy is {}%'.format(labels_set, acc))
        print('Feature set {} was best, accuracy is {}%'.format(cur_high_accuracy_idxes, cur_max_accuracy))
        max_accuracy_idx = len(max_accuracy) if len(max_accuracy) == 0 or cur_max_accuracy > max(max_accuracy) else max_accuracy_idx
        max_accuracy.append(cur_max_accuracy)
        high_accuracy_idxes.append(cur_high_accuracy_idxes)
        q.clear()
        for i in range(0, len(high_accuracy_idxes[-1])):
            t = deepcopy(high_accuracy_idxes[-1])
            t.pop(i)
            if len(t) > 0:
                q.append(t)

    print('Finished search!! The best feature subset is {}, which has an accuracy of {}%'.format(
        high_accuracy_idxes[max_accuracy_idx], max(max_accuracy)))
    return max_accuracy, high_accuracy_idxes


def gen_graph(x, acc_data, title):
    x_labels = [(str(i)) for i in x]
    # x_labels = list()
    # t = 0
    # for i in x:
    #     if len(i) < 5:
    #         x_labels.append(str(i))
    #     else:
    #         x_labels.append(str(t))
    #         t += 1

    my_plt = px.bar(x=x_labels, y=acc_data,
                    title=title,
                    labels={'x': 'current_feature_set', 'y': 'accuracy'})
    my_plt.show()


if __name__ == '__main__':
    print('Welcome to Bertie Woosters Feature Selection Algorithm.')
    # input file
    file_name = input('Type in the name of the file to test : ')
    file_path = 'Project_2_data/{}'.format(file_name)
    data = np.loadtxt(file_path)
    data_size = len(data)

    classified_type = data[:, 0]
    normalized_data = data[:, 1:]  # stats.zscore(data[:, 1:])
    features_cnt = len(normalized_data[0])

    # Choose selection method
    search_method = input('Type the number of the algorithm you want to run. \n 1) Forward Selection \n 2) Backward Elimination \n ')
    print('This dataset has {} features (not including the class attribute), with {} instances.'.format(features_cnt, len(normalized_data)))
    time_start = time.time()
    max_accuracy_list, max_accuracy_idxes_list = list(), list()
    method_name = None
    if search_method == '1':
        max_accuracy_list, max_accuracy_idxes_list = forward_selection(classified_type, normalized_data, features_cnt)
        method_name = 'Forward Selection'
    elif search_method == '2':
        max_accuracy_list, max_accuracy_idxes_list = backward_elimination(classified_type, normalized_data, features_cnt)
        method_name = 'Backward Elimination'
    time_end = time.time()
    print('Search running time: {}'.format(time_end - time_start))

    gen_graph(max_accuracy_idxes_list, max_accuracy_list, method_name)
