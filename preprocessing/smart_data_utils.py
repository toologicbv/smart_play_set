import numpy as np
from preprocessing.process_data import get_data
from utils.smart_utils import get_other_label
from tensorflow.contrib.learn.python.learn.datasets import base


def one_hot_encoding(label):
    """convert label from dense to one hot
      argument:
        label: ndarray dense label ,shape: [sample_num,1]
      return:
        one_hot_label: ndarray  one hot, shape: [sample_num,n_class]
    """
    label_num = label.shape[0]
    new_label = label.reshape(label_num)  # shape : [sample_num]
    new_label = new_label.astype(int)
    # because max is n_classes-1, we add 1
    n_values = np.max(new_label) + 1
    n_values = n_values.astype(int)
    one_hot_labels = np.eye(n_values)[new_label].astype(int)
    return one_hot_labels


def get_class_freq(data, labels):

    classes = np.unique(labels)
    N = data.shape[0]
    c_counts = []
    c_percs = []
    c_indices = {}
    for c in classes:
        c_indices[c] = np.where(labels==c)[0]
        class_count = np.sum(np.where(labels==c)[0].shape[0])
        c_perc = class_count / float(N)
        c_percs.append(c_perc)
        c_counts.append(class_count)
    return np.array(c_percs), np.array(c_counts), c_indices


def create_class_freqs(classes, class_percs, N):
    c_counts = []
    for i in np.arange(len(classes)-1):
        c_counts.append(int(np.ceil(N * class_percs[i])))

    c_counts.append(N - np.sum(np.array(c_counts)))
    return np.array(c_counts)


def create_train_test_set(e_date, data_label, split_perc=0.1, force=False, apply_window_func=False,
                          optimal_w_size=False, calc_mag=False, inc_game_state=False, inc_linear_acc=False,
                          use_level_labels = False,
                          f_type='low', lowcut=8, highcut=0.3, b_order=5,  binary=False, one_hot=False):

    if use_level_labels and binary:
        raise ValueError("ERROR - Combination use_level_labels & binary equal to TRUE is not allowed")

    data, labels, other_labels, dta_descr_dict = get_data(e_date=e_date, force=force,
                                                          apply_window_func=apply_window_func,
                                                          extra_label=data_label,
                                                          optimal_w_size=optimal_w_size, calc_mag=calc_mag,
                                                          f_type=f_type, lowcut=lowcut, highcut=highcut,
                                                          b_order=b_order,
                                                          nn_switch=True)

    if use_level_labels:
        # use labels that indicate the level of the game
        labels = get_other_label(other_labels, "level")
        # subtract 1 because level labels range from 1-3 and for one-hot encoding we better use 0-2
        labels -= 1
        labels = np.expand_dims(labels, axis=1)
        print("IMPORTANT - Using level labels instead of motor skill labels")

    # the original data contains 8 columns
    # 1-3: 3-axial acc data, butterworth filter/low 8Hz
    # 4: norm of column 1-3
    # 5-7: 3-axial acc data without gravity
    # 8: cosine similarity measure
    if binary:
        # the original data set has 3 classes, when binary is True we merge class 1 and 2 (motor deficiencies)
        # ending up with class 0 = normal motor skills; class 1 = likely motor deficiencies
        labels[labels == 2] = 1

    if not inc_game_state:
        # omit the fourth channel in the data which holds the cosine similarity data
        data = data[:, :, 0:-1]

    if not inc_linear_acc:
        # don't include 3 channels that contain only linear acceleration
        new_data = np.zeros((data.shape[0], data.shape[1], data.shape[2] - 3))
        new_data[:, :, 0:4] = data[:, :, 0:4]
        new_data[:, :, 4:] = data[:, :, 7:]
        data = new_data

    classes = np.unique(labels)
    N = data.shape[0]
    test_N = int(N * split_perc)
    train_N = N - test_N

    class_percs, class_counts, class_indices = get_class_freq(data, labels)
    new_cc_train = create_class_freqs(classes, class_percs, train_N)
    new_cc_test = class_counts - new_cc_train
    if data.ndim == 2:
        x_train = np.zeros((train_N, data.shape[1]))
        x_test = np.zeros((test_N, data.shape[1]))
    elif data.ndim == 3:
        x_train = np.zeros((train_N, data.shape[1], data.shape[2]))
        x_test = np.zeros((test_N, data.shape[1], data.shape[2]))
    else:
        raise ValueError("data tensor dimensionality %d is not supported" % data.ndim)
    y_train = np.zeros((train_N, labels.shape[1]))
    y_test = np.zeros((test_N, labels.shape[1]))
    print("total N/N-train/N-test: %d/%d/%d" % (N, train_N, test_N))
    print("Original class counts ", class_counts)
    print("Train class counts ", new_cc_train)
    print("Test class counts ", new_cc_test)
    start_tr, start_tt = [0, 0]
    for i,c in enumerate(classes):
        end_tr = start_tr + new_cc_train[i]
        end_tt = start_tt + new_cc_test[i]

        x_train[start_tr:end_tr] = data[class_indices[c][0:new_cc_train[i]]]
        y_train[start_tr:end_tr] = labels[class_indices[c][0:new_cc_train[i]]]
        x_test[start_tt:end_tt] = data[class_indices[c][new_cc_train[i]:new_cc_train[i]+new_cc_test[i]]]
        y_test[start_tt:end_tt] = labels[class_indices[c][new_cc_train[i]:new_cc_train[i]+new_cc_test[i]]]
        start_tr = end_tr
        start_tt = end_tt

    # normalize sets internally
    x_train -= np.mean(x_train, axis=(0, 1))
    x_train /= np.std(x_train, axis=(0, 1))
    x_test -= np.mean(x_test, axis=(0, 1))
    x_test /= np.std(x_test, axis=(0, 1))
    # print("----- Mean/Stddev of train/test set")
    # print(np.mean(x_train, axis=(0,1)), np.std(x_train, axis=(0,1)))
    # print(np.mean(x_test, axis=(0, 1)), np.std(x_test, axis=(0, 1)))
    # permute the two sets
    perm_tr = np.arange(train_N)
    perm_tt = np.arange(test_N)
    np.random.shuffle(perm_tr)
    np.random.shuffle(perm_tt)
    x_train = x_train[perm_tr]
    y_train = y_train[perm_tr]
    x_test = x_test[perm_tt]
    y_test = y_test[perm_tt]

    if one_hot:
        y_train = one_hot_encoding(y_train)
        y_test = one_hot_encoding(y_test)

    return base.Datasets(train=DataSet(x_train, y_train, one_hot), validation=None,
                         test=DataSet(x_test, y_test, one_hot))


class DataSet(object):
    """
    Utility class to handle dataset structure.
    """

    def __init__(self, x_data, labels, one_hot=False):
        """
        Builds dataset with images and labels.
        Args:
          images: Images data.
          labels: Labels data
        """
        assert x_data.shape[0] == labels.shape[0], (
            "images.shape: {0}, labels.shape: {1}".format(str(x_data.shape), str(labels.shape)))

        self._num_examples = x_data.shape[0]
        self._data = x_data
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        if one_hot:
            dense_labels = np.reshape(np.argmax(labels, axis=1), labels.shape[0], 1)
            self._class_percs, self._class_counts, self._class_indices = get_class_freq(self._data, dense_labels)
        else:
            self._class_percs, self._class_counts, self._class_indices = get_class_freq(self._data, self._labels)
        self._num_of_classes = len(self._class_counts)

    @property
    def class_percs(self):
        return self._class_percs

    @property
    def num_of_classes(self):
        return self._num_of_classes

    @property
    def class_counts(self):
        return self._class_counts

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """
        Return the next `batch_size` examples from this data set.
        Args:
          batch_size: Batch size.
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1

            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._data = self._data[perm]
            self._labels = self._labels[perm]

            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        # print(">>>>>>>>>>>>> NEXT BATCH: %d-%d" % (start, end))
        # x_batch, y_batch = self._data[start:end], self._labels[start:end]
        # batch_class_percs, batch_class_counts, _ = get_class_freq(x_batch, y_batch)
        # print("======> batch class dist: ", batch_class_counts)
        return self._data[start:end], self._labels[start:end]


# 20hz_1axis_low8hz_330_12_True   20hz_1axis_low8hz_4731_128_False
# "20hz_1axis_low8hz_3245_123_False"   20hz_1axis_low8hz_6377_64_False
# smart_play = create_train_test_set(e_date='20161206', data_label="20hz_1axis_low8hz_6267_128_False",
#                                   split_perc=0.15, binary=False, one_hot=True, use_level_labels=True,
#                                   optimal_w_size=True, inc_game_state=True, inc_linear_acc=False,
#                                   f_type="low", lowcut=8., highcut=0.)
# print(np.where(smart_play.train.labels==0)[0].shape[0])
# print(np.where(smart_play.train.labels==1)[0].shape[0])
# print(np.where(smart_play.train.labels==2)[0].shape[0])
# print(smart_play.train.data.shape)
# print(smart_play.test.data.shape)
# print(smart_play.train.num_of_classes)
