import h5py
import numpy as np
import pandas as pd
import os
import glob
import json
from scipy.signal import butter, lfilter
from data.config import config
from preprocessing.settings import DEBUG_LEVEL, DATA_ARRAY, LABEL_ARRAY, RAW_DATA_ARRAY, CUT_OFF_LENGTH, \
                                    MEAN_FILE_LENGTH, OVERLAP_COEFFICIENT, LEVEL_TIME_INTERVALS, LABELS


def create_row_mask(row_set, length):
    return np.array(row_set * length, dtype=bool)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def apply_butter_filter(signal, fs, lowcut, highcut, f_type, order):
    """
        the butterworth filter is always applied to the last axis of a tensor
        in the procedures above (e.g. butter_lowpass_filter) the "lfilter" functions
        has the default "axis=-1"
    :param signal:
    :param fs:
    :param lowcut:
    :param highcut:
    :param f_type:
    :param order:
    :return:
    """

    if f_type == 'band':
        y = butter_bandpass_filter(signal, lowcut, highcut, fs, order=order)
        p_label = 'Filtered signal (bandpass: %g/%g Hz order %d)' % (lowcut, highcut, order)
    elif f_type == 'low':
        y = butter_lowpass_filter(signal, lowcut, fs, order=order)
        p_label = 'Filtered signal (lowpass: %g Hz order %d)' % (lowcut, order)
    elif f_type == 'high':
        y = butter_highpass_filter(signal, highcut, fs, order=order)
        p_label = 'Filtered signal (highpass: %g Hz order %d)' % (highcut, order)
    elif f_type == 'lowhigh':
        y_t = butter_lowpass_filter(signal, lowcut, fs, order=order)
        y = butter_highpass_filter(signal, highcut, fs, order=order)
        p_label = 'Filtered signal (low/high: %g/%g Hz order %d)' % (lowcut, highcut, order)

    if signal.ndim <= 2:
        y = np.reshape(y, (signal.shape[0], 1))
    elif signal.ndim == 3:
        y = np.reshape(y, (signal.shape[0], signal.shape[1], signal.shape[2]))
    else:
        # Error number of dimensions is not supported, to be implemented
        print("**** ERROR: Number of dimensions of signal is not supported ***")
    return y , p_label


def load_file_to_pandas(device, game, filename, abs_path=False):

    file_ext = filename[(filename.find(".") + 1):]
    if not abs_path:
        root_dir = get_dir_path(device, game)
        df = None
        if root_dir.replace("..", "") not in os.getcwd():

            os.chdir(root_dir)

        abs_dir_path = os.getcwd() + "/"
        filename = os.path.join(abs_dir_path, filename)

    if file_ext == 'csv':
        df = pd.read_csv(filename)
    elif file_ext == 'xlsx':
        excel_file = pd.ExcelFile(filename)
        df = excel_file.parse(excel_file.sheet_names[0])
    # only pass column 1,2 and 3 (x, y and z axis)

    return df.iloc[:, 1:4]


def tensor_to_pandas(d_tensor):
    d_tensor = d_tensor.reshape((d_tensor.shape[0] * d_tensor.shape[1], d_tensor.shape[2]))
    return pd.DataFrame(d_tensor, columns=['x', 'y', 'z'])


def get_dir_path(device, game):

    root_folder = config.get_datapath("SMART_PLAY_SET")
    return os.path.join(os.path.join(root_folder, device), game)


def load_hdf5_file(filename):

    with h5py.File(filename + ".h5", 'r') as hf:
        data = hf.get(RAW_DATA_ARRAY)
        data = np.array(data)
    return data


def load_data(filename):

    with h5py.File(filename + ".h5", 'r') as hf:
        if DEBUG_LEVEL >= 1:
            print('INFO - List of arrays in this file: \n', hf.keys())
        data = hf.get(DATA_ARRAY)
        data = np.array(data)
        labels = hf.get(LABEL_ARRAY)
        labels = np.array(labels)

    with open(filename + ".json", 'r') as fp:
        if DEBUG_LEVEL >= 1:
            print('INFO - Loading data description from json.')
        dta_dict = json.load(fp)

    return data, labels, dta_dict


def store_data(f_data, l_data, out_file, out_loc, descr='None'):

    output_file_hd5 = out_loc + out_file + ".h5"
    h5f = h5py.File(output_file_hd5, 'w')
    h5f.create_dataset(DATA_ARRAY, data=f_data)
    h5f.create_dataset(LABEL_ARRAY, data=l_data)
    h5f.close()

    # store the dictionary that contains the description of the data by means of json
    output_file_json = out_loc + out_file + ".json"
    with open(output_file_json, 'w') as fp:
        json.dump(descr, fp)

    if DEBUG_LEVEL >= 1:
        print("INFO - Successfully saved data to %s" % (out_loc + out_file))


def get_array_filenames(e_date, device='futurocube', game='roadrunner', sequence=1, file_ext='csv'):

        root_dir = get_dir_path(device, game)
        if os.getcwd().find('data') == -1:
            os.chdir(root_dir)
        abs_dir_path = os.getcwd() + "/"
        files_to_load = glob.glob(e_date + "*." + file_ext)
        filenames = []
        for f_name in files_to_load:
            filenames.append(os.path.join(abs_dir_path, f_name))

        return filenames


def make_data_description(fs, w_size, num_windows, window_func, b_filter, filter_specs, num_of_files, features,
                          id_attributes=[]):

    d_dict = {
        "frequency": fs,
        "window_size": w_size,
        "num_of_windows": int(num_windows),
        "window_func": window_func,
        "filter": b_filter,
        "filter_specs": filter_specs,
        "num_of_files": int(num_of_files),
        "features": features,
        "cut_off_sec": CUT_OFF_LENGTH,
        "MEAN_FILE_LENGTH": MEAN_FILE_LENGTH,
        "OVERLAP_COEFFICIENT": OVERLAP_COEFFICIENT,
        "id_attributes": id_attributes,
        "LEVEL_TIME_INTERVALS": LEVEL_TIME_INTERVALS,
        "LABELS": LABELS
        }
    return d_dict


def get_file_label_info(filename):
    labels = filename[filename.index('[') + 1:filename.index(']')].split(':')
    file_dict = {}
    for i, label in enumerate(labels):
        file_dict[LABELS[i]] = label

    return file_dict


def split_on_classes(d_tensor, d_labels, num_classes=2):
    res_dict = {}
    for cls in np.arange(num_classes):
        res_dict[cls] = d_tensor[d_labels[:, 0] == cls]

    return res_dict

# a = get_array_filenames('20160921', device='futurocube', game='roadrunner', sequence=1, file_ext='csv')


