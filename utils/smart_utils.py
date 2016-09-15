import h5py
import numpy as np
import os
from data.config import config
from preprocessing.process_data import import_data, get_exprt_label
from preprocessing.settings import DEBUG_LEVEL, DATA_ARRAY, LABEL_ARRAY, FEATURE_ARRAY, DATA_DESCR


def get_dir_path(device, game):

    root_folder = config.get_datapath("SMART_PLAY_SET")
    return os.path.join(os.path.join(root_folder, device), game)


def load_hdf5_data(filename):

    with h5py.File(filename, 'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        data = hf.get(DATA_ARRAY)
        data = np.array(data)
        labels = hf.get(LABEL_ARRAY)
        labels = np.array(labels)
        feature_list = list(hf.get(FEATURE_ARRAY))
        descr = str(hf.get(DATA_DESCR))

    return data, labels, feature_list, descr


def get_data(e_date, device='futurocube', game='roadrunner', sequence=1, file_ext='csv', force=False):

    data_label = get_exprt_label(e_date, device, game, sequence)
    root_dir = get_dir_path(device, game)

    os.chdir(root_dir)
    abs_file_path = os.path.join(os.getcwd(), data_label) + ".h5"

    if os.path.isfile(abs_file_path) and not force:
        if DEBUG_LEVEL >= 1:
            print("INFO Loading matrices from h5 file %s" % abs_file_path)
            data, labels, feature_list, descr = load_hdf5_data(abs_file_path)
            print(feature_list)
            return data, labels
    else:
        if DEBUG_LEVEL >= 1:
            print("INFO - Need to process raw data...")
        return import_data(device, game, root_dir, file_ext)


train_data, train_labels = get_data('11092016', force=True)


