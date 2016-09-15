from __future__ import print_function

import os
import glob
import pandas as pd
import numpy as np
import scipy.stats as sc
import h5py
from datetime import datetime

from settings import SAMPLE_FREQUENCY_FUTUROCUBE, DEBUG_LEVEL, WINDOW_SIZE, OVERLAP_COEFFICIENT, \
    MAX_NUM_WINDOWS, FEATURE_LIST, DATA_ARRAY, LABEL_ARRAY, FEATURE_ARRAY, DATA_DESCR, GAME1

"""
    The following assumptions are made:

    Sampling frequency of futuro cube = 70 Hz

    Lay-out of Excel/csv files:
    ---------------------------
        first column        = index number
        second column       = x-axis (of accelerometer)
        third column        = y-axis
        fourth column       = z-axis

    Label information:
    ------------------
        should be incorporated into the filename
        assuming, one file contains ONE game with a specific game device, done by one person
        Example:
        =========
        filename = "20160907_roadrunner_futurocube_[ID5:0:age8]_acc.csv"
            # label info is contained in brackets [..:..] separated by dots
            # 1. ID of child/adult
            # 2. classification/fitness label (currently binary 0 = normal, 1 = unnormal (bad)
            # 3. age of person
"""

# ================================= Procedures ======================================


def get_exprt_label(e_date, device='futurocube', game='roadrunner', sequence=1):

    # centralize the construction of the experiment labelling
    # used to store and load matrices

    return e_date + "_" + device + "_" + game + "_" + "s" + str(sequence)


def store_data_as_hdf5(f_data, l_data, out_file, features, out_loc, descr='None'):

    output_file = out_loc + out_file + ".h5"
    h5f = h5py.File(output_file, 'w')
    h5f.create_dataset(DATA_ARRAY, data=f_data)
    h5f.create_dataset(LABEL_ARRAY, data=l_data)
    h5f.create_dataset(FEATURE_ARRAY, data=features)
    h5f.create_dataset(DATA_DESCR, data=descr)
    h5f.close()

    if DEBUG_LEVEL >= 1:
        print("INFO - Successfully saved data to %s" % output_file)


def extract_label_info(label_list, filename, num_windows):
    # current assumption about labels, please explanations above
    # we are only using the "fitness" indication at the moment to train the classifier
    #

    labels = filename[filename.index('[') + 1:filename.index(']')].split(':')
    label_list.extend([labels[1] for i in range(num_windows)])
    return label_list


def normalize_features(d_tensor):
    """
    Assuming the passed tensor contains 3 axis
    axis 0: number of samples
    axis 1: number of features
    axis 2: channels/xyz-axis of accelerometer

    Normalization per feature, over all 3 channels
    :param d_tensor:
    :return: normalized tensor over all samples and features
    """
    # first average over all channels (axis=2) then over all samples (axis=0)
    # reshape result to fit for broadcasting operation
    mean_feature = np.reshape(np.mean(np.mean(d_tensor, axis=2), axis=0), (1, d_tensor.shape[1], 1))
    std_x = np.reshape(np.sqrt(np.mean(np.mean(np.abs(d_tensor - mean_feature) ** 2, axis=2), axis=0)),
                       (1, d_tensor.shape[1], 1))
    d_tensor[:][:] -= mean_feature
    d_tensor[:][:] /= std_x

    return d_tensor


def convert_to_window_size(expt_data, win_size, overlap_coeff=OVERLAP_COEFFICIENT,
                           max_num_windows=MAX_NUM_WINDOWS):
    """
    Creates chunks of original raw data. Chunk size = window size
    :param expt_data:
    :param win_size:
    :param overlap_coeff
    :param max_num_windows:
    :return:
    """
    window_lists = []
    total_samples = expt_data.shape[0]
    num_samples_so_far = 0
    while num_samples_so_far < total_samples:
        window = expt_data[num_samples_so_far:num_samples_so_far + win_size]
        if window.shape[0] < win_size:
            break
        window_lists.append(window)
        # remember, we want the time frames (windows) to overlap
        # previous research has revealed that a 50% overlap works well
        # whether this will work in our case as well?
        num_samples_so_far += int(win_size * overlap_coeff)

    num_of_windows = len(window_lists)
    if num_of_windows > max_num_windows:
        # cut off some blocks
        window_lists = window_lists[:max_num_windows]
    else:
        # if experimental data contains less than maximum num of windows
        # then we just pass what we have so far
        pass
    return np.array(window_lists)


def calculate_features(d_tensor, d_axis=1):
    """

    :param d_tensor:
    :param d_axis: along which the features need to be calculated
    :return:
    """

    # Features of the time domain
    # maximum value for each of the features over the window
    # minimum value for each of the features
    # standard deviation

    dim1 = d_tensor.shape[0]
    dim2 = d_tensor.shape[1]  # this is the actual window size
    dim3 = d_tensor.shape[2]

    maxf = np.reshape(np.amax(d_tensor, axis=d_axis), (dim1, 1, dim3))
    minf = np.reshape(np.amin(d_tensor, axis=d_axis), (dim1, 1, dim3))
    mean = np.reshape(np.mean(d_tensor, axis=d_axis), (dim1, 1, dim3))
    std = np.reshape(np.std(d_tensor, axis=d_axis), (dim1, 1, dim3))
    median = np.reshape(np.median(d_tensor, axis=d_axis), (dim1, 1, dim3))

    # Features of the frequency domain
    fd = np.fft.fft(d_tensor, axis=1)  # frequency domain
    # DC or zero Hz component, is the first component of the N (window sample size) components
    # -----------------------
    # for each window (first axis) take the first component, reshape so we can stack later
    dc = np.reshape(np.real(fd[:, 0]), (dim1, 1, dim3))
    # Power Spectrum  = Power Spectral Density (PSD)
    # ----------------------------------------------
    # When computing the PS we omit the DC component
    # We end up with (N-1) component values (again N is window size)
    # But isn't it necessary to average(?) over these values c.q. the window values??? ask Pascal
    power_spec_not_avg = np.abs(fd[:, 1:]) ** 2

    # Energy (following Bao and Intille, is this correct?)
    # ----------------------------------------------------
    # Questions:
    #   according to Boa et al. we need to omit the DC component, so do we need to normalize by N-1 right?
    energy = 1/(float(dim2) - 1) * np.reshape(np.sum(np.abs(fd[:, 1:]) ** 2, axis=d_axis), (dim1, 1, dim3))

    # Power Spectral Entropy (PSE)
    # ----------------------------------
    # (1) first step, normalize the PSD. Note, we are summing over the window axis, therefore we end up with a
    #       sum-component for each window and channel/axis which is used as normalizing coefficient
    norm_power_spec = power_spec_not_avg / np.reshape(np.sum(power_spec_not_avg, axis=d_axis), (dim1, 1, dim3))
    # (2) We then average again the PSE over the window size for each window/channel
    #       calculate entropy, don't use scipy.stats because we can't influence the summing over axis
    #        use numpy log function with base "e"
    power_spec_entropy = - np.sum(norm_power_spec * np.log(norm_power_spec), axis=d_axis)
    # (3) Reshape in order to stack features at the end
    power_spec_entropy = np.reshape(power_spec_entropy, (dim1, 1, dim3))

    # concatenate the features along axis 1, which is 1 for all tensors
    res_tensor = np.concatenate((maxf, minf, mean, std, median, dc, energy, power_spec_entropy), axis=1)
    if DEBUG_LEVEL >= 1:
        print("INFO - calculating features -shape of result tensor ", res_tensor.shape)
        # print(res_tensor)

    return res_tensor


def import_data(device, game, root_path, file_ext='csv'):
    """
        Parameters:
            root_path
            file_ext
            w_size: size of sliding window in seconds (or fraction of second
    """

    if device == GAME1:  # futurocube specific processing steps
        sample_frequency = SAMPLE_FREQUENCY_FUTUROCUBE
        # for efficiency reasons make window size a multiple of 2
        exp_2 = np.ceil(np.log2(WINDOW_SIZE * sample_frequency))
        window_size_samples = int(2 ** exp_2)
        feature_list = FEATURE_LIST

    else:
        print("NOT IMPLEMENTED YET")
        quit()

    if DEBUG_LEVEL >= 1:
        print("-------------------------------------------------------------------------")
        print("INFO - Running feature calculation with the following parameter settings:")
        print("-------------------------------------------------------------------------")
        print("Game device *** %s  %s ***" % (device, game))
        print("Assuming sample frequency of device: %d" % sample_frequency)
        print("Calculated window size for feature extraction %d" % window_size_samples)
        print("Length of window is approx %.2f secs" % (window_size_samples / float(sample_frequency)))
        print("Restrict # of windows per file to %d" % MAX_NUM_WINDOWS)
        print("")
        print("List of features")
        print(feature_list)
        print("-------------------------------------------------------------------------")
        print("")

    # only change directory if we are not already in ../data/... path
    if os.getcwd().find('data') == -1:
        os.chdir(root_path)
    abs_dir_path = os.getcwd() + "/"
    files_to_load = glob.glob("*." + file_ext)
    if DEBUG_LEVEL >= 1:
        print("INFO - Loading accelerometer %d files from: %s" % (len(files_to_load), abs_dir_path))
    num_of_files = 0
    num_of_files_skipped = 0
    feature_data = None
    label_data = []

    for f_name in files_to_load:
        acc_file = os.path.join(abs_dir_path, f_name)
        if DEBUG_LEVEL > 1:
            print("INFO - processing file %s" % acc_file)
        try:
            # read file to pandas dataframe, acc_file contains the absolute path to import file
            # can process csv-format of xlsx file extension. in the last case
            # only the first sheet is imported
            df = None
            if file_ext == 'csv':
                df = pd.read_csv(acc_file)
            elif file_ext == 'xlsx':
                excel_file = pd.ExcelFile(acc_file)
                df = excel_file.parse(excel_file.sheet_names[0])
            num_of_files += 1
            # convert the pandas dataframe to a numpy array
            # currently assuming that the dataframe columns 1 = x-axis, 2 = y-axis, 3 = z-axis
            # remember, indexing of columns starts at 0
            expt_data = df.iloc[:, 1:4].as_matrix()
            # dimensionality of expt_data = (total number of samples, num of channels(x,y,z))
            np_windows = convert_to_window_size(expt_data, win_size=window_size_samples)

            # previous function returns a numpy array with 3 axis:
            #   axis 0 = number of windows
            #   axis 1 = number of samples per window
            #   axis 2 = number of channels, e.g. 3 for accelerometer data (x,y,z axis)
            # we are calculating the features for the tuple(window/channel-axis)
            # and therefore aggregating over axis 1 (2nd parameter to calculate_features)
            np_windows = calculate_features(np_windows, 1)
            # concatenate the contents of the files (transformed as numpy arrays)
            if feature_data is None:
                feature_data = np_windows
            else:
                # Concatenate along axis 0...the windows
                feature_data = np.concatenate((feature_data, np_windows), axis=0)

            if DEBUG_LEVEL > 1:
                print("INFO - total length=%d, num of windows=%d, samples/window=%d, channels=%d" %
                  (expt_data.shape[0], np_windows.shape[0], np_windows.shape[1], np_windows.shape[2]))
            # get label information
            label_data = extract_label_info(label_data, f_name, np_windows.shape[0])

        except IOError as e:
            print('WARNING ****** Could not read:', acc_file, ':', e, '- it\'s ok, skipping. *******')
            num_of_files_skipped += 1

    # finally normalize the calculated features
    # Note: each feature is normalized separately, but over all 3 axis
    feature_data = normalize_features(feature_data)
    label_data = np.reshape(np.array(label_data), (len(label_data), 1))
    print("INFO - %d files loaded successfully! Skipped %d" % (num_of_files, num_of_files_skipped))
    if DEBUG_LEVEL > 1:
        print("INFO - Feature data shape ", feature_data.shape, " / label data shape ", label_data.shape)
    # finally store matrices in hdf5 format
    data_label = get_exprt_label("{:%d%m%Y}".format(datetime.now()), device, game, 1)
    store_data_as_hdf5(feature_data, label_data, data_label, feature_list, out_loc=abs_dir_path)
    return feature_data, label_data
