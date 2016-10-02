from __future__ import print_function

import os
import glob
import pandas as pd
import numpy as np
import h5py
from sklearn.preprocessing import normalize
# from datetime import datetime

from utils.smart_utils import apply_butter_filter, store_data, get_dir_path, load_data, make_data_description, \
                                get_file_label_info, split_on_classes
from settings import SAMPLE_FREQUENCY_FUTUROCUBE, DEBUG_LEVEL, WINDOW_SIZE, OVERLAP_COEFFICIENT, \
    FEATURE_LIST, GAME1, CUT_OFF_LENGTH, MEAN_FILE_LENGTH, RAW_DATA_ARRAY, LABELS

"""
    The following assumptions are made:

    Sampling frequency of futuro cube = 20 Hz

    Lay-out of Excel/csv files:
    ---------------------------
        first column        = index number
        second column       = x-axis (of accelerometer)
        third column        = y-axis
        fourth column       = z-axis
        fifth column        = error measure (dx + dy)

    Label information:
    ------------------
        should be incorporated into the filename
        assuming, one file contains ONE game with a specific game device, done by one person
        Example:
        =========
        filename = "20160907_roadrunner_futurocube_[ID5:0:age8]_acc.csv"
            # label info is contained in brackets [..:..] separated by dots
            # 1. ID of child/adult
            # 2. classification/fitness label (currently binary 0 = normal, 1 = derailed (bad)
            # 3. age of person
            # 4. female=0 or male=1
            # 5. left=0   or right=1 handed


    Sliding window approach:
    ------------------------
        The constant WINDOW_SIZE (specified in settings.py) determines length of window in SECONDS e.g. 6 sec
            - window length (measured in samples) = WINDOW_SIZE * SAMPLE_FREQUENCY (device)
            - for efficiency reasons (fft) we want the window length to be a multiple of 2
            - the constant OVERLAP_COEFFICIENT (e.g. 0.5) determines the level of overlap between windows
            - previous research revealed 50% to be a good factor
            - for calculation purposes we want all our experimental data to have the same number of windows
               therefore we use the constant MEAN_FILE_LENGTH to restrict the total length of the data (each file)
            - we assume, that all files have nearly the same length (e.g. 2500 samples)
            - calculating the
"""

# ================================= Procedures ======================================


def get_exprt_label(e_date, device='futurocube', game='roadrunner', s_label=''):

    # centralize the construction of the experiment labelling
    # used to store and load matrices

    return e_date + "_" + device + "_" + game + "_" + str(s_label)


def save_one_array(d_array, out_file, out_loc):
    output_file = out_loc + out_file + ".h5"
    h5f = h5py.File(output_file, 'w')
    h5f.create_dataset(RAW_DATA_ARRAY, data=d_array)
    h5f.close()


def extract_label_info(label_list, filename, num_windows):
    # current assumption about labels, please explanations above
    # we are only using the "fitness" indication which should be positioned at 2nd label position
    # at the moment to train the classifier

    labels = filename[filename.index('[') + 1:filename.index(']')].split(':')
    label_list.extend([int(labels[1]) for i in range(num_windows)])
    return label_list


def normalize_features(d_tensor, use_scikit=False):
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
    dim0 = d_tensor.shape[0]
    dim1 = d_tensor.shape[1]
    dim2 = d_tensor.shape[2]
    if not use_scikit:
        mean_feature = np.reshape(np.mean(np.mean(d_tensor, axis=2), axis=0), (1, d_tensor.shape[1], 1))
        std_x = np.reshape(np.sqrt(np.mean(np.mean(np.abs(d_tensor - mean_feature) ** 2, axis=2), axis=0)),
                           (1, d_tensor.shape[1], 1))
        d_tensor[:][:] -= mean_feature
        d_tensor[:][:] /= std_x
    else:
        d_tensor = normalize(np.reshape(d_tensor, (dim0, dim1 * dim2)), norm='l2')
        d_tensor = np.reshape(d_tensor, (dim0, dim1, dim2))

    return d_tensor


def convert_to_window_size(expt_data, win_size, overlap_coeff=OVERLAP_COEFFICIENT,
                           max_num_windows=20):
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


def calculate_features(d_tensor, window_func=False, d_axis=1, low_offset=0, high_offset=0):
    """

    :param d_tensor:
    :param d_axis: along which the features need to be calculated
    :return:
    """

    # Features of the time domain
    # maximum value for each of the features over the window
    # minimum value for each of the features
    # standard deviation

    if d_tensor.ndim == 3:
        dim1 = d_tensor.shape[0]
        dim2 = d_tensor.shape[1]  # this is the actual window size
        dim3 = d_tensor.shape[2]
    else:
        raise ValueError("tensor has less or more than 3 dimensions: %d" % d_tensor.ndim)

    minf = np.reshape(np.amin(d_tensor, axis=d_axis), (dim1, 1, dim3))
    maxf = np.reshape(np.amax(d_tensor, axis=d_axis), (dim1, 1, dim3))
    mean = np.reshape(np.mean(d_tensor, axis=d_axis), (dim1, 1, dim3))
    std = np.reshape(np.std(d_tensor, axis=d_axis), (dim1, 1, dim3))
    median = np.reshape(np.median(d_tensor, axis=d_axis), (dim1, 1, dim3))

    # Features of the frequency domain
    # First, apply Hamming window in order to prevent frequency leakage
    if window_func is not None:
        d_tensor = d_tensor * np.reshape(window_func, (1, len(window_func), 1))

    fd = np.fft.fft(d_tensor, axis=1)  # frequency domain
    # now skip fft coefficients that are outside of the low/high pass filter
    if low_offset != 0 or high_offset != 0:
        if low_offset != 0:
            fd = fd[:, 0:low_offset, :]
            dim2 = fd.shape[1]

        if high_offset != 0:
            fd = fd[:, high_offset:, :]
            dim2 = fd.shape[1]

    else:
        # no butterworth filtering
        pass

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
    res_tensor = np.concatenate((minf, maxf, mean, std, median, dc, energy, power_spec_entropy), axis=1)
    if DEBUG_LEVEL >= 1:
        print("INFO - calculating features -shape of result tensor ", res_tensor.shape)
        # print(res_tensor)

    return res_tensor


def import_data(edate, device, game, root_path, file_ext='csv', save_raw_files=True, calc_mag=False,
                f_type=None, lowcut=0., highcut=0., b_order=5,
                apply_window_func=False, extra_label='', optimal_w_size=True):
    """
        Parameters:
            device:
            game:
            root_path
            file_ext
            save_raw_files
            calc_mag: calculate the magnitude of the signal BEFORE computing the features!
            f_type: low, high, band or None
            lowcut:
            highcut:
            b_order: order of butterworth filter
            apply_window_func
            extra_label: a string that will be concatenated to form the filename of the
                         output file
    """

    if device == GAME1:  # futurocube specific processing steps
        freq = SAMPLE_FREQUENCY_FUTUROCUBE
        if optimal_w_size:
            # for efficiency reasons make window size a multiple of 2
            exp_2 = np.floor(np.log2(WINDOW_SIZE * freq))
            window_size_samples = int(2 ** exp_2)
        else:
            window_size_samples = int(WINDOW_SIZE * freq)

        feature_list = FEATURE_LIST

    else:
        print("NOT IMPLEMENTED YET")
        quit()

    # calculate the maximum length (in samples) per file
    signal_offset = int(CUT_OFF_LENGTH * freq)
    max_windows = np.floor(((MEAN_FILE_LENGTH - window_size_samples - signal_offset) /
                            float(OVERLAP_COEFFICIENT * window_size_samples)) + 1)
    max_file_length = window_size_samples + ((max_windows - 1) * OVERLAP_COEFFICIENT * window_size_samples)
    if DEBUG_LEVEL >= 1:
        print("-------------------------------------------------------------------------")
        print("INFO - Running feature calculation with the following parameter settings:")
        print("-------------------------------------------------------------------------")
        print("Game device *** %s  %s ***" % (device, game))
        print("Assuming sample frequency of device: %d" % freq)
        print("Cutting off the first samples: %g" % signal_offset)
        print("Calculated window size for feature extraction: %d" % window_size_samples)
        print("Length of window is approx %.2f secs" % (window_size_samples / float(freq)))
        print("Restrict # of windows per file to %d = %g seconds" % (max_windows, max_file_length))
        print("")
        print("Compute features based on signal magnitude %s" % calc_mag)
        print("IMPORTANT - Applying filtering - Butterworth type %s" % f_type)
        if f_type is not None:
            print("Filter details lowcut/highcut/order %g/%g/%g" % (lowcut, highcut, b_order))
        print("")
        print("IMPORTANT - Applying window function - Hamming: %s" % apply_window_func)
        print("")
        print("List of features")
        print(feature_list)
        print("---------------------------------------------------------------------------")
        print("")

    # only change directory if we are not already in ../data/... path
    if os.getcwd().find('data') == -1:
        os.chdir(root_path)
    abs_dir_path = os.getcwd() + "/"
    files_to_load = glob.glob(edate + "*." + file_ext)
    if DEBUG_LEVEL >= 1:
        print("INFO - Loading accelerometer %d files from: %s" % (len(files_to_load), abs_dir_path))
    num_of_files = 0
    num_of_files_skipped = 0
    feature_data = None
    label_data = []
    id_attributes = []

    if apply_window_func:
        hamming_w = np.hamming(window_size_samples)
    else:
        hamming_w = None

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

            # we assume that the beginning and the end of the raw signal contains to much
            # noise, therefore we are cutting of a piece in the beginning and in the end
            expt_data = expt_data[signal_offset:expt_data.shape[0] - signal_offset, :]
            # apply a butterworth filter if specified
            low_offset = 0
            high_offset = 0
            if f_type is not None:
                # apply butterworth filter or filters
                expt_data, _ = apply_butter_filter(expt_data, freq, lowcut, highcut, f_type, b_order)
                low_offset = int(freq * lowcut)
                high_offset = int(freq * highcut)
                if DEBUG_LEVEL >= 1:
                    print("INFO - Dimension of tensor after filtering ", expt_data.shape)

            # calculate signal magnitudes
            if calc_mag:
                expt_data = np.reshape(np.sqrt(expt_data[:, 0]**2 + expt_data[:, 1]**2 + expt_data[:, 2]**2),
                                            (expt_data.shape[0], 1))

            # segmentation of signal based on sliding window approach
            np_windows = convert_to_window_size(expt_data, win_size=window_size_samples, max_num_windows=max_windows)
            if save_raw_files:
                save_filename = f_name[:f_name.find(".")]
                save_one_array(np_windows, out_file=save_filename, out_loc=abs_dir_path)
            # previous function returns a numpy array with 3 axis:
            #   axis 0 = number of windows
            #   axis 1 = number of samples per window
            #   axis 2 = number of channels, e.g. 3 for accelerometer data (x,y,z axis)
            # we are calculating the features for the tuple(window/channel-axis)
            # and therefore aggregating over axis 1 (2nd parameter to calculate_features)
            np_windows = calculate_features(np_windows, window_func=hamming_w, d_axis=1, low_offset=low_offset,
                                            high_offset=high_offset)
            # concatenate the contents of the files (transformed as numpy arrays)
            if feature_data is None:
                feature_data = np_windows
            else:
                # Concatenate along axis 0...the windows
                feature_data = np.concatenate((feature_data, np_windows), axis=0)

            if DEBUG_LEVEL > 1:
                print("INFO - total length file=%d, num of windows=%d, num of features=%d, channels=%d" %
                  (expt_data.shape[0], np_windows.shape[0], np_windows.shape[1], np_windows.shape[2]))
            # get label information
            label_data = extract_label_info(label_data, f_name, np_windows.shape[0])
            # get dictionary with label info for this file
            label_dict = get_file_label_info(f_name)
            id_attributes.append(label_dict)

        except IOError as e:
            print('WARNING ****** Could not read:', acc_file, ':', e, '- it\'s ok, skipping. *******')
            num_of_files_skipped += 1

    # finally normalize the calculated features
    # Note: each feature is normalized separately, but over all 3 axis
    feature_data = normalize_features(feature_data, use_scikit=False)
    label_data = np.reshape(np.array(label_data), (len(label_data), 1))
    print("INFO - %d files loaded successfully! Skipped %d" % (num_of_files, num_of_files_skipped))
    if DEBUG_LEVEL > 1:
        print("INFO - Feature data shape ", feature_data.shape, " / label data shape ", label_data.shape)
    # finally store matrices in hdf5 format
    # data_label = get_exprt_label("{:%d%m%Y}".format(datetime.now()), device, game, extra_label, 1)
    # extend extra label with dimensions of feature data, that helps identifying the contents of the different
    # files
    extra_label = extra_label + "_" + str(feature_data.shape[0]) + "_" + str(feature_data.shape[1]) + "_" + \
                    str(feature_data.shape[2])
    data_label = get_exprt_label(edate, device, game, extra_label)
    d_dict = make_data_description(freq, window_size_samples, max_windows, apply_window_func, f_type,
                                   [lowcut, highcut, b_order], num_of_files, feature_list, id_attributes)
    store_data(feature_data, label_data, data_label, out_loc=abs_dir_path, descr=d_dict)
    return feature_data, label_data, d_dict


def get_data(e_date, device='futurocube', game='roadrunner', file_ext='csv', calc_mag=False,
             apply_window_func=True, extra_label='', force=False,
             f_type=None, lowcut=8, highcut=0.5, b_order=5, optimal_w_size=True):

    data_label = get_exprt_label(e_date, device, game, extra_label)
    root_dir = get_dir_path(device, game)
    if os.getcwd().find('data') == -1:
        os.chdir(root_dir)
    abs_file_path = os.path.join(os.getcwd(), data_label)
    if DEBUG_LEVEL >= 1:
        print("INFO - Used data label %s" % data_label)
    if os.path.isfile(abs_file_path + ".h5") and not force:
        if DEBUG_LEVEL >= 1:
            print("INFO Loading matrices from h5 file %s" % abs_file_path + ".h5")
            data, labels, d_dict = load_data(abs_file_path)
            return data, labels, d_dict
    else:
        if DEBUG_LEVEL >= 1:
            print("INFO - Need to process raw data...")
        return import_data(e_date, device, game, root_dir, file_ext, apply_window_func=apply_window_func,
                           calc_mag=calc_mag,
                           f_type=f_type, lowcut=lowcut, highcut=highcut, b_order=b_order,
                           extra_label=extra_label, optimal_w_size=optimal_w_size)


# train_data, train_labels, mydict = get_data('20160921', force=False, apply_window_func=True,
#                                            extra_label="20hz_1axis_try",
#                                             optimal_w_size=False, calc_mag=True,
#                                            f_type='lowhigh', lowcut=2, highcut=0.5, b_order=5)

# res = split_on_classes(train_data, train_labels)
