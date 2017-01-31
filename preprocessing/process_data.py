from __future__ import print_function

import os
import glob
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from scipy.ndimage import convolve1d

from utils.smart_utils import apply_butter_filter, store_data, get_dir_path, load_data, make_data_description, \
                                get_file_label_info, FuturoCube, calc_cos_sim
from settings import config

"""
    The following assumptions are made:

    Sampling frequency of futuro cube = 20 Hz

    Lay-out of Excel/csv files:
    ---------------------------
        first column        = sequence number
        second column       = x-axis (of accelerometer)
        third column        = y-axis
        fourth column       = z-axis
        fifth column        = cube index of walker

    Label information:
    ------------------
        should be incorporated into the filename
        assuming, one file contains ONE game with a specific game device, done by one person
        Example:
        =========
        filename = "20161206_futurocube_roadrunner_[ID1:0:Age7:1:1:A]_acc.csv"
            # label info is contained in brackets [..:..] separated by dots
            # 1. ID of child/adult
            # 2. classification/fitness label (currently binary 0 = normal, 1 = likely to have delayed fine motor skills
                                                                            2 = very likely delayed fine motor skills
            # 3. age of person
            # 4. female=0 or male=1
            # 5. left=0   or right=1 handed
            # 6. Permutations of levels of difficulty:
                A, B, C, D, E, F, G, H, I, J

                A:  1-2-3-1-2-3
                B:  1-2-3-1-3-2
                C:  1-2-3-2-3-1
                D:  1-2-3-2-1-3

                E:  1-3-2-1-2-3
                F:  1-3-2-1-3-2
                G:  1-3-2-3-1-2
                H:  1-3-2-3-2-1

                I:  1-2-1-3-2-3

                J:  1-3-1-2-3-2



    Sliding window approach:
    ------------------------
        The constant WINDOW_SIZE (specified in settings.py) determines length of window in SECONDS e.g. 6 sec
            - window length (measured in samples) = WINDOW_SIZE * SAMPLE_FREQUENCY (device)
            - for efficiency reasons (fft) we want the window length to be a multiple of 2
            - the constant OVERLAP_COEFFICIENT (e.g. 0.5) determines the level of overlap between windows
            - previous research revealed 50% to be a good factor
            - for calculation purposes we want all our experimental data to have the same number of windows
               therefore we use the constant MEAN_FILE_LENGTH to restrict the total length of the data (each file)
            - we assume, that all files have nearly the same length (e.g. last experiments 3750 samples, 20.8 Hz)
            - Note: in the final analysis, sliding window was not used (OVERLAP_COEFFICIENT = 1), used fix size
              window of 30 seconds.
"""

# ================================= Procedures ======================================

Cube = FuturoCube()
game_level_dict = {'A': [1, 2, 3, 1, 2, 3],
                   'B': [1, 2, 3, 1, 3, 2],
                   'C': [1, 2, 3, 2, 3, 1],
                   'D': [1, 2, 3, 2, 1, 3],
                   'E': [1, 3, 2, 1, 2, 3],
                   'F': [1, 3, 2, 1, 3, 2],
                   'G': [1, 3, 2, 3, 1, 2],
                   'H': [1, 3, 2, 3, 2, 1],
                   'I': [1, 2, 1, 3, 2, 3],
                   'J': [1, 3, 1, 2, 3, 2],
                   'W': [1, 2, 3, 1, 2, 3]  # actually W=warm-up uses slightly uncomparable levels, not usable as label
                }


def calc_cosine_similarity(cube_index_tensor, raw_acc_signal):
    """
    Calculate the cosine similarity measure between walker and assumed upward position of the cube.
    Please see final report for a more detailed description.

    :param cube_index_tensor: 2D tensor that holds walker location (index number) for each sample (num of windows
    x window size).
    :param raw_acc_signal:
    :return: 3D tensor (num of windows, window size, 1) of cosine similarity measures
    """
    dim1 = cube_index_tensor.shape[0]  # number of windows (segmentation)
    dim2 = cube_index_tensor.shape[1]  # size of each window
    cos_sims = np.zeros((dim1, dim2, 1))
    for w in np.arange(dim1):
        idxs = cube_index_tensor[w]
        # look up the coordinate vectors in Cube
        sq_vecs = Cube.f_g_map[idxs.astype(int)]
        # unfortunately we also need to iterate through window samples each
        # because cos-sim function only accepts 1d vectors
        for s in np.arange(dim2):
            cos_sims[w, s, :] = calc_cos_sim(sq_vecs[s], raw_acc_signal[w, s])

    return cos_sims


def get_exprt_label(e_date, device='futurocube', game='roadrunner', s_label=''):
    """
    centralize the construction of the experiment labelling
    used to store and load matrices
        :param e_date: YYYYMMDD identifying the date of the experiments
        :param device: constant for now
        :param game: constant for now
        :param s_label: extra suffix label to identify run
    """

    return e_date + "_" + device + "_" + game + "_" + str(s_label)


def extract_label_info(label_list1, other_labels, filename, num_windows, windows_end_indices=None):
    """
    Extract for each file the label information which is explained in the header of this file.

    :param label_list1: a list that will be extended each time we call the procedure
    :param other_labels: a dictionary of lists with 3 entries 'permutations', 'levels', 'ID of child'.
    The lists will be extended each time we call the procedure
    :param filename: the relative filename (without directory naming) we're currently processing.
    :param num_windows: Number of windows
    :param windows_end_indices: Actually this is not in use. I experimented around and used this to omit the
    samples in the level changes, so only processing the samples that I assumed the children were adjusted to the
    game level.
    :return: the objects label_list1, other_labels that are extended each time we call the procedure
    """
    # current assumption about labels, please see explanations above

    if len(other_labels) == 0:
        other_labels = {'perm': [],
                        'level': [],
                        'ID': []}

    labels = filename[filename.index('[') + 1:filename.index(']')].split(':')
    label_list1.extend([int(labels[1]) for i in range(num_windows)])

    # in the 2nd experiments we permutated the 3 different levels of difficulty 1,2,3 over the game (300 seconds
    # long). One level lasted for approximately 30 seconds. So basically we have 6 intervals. The different
    # permutations are stored in the dict "game_level_dict". The last label in the filename
    # 20161206_futurocube_roadrunner_[ID3:1:Age7:1:1:C]_acc.csv (in this case "C") indicates the permutation.
    # Therefore we can extend the label_list2 just with the array stored in game_level_dict for the specific
    # permutation.
    other_labels['perm'].extend([labels[5] for i in range(num_windows)])

    if num_windows == len(game_level_dict[labels[5]]):   # labels[5] equal to A..W, game level permutations
        # number of windows fits exactly the number of different game levels (e.g. 6 windows and 6 game levels)
        n_times = num_windows / len(game_level_dict[labels[5]])
        for i in game_level_dict[labels[5]]:
            other_labels['level'].extend([i] * n_times)

    else:
        # can't derive game level with this segmentation, fill with zeros
        # we could do this once...but lazy here, so do it for each file
        max_intervals = len(config.LEVEL_TIME_INTERVALS)
        game_levels = game_level_dict[labels[5]]
        for end_idx in windows_end_indices:
            for i, level_end_idx in enumerate(config.LEVEL_TIME_INTERVALS):
                if end_idx <= level_end_idx:
                    other_labels['level'].extend([game_levels[i]])
                    break
                elif max_intervals == i+1:
                    other_labels['level'].extend([game_levels[i]])
                    break
                else:
                    pass
    # labels[0] contains IDx, we filter out only the actual ID of the child
    other_labels['ID'].extend([int(labels[0][2:]) for i in range(num_windows)])

    return label_list1, other_labels


def normalize_features(d_tensor, norm_stddev=True, use_scikit=False):
    """
    Assuming the passed tensor contains 3 axis
    axis 0: number of samples
    axis 1: number of features
    axis 2: channels/xyz-axis of accelerometer

    Normalization per feature, over all 3 channels
    :param d_tensor:
    :param norm_stddev: make sure each channel has stddev = 1
    :param use_scikit: use the normalization function from skicit-learn libraries (used to validate computations)
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
        if norm_stddev:
            d_tensor[:][:] /= std_x
    else:
        d_tensor = normalize(np.reshape(d_tensor, (dim0, dim1 * dim2)), norm='l2')
        d_tensor = np.reshape(d_tensor, (dim0, dim1, dim2))

    return d_tensor


def get_windows_level_changes(signal, game_state_signal, win_length):
    """
    Assuming that signal and game_state_signal are already segmented into windows that have a fixed game level
    i.e. this only works if window length = game level length (e.g. in exp2 this was 30 seconds)
    Basically what I do here is trying to extract only the game level changes. This relates to the assumption
    that these periods reveal important information about the motor skill abilities of a child (how fast do
    they adjust to the new level, how many mistakes to they make in these periods).

    NOTE: this is currently not in use. Relates to the story around object "windows_end_indices" in procedure
          extract_label_info.

    :param signal:
    :param game_state_signal:
    :param win_length:
    :return:
    """

    signal_init_seg = np.zeros((signal.shape[0], win_length, signal.shape[2]))
    game_init_seq = np.zeros((game_state_signal.shape[0], win_length, game_state_signal.shape[2]))
    signal_init_seg[:] = signal[:, 0:win_length, :]
    game_init_seq[:] = game_state_signal[:, 0:win_length, :]

    return signal_init_seg, game_init_seq


def convert_to_window_size(expt_data, game_state, win_size, overlap_coeff=config.OVERLAP_COEFFICIENT,
                           max_num_windows=20, off_set=0):
    """
    Creates chunks of original raw data. Chunk size = window size
    :param expt_data: 3D tensor containing the original signal data
    :param game_state: 2D tensor containing the game state information
    :param win_size: number of samples per window
    :param overlap_coeff: how much do windows overlap, we used 0.5 and 1 (= no overlapping)
    :param max_num_windows: everything after this number of windows will be omitted
    :return: (1) 3D tensor (num of windows, window size, num of channels) for acc signal data
             (2) 3D tensor (num of windows, window size, 1) for game state measure
             (3) list of indices that indicate end of window index. Currently not used in program.
                 Idea was to use this to determine the game level when using a sliding window approach.
                 Therefore the list in used in the procedure that extracts the label information.
    """
    window_lists1 = []
    window_lists2 = []
    # stores the final index of a sliding window segment. we'll use this later to determine the label for the
    # game level (1,2,3)
    windows_final_idx = []
    total_samples = expt_data.shape[0]
    num_samples_so_far = 0
    while num_samples_so_far < total_samples:
        window1 = expt_data[num_samples_so_far+off_set:num_samples_so_far + win_size]
        window2 = game_state[num_samples_so_far+off_set:num_samples_so_far + win_size]
        # print("num_samples_so_far %d / window1 %d" % (num_samples_so_far, len(window1)))
        if window1.shape[0] < win_size - off_set:
            break
        window_lists1.append(window1)
        window_lists2.append(window2)
        # remember, we want the time frames (windows) to overlap
        # previous research has revealed that a 50% overlap works well
        num_samples_so_far += int(win_size * overlap_coeff)
        windows_final_idx.append(num_samples_so_far)

    num_of_windows = len(window_lists1)
    # print(num_of_windows, max_num_windows)
    if num_of_windows > max_num_windows:
        # cut off some blocks
        window_lists1 = window_lists1[:max_num_windows]
        window_lists2 = window_lists2[:max_num_windows]
        windows_final_idx = windows_final_idx[:max_num_windows]
    else:
        # if experimental data contains less than maximum num of windows
        # then we just pass what we have so far
        pass
    return np.array(window_lists1), np.array(window_lists2), windows_final_idx


def calculate_features(d_tensor, d_game_state, d_signal_3axis, freq_bins,
                       window_func, d_axis=1):
    """
    Important part where we calculate the different features used for the analysis.

    :param d_tensor: 3D tensor that contains acc signal data (num of windows, window size, num of channels)
    Note, if the magnitude of the signal was calculated before, num of channels is equal to 1, otherwise 3.
    :param d_game_state: 3D tensor that contains game state data (num of windows, window size, 1)
    :param d_signal_3axis: in case the magnitude was calculated for the acc signal we still need the original 3 axis
    signal in order to determine the cosine similarity measure.
    :param freq_bins:
    :param window_func: We only use Hamming window function that is applied to acc signal
    :param d_axis: along which the features need to be calculated (default is 1 == over window length)
    :return: (1) 3D tensor (num of windows, number of features, 1) that contains all non-game features
             (2) 3D tensor (num of windows, number of game features, 1) contains game features
    """

    if d_tensor.ndim == 3:
        dim1 = d_tensor.shape[0]  # number of windows (segments)
        dim2 = d_tensor.shape[1]  # this is the actual window size
        dim3 = d_tensor.shape[2]  # equal to 1 (magnitude signal) or 3 (3 axial accelerometer data)
    else:
        raise ValueError("tensor has less or more than 3 dimensions: %d" % d_tensor.ndim)

    res_tensor = None
    res_game_tensor = None

    """
        Note: little subtlety, make sure that FEATURE_LIST (set in settings.py) always contains the "minf" feature
        otherwise you need to adjust the first time when res_tensor object is set (because after the 1st time we
        concatenate) and I was lazy checking this each time.
    """

    # Envelope metrics in time domain
    if 'minf' in config.FEATURE_LIST:
        minf = np.reshape(np.amin(d_tensor, axis=d_axis), (dim1, 1, dim3))
        res_tensor = minf

    if 'rms' in config.FEATURE_LIST:
        rms = np.reshape(np.sqrt(1/float(dim2) * np.sum(d_tensor**2, axis=d_axis)), (dim1, 1, dim3))
        res_tensor = np.append(res_tensor, rms, axis=1)

    if 'mean_squared_jerk' in config.FEATURE_LIST or 'int_squared_jerk' in config.FEATURE_LIST:
        d_filter = np.array([1.0, -1.0], np.float32)
        if dim3 > 1:
            # convolve function can only take 2D tensor, so we need to convolve each dimension separately
            signal_jerk_x = np.sum(convolve1d(np.reshape(d_tensor[:, :, 0], (dim1, dim2 * 1)),
                                            d_filter, axis=1)**2, axis=1)
            signal_jerk_y = np.sum(convolve1d(np.reshape(d_tensor[:, :, 1], (dim1, dim2 * 1)),
                                            d_filter, axis=1) ** 2, axis=1)
            signal_jerk_z = np.sum(convolve1d(np.reshape(d_tensor[:, :, 2], (dim1, dim2 * 1)),
                                            d_filter, axis=1) ** 2, axis=1)

            signal_jerk_x = np.reshape(signal_jerk_x, (dim1, 1, 1))
            signal_jerk_y = np.reshape(signal_jerk_y, (dim1, 1, 1))
            signal_jerk_z = np.reshape(signal_jerk_z, (dim1, 1, 1))
            m_sq_jerk_x = 1/float(dim2) * signal_jerk_x
            m_sq_jerk_y = 1/float(dim2) * signal_jerk_y
            m_sq_jerk_z = 1/float(dim2) * signal_jerk_z

            mean_sq_jerk = np.concatenate((m_sq_jerk_x, m_sq_jerk_y, m_sq_jerk_z), axis=2)
            int_mean_jerk = np.concatenate((signal_jerk_x, signal_jerk_y, signal_jerk_z), axis=2)

        else:
            signal_jerk = np.sum(convolve1d(np.reshape(d_tensor, (dim1, dim2 * dim3)), d_filter, axis=1)**2,
                                    axis=1)
            int_mean_jerk = np.reshape(signal_jerk, (dim1, 1, 1))
            mean_sq_jerk = 1 / float(dim2) * int_mean_jerk

        if 'mean_squared_jerk' in config.FEATURE_LIST:
            res_tensor = np.append(res_tensor, mean_sq_jerk, axis=1)
        # currently not using this
        if 'int_squared_jerk' in config.FEATURE_LIST:
            res_tensor = np.append(res_tensor, int_mean_jerk, axis=1)

    ###############################################################################################
    #                            NOTE FREQUENCY DOMAIN FEATURES                                   #

    # Features of the frequency domain
    # First, apply Hamming window in order to prevent frequency leakage
    if window_func is not None:
        d_tensor = d_tensor * np.reshape(window_func, (1, len(window_func), 1))

    fd = np.fft.fft(d_tensor, axis=1)  # frequency domain
    # DC or zero Hz component, is the first component of the N (window sample size) components
    # -----------------------
    # for each window (first axis) take the first component, reshape so we can stack later
    if 'dc' in config.FEATURE_LIST:
        dc = np.reshape(np.real(fd[:, 0]), (dim1, 1, dim3))
        res_tensor = np.append(res_tensor, dc, axis=1)

    # Power Spectrum  = Power Spectral Density (PSD)
    # ----------------------------------------------
    # When computing the PS we omit the DC component
    # We end up with (N-1) component values (again N is window size)
    #
    power_spec_not_avg = np.abs(fd[:, 1:]) ** 2

    # Energy (following Bao and Intille)
    # ----------------------------------------------------
    if 'energy' in config.FEATURE_LIST:
        energy = 1/(float(dim2) - 1) * np.reshape(np.sum(np.abs(fd[:, 1:]) ** 2, axis=d_axis), (dim1, 1, dim3))
        res_tensor = np.append(res_tensor, energy, axis=1)

    # Power Spectral Entropy (PSE)
    # Please note that we follow the description given in
    # "Feature Extraction of EEG Signals Using Power Spectral Entropy" from Zhang, Yang, Huang (2008)
    # See final report for more details
    # ----------------------------------
    # (1) first step, normalize the PSD. Note, we are summing over the window axis, therefore we end up with a
    #       sum-component for each window and channel/axis which is used as normalizing coefficient
    norm_power_spec = power_spec_not_avg / np.reshape(np.sum(power_spec_not_avg, axis=d_axis), (dim1, 1, dim3))
    # (2) We then average again the PSE over the window size for each window/channel
    #       calculate entropy, don't use scipy.stats because we can't influence the summing over axis
    #        use numpy log function with base "e"
    if 'power_spec_entropy' in config.FEATURE_LIST:
        power_spec_entropy = - np.sum(norm_power_spec * np.log(norm_power_spec), axis=d_axis)
        # (3) Reshape in order to stack features at the end
        power_spec_entropy = np.reshape(power_spec_entropy, (dim1, 1, dim3))
        res_tensor = np.append(res_tensor, power_spec_entropy, axis=1)

    if 'dominant freq' in config.FEATURE_LIST:
        if window_func is not None:
            freq_offset = 2 # skip DC + 1 freq components (still a mystery why 2nd is so large
        else:
            freq_offset = 1

        # because skipping DC (+1) component we also need to skip first bins
        freq_b = freq_bins[freq_offset:]
        dominant_freq = freq_b[np.argmax(np.abs(fd[:, freq_offset:(dim2/2)]), axis=d_axis)]
        dominant_freq = np.reshape(dominant_freq, (dim1, 1, dim3))
        res_tensor = np.append(res_tensor, dominant_freq, axis=1)
    ###############################################################################################
    #                            GAME STATE FEATURES
    # dx_dy_error was only used in first experiments                                           #
    if 'dxdy_error' in config.FEATURE_LIST:

        error_measure = np.reshape(np.sum(d_game_state[:, :, 0]**2, axis=d_axis), (d_game_state.shape[0], 1, 1))
        res_tensor = np.append(res_tensor, error_measure, axis=1)
        # u_values, u_counts = np.unique(error_measure, return_counts=True)
        # print("error_measure unique counts ", u_values, u_counts)

    if 'cos_sim' in config.FEATURE_LIST:
        # unfortunately need to loop through windows, compute cosine similarity for each window
        # separately
        cos_sims = calc_cosine_similarity(d_game_state, d_signal_3axis)

        #   January 2017:
        #   ==============
        #   Decided to use sum of squared cosine-sim values. Why? I attenuate errors between 0-1 which
        #   happen easily and values up to 0.5 are just due to noise in the data because walker index almost always
        #   deviates at least one square on the cube to the gravity signal of the accelerometer (just look at the
        #   raw output data, you'll notice that this is almost always the case)
        error_measure = np.reshape(np.sum(cos_sims**2, axis=d_axis), (d_game_state.shape[0], 1, 1))
        if res_game_tensor is None:
            res_game_tensor = error_measure
            # res_game_tensor = np.append(res_game_tensor, cos_sim_avg, axis=1)
        else:
            res_game_tensor = np.append(res_game_tensor, error_measure, axis=1)
            # res_game_tensor = np.append(res_game_tensor, cos_sim_avg, axis=1)

    if 'maxf' in config.FEATURE_LIST:
        maxf = np.reshape(np.amax(d_tensor, axis=d_axis), (dim1, 1, dim3))
        res_tensor = np.append(res_tensor, maxf, axis=1)
    if 'mean' in config.FEATURE_LIST:
        mean = np.reshape(np.mean(d_tensor, axis=d_axis), (dim1, 1, dim3))
        res_tensor = np.append(res_tensor, mean, axis=1)
    if 'std' in config.FEATURE_LIST:
        std = np.reshape(np.std(d_tensor, axis=d_axis), (dim1, 1, dim3))
        res_tensor = np.append(res_tensor, std, axis=1)
    if 'median' in config.FEATURE_LIST:
        median = np.reshape(np.median(d_tensor, axis=d_axis), (dim1, 1, dim3))
        res_tensor = np.append(res_tensor, median, axis=1)
    if 'range' in config.FEATURE_LIST:
        range = maxf - minf
        res_tensor = np.append(res_tensor, range, axis=1)

    if config.DEBUG_LEVEL >= 1:
        print("INFO - calculating features -shape of result tensor ", res_tensor.shape)
        # print(res_tensor)

    return res_tensor, res_game_tensor


def import_data(edate, device, game, root_path, file_ext='csv', calc_mag=False,
                f_type=None, lowcut=0., highcut=0., b_order=5,
                apply_window_func=False, extra_label='', optimal_w_size=True, nn_switch=False,
                skip_level_chg=False):
    """
        Probably it would have been wiser (by hindsight) to split this procedure in smaller chunks

        Parameters:
            device: actually the intention was to use different game devices, but analysis is limited
                    to "futurocube"
            game: in the end limited to "roadrunner" game
            root_path: root directory in which to search for csv files
            file_ext: actually only used 'csv' for the analysis, so could be omitted.
            calc_mag: calculate the magnitude of the signal BEFORE computing the features!
            f_type: low, high, band or None
            lowcut: for butterworth filter low cut frequency
            highcut: for butterworth filter high cut frequency
            b_order: order of butterworth filter
            apply_window_func: boolean indicating whether or not to use Hamming window
            extra_label: a string that will be concatenated to form the filename of the
                         output file
            optimal_w_size: boolean, if true window size must be of size 2^n, where n fits the data
            nn_switch: boolean, when true the data is pre-processed for the neural network / RNN version
                       which basically means that features are not calculated and the data is as "raw" as possible
            skip_level_chg: currently not in use. Was used to omit the game level transitions from the analysis
                            and process them separately.

            Returns: (1) feature_data:
                     (2) label_data: 2D tensor with motor skill labels (exp2 0=normal, 1=likely deficiency, 2 = most
                     likely motor skill deficiency
                     (3) label_other: dictionary of 2D tensors with labels a) permutations ('perm')
                                                                           b) game levels ('level')
                                                                           c) ID of children ('ID')
                     (4) d_dict: dictionary which basically contains all the settings  (e.g. features calculated etc)

            The procedure also creates two output files in the same directory where the original experimental data
            can be found (root_path). Example:
            (1) File that contains:
                a) 3D tensor with segments and calculated features (number of segments, features, 1)
                b) 2D tensor with segments and motor skill labels
                c) Dictionary of three 2D tensors with other labels, see above

                e.g. 20161206_futurocube_roadrunner_20hz_1axis_low8hz_330_12_True.h5
                where "20hz_1axis_low8hz" = extra_label
                      330 = number of segments
                      12  = number of features
                      True = calculate magnitude

            (2) json file that contains a dictionary that describes the data. Looking back, this could easily
                been integrated with other objects in one file.

            The procedure smart_utils.load_data implements the load of the data from these 2 files.
    """

    if nn_switch and calc_mag:
        raise ValueError("ERROR - Preprocessing option NN cannot be combined with 'calculate magnitude'")

    if device == config.GAME1:  # futurocube specific processing steps
        freq = config.SAMPLE_FREQUENCY_FUTUROCUBE
        if optimal_w_size:
            # for efficiency reasons make window size a multiple of 2
            exp_2 = np.floor(np.log2(config.WINDOW_SIZE * freq))
            window_size_samples = int(2 ** exp_2)
        else:
            window_size_samples = int(config.WINDOW_SIZE * freq)

        feature_list = config.FEATURE_LIST

    else:
        raise NotImplementedError("NOT IMPLEMENTED YET")

    # calculate the maximum length (in samples) per file
    signal_offset = int(config.CUT_OFF_LENGTH * freq)
    if config.OVERLAP_COEFFICIENT != 1:
        max_windows = np.floor(((config.MAX_FILE_LENGTH - window_size_samples - signal_offset) /
                                float(config.OVERLAP_COEFFICIENT * window_size_samples)) + 1)
    else:
        # OVERLAP_COEFFICIENT = 0, means no sliding window approach
        max_windows = np.floor(config.MAX_FILE_LENGTH / float(window_size_samples))

    max_windows = int(max_windows)
    freq_bins = np.fft.fftfreq(window_size_samples, 1/freq)[:window_size_samples/2]
    max_file_length = window_size_samples + ((max_windows - 1) * config.OVERLAP_COEFFICIENT * window_size_samples)
    if config.DEBUG_LEVEL >= 1:
        print("-------------------------------------------------------------------------")
        print("INFO - Running feature calculation with the following parameter settings:")
        print("-------------------------------------------------------------------------")
        print("Extraction mode for NN? %s" % nn_switch)
        print("Game device *** %s  %s ***" % (device, game))
        print("Assuming sample frequency of device: %.2f" % freq)
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

    # Not yet fully developed, if set to true, we skip the first (4 seconds in this case) seconds of
    # each game level interval in order to skip the level changes
    skip_level_chg = False
    if skip_level_chg:
        off_set = int(freq * 4)
    else:
        off_set = 0

    # only change directory if we are not already in ../data/... path
    if os.getcwd().find('data') == -1:
        os.chdir(root_path)
    abs_dir_path = os.getcwd() + "/"
    files_to_load = glob.glob(edate + "*." + file_ext)
    if config.DEBUG_LEVEL >= 1:
        print("INFO - Loading accelerometer %d files from: %s" % (len(files_to_load), abs_dir_path))
    num_of_files = 0
    num_of_files_skipped = 0
    # two tensors that will held the signal features and the game state features
    feature_data = None
    game_feature_data = None
    # following two matrices are used for signal segments that hold the game level transition data
    # we can use those segments to calculate features specifically for those transitions
    feature_data_trans = None
    game_feature_data_trans = None
    # label_data stores the class labels for the different motor skill levels (e.g. 0, 1)
    label_data = []
    # label_game_level stores the class labels for the different game levels of difficulty
    other_labels = {}
    id_attributes = []

    if apply_window_func:
        hamming_w = np.hamming(window_size_samples-off_set)
    else:
        hamming_w = None

    for f_name in files_to_load:
        acc_file = os.path.join(abs_dir_path, f_name)
        if config.DEBUG_LEVEL > 1:
            print("INFO - processing file %s" % acc_file)
        try:
            # read file to pandas dataframe, acc_file contains the absolute path to import file
            # can process csv-format of xlsx file extension. in the last case
            # only the first sheet is imported
            df = None
            if file_ext == 'csv':
                # pos, ax,  ay,  az, error
                df = pd.read_csv(acc_file,
                                 converters={"ax": np.int64, "ay": np.int64, "az": np.int64, "error": np.int64},
                                 engine='python')
            elif file_ext == 'xlsx':
                excel_file = pd.ExcelFile(acc_file)
                df = excel_file.parse(excel_file.sheet_names[0])
            num_of_files += 1
            # convert the pandas dataframe to a numpy array
            # currently assuming that the dataframe columns 1 = x-axis, 2 = y-axis, 3 = z-axis
            df.columns = config.IMPORT_COLUMNS
            # Experiments 1 when we used the unreliable dx/dy error measure. We surmised that we could rescue the
            # score by "wrapping" it back into the 16 bit range...anyway not used anymore
            if 'dxdy_error' in df.columns:
                df.loc[df.dxdy_error > 20, 'dxdy_error'] = np.abs((df.dxdy_error - 2**16))
            raw_data = df.iloc[:, 1:4].as_matrix()
            # assuming that after the first four columns (package_id, x, y, z) we store the game state info
            # IMPORTANT, CHANGE IF MORE GAME STATE FEATURES!!!
            game_state_dta = df.iloc[:, 4:].as_matrix()
            # dimensionality of raw_data = (total number of samples, num of channels(x,y,z))
            # we assume that the beginning and the end of the raw signal contains too much
            # noise, therefore we are cutting of a piece in the beginning and in the end
            # Note: we did not use offset in the end
            raw_data = raw_data[signal_offset:raw_data.shape[0] - signal_offset, :]
            # apply a butterworth filter if specified
            if f_type is not None:
                # apply butterworth filter
                signal_data, _ = apply_butter_filter(raw_data, freq, lowcut, highcut, f_type, b_order)
                if config.DEBUG_LEVEL >= 1:
                    print("INFO - Dimension of tensor after filtering ", signal_data.shape)
            else:
                signal_data = raw_data

            # segmentation of signal based on sliding window approach or fixed window size
            np_signal, np_game_state, window_final_idx = convert_to_window_size(signal_data, game_state_dta,
                                                                                win_size=window_size_samples,
                                                                                max_num_windows=max_windows,
                                                                                off_set=off_set)
            if config.WINDOW_SIZE == 30:
                """
                    Note: This piece of coding refers to earlier remarks w.r.t. the game level transitions.
                    We experimented with the extraction of features only from these periods.
                    In the current version of this program the final matrices:
                        m_features_trans
                        m_game_features_trans
                    are not saved to file and therefore not used.

                """
                # get the game level transition windows (w_length). We can use these segments to calculate
                # features for these specific windows when children have to cope with level transitions
                win_length = int(freq * 5)  # length of window
                np_signal_trans, np_game_state_trans = get_windows_level_changes(np_signal, np_game_state,
                                                                                 win_length)
                if apply_window_func:
                    hamming_w_trans = np.hamming(win_length)
                else:
                    hamming_w_trans = None
            else:
                np_signal_trans = None
                np_game_state_trans = None

            # we need to keep 3-axis acc data for calculation of cosine similarity
            np_signal_3axis = np.zeros_like(np_signal)
            np_signal_3axis[:] = np_signal[:]
            if np_signal_trans is not None:
                np_signal_trans_3axis = np.zeros_like(np_signal_trans)
                np_signal_trans_3axis[:] = np_signal_trans[:]

            # calculate signal magnitudes
            if calc_mag:
                np_signal = np.reshape(np.sqrt(np_signal[:, :, 0]**2 + np_signal[:, :, 1]**2 + np_signal[:, :, 2]**2),
                                            (np_signal.shape[0], np_signal.shape[1], 1))
                if np_signal_trans is not None:
                    np_signal_trans = np.reshape(np.sqrt(np_signal_trans[:, :, 0] ** 2 + np_signal_trans[:, :, 1] ** 2 +
                                                         np_signal_trans[:, :, 2] ** 2),
                                                        (np_signal_trans.shape[0], np_signal_trans.shape[1], 1))

            # previous function returns a numpy array with 3 axis:
            #   axis 0 = number of windows
            #   axis 1 = number of samples per window
            #   axis 2 = number of channels, e.g. 3 for accelerometer data (x,y,z axis)
            # we are calculating the features for the tuple(window/channel-axis)
            # and therefore aggregating over axis 1 (2nd parameter to calculate_features)
            # 01-2017: added switch for processing data in case of NN/RNN usage, don't calculate features
            #          except for cosine-similarity measure.
            if not nn_switch:
                m_features, m_game_features = calculate_features(np_signal, np_game_state, np_signal_3axis,
                                                                 window_func=hamming_w, d_axis=1,
                                                                 freq_bins=freq_bins)
                if np_signal_trans is not None:
                    m_features_trans, m_game_features_trans = calculate_features(np_signal_trans, np_game_state_trans,
                                                                                 np_signal_trans_3axis,
                                                                                 window_func=hamming_w_trans, d_axis=1,
                                                                                 freq_bins=freq_bins)
                else:
                    m_features_trans = None
                    m_game_features_trans = None

            else:
                # pre-processing for Neural Network of RNN usage only
                # (1) we use the low-pass-filtered accelerometer data
                m_features_trans = None
                m_game_features_trans = None
                gravity, _ = apply_butter_filter(np_signal, freq, lowcut=0.3, highcut=0, f_type='low', order=3)
                linear_acc = np_signal - gravity
                norm_signal = np.reshape(np.sqrt(np_signal[:, :, 0] ** 2 + np_signal[:, :, 1] ** 2 + np_signal[:, :, 2] ** 2),
                                      (np_signal.shape[0], np_signal.shape[1], 1))
                m_features = np.concatenate((np_signal, norm_signal, linear_acc), axis=2)
                # (2) we convert the walker index (assuming this is the only game state feature for this experiment
                #     to a "cube" vector and calculate the cosine similarity based on accelerometer vector
                m_game_features = calc_cosine_similarity(np_game_state, np_signal)

            # concatenate the contents of the files (transformed as numpy arrays)
            if feature_data is None:
                feature_data = m_features
                if feature_data_trans is None and m_features_trans is not None:
                    feature_data_trans = m_features_trans
            else:
                # Concatenate along axis 0...the windows
                feature_data = np.concatenate((feature_data, m_features), axis=0)
                if m_features_trans is not None:
                    feature_data_trans = np.concatenate((feature_data_trans, m_features_trans), axis=0)

            if game_feature_data is None:
                game_feature_data = m_game_features
                if game_feature_data_trans is None and m_game_features_trans is not None:
                    game_feature_data_trans = m_game_features_trans
            else:
                # Concatenate along axis 0...the windows
                game_feature_data = np.concatenate((game_feature_data, m_game_features), axis=0)
                if m_game_features_trans is not None:
                    game_feature_data_trans = np.concatenate((game_feature_data_trans, m_game_features_trans), axis=0)

            if config.DEBUG_LEVEL > 1:
                print("INFO - total length file=%d, num of windows=%d, num of features=%d, channels=%d" %
                  (signal_data.shape[0], m_features.shape[0], m_features.shape[1], m_features.shape[2]))
            # get label information
            label_data, other_labels = extract_label_info(label_data, other_labels, f_name, m_features.shape[0],
                                                          window_final_idx)
            # get dictionary with label info for this file
            label_dict = get_file_label_info(f_name)
            id_attributes.append(label_dict)

        except IOError as e:
            print('WARNING ****** Could not read:', acc_file, ':', e, '- it\'s ok, skipping. *******')
            num_of_files_skipped += 1

    # finally normalize the calculated features
    # Note: each feature is normalized separately, but over all 3 axis
    if not nn_switch:
        # NOTE: this code is disabled. Was used to experiment with features only from game level transition periods

        # if feature_data_trans is not None:
        #    print("!!!!!!!!!!!!!!!!!!!! Using transition matrices !!!!!!!!!!!!!!!!!!!!")
        #    feature_data = feature_data_trans
        #    game_feature_data = game_feature_data_trans

        feature_data = normalize_features(feature_data, use_scikit=False)
        game_feature_data = normalize_features(game_feature_data, use_scikit=False)
        # reshape to 2 dim tensor
        feature_data = np.reshape(feature_data, (feature_data.shape[0], feature_data.shape[1] * feature_data.shape[2]))
        game_feature_data = np.reshape(game_feature_data, (game_feature_data.shape[0],
                                                           game_feature_data.shape[1] * game_feature_data.shape[2]))
        feature_data = np.concatenate((feature_data, game_feature_data), axis=1)
    else:
        # for nn pre-processing we're not normalizing the data here. This is done in a later stage (smart_data_utils)
        # when we're construct the training & test set. These will be normalized then.
        feature_data = np.concatenate((feature_data, game_feature_data), axis=2)

    label_data = np.reshape(np.array(label_data), (len(label_data), 1))
    # concatenate all other label information into a matrix of N x num-of-labels
    print(np.array(other_labels['level']).shape, np.array(other_labels['perm']).shape, np.array(other_labels['ID']).shape)
    label_other = np.concatenate((np.expand_dims(np.array(other_labels['ID']), 1),
                                  np.expand_dims(np.array(other_labels['perm']), 1),
                                  np.expand_dims(np.array(other_labels['level']), 1)), axis=1)

    print("INFO - %d files loaded successfully! Skipped %d" % (num_of_files, num_of_files_skipped))
    if config.DEBUG_LEVEL > 1:
        print("INFO - Feature data shape ", feature_data.shape, " / label data shape ", label_data.shape,
              " label game levels shape ", label_other.shape, " shape game state features ", game_feature_data.shape)
    # finally store matrices in hdf5 format
    # extend extra label with dimensions of feature data, that helps identifying the contents of the different
    # files
    extra_label = extra_label + "_" + str(feature_data.shape[0]) + "_" + str(feature_data.shape[1]) + "_" + \
                  str(calc_mag)
    data_label = get_exprt_label(edate, device, game, extra_label)
    d_dict = make_data_description(freq, window_size_samples, max_windows, apply_window_func, f_type,
                                   [lowcut, highcut, b_order], num_of_files, feature_list, id_attributes)
    store_data(feature_data, label_data, label_other, data_label, out_loc=abs_dir_path, descr=d_dict)

    return feature_data, label_data, label_other, d_dict


def get_data(e_date, device='futurocube', game='roadrunner', file_ext='csv', calc_mag=False,
             apply_window_func=True, extra_label='', force=False,
             f_type=None, lowcut=8, highcut=0., b_order=5, optimal_w_size=True, nn_switch=False):
    """
    Most of the parameters were already described in the method "import_data".
    This is a wrapper function. When the data was already pre-processed we're just trying to open the specific
    h5 file e.g. "20161206_futurocube_roadrunner_20hz_1axis_low8hz_330_12_True.h5".
    Otherwise we call import_data to process the experimental data and store the final tensors.

    :param e_date: used to construct the prefix of the files that need to be processed, see filename example above
    :param device: see import_data
    :param game: see import_data
    :param file_ext: see import_data
    :param calc_mag: see import_data
    :param apply_window_func: see import_data
    :param extra_label: see import_data
    :param force: boolean, if set to True, the experimental data will be pre-processed and features will be calculated
    :param f_type: see import_data
    :param lowcut: see import_data
    :param highcut: see import_data
    :param b_order: see import_data
    :param optimal_w_size: see import_data
    :param nn_switch: see import_data
    :return: the same objects as procedure import_data
    """

    data_label = get_exprt_label(e_date, device, game, extra_label)
    root_dir = get_dir_path(device, game)
    if os.getcwd().find('data') == -1:
        os.chdir(root_dir)
    abs_file_path = os.path.join(os.getcwd(), data_label)
    if config.DEBUG_LEVEL >= 1:
        print("INFO - Used data label %s" % data_label)
    if os.path.isfile(abs_file_path + ".h5") and not force:
        if config.DEBUG_LEVEL >= 1:
            print("INFO Loading matrices from h5 file %s" % abs_file_path + ".h5")
            data, labels, labels_game_level, d_dict = load_data(abs_file_path)
            return data, labels, labels_game_level, d_dict
    else:
        if config.DEBUG_LEVEL >= 1:
            print("INFO - Need to process raw data...")
        return import_data(e_date, device, game, root_dir, file_ext, apply_window_func=apply_window_func,
                           calc_mag=calc_mag,
                           f_type=f_type, lowcut=lowcut, highcut=highcut, b_order=b_order,
                           extra_label=extra_label, optimal_w_size=optimal_w_size, nn_switch=nn_switch)


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    print_flags()
    get_data(FLAGS.exp_date, force=FLAGS.force, apply_window_func=FLAGS.apply_w_func,
             extra_label=FLAGS.extra_label,
             optimal_w_size=FLAGS.optimal_w_size, calc_mag=FLAGS.calc_magnitude,
             f_type='low', lowcut=8, b_order=5,
             nn_switch=FLAGS.nn_preprocess)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_date', type=str, default='20161206',
                        help='Date (YYYYMMDD) of experiment, to distinguish between data files.')
    parser.add_argument('--extra_label', type=str, default="void_20hz_1axis_low8hz",
                        help='Extra label to distinguish between different pre-processing settings.')
    parser.add_argument('--force', action='store_true',
                        help='Indicating whether to force (re-) preprocessing of experimental data.')
    parser.add_argument('--apply_w_func', action='store_false',
                        help='Indicating whether apply Hamming window function to signal data.')
    parser.add_argument('--optimal_w_size', action='store_true',
                        help='Indicating whether to use segmentation length that is power of 2.')
    parser.add_argument('--calc_magnitude', action='store_false',
                        help='Indicating whether to calculate the magnitude signal (1 channel instead of 3).')
    parser.add_argument('--nn_preprocess', action='store_true',
                        help='Indicating whether use pre-processing for (R)neural networks.')
    FLAGS, unparsed = parser.parse_known_args()
    main()