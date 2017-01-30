import numpy as np
"""
    Defining some global constants

"""

IMPORT_COLUMNS = ['pid', 'ax', 'ay', 'az', 'error']
CUT_OFF_LENGTH = 0              # cut-off lengths in seconds, raw signal will be shortened
GAME1 = 'futurocube'
TIME_INTERVALS = [30, 60, 90, 120, 150, 180]   # in seconds between game levels
DEBUG_LEVEL = 2
SAMPLE_FREQUENCY_FUTUROCUBE = 20.75  # sampling frequency
LEVEL_TIME_INTERVALS = list(np.array(TIME_INTERVALS) * SAMPLE_FREQUENCY_FUTUROCUBE)

# if we want to calculate the features for one window across the whole file length
# we need to approximate "exp_2 = np.floor(np.log2(WINDOW_SIZE * freq))"
WINDOW_SIZE = 30                 # in seconds
OVERLAP_COEFFICIENT = 1          # 50% overlap of windows, important parameter!
                                 # 1 means no sliding window approach, can be used for hard cuts
MEAN_FILE_LENGTH = 3750             # in samples
# ['min', 'max', 'mean', 'std', 'median', 'rms', 'range', 'dc', 'energy', 'power_spec_entropy',
#           'dominant freq', "cos_sim", "dxdy_error"]
FEATURE_LIST = ['minf', 'rms', 'mean_squared_jerk', 'dc',
                'energy', 'power_spec_entropy', 'cos_sim', 'maxf', 'mean', 'std', 'median', 'range',]

LABELS = ['ID', 'CLASS', 'AGE', 'SEX', 'HANDED', 'PERM']
RAW_DATA_ARRAY = 'raw_data'
DATA_ARRAY = 'feature_data'
LABEL_ARRAY = 'label_data'
LABEL_GAME_ARRAY = 'label_game_level'
