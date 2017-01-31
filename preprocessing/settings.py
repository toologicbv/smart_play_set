import numpy as np
"""
    Defining some global constants

"""


class Config(object):

    def __init__(self):
        self.IMPORT_COLUMNS = ['pid', 'ax', 'ay', 'az', 'error']
        self.CUT_OFF_LENGTH = 0              # cut-off lengths in seconds, raw signal will be shortened
        self.GAME1 = 'futurocube'
        self.TIME_INTERVALS = [30, 60, 90, 120, 150, 180]   # in seconds between game levels
        self.DEBUG_LEVEL = 2
        self.SAMPLE_FREQUENCY_FUTUROCUBE = 20.75  # sampling frequency
        self.LEVEL_TIME_INTERVALS = list(np.array(self.TIME_INTERVALS) * self.SAMPLE_FREQUENCY_FUTUROCUBE)

        # if we want to calculate the features for one window across the whole file length
        # we need to approximate "exp_2 = np.floor(np.log2(WINDOW_SIZE * freq))"
        self.WINDOW_SIZE = 30                 # in seconds
        # Sliding window overlap, e.g. 0.5 = 50% overlap
        # 1 means no sliding window approach, can be used for hard cuts
        self.OVERLAP_COEFFICIENT = 1
        # maximum number of samples in file
        self.MAX_FILE_LENGTH = 3750
        # ['min', 'max', 'mean', 'std', 'median', 'rms', 'range', 'dc', 'energy', 'power_spec_entropy',
        #           'dominant freq', "cos_sim", "dxdy_error"]
        self.FEATURE_LIST = ['minf', 'rms', 'mean_squared_jerk', 'dc',
                        'energy', 'power_spec_entropy', 'cos_sim', 'maxf', 'mean', 'std', 'median', 'range',]

        self.LABELS = ['ID', 'CLASS', 'AGE', 'SEX', 'HANDED', 'PERM']
        self.RAW_DATA_ARRAY = 'raw_data'
        self.DATA_ARRAY = 'feature_data'
        self.LABEL_ARRAY = 'label_data'
        self.LABEL_GAME_ARRAY = 'label_game_level'

config = Config()