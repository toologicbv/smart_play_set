"""
    Defining some global constants

"""

IMPORT_COLUMNS = ['pid', 'ax', 'ay', 'az', 'dxdy_error']
CUT_OFF_LENGTH = 0              # cut-off lengths in seconds, raw signal will be shortened
GAME1 = 'futurocube'
LEVEL_TIME_INTERVALS = [40, 80, 120]   # in seconds between game levels
DEBUG_LEVEL = 2
SAMPLE_FREQUENCY_FUTUROCUBE = 20.5  # 70 Hz for futurocube according to last info of Antoine/Pascal
                                  # needs to be adjusted per game device

# if we want to calculate the features for one window across the whole file length
# we need to approximate "exp_2 = np.floor(np.log2(WINDOW_SIZE * freq))"
WINDOW_SIZE = 40                     # in seconds
OVERLAP_COEFFICIENT = 1          # 50% overlap of windows, important parameter!
                                 # 1 means no sliding window approach, can be used for hard cuts
MEAN_FILE_LENGTH = 2460             # in samples
# ['min', 'max', 'mean', 'std', 'median', 'rms', 'range', 'dc', 'energy', 'power_spec_entropy',
#           'dominant freq', "dxdy_error"]
FEATURE_LIST = ['minf', 'maxf', 'mean', 'std', 'median', 'range', 'rms', 'mean_squared_jerk', 'dc',
                'energy', 'power_spec_entropy']

LABELS = ['ID', 'CLASS', 'AGE', 'SEX', 'HANDED']
RAW_DATA_ARRAY = 'raw_data'
DATA_ARRAY = 'feature_data'
LABEL_ARRAY = 'label_data'

