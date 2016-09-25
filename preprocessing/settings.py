"""
    Defining some global constants

"""

CUT_OFF_LENGTH = 5              # cut-off lengths in seconds, raw signal will be shortened
GAME1 = 'futurocube'
LEVEL_TIME_INTERVALS = [40, 80, 120]   # in seconds between game levels
DEBUG_LEVEL = 2
SAMPLE_FREQUENCY_FUTUROCUBE = 20  # 70 Hz for futurocube according to last info of Antoine/Pascal
                                  # needs to be adjusted per game device

WINDOW_SIZE = 6                     # in seconds
OVERLAP_COEFFICIENT = 0.5           # 50% overlap of windows, important parameter!
MEAN_FILE_LENGTH = 2400             # in samples
FEATURE_LIST = ['min', 'max', 'mean', 'std', 'median', 'dc', 'energy', 'power_spec_entropy']

LABELS = ['ID', 'CLASS', 'AGE', 'SEX', 'HANDED']
RAW_DATA_ARRAY = 'raw_data'
DATA_ARRAY = 'feature_data'
LABEL_ARRAY = 'label_data'

