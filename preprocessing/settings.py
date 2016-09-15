"""
    Defining some global constants

"""

GAME1 = 'futurocube'
DEBUG_LEVEL = 2
SAMPLE_FREQUENCY_FUTUROCUBE = 5  # 70 Hz for futurocube according to last info of Antoine/Pascal
                                  # needs to be adjusted per game device

WINDOW_SIZE = 6                     # in seconds
OVERLAP_COEFFICIENT = 0.5           # 50% overlap of windows, important parameter!
MAX_NUM_WINDOWS = 10
FEATURE_LIST = ['max', 'min', 'mean', 'std', 'median', 'dc', 'energy', 'power_spec_entropy']

DATA_ARRAY = 'feature_data'
LABEL_ARRAY = 'label_data'
FEATURE_ARRAY = 'features'
DATA_DESCR = 'descr'
