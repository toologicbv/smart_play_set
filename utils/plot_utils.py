import matplotlib as plt
import pandas as pd


def plot_3axis_raw_signal_1(r_signal, seperate=True):

    r_signal.plot(subplots=seperate)
