import os
import matplotlib.pyplot as plt
from utils.smart_utils import get_dir_path, tensor_to_pandas, load_hdf5_file
from utils.smart_utils import butter_bandpass, butter_lowpass, butter_highpass
from utils.smart_utils import apply_butter_filter, load_file_to_pandas
from scipy.signal import freqz

import numpy as np
from scipy import fft, arange
from scipy.signal import hamming


def plot_fft(signal, p_title="", apply_window_func=False):
    n = signal.shape[0]
    frq = np.arange(n)[range(n / 2)]

    if apply_window_func:
        hamming_w = np.hamming(signal.shape[0])
    else:
        hamming_w = 1
    sig_spec_x = 2. / n * np.abs(np.fft.fft(signal[:, 0] * hamming_w, axis=0))[range(n / 2)]
    sig_spec_y = 2. / n * np.abs(np.fft.fft(signal[:, 1] * hamming_w, axis=0))[range(n / 2)]
    sig_spec_z = 2. / n * np.abs(np.fft.fft(signal[:, 2] * hamming_w, axis=0))[range(n / 2)]
    print("shapes ", sig_spec_x.shape, frq.shape)

    ax1 = plt.subplot(3, 1, 1)
    plt.title(p_title, y=1.08)
    plt.plot(frq, sig_spec_x, color='r', label='x-axis')
    plt.ylabel('|Amplitude|')
    plt.legend(loc="best")
    plt.subplot(3, 1, 2, sharex=ax1)

    # plt.title('signal y-axis')
    plt.plot(frq, sig_spec_y, color='g', label='y-axis')
    plt.ylabel('|Amplitude|')
    plt.legend(loc="best")
    plt.subplot(3, 1, 3, sharex=ax1)
    # plt.title('signal z-axis')
    plt.plot(frq, sig_spec_z, color='k', label='z-axis')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|Amplitude|')
    plt.legend(loc="best")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def plot_spectra_1axis(signal, sampling_freq, p_title, apply_window_func=False, skip_dc=False,
                       p_label='mag-signal', width=20, height=10):

    # n = number of time stamps
    n = signal.shape[0]
    k = arange(n)
    # T = time in seconds
    T = n/float(sampling_freq)
    frq = (k/T)[range(n/2)]     # one side frequency range
    if apply_window_func:
        hamming_w = np.reshape(hamming(signal.shape[0]), (signal.shape[0], 1))
    else:
        hamming_w = 1

    sig_spec_x = 2. / n * np.abs(np.fft.fft(signal * hamming_w, signal.shape[0], axis=0))[range(n / 2)]

    if skip_dc and not apply_window_func:
        print("Note")
        print("FFT without hamming window - first 4 freq coefficients: %.3f/%.3f/%.3f/%.3f" %
              (sig_spec_x[0], sig_spec_x[1], sig_spec_x[2], sig_spec_x[3]))
        sig_spec_x = sig_spec_x[1:]
        frq = frq[1:]

    elif skip_dc and apply_window_func:
        print("Note")
        print("FFT with hamming window - first 4 freq coefficients: %.3f/%.3f/%.3f/%.3f" %
              (sig_spec_x[0], sig_spec_x[1], sig_spec_x[2], sig_spec_x[3]))
        sig_spec_x = sig_spec_x[1:]
        frq = frq[1:]


    plt.figure(figsize=(width, height))
    plt.title(p_title, y=1.08)
    plt.plot(frq, sig_spec_x, color='b', label=p_label)
    plt.ylabel('|Amplitude|')
    plt.xlabel('Freq (Hz)')
    plt.legend(loc="best")
    plt.show()


def plot_spectra_3axis(signal, sampling_freq, p_title, apply_window_func=True, skip_dc=False,
                       width=20, height=10):

    # n = number of time stamps
    n = signal.shape[0]
    k = arange(n)
    # T = time in seconds
    T = n/float(sampling_freq)
    # T = 1.0 / n
    frq = (k/T)[range(n/2)]     # one side frequency range
    if apply_window_func:
        hamming_w = np.hamming(signal.shape[0])
    else:
        hamming_w = 1
    sig_spec_x = 2./n * np.abs(np.fft.fft(signal[:, 0] * hamming_w, signal.shape[0], axis=0))[range(n/2)]
    sig_spec_y = 2./n * np.abs(np.fft.fft(signal[:, 1] * hamming_w, signal.shape[0], axis=0))[range(n/2)]
    sig_spec_z = 2./n * np.abs(np.fft.fft(signal[:, 2] * hamming_w, signal.shape[0], axis=0))[range(n/2)]

    if skip_dc:
        sig_spec_x = sig_spec_x[1:]
        sig_spec_y = sig_spec_y[1:]
        sig_spec_z = sig_spec_z[1:]
        frq = frq[1:]

    plt.figure(figsize=(width, height))
    ax1 = plt.subplot(3, 1, 1)
    plt.title(p_title, y=1.08)
    plt.plot(frq, sig_spec_x, color='r', label='x-axis')
    plt.ylabel('|Amplitude|')
    plt.legend(loc="best")
    plt.subplot(3, 1, 2, sharex=ax1)

    # plt.title('signal y-axis')
    plt.plot(frq, sig_spec_y, color='g', label='y-axis')
    plt.ylabel('|Amplitude|')
    plt.legend(loc="best")
    plt.subplot(3, 1, 3, sharex=ax1)
    # plt.title('signal z-axis')
    plt.plot(frq, sig_spec_z, color='k', label='z-axis')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|Amplitude|')
    plt.legend(loc="best")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def plot_3axis_raw_signal_1(df_signal, p_title="", separate=True):

    # Assuming that pandas dataframe r_signal contains three columns
    # renaming columns

    df_signal.columns = ['x-axis', 'y-axis', 'z-axis']

    ax1 = plt.subplot(311)
    plt.title(p_title, y=1.08)
    df_signal['x-axis'].plot(color='r', label='x-axis')
    plt.legend(loc="best")
    ax2 = plt.subplot(3, 1, 2, sharex=ax1, sharey=ax1)
    df_signal['y-axis'].plot(color='g')
    plt.legend(loc="best")
    ax3 = plt.subplot(3, 1, 3, sharex=ax1, sharey=ax1)
    df_signal['z-axis'].plot(color='k')
    plt.legend(loc="best")

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def plot_3axis_raw_signal_compare(df_signal1, df_signal2, p_title=""):

    df_signal1.columns = ['x-axis', 'y-axis', 'z-axis']
    df_signal2.columns = ['x-axis', 'y-axis', 'z-axis']

    ax1 = plt.subplot(321)
    plt.title(p_title, y=1.08)
    df_signal1['x-axis'].plot(color='r', label='x-axis')
    plt.legend(loc="best")
    ax2 = plt.subplot(3, 2, 3, sharex=ax1)
    df_signal1['y-axis'].plot(color='g')
    plt.legend(loc="best")
    ax3 = plt.subplot(3, 2, 5, sharex=ax1)
    df_signal1['z-axis'].plot(color='k')
    plt.legend(loc="best")

    ax4 = plt.subplot(3, 2, 2, sharey=ax1)
    df_signal2['x-axis'].plot(color='r', label='x-axis')
    plt.legend(loc="best")
    plt.subplot(3, 2, 4, sharex=ax4, sharey=ax2)
    df_signal2['y-axis'].plot(color='g')
    plt.legend(loc="best")
    plt.subplot(3, 2, 6, sharex=ax4, sharey=ax3)
    df_signal2['z-axis'].plot(color='k')
    plt.legend(loc="best")

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def plot_butterworth_filter(fs, lowcut=1, highcut=10, f_type='band', o_range=[2,4,5]):

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in o_range:
        if f_type == 'band':
            b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        elif f_type == 'low':
            b, a = butter_lowpass(lowcut, fs, order=order)
        elif f_type == 'high':
            b, a = butter_highpass(highcut, fs, order=order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],  '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()


def single_file_plots(r_signal, fs, lowcut=2, highcut=0.5, f_type=None, b_order=4, plot_type=3,
                      width=20, height=10, add_to_title="", apply_w_func=False, skip_dc=False, p_legend=False,
                      use_raw_sig=False, use_mag=False, plot_sig=[1,2]):

    freq = fs


    if f_type is not None:
        f_signal_x, p_label = apply_butter_filter(r_signal[:, 0], fs=freq, lowcut=lowcut, highcut=highcut,
                                                  f_type=f_type, order=b_order)
        # print("r_signal[0:5, 0] ", r_signal[0:5, 0])
        # print("f_signal_x[0:5] ", f_signal_x[0:5])
        f_signal_y, _ = apply_butter_filter(r_signal[:, 1], fs=freq, lowcut=lowcut, highcut=highcut,
                                            f_type=f_type, order=b_order)
        # print("r_signal[0:5, 1] ", r_signal[0:5, 1])
        # print("f_signaly[0:5] ", f_signal_y[0:5])
        f_signal_z, _ = apply_butter_filter(r_signal[:, 2], fs=freq, lowcut=lowcut, highcut=highcut,
                                            f_type=f_type, order=b_order)
        # print("r_signal[0:5, 2] ", r_signal[0:5, 2])
        # print("f_signal_z[0:5] ", f_signal_z[0:5])
        f_signal = np.concatenate((f_signal_x, f_signal_y, f_signal_z), axis=1)
        f_signal_m = np.reshape(np.sqrt(f_signal_x**2 + f_signal_y**2 + f_signal_z**2), (f_signal_z.shape[0], 1))
        f_msg = p_label
    else:
        f_msg = "No filtering"

    r_signal_m = np.reshape(np.sqrt(r_signal[:, 0]**2 + r_signal[:, 1]**2 + r_signal[:, 2]**2), (r_signal.shape[0], 1))

    if apply_w_func:
        f_msg += " apply Hamming-W"
    else:
        pass

    print(f_msg)
    if plot_type == 1:
        if len(plot_sig) > 1:
            p_title = add_to_title + p_label
        else:
            p_title = add_to_title

        if not use_mag:
            plot_butter_effect_3axis(r_signal, f_signal, p_title=p_title, width=width, height=height,
                                        p_legend=p_legend, plot_sig=plot_sig)
        else:
            plot_butter_effect_1axis(r_signal_m, f_signal_m, p_title=p_title, label="filtered", p_legend=True,
                                     width=width, height=height, plot_sig=plot_sig)

    # plot_fft(d_array_r)
    if plot_type == 2:
        # Only plotting magnitude frequency spectrum

        if use_raw_sig:
            p_title = add_to_title + "Frequency spectrum (use raw signal %s)" % use_raw_sig
            y = r_signal_m
        else:
            p_title = add_to_title + "Frequency spectrum (wfunc=%s/skip_dc=%s) / %s" % (apply_w_func, skip_dc, p_label)
            y = f_signal_m
        plot_spectra_1axis(y, freq, p_title,
                            apply_window_func=apply_w_func, skip_dc=skip_dc,
                            width=width, height=height)
    if plot_type == 3:

        if use_raw_sig:
            p_title = add_to_title + "Frequency spectrum (use raw signal %s)" % use_raw_sig
            y = r_signal
        else:
            p_title = add_to_title + "Frequency spectrum (wfunc=%s/skip_dc=%s) / %s" % (apply_w_func, skip_dc, p_label)
            y = f_signal
        plot_spectra_3axis(y, freq, p_title,
                            apply_window_func=apply_w_func, skip_dc=skip_dc,
                            width=width, height=height)


def plot_butter_effect_1axis(r_signal, f_signal, label="filtered", p_title="", p_legend=False, height=10, width=20,
                                plot_sig=[1, 2]):

    t = np.arange(r_signal.shape[0])

    plt.figure(figsize=(width, height))
    plt.title(p_title, y=1.08)

    if 1 in plot_sig:
        plt.plot(t, r_signal, label='original signal')
    if 2 in plot_sig:
        plt.plot(t, f_signal, label=label, color='r')
    plt.xlabel('time (seconds)')
    plt.grid(True)
    plt.axis('tight')
    if p_legend:
        # ax1.legend([sig1, sig2], loc="best")
        plt.legend(loc="best")

    plt.show()


def plot_butter_effect_3axis(r_signal, f_signal, label="filtered", p_title="", p_legend=False, height=10, width=20,
                                plot_sig=[1, 2]):

    r_x = r_signal[:, 0]
    r_y = r_signal[:, 1]
    r_z = r_signal[:, 2]
    f_x = f_signal[:, 0]
    f_y = f_signal[:, 1]
    f_z = f_signal[:, 2]
    t = np.arange(r_signal.shape[0])

    plt.figure(figsize=(width, height))
    ax1 = plt.subplot(3, 1, 1)
    plt.title(p_title, y=1.08)
    if 1 in plot_sig:
        sig1, = plt.plot(t, r_x, label='original signal')
    if 2 in plot_sig:
        sig2, = plt.plot(t, f_x, label=label, color='r')
    plt.xlabel('time (seconds)')
    plt.grid(True)
    plt.ylabel('|Amplitude|')
    if p_legend:
        # ax1.legend([sig1, sig2], loc="best")
        ax1.legend(loc="best")

    plt.subplot(3, 1, 2, sharex=ax1)
    if 1 in plot_sig:
        plt.plot(t, r_y, label='Noisy signal')
    if 2 in plot_sig:
        plt.plot(t, f_y, label=label, color='r')
    plt.xlabel('time (seconds)')
    plt.grid(True)
    plt.ylabel('|Amplitude|')

    plt.subplot(3, 1, 3, sharex=ax1)
    if 1 in plot_sig:
        plt.plot(t, r_z, label='Noisy signal')
    if 2 in plot_sig:
        plt.plot(t, f_z, label=label, color='r')
    plt.xlabel('time (seconds)')
    plt.grid(True)
    plt.ylabel('|Amplitude|')

    plt.axis('tight')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def plot_hamming_effect():
    root_dir = get_dir_path("futurocube", "roadrunner")
    d_array1 = load_hdf5_file(os.path.join(root_dir, "20160916_roadrunner_futurocube_[ID9:0:age47]_acc"))
    df1 = tensor_to_pandas(d_array1)
    hamming_w = np.hamming(d_array1.shape[1])
    f_array1 = d_array1 * np.reshape(hamming_w, (1, len(hamming_w), 1))
    df2 = tensor_to_pandas(f_array1)
    plot_3axis_raw_signal_compare(df1, df2)


def plot_sinosoidal_example1(apply_w_func=False):
    # Number of samplepoints
    N = 600
    # sample spacing
    T = 1.0 / 600.0
    x = np.linspace(0.0, N*T, N)
    y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    yf = np.reshape(y, (N, 1))
    # ext_yf = np.concatenate((yf, yf, yf), axis=1)
    plot_spectra_1axis(yf, N, 'without window_func - non-Smooth DFT spectrum x,y and z-axis',
                 apply_window_func=apply_w_func)


def plot_sinosoidal_example2(apply_w_func=False):
    # Number of samplepoints
    N = 600
    # sample spacing
    T = 1.0 / 600.0
    x = np.linspace(0.0, N*T, N)
    y1 = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    y2 = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    y = np.concatenate((y1, y2), axis=0)
    yf = np.reshape(y, (2*N, 1))

    print(yf.shape)
    plot_spectra_1axis(yf, N, 'without window_func - non-Smooth DFT spectrum x,y and z-axis',
                 apply_window_func=apply_w_func)


def plot_single_filter_example(fs, lowcut, highcut, f_type):
    freq = 20  # 20 Hz
    r_signal = (load_file_to_pandas("futurocube", "roadrunner", "20160921_futurocube_roadrunner_[ID4:0:Age7:0:1]_acc.csv")
                        ).as_matrix()

    b_order = 4
    f_signal_x, p_label = apply_butter_filter(r_signal[:, 0], fs=freq, lowcut=lowcut, highcut=highcut,
                                              f_type=f_type, order=b_order)
    f_signal_y, _ = apply_butter_filter(r_signal[:, 1], fs=freq, lowcut=lowcut, highcut=highcut,
                                                f_type=f_type, order=b_order)
    f_signal_z, _ = apply_butter_filter(r_signal[:, 2], fs=freq, lowcut=lowcut, highcut=highcut,
                                                f_type=f_type, order=b_order)
    f_signal = np.concatenate((f_signal_x, f_signal_y, f_signal_z), axis=1)
    plot_butter_effect_3axis(r_signal, f_signal, p_label, p_label)


# single_file_plot()
# plot_butterworth_filter(20, lowcut=0.5, highcut=7, f_type='high')
# plot_sinosoidal_example1()
# plot_single_filter_example(fs=20, lowcut=2, highcut=0.5, f_type='lowhigh')

# freq = 20   # 20 Hz
# r_signal = (load_file_to_pandas("futurocube", "roadrunner", "20160921_futurocube_roadrunner_[ID6:0:Age8:0:1]_acc.csv")
#       ).as_matrix()
# t_cufoff = 15*freq
# signal_fraction = r_signal[t_cufoff:t_cufoff+360, :]
# single_file_plots(signal_fraction, 20, lowcut=2, highcut=0.1, f_type='lowhigh', b_order=5, plot_type=2)


#

# d_array1 = (load_file_to_pandas("futurocube", "roadrunner", "20160921_futurocube_roadrunner_[ID1:1:Age7:1:1]_acc.csv")
#           ).as_matrix()
# d_array_mag = np.sqrt(d_array1[:, 0]**2 + d_array1[:,1]**2 + d_array1[:,2]**2)
# plot_spectra_1axis(d_array_mag, sample_freq, 'without window_func - non-Smooth DFT spectrum x,y and z-axis',
#            apply_window_func=False)
# plot_sinosoidal_example1(False)