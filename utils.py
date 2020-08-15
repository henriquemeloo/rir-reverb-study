from IPython import display
from scipy import signal
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_wav(path, fs, plot=False, play=False):
    audio = librosa.load(path, mono=True, sr=fs)[0]
    t = np.arange(0, len(audio)*1/fs, 1/fs)
    if plot:
        plt.plot(t, audio)
        plt.show()
    if play:
        return audio, display.Audio(audio, rate=fs)
    else:
        return audio


def oct_bandpass_filter(fc, fs):
    fl = fc / 2 ** (1/6)
    fu = fc * 2 ** (1/6)
    sos = signal.butter(N=4, Wn=[fl, fu], btype='bandpass',
                        fs=fs, output='sos')
    return sos


def plot_filter_bank(fs):
    fig, ax = plt.subplots(figsize=(14,8))
    for fc in [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]:
        sos = oct_bandpass_filter(fc, fs)
        w, H = signal.sosfreqz(sos, fs=fs, worN=2**15)
        ax.plot(w, 20*np.log10(np.abs(H)))
    plt.xlim((10**1, 10**4.5))
    plt.ylim((-100, 10))
    plt.xscale('log')
    plt.xlabel('$f$ (Hz)')
    plt.ylabel('$|H(e^{j2\pi f/f_s})|$ (dB)')
    plt.title('Designed filter bank')


def calculate_t60(rir, fs, fc=None, plot=False):
    if fc:
        sos = oct_bandpass_filter(fc, fs)
        rir = signal.sosfilt(sos, rir)
    abs_signal = np.abs(rir)
    # Schroeder integration
    sch = np.cumsum(abs_signal[::-1]**2)[::-1]
    sch_db = 10 * np.log10(sch / np.max(sch))
    if plot:
        plt.plot(sch_db)
    a = sch_db[np.argmax(sch_db):]
    start_sample = np.argmax(a <= -5)
    stop_sample = np.argmax(a <= -35)
    t60 = 2 * (stop_sample - start_sample) / fs
    return t60


def calculate_c80(rir, fs, fc=None):
    # Remove silence in beggining
    rir = rir[np.argmax(rir):]
    if fc:
        sos = oct_bandpass_filter(fc, fs)
        rir = signal.sosfilt(sos, rir)
    index_80 = int(80e-3 * fs)
    c80 = np.sum(rir[:index_80]**2) / np.sum(rir[index_80:]**2)
    c80dB = 10 * np.log10(c80)
    return c80dB