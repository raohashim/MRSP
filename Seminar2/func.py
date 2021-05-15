import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import wave
import scipy.io.wavfile as wav
import pyaudio
import scipy.signal as sig

def filter_bank(order, subband, windowtype):
    gap = 0.01 #### we keep the same gap value we are still working with 8 subbands
    n = np.arange(order)
    ws = (1.0 / subband) + gap ### deciding upon the cutoff
    ws2 = (1.0 / subband) / 2.0 + gap

    # ideal impulse response (Low-Pass):
    IR = np.sin(ws * np.pi * (n - ((order - 1) / 2))) / (np.pi * (n - ((order - 1) / 2)))
    # ideal impulse response (Low-Pass) for band-pass:
    IR2 = np.sin(ws2 * np.pi * (n - ((order - 1) / 2))) / (np.pi * (n - ((order - 1) / 2)))

    # multiply ideal filter and window:
    hfilt = IR * windowtype
    hfilt2 = IR2 * windowtype ##shorter attenuation regions for bandpass lead to less noise...

    # Nyquist frequency is normalized to 0.5!:
    filterbank = np.empty((subband, order))
    for i in range(subband):
        # this is the lowpass
        if i == 0:
            filterbank[i] = hfilt
        # this is for the highpass
        elif i == subband - 1:
            mod = np.cos(np.pi * np.arange(order))
            filterbank[i] = hfilt * mod
        # this is for the middlebands for which we will use modulation
        else:
            mod = np.cos((((2 * i) + 1) / (2 * subband)) * np.pi * np.arange(order))
            filterbank[i] = hfilt2 * mod
    return filterbank


def filt(input, filterbank, subband):
    for i in range(subband):
        if i == 0:
            if input.ndim == 1:
                conv = np.convolve(input, filterbank[0])
            else:
                conv = np.convolve(input[0], filterbank[0])
            bands = np.empty((subband, conv.size))
            bands[0] = conv
        else:
            if input.ndim == 1:
                bands[i] = np.convolve(input, filterbank[i])
            else:
                bands[i] = np.convolve(input[i], filterbank[i])

    return bands

def reconstructed(subbands, subband_nr):
    for i in range(subband_nr):
        if i == 0:
            reconstSig = subbands[0]
        else:
            reconstSig += subbands[i]

    return reconstSig

def plot(filterBank, numBands, fs):
 
    for i in range(numBands):
        w, H = sig.freqz(filterBank[i])
        # Plot magnitude in dB:
        plt.plot((w / (2 * np.pi)) * fs, 20 * np.log10(np.abs(H)))

