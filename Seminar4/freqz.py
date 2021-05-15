import scipy.signal as sig
from scipy import signal as sp
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
from sound import sound

def freqz(b, a=1, whole = False, axisFreqz = None, axisPhase = None):
    
    w, h = sp.freqz(b, a, worN=512, whole=whole)
    #w = w/np.pi
    fig = plt.figure()
    plt.title('Digital filter frequency response')
    plt.subplot(2,1,1)
    
    plt.plot(w, 20 * np.log10(abs(h)), 'm')
    plt.ylabel('Amplitude (dB)')
    plt.xlabel('Normalized Frequency')
    plt.grid()
    if axisFreqz is not None:
        plt.axis(axisFreqz)
    
    plt.subplot(2,1,2)
    #angles = np.unwrap(np.angle(h))
    angles = np.angle(h)
    plt.plot(w, angles, 'r')
    plt.ylabel('Angle (radians)')
    plt.xlabel('Normalized Frequency')
    plt.grid()

    if axisPhase is not None:
        plt.axis(axisPhase)
    
    plt.show()
    return h
    
def fft_fr(filterBank, numBands, rate, string):
    plt.figure(string)
    plt.xlabel("Frequency in Hz")
    plt.ylabel("Magnitude of frequency response in dB")
    axes = plt.gca()
    axes.set_ylim([-10, 30])
    #axes.set_xlim([0, 16000])
    for i in range(numBands):
        w, H = sig.freqz(np.flipud(filterBank[:, i]), whole=True)
        # Plot magnitude in dB:
        plt.plot((w / (2 * np.pi)) * rate, 20 * np.log10(np.abs(H)))

