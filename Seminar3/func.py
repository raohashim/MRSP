import sound
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
import scipy.optimize as opt


numBands = 8
numTaps = 32
numfreqsamples = 512
gap = 0.0003
n = np.arange(numTaps)
ws = (1.0/numBands)/2.0 + gap
IR = np.sin(ws*np.pi*(n-((numTaps-1)/2)))/(np.pi*(n-((numTaps-1)/2)))   

def createFilterBank(numBands, numTaps, fs):
    #Create window by means of Optimization
    minout = opt.minimize(errfunc,np.random.rand(numTaps))
    hfilt = IR*minout.x
        
    #Nyquist frequency is normalized to 0.5!:
    h = np.empty((numBands, numTaps))
    for i in range(numBands):
        #If first Band --> Create Lowpass
        if i == 0:
            h[i] = hfilt
        #If last Band --> Create Highpass
        elif i == numBands-1:
            #mod = np.cos(np.pi*np.arange(numTaps))
            mod = np.cos((np.pi/numTaps)*np.arange(numTaps)*(numTaps+0.5))
            h[i] = hfilt*mod
        #If middle band --> Create Bandpass
        else:
            #mod = np.cos((((2*i)+1)/(2*numBands))*np.pi*np.arange(numTaps))
            mod = np.cos((np.pi/numTaps)*np.arange(numTaps)*(((2*i)+1)*(numTaps/(2*numBands))+0.5))
            h[i] = hfilt*mod    
    
    return h,minout.x


def plotIRFR(filterBank, numBands, fs, string):
    
    plt.figure(string)
    #Impulse response:
    plt.subplot(2,1,1)
    plt.xlabel("Number of filter-taps [n]")
    plt.ylabel("Filter coefficients")
    for i in range(numBands):
        plt.plot(filterBank[i])

    #Frequency response:
    plt.subplot(2,1,2)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude of FR [dB]")
    for i in range(numBands):
        w,H=sig.freqz(filterBank[i])
        #Plot magnitude in dB:
        plt.plot((w/(2*np.pi))*fs,20*np.log10(np.abs(H)))
        
def plotIRFR2(signal, fs, string):
    
    plt.figure(string)
    #Impulse response:
    plt.subplot(2,1,1)
    plt.xlabel("Samples [n]")
    plt.ylabel("Amplitude")
    plt.plot(signal)

    #Frequency response:
    plt.subplot(2,1,2)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude of FR [dB]")
    w,H=sig.freqz(signal)
    #Plot magnitude in dB:
    plt.plot((w/(2*np.pi))*fs,20*np.log10(np.abs(H)))
        

def filtering(sig, filterBank, numBands):
    for i in range(numBands):
        if i == 0:
            if sig.ndim == 1:
                conv = np.convolve(sig, filterBank[0])
            else:
                conv = np.convolve(sig[0], filterBank[0])
            bands = np.empty((numBands, conv.size))
            bands[0] = conv
        else:
            if sig.ndim == 1:
                bands[i] = np.convolve(sig, filterBank[i])
            else:
                bands[i] = np.convolve(sig[i], filterBank[i])
            
    return bands

def reconstruct(bandsInfo, numBands):
    for i in range(numBands):
        if i == 0:
            reconstSig = bandsInfo[0]
        else:
            reconstSig += bandsInfo[i]
            
    return reconstSig  

def filtering(sig, filterBank, numBands):
    for i in range(numBands):
        if i == 0:
            if sig.ndim == 1:
                conv = np.convolve(sig, filterBank[0])
            else:
                conv = np.convolve(sig[0], filterBank[0])
            bands = np.empty((numBands, conv.size))
            bands[0] = conv
        else:
            if sig.ndim == 1:
                bands[i] = np.convolve(sig, filterBank[i])
            else:
                bands[i] = np.convolve(sig[i], filterBank[i])
            
    return bands

def reconstruct(bandsInfo, numBands):
    for i in range(numBands):
        if i == 0:
            reconstSig = bandsInfo[0]
        else:
            reconstSig += bandsInfo[i]
            
    return reconstSig

def errfunc(h):
    #desired passband:
    pb=int(numfreqsamples/8)
    tb=int(numfreqsamples/32)
    #t =h*IR
    w, H = sig.freqz(h*IR,1,numfreqsamples)
    print("H len", len(H))
    H_desired=np.concatenate((1*np.ones(pb), 0*np.ones(numfreqsamples-pb)))
    sig.freqz(H_desired,1,numfreqsamples)
    weights = np.concatenate((np.ones(pb), 1000*np.ones(numfreqsamples-pb)))
    err = np.sum(np.abs(H-H_desired)*weights)
    return err
