import sound
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
import scipy.optimize as opt
from func import *

numBands = 8
numTaps = 32 #filter order
    
#snd, fs = sound.wavread('Track32.wav')
fs, data = wav.read('Track32.wav')
snd = data[:,0]
#print("PLAYING: ORIGINAL SIGNAL")
#sound.sound(snd,fs)
    
#Filter bank for each window
filterBank,win = createFilterBank(numBands, numTaps, fs)
   
#Separation of bands
bands = filtering(snd, filterBank, numBands)
    
#Downsampling
downsample = bands[:,::numBands]
    
#Upsampling
upsample = np.empty((numBands, np.size(bands,1)))
upsample[:,::numBands] = downsample
    
#Final filtering
reconst = filtering(upsample, filterBank, numBands)
    
#Reconstructed signal
sigReconst = reconstruct(reconst, numBands)
            
#Play each band after downsampling
#print("PLAYING: 1ST BAND (LOW FREQS) - AFTER DOWNSAMPLING")
#sound.sound(downsample[0],int(fs/numBands))
#print("PLAYING: 2ND BAND - AFTER DOWNSAMPLING")
#sound.sound(downsample[1],int(fs/numBands))
#print("PLAYING: 3RD BAND - AFTER DOWNSAMPLING")
#sound.sound(downsample[2],int(fs/numBands))
#print("PLAYING: 4TH BAND (HIGH FREQS) - AFTER DOWNSAMPLING")
#sound.sound(20*downsample[3],int(fs/numBands))
#print("PLAYING: RECONSTRUCTED SIGNAL - OPTIMIZED WINDOW")
#sound.sound(sigReconst,32000)
    
#Plot Window Function
plotIRFR2(win, fs, "Optimized Winodow Function")
 
#Plot impulse and frecuency response of filter bank
plotIRFR(filterBank, numBands, fs, "Filter Bank")
    
#Plot Original and Reconstructed Signal
#plotIRFR2(snd/np.max(np.abs(snd)), fs, "Original Sound Signal")
#plotIRFR2(sigReconst/np.max(np.abs(sigReconst)), fs, "Reconstructed Sound Signal")
plt.figure("Sound Signals")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.plot(snd/np.max(snd),'b',label='Orignal')
plt.plot(sigReconst/np.max(sigReconst),'g--',label='Recons')
plt.legend()
plt.show()
