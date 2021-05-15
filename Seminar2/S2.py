import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import wave
import scipy.io.wavfile as wav
import pyaudio
import scipy.signal as sig
import sound
from func import *


fs, data = wav.read("Track32.wav")
chan = data[:, 1]
# downsampling upsampling value
P = 8 
# subband number
M = 8 
#filter order
N = 64 

#Windows definition
#Rectangular window:
rect=1
#Hanning Window:
hann = 0.5-(0.5*np.cos((2*np.pi/N)*(np.arange(N)+0.5)))
#Sine Window:
sine = np.sin((np.pi/N)*(np.arange(N)+0.5))
#Kaiser Window:
kaiser = np.kaiser(N,8) # where 8 is the beta coeff that deals with the attenuation

#Creating Filter Banks
fb_Rect = filter_bank(N, M,  rect)
fb_Hann = filter_bank(N, M,  hann)
fb_Sine = filter_bank(N, M,  sine)
fb_Kais = filter_bank(N, M,  kaiser)

#subband separation and Convolution of Filter with I/P Signal
bands_Rect = filt(chan, fb_Rect, M)
bands_Hann = filt(chan, fb_Hann, M)
bands_Sine = filt(chan, fb_Sine, M)
bands_Kais = filt(chan, fb_Kais, M)

#Downsampling
down_Rect = bands_Rect[:,::P]
down_Hann = bands_Hann[:,::P]
down_Sine = bands_Sine[:,::P]
down_Kais = bands_Kais[:,::P]

#Upsampling
up_Rect = np.empty((M, np.size(bands_Rect,1)))
up_Hann = np.empty((M, np.size(bands_Hann,1)))
up_Sine = np.empty((M, np.size(bands_Sine,1)))
up_Kais = np.empty((M, np.size(bands_Kais,1)))
up_Rect[:,::P] = down_Rect
up_Hann[:,::P] = down_Hann
up_Sine[:,::P] = down_Sine
up_Kais[:,::P] = down_Kais


reco_Rect = filt(up_Rect, fb_Rect, M)
reco_Hann = filt(up_Hann, fb_Hann, M)
reco_Sine = filt(up_Sine, fb_Sine, M)
reco_Kais = filt(up_Kais, fb_Kais, M)

#Reconstructed signal
reconstructed_Rect = reconstructed(reco_Rect, M)
reconstructed_Hann = reconstructed(reco_Hann, M)
reconstructed_Sine = reconstructed(reco_Sine, M)
reconstructed_Kais = reconstructed(reco_Kais, M)

#sound.sound(chan, 32000)
#sound.sound(reconstructed_Rect , 32000)
#sound.sound(reconstructed_Hann, 32000)
#sound.sound(reconstructed_Sine, 32000)
sound.sound(reconstructed_Kais, 32000)


#Plot the LP Freq Resp. of all filters
plt.figure("LP FILTER Freq. Resp. of All Filters")
plt.xlabel("Number of filter-taps [n]")
plt.ylabel("Filter coefficients")
#LP Freq. Resp. of Rect Filt.
wr,hr = sig.freqz(fb_Rect[0])
plt.plot((wr / (2 * np.pi)) * fs, 20 * np.log10(np.abs(hr)),'b-', label='LP Rect')
#LP Freq. Resp. of Sine Filt.
ws,hs = sig.freqz(fb_Sine[0])
plt.plot((ws / (2 * np.pi)) * fs, 20 * np.log10(np.abs(hs)),'g-', label='LP Sine')
#LP Freq. Resp. of Hanning Filt.
wh,hh = sig.freqz(fb_Hann[0])
plt.plot((wh / (2 * np.pi)) * fs, 20 * np.log10(np.abs(hh)),'r-', label='LP Hanning')
#LP Freq. Resp. of Kaiser Filt.
wk,hk = sig.freqz(fb_Kais[0])
plt.plot((wk / (2 * np.pi)) * fs, 20 * np.log10(np.abs(hk)),'m-', label='LP Kaiser')
plt.legend()

#Frequency response of all Filters
plt.figure("Frquency reponse of all Filters")
plt.subplot(411)
plt.subplot(411).set_title('Rectangular Window')
plt.subplot(411).set(ylabel='[dB]')
plot(fb_Rect, M, fs )

plt.subplot(412)
plt.subplot(412).set_title('Sine Window')
plt.subplot(412).set(ylabel='[dB]')
plot(fb_Sine, M, fs )

plt.subplot(413)
plt.subplot(413).set_title('Hanning Window')
plt.subplot(413).set(ylabel='[dB]')
plot(fb_Hann, M, fs )

plt.subplot(414)
plt.subplot(414).set_title('Kaiser Window')
plt.subplot(414).set(ylabel='[dB]')
plot(fb_Kais, M, fs )
plt.xlabel("[Hz]")

#Orignla and Reconstructed
plt.figure("Sound Signals")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.plot(chan/np.max(chan),'b',label='Orignal')
plt.plot(reconstructed_Rect/np.max(reconstructed_Rect),'r-',label='Rect Rec.')
plt.plot(reconstructed_Hann/np.max(reconstructed_Hann),'g--',label='Hann Rec')
plt.plot(reconstructed_Sine/np.max(reconstructed_Sine),'y-',label='Sine Rec')
plt.plot(reconstructed_Kais/np.max(reconstructed_Kais),'m--',label='Kaiser rec')
plt.legend()
plt.show()

