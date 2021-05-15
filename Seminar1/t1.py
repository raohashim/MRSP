import numpy as np
import scipy.signal as signal
import sound 
import matplotlib.pyplot as plt

# loading sound file as array
[s,r] = sound.wavread('Track32.wav')

# Taking first chanel of audio
data = s[:, 0]

# filter
#First is a Low Pass Filter
h1 = signal.remez(64, [0,0.0625, 0.0627, 0.5], [1,0], [1, 100])# nyquist frequency  normalized to 0.5
#Band Pass filters
h2 = signal.remez(64, [0, 0.0625, 0.0627, 0.125, 0.127, 0.5], [0, 1, 0], [100, 1, 100])
h3 = signal.remez(64, [0, 0.125, 0.127, 0.1875, 0.1877, 0.5], [0, 1, 0], [100, 1, 100])
h4 = signal.remez(64, [0, 0.1875, 0.1877, 0.25, 0.27, 0.5], [0, 1, 0], [100, 1, 100])
h5 = signal.remez(64, [0, 0.25, 0.27, 0.3125, 0.3127, 0.5], [0, 1, 0], [100, 1, 100])
h6 = signal.remez(64, [0, 0.3125, 0.3127, 0.375, 0.377, 0.5], [0, 1, 0], [100, 1, 100])
h7 = signal.remez(64, [0, 0.375, 0.377, 0.4375, 0.4377, 0.5], [0, 1, 0], [100, 1, 100])
#High Pass filter
h8 = signal.remez(64, [0, 0.4375, 0.4377, 0.5], [0,1], [100, 1])

# filter implementation
s1 = signal.lfilter(h1,1,data)
s2 = signal.lfilter(h2,1,data)
s3 = signal.lfilter(h3,1,data)
s4 = signal.lfilter(h4,1,data)
s5 = signal.lfilter(h5,1,data)
s6 = signal.lfilter(h6,1,data)
s7 = signal.lfilter(h7,1,data)
s8 = signal.lfilter(h8,1,data)

#Downsampling
s1_ds = s1[0::8]
s2_ds = s2[0::8]
s3_ds = s3[0::8]
s4_ds = s4[0::8]
s5_ds = s5[0::8]
s6_ds = s6[0::8]
s7_ds = s7[0::8]
s8_ds = s8[0::8]

# playing filtered sound of subband 1
#print ("Playinf filtered sound of SB1")
#sound.sound(s1, 32000)
# playing filtered sound of sb 1 after downsampling
#print ("Playinf filtered and ds sound of SB1")
#sound.sound(s1_ds, 4000)

# playing filtered sound of subband 4
print ("Playinf filtered sound of SB4")
sound.sound(s4, 32000)
# playing filtered sound of sb 1 after downsampling
#print ("Playinf filtered and ds sound of SB4")
#sound.sound(s4_ds, 4000)

# frequency response
w, H1 = signal.freqz(h1)
w, H2 = signal.freqz(h2)
w, H3 = signal.freqz(h3)
w, H4 = signal.freqz(h4)
w, H5 = signal.freqz(h5)
w, H6 = signal.freqz(h6)
w, H7 = signal.freqz(h7)
w, H8 = signal.freqz(h8)

# plotting Impulse response and frequency response
figure,(f1,f2) = plt.subplots(2)
f1.plot(h2)
f1.set_xlabel('Sample')
f1.set_ylabel('Value')
f1.set_title('Impulse Response ')

#Freq. Response of all filters
f2.plot(w,20*np.log10(np.abs(H1)),w,20*np.log10(np.abs(H2)),w,20*np.log10(np.abs(H3)),w,20*np.log10(np.abs(H4)),w,20*np.log10(np.abs(H5)),w,20*np.log10(np.abs(H6)),w,20*np.log10(np.abs(H7)),w,20*np.log10(np.abs(H8)))

#f2.plot(w,20*np.log10(np.abs(H2)))
f2.set_xlabel('Normalized frequency')
f2.set_ylabel('Magnitude in dB')
f2.set_title('Magnitude Frequency response')

plt.show()

########################SYNTHESIS#####################
# up-sampling

up_1 = np.zeros(len(data))
up_2 = np.zeros(len(data))
up_3 = np.zeros(len(data))
up_4 = np.zeros(len(data))
up_5 = np.zeros(len(data))
up_6 = np.zeros(len(data))
up_7 = np.zeros(len(data))
up_8 = np.zeros(len(data))

up_1[::8] = s1_ds
up_2[::8] = s2_ds
up_3[::8] = s3_ds
up_4[::8] = s4_ds
up_5[::8] = s5_ds
up_6[::8] = s6_ds
up_7[::8] = s7_ds
up_8[::8] = s8_ds

# synthesis filter
filsyn1 = signal.lfilter(h1,1,up_1)
filsyn2 = signal.lfilter(h2,1,up_2)
filsyn3 = signal.lfilter(h3,1,up_3)
filsyn4 = signal.lfilter(h4,1,up_4)
filsyn5 = signal.lfilter(h5,1,up_5)
filsyn6 = signal.lfilter(h6,1,up_6)
filsyn7 = signal.lfilter(h7,1,up_7)
filsyn8 = signal.lfilter(h8,1,up_8)
#Reconstructing the signal
rec_sig = filsyn1 + filsyn2 + filsyn3 + filsyn4 + filsyn5 + filsyn6 + filsyn7 + filsyn8
#Play reconstructed signal
sound.sound(rec_sig, 32000)
