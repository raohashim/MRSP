import scipy.signal as sig
from scipy import signal as sp
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
from sound import sound
from freqz import *


if __name__ == "__main__":
    # Defining the number of subbands equal to 16.
    numBands = 16
    # Defining the block length N as
    N = 16
    # Reading the audio file
    rate, snd = wav.read('Track32.wav')
    snd = snd[:, 0]
    print("Playing original signal")
    #sound(snd, rate)
    # Reshaping it to match the parameters for the dot product
    block_sound = snd.reshape((-1, N))
    print ("Shape", block_sound.shape)
    #print(block_sound.shape)
    # Obtain the Transform Matrix of size of length N
    T = np.fft.fft(np.eye(N))
    #print("T",T.shape) 
    # Transform blocks
    yt = np.dot(block_sound, T)
    print ("yt", yt.shape)
    q = np.asarray(yt).reshape(-1)
    print ("q", q.shape)
    freqz(q)
    # Reconstruction of the signal
    block_recon_sig = np.dot(yt, np.linalg.inv(T))
    #block_recon_sig1 = np.dot(yt,T)
    #block_recon_sig =np.fft.ifft(block_recon_sig1)
    print ("block_recon_sig", block_recon_sig.shape)
    recon_sig = block_recon_sig.reshape(-1)
    print('Reconstructed Signal')
    #sound(recon_sig, rate)

    # freqz(yt)
    fft_fr(T, numBands, rate, "Frequency response of equivalent Filter Bank")
    plt.figure("Original vs Reconstructed Signal")
    plt.plot(snd / np.max(np.abs(snd)), "r-", label="Original signal")
    plt.plot(recon_sig / np.max(np.abs(recon_sig)), "g--", label="Reconstructed signal ")

    plt.show()
