
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
import pyaudio

def play_file(audio, sampling_rate, channels):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sampling_rate,
                    output=True)
    sound = (audio.astype(np.int16).tostring())
    stream.write(sound)
    stream.stop_stream()
    stream.close()
    p.terminate()
    return

def plotIRFR(filterBank, numBands, fs, string):
    plt.figure(string)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude of FR [dB]")
    axes = plt.gca()
    axes.set_ylim([-20, 5])
    axes.set_xlim([0, 16000])
    for i in range(numBands):
        w, H = sig.freqz(np.flipud(filterBank[:, i]), whole=True)
        # Plot magnitude in dB:
        plt.plot((w / (2 * np.pi)) * fs, 20 * np.log10(np.abs(H)))


def plotIRFR2(signal, fs, string):
    plt.figure(string)
    # Impulse response:
    plt.subplot(2, 1, 1)
    plt.xlabel("Samples [n]")
    plt.ylabel("Amplitude")
    plt.plot(signal)
    plt.subplot(2, 1, 2)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude of FR [dB]")
    w, H = sig.freqz(signal)
    # Plot magnitude in dB:
    plt.plot((w / (2 * np.pi)) * fs, 20 * np.log10(np.abs(H)))


def ramp(x):
    return np.maximum(0, x)


if __name__ == "__main__":
    f = open('original_signal.txt', 'wb')
    f2 = open('compressed_signal.txt', 'wb')

    numBands = 8
    # numTaps = 32 #filter order
    N = 8
    bandElimination = False

    # snd, fs = sound.wavread('Track32.wav')
    fs, snd = wav.read('Track32.wav')
    snd = snd[:, 0].astype(float)
    # get a ramp function for the testing part as well
    rampSig = ramp(np.linspace(-3, 4, 8))
    # this we do to separate the signal into blocks
    blockSnd = snd.reshape((-1, N))

    # Obtain the DCT4 Transform Matrix for the subbands
    T = np.zeros((N, N))
    for n in range(N):
        for k in range(N):
            T[n, k] = 0.5 * np.cos((np.pi / N) * (n + 0.5) * (k + 0.5)) * np.sqrt(2.0 / N)


    # Transformed Blocks (Analysis) multiply the signal with the dct4 matrix
    audio = np.dot(blockSnd, T)
    ramp = np.dot(rampSig, T)
    if bandElimination == True:
        audio[:, 3::] = 0

    # Block Reconstruction (synthesis)
    blockReconstSig = np.dot(audio, T)
    reconstRamp = np.dot(ramp, T)

    # Reconstruction of signal
    sigReconst = blockReconstSig.reshape(-1)

    # Play reconstructed signal
    play_file(sigReconst, fs, 1)
    play_file(snd, fs, 1)

    # Plot frecuency response of filter bank and each Subband
    plotIRFR(T, numBands, fs, "Analysis Filter Bank")  #
    plotIRFR(T.T, numBands, fs, "Synthesis Filter Bank")
    # for i in range(numBands):
    #    plotIRFR3(yt[:,i], fs, "Transformed Subband {}".format(i))

    # Plot Original and Reconstructed Signal
    plt.figure("Sound Signals")
    plt.plot(snd / np.max(np.abs(snd)), "b", label="original")
    plt.plot(sigReconst / np.max(np.abs(sigReconst)), "g", label="reconstructed ")
    plt.legend(loc="upper left")

    plt.show()
