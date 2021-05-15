
import sound
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import pyaudio
from x2polyphase import *
from polmatmult import *
from Fafoldingmatrix import *
from polyphase2x import *


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
    plt.plot((w / (2 * np.pi)) * fs, 20 * np.log10(np.abs(H)))


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

def ramp(x):
    return np.maximum(0, x)


if __name__ == "__main__":
    numBands = 8
    # numTaps = 32 #filter order
    N = 8
    bandElimination = False
    fs, snd = wav.read('Track32.wav')
    snd = snd[:, 0]

    rampSig = ramp(np.linspace(-3, 4, 8))

    # MDCT sine window:
    h = np.sin(np.pi / (2 * N) * (np.arange(2 * N) + 0.5))

    # Folding Matrix:
    Fa = Famatrix(h)

    # Delay Matrix D(z):
    Dp = np.zeros((N, N, 2))
    Dp[:, :, 0] = np.diag(np.hstack((np.zeros(int(N / 2)), np.ones(int(N / 2)))))
    print ("Do", Dp[:, :, 0] )
    Dp[:, :, 1] = np.diag(np.hstack((np.ones(int(N / 2)), np.zeros(int(N / 2)))))
    print ("Do", Dp[:, :, 1] )

    # Fa*D(z):
    Faz = polmatmult(Fa, Dp)

    T = np.zeros((N, N, 1))
    for n in range(N):
        for k in range(N):
            T[n, k] = 0.5 * np.cos((np.pi / N) * (n + 0.5) * (k + 0.5)) * np.sqrt(2.0 / N)

    # Obtain analysis Matrix Hz
    Hz = polmatmult(Faz, T)

    # Polyphase representation of Input signal
    xp_audio = x2polyphase(snd, N)
    xp_ramp = x2polyphase(rampSig, N)

    # Apply analysis filter
    yt_audio = polmatmult(xp_audio, Hz)
    yt_ramp = polmatmult(xp_ramp, Hz)



    if bandElimination == True:
        yt_audio[:, 3::] = 0

    # Compute the inverse folding matrix for the Synthesis:
    Fs = np.zeros(Fa.shape)
    Fs[:, :, 0] = np.linalg.inv(Fa[:, :, 0])


    # Inverse Delay Matrix with delay:
    Dpi = np.zeros((N, N, 2))
    Dpi[:, :, 1] = np.diag(np.hstack((np.zeros(int(N / 2)), np.ones(int(N / 2)))))
    Dpi[:, :, 0] = np.diag(np.hstack((np.ones(int(N / 2)), np.zeros(int(N / 2)))))

    # Obtain inverse DCT4 transform matrix
    invT = np.zeros(T.shape)
    invT[:, :, 0] = np.linalg.inv(T[:, :, 0])

    # Obtain Synthesis Matrix Gz
    Gz = polmatmult(polmatmult(invT, Dpi), Fs)


    # Obtain reconstructed signals
    xrec_audio = polmatmult(yt_audio, Gz)
    xrec_ramp = polmatmult(yt_ramp, Gz)
    sigReconst = polyphase2x(xrec_audio)
    sigReconst = sigReconst.flatten()
    reconstRamp = polyphase2x(xrec_ramp)
    reconstRamp = reconstRamp.flatten()


    # Play reconstructed signal

    play_file(snd, fs, 1)
    play_file(sigReconst, fs, 1)
    # Plot frecuency response of filter bank and each Subband
    plotIRFR(Hz[:, :, 1], numBands, fs, "Analysis Filter Bank")
    plotIRFR(Gz[:, :, 1].T, numBands, fs, "Synthesis Filter Bank")
    # for i in range(numBands):
    #    plotIRFR3(yt[:,i], fs, "Transformed Subband {}".format(i))

    # Plot Original and Reconstructed Signal
    plt.figure("Sound Signals")
    plt.plot(snd / np.max(np.abs(snd)), "b", label="original")
    plt.plot(sigReconst / np.max(np.abs(sigReconst)), "g", label="reconstructed ")
    plt.legend(loc="upper left")

    plt.show()
