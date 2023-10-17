import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import IPython.display as ipd
from IPython.display import Audio

matplotlib.use('Agg')

AIR_EMBOLISM_FILE = 'C:/Users/harry/PycharmProjects/Air-Embolism/DBS2.mp3'

data, sr = librosa.load(AIR_EMBOLISM_FILE)


def plot_magnitude_data(signal, title, sr, f_ratio=1):
    ft = np.fft.fft(signal)
    magnitude_data = np.abs(ft)

    plt.figure(figsize=(18, 5))

    frequency = np.linspace(0, sr, len(magnitude_data))
    num_frequency_bins = int(len(frequency) * f_ratio)

    plt.plot(frequency[:num_frequency_bins], magnitude_data[:num_frequency_bins])

    plt.xlabel("Frequency (Hz)")
    plt.title(title)
    plt.savefig("magnitude_plot.png")


def plot_amplitude_data(signal, title, sr):
    plt.figure(figsize=(18, 5))
    librosa.display.waveshow(signal, sr=sr, color='purple')

    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig("amplitude_plot.png")


def plot_spectrogram(signal, title, sr):
    # Compute the Short-Time Fourier Transform (STFT)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)

    # Display the spectrogram
    plt.figure(figsize=(18, 5))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.savefig("spectrogram_plot.png")


plot_spectrogram(data, "AIR_EMBOLISM_SPECTROGRAM", sr)
plot_magnitude_data(data, "AIR_EMBOLISM_FREQUENCY_MAGNITUDE", sr, 0.4)
plot_amplitude_data(data, "AIR_EMBOLISM_AMPLITUDE_TIME", sr)