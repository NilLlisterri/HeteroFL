import torch
import matplotlib.pyplot as plt
import librosa


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    time_axis = torch.arange(0, 16000) / sr

    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.plot(time_axis, waveform, linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


def plot_audio(waveform, mfccs):
    fig, axs = plt.subplots(2, 1)
    plot_waveform(waveform, 16000, title="Original waveform", ax=axs[0])
    plot_spectrogram(mfccs, title="spectrogram", ax=axs[1])
    fig.tight_layout()
    plt.show()
