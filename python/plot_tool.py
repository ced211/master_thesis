import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
from librosa.display import specshow
import librosa


def plot_audio(rec_audio, true_audio, path):
    fig, ax = plt.subplots()
    ax.plot(np.arange(true_audio.size), true_audio, 'b-', label='true_audio')
    ax.plot(np.arange(rec_audio.size), rec_audio, 'r-', label='reconstruction')
    plt.legend()
    plt.savefig(path)
    plt.close()


def plot_audios(rec_audio_batch, true_audio_batch, out_folder):
    for i in range(len(rec_audio_batch)):
        rec_audio = rec_audio_batch[i]
        true_audio = true_audio_batch[i]
        path = out_folder + 'rec_vs_original_audio_sample_' + str(i) + '.png'
        plot_audio(rec_audio, true_audio, path)

def plot_spectrum_pred(rec_spec, true_spec, path, sr, hop_length):
    rec_spec = np.swapaxes(rec_spec, 0, 1)
    true_spec = np.swapaxes(true_spec, 0, 1)
    rec_spec = tf.squeeze(rec_spec)
    true_spec = tf.squeeze(true_spec)
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    plt.xlabel("Frequency")
    plt.ylabel("Time in sample")
    ax1, ax2 = axes
    ax1.set_title("reconstruction")
    specshow(librosa.amplitude_to_db(rec_spec, ref=np.max), ax=ax1, sr=sr, hop_length=hop_length, y_axis='linear',
             x_axis='s')
    ax2.set_title("ground truth")
    specshow(librosa.amplitude_to_db(true_spec, ref=np.max), ax=ax2, sr=sr, hop_length=hop_length, y_axis='linear',
             x_axis='s')
    plt.savefig(path)
    plt.close()

def plot_spectrum_inp(rec_spec, true_spec, path, sr, hop_length):
    rec_spec = np.swapaxes(rec_spec, 0, 1)
    true_spec = np.swapaxes(true_spec, 0, 1)
    rec_spec = tf.squeeze(rec_spec)
    true_spec = tf.squeeze(true_spec)
    fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True)
    plt.xlabel("Frequency")
    plt.ylabel("Time in sample")
    ax3, ax1, ax2 = axes
    ax1.set_title("reconstruction")
    specshow(librosa.amplitude_to_db(rec_spec, ref=np.max), ax=ax1, sr=sr, hop_length=hop_length, y_axis='linear',
             x_axis='s')
    ax2.set_title("ground truth")
    specshow(librosa.amplitude_to_db(true_spec, ref=np.max), ax=ax2, sr=sr, hop_length=hop_length, y_axis='linear',
             x_axis='s')
    ax3.set_title("hole")
    time_idx = true_spec.shape[1] // 3
    mask1 = tf.ones(tf.shape(true_spec[:, 0:time_idx]))
    mask2 = tf.zeros(tf.shape(true_spec[:, time_idx:2 * time_idx]))
    mask3 = tf.ones(tf.shape(true_spec[:, 2 * time_idx:]))
    true_spec = true_spec * tf.concat((mask1, mask2, mask3), 1)
    specshow(librosa.amplitude_to_db(true_spec, ref=np.max), ax=ax3, sr=sr, hop_length=hop_length*0.75, y_axis='linear',
             x_axis='s')
    fig.colorbar(plt.cm.ScalarMappable())
    plt.savefig(path)
    plt.close()


def plot_spectrums(rec_spec_batch, true_spec_batch, out_folder, sr, hop_length, prediction):
    for i in range(rec_spec_batch.shape[0]):
        path = out_folder + 'rec_vs_original_spectrum_sample_' + str(i) + '.png'
        if prediction:
            plot_spectrum_pred(rec_spec_batch[i], true_spec_batch[i], path, sr, hop_length)
        else:
            plot_spectrum_inp(rec_spec_batch[i], true_spec_batch[i], path, sr, hop_length)


def write_audio_batch(rec_audios, true_audios, out_folder, sr):
    for i in range(rec_audios.shape[0]):
        scipy.io.wavfile.write(out_folder + 'rec_sample_' + str(i) + '.wav', sr, rec_audios[i])
        scipy.io.wavfile.write(out_folder + 'or_sample_' + str(i) + '.wav', sr, true_audios[i])
