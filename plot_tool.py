import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy
from librosa.display import specshow
import librosa

def plot_audio(rec_audio, true_audio, path):
    fig, ax = plt.subplots()
    ax.plot(np.arange(true_audio.size), true_audio, 'b-', label='true_audio')
    ax.plot(np.arange(rec_audio.size), rec_audio, 'r-', label='reconstruction')
    plt.legend()
    plt.savefig(path)

def plot_audios(rec_audio_batch, true_audio_batch, out_folder):
    for i in range(len(rec_audio_batch)):
        rec_audio = rec_audio_batch[i]
        true_audio = true_audio_batch[i]
        path = out_folder + 'rec_vs_original_audio' + str(i) + '.png'
        plot_audio(rec_audio, true_audio)

def plot_spectrum(rec_spec, true_spec, path):
    rec_spec = np.swapaxes( 0, 1)
    true_spec = np.swapaxes( 0, 1)
    rec_spec = tf.squeeze(rec_spec)
    true_spec = tf.squeeze(true_spec)
    fig, axes = plt.subplots(ncols=3)
    plt.xlabel("Frequency")
    plt.ylabel("Time in sample")
    ax3, ax1, ax2 = axes
    ax1.set_title("reconstruction spectrum")
    specshow(librosa.amplitude_to_db(rec_spec, ref=np.max), ax=ax1, sr=sr, hop_length=hop_length, y_axis='linear',
             x_axis='s')
    ax2.set_title("true spectrum")
    specshow(librosa.amplitude_to_db(true_spec, ref=np.max), ax=ax2, sr=sr, hop_length=hop_length, y_axis='linear',
             x_axis='s')
    ax3.set_title("hole spectrum")
    time_idx = true_spec.shape[1] // 3
    mask1 = tf.ones(tf.shape(true_spec[:, 0:time_idx]))
    mask2 = tf.zeros(tf.shape(true_spec[:, time_idx:2 * time_idx]))
    mask3 = tf.ones(tf.shape(true_spec[:, 2 * time_idx:]))
    true_spec = true_spec * tf.concat((mask1, mask2, mask3), 1)
    specshow(librosa.amplitude_to_db(true_spec, ref=np.max), ax=ax3, sr=sr, hop_length=hop_length, y_axis='linear',
             x_axis='s')
    fig.colorbar(plt.cm.ScalarMappable())
    plt.savefig(path)
    plt.close()

def plot_spectrums(rec_spec_batch, true_spec_batch, out_folder, sr=None, hop_length=None):
    for i in range(rec_spec_batch.shape[0]):
        path = out_folder + 'rec_vs_original_spectrum_sample_' + str(i) + '.png'
        plot_spectrum(rec_spec_batch[i], true_spec_batch[i])
