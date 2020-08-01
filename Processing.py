import tensorflow as tf
import numpy as np
import librosa
import scipy
from metrics import log
from metrics import *
import math
import SpecgramsHelper

class Processeur(SpecgramsHelper.SpecgramsHelper, tf.keras.utils.Sequence):

    def __init__(self, sr, dataset, dataset_length, audio_length, window_size=0.040, overlap=0.75, get_audio=False, prediction_only=False):
        """
        Input:
            - Int sr: sample rate of the audio in dataset
            - tf.Dataset dataset: dataset of waveform
            - Int dataset_length: nb of batch in the dataset
            - Float audio_length: length of a sample in the dataset in second
            - Float windows_size: Desired size of the analysis windows of the stft in second.
                The effective size will be two times the closest power equal or bigger than the size of the windows in sample.
            - Float overlap: Overlap when sliding the analysis windows during the stft. Must be between 0 and 1.
            - Boolean get_audio: Whether to also get the audio when calling self.__getitem__().
            - Boolean prediction only: Whether to give the next frame. True for prediction and false for inpainting.
        """
        window_size = int(sr * window_size)
        self.length = dataset_length
        self.dataset_ite = iter(dataset)
        self.get_audio = get_audio
        self.sr = sr
        audio_length *= sr
        audio_length //= 3
        window_size = 2 * 2 ** math.ceil(math.log(window_size, 2) - 0.0000001)
        hop_size = int(window_size * (1. - overlap))
        self.audio_length = int(audio_length)
        self.prediction = prediction_only
        # init specgram helper
        super().__init__(window_size, hop_size, overlap=overlap,
                         sample_rate=sr)

    def process_data(self, batch):
        """"Process the audio batch and transform it into feature
        Input:
            - Tensor batch: batch of audio to process of shape [batch, time]
        Output:
            - Tensor processed_data: spectrum of shape [batch, time, freq, 1]"""
        return self.waves_to_lin_spectrum(tf.expand_dims(batch, -1))

    def __getitem__(self, item):
        """Give a batch of:
            x, y, true_audio if self.get_audio == True
            or x, y if self.get_audio == False.
            x is the neural network input and y its target.
            x = (specs1, specs3) in case of inpainting, when self.prediction_only == false
            x = specs1 in caseof prediction, when self.prediction_only == true
            y = specs2
        Input:
            - Int item: unused
        Output:
            - Tensor spec1 : spectrum of the first third of the audios
            - Tensor spec2: spectrum of the second third of the audios
            - Tensor specs3: spectrum of the last third of the audios
            - Tensor true_audio: the waveform of the second third of the audio
        Note:
            - The audio are the ones inside the dataset given during the object creation
            """
        audio_batch = next(self.dataset_ite)
        audio1 = audio_batch[:, 0:self.audio_length]
        audio2 = audio_batch[:, self.audio_length: 2 * self.audio_length]
        audio3 = audio_batch[:, 2 * self.audio_length: 3 * self.audio_length]
        specs1 = self.process_data(audio1)
        specs3 = self.process_data(audio3)
        specs2 = self.process_data(audio2)
        if not self.prediction:
            if not self.get_audio:
                return (specs1, specs3), specs2
            else:
                return (specs1, specs3), specs2, audio2
        else:
            if not self.get_audio:
                return specs1, specs2
            else:
                return specs1, specs2, audio2

    def __len__(self):
        """Return the number of batch in one epoch
        Output:
            - Int nb_batch: number of batch in one epoch"""
        return self.length

    def to_audio(self, spectrum_batch):
        """Convert spectrum back to audio. Use Griffinlin for phase estimation.
        Input:
            - Tensor spectrum: spectrum to convert
        Output:
            - Tensor audio: batch of reconstructed audio"""
        return self.lin_spectrum_to_waves(spectrum_batch)
