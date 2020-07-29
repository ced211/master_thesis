import tensorflow as tf
import numpy as np
import librosa
import scipy
from metrics import log
from specgrams_helper import SpecgramsHelper
from metrics import *
import math


class ThreeSpecFrame(SpecgramsHelper, tf.keras.utils.Sequence):

    def __init__(self, sr, dataset, dataset_length, audio_length, window_size=0.040, overlap=0.75, get_audio=False,
                 process_name='linear_spectrum', normalize=False, prediction_only=False, normalization_process = 'standarize',
                 frame_size=None):
        """
        process name: linear_spectrum ifreq or stft#
        """
        self.frame_size = frame_size
        window_size = int(sr * window_size)
        self.length = dataset_length
        self.dataset_ite = iter(dataset)
        self.get_audio = get_audio
        self.sr = sr
        audio_length *= sr
        audio_length //= 3
        window_size = 2 * 2 ** math.ceil(math.log(window_size, 2) - 0.0000001)
        hop_size = int(window_size * (1. - overlap))
        self.process_name = process_name
        self.normalize = False
        self.audio_length = int(audio_length)
        self.prediction = prediction_only
        self.norm_process = normalization_process
        print("Gap length: " + str(self.audio_length))
        # init specgram helper
        ifreq = False
        if process_name == 'ifreq':
            ifreq = True
        super().__init__(window_size, hop_size, overlap=overlap,
                         sample_rate=sr, ifreq=ifreq)
        if normalize and self.norm_process == 'standarize':
            self.mean, self.std = self.fit_normalize()
            self.normalize = True
        elif normalize and self.norm_process == 'minmax':
            self.min, self.max = self.fit_normalize()
            self.normalize = True


    def normalize_data(self, batch):
        eps = 0.000000001
        if self.norm_process == 'standarize':
            if self.process_name == 'stft' or self.process_name == 'ifreq':
                out = tf.stack((
                    (batch[:, :, :, 0] - self.mean[0]) / self.std[0],
                    (batch[:, :, :, 1] - self.mean[1]) / self.std[1]),
                    axis=-1
                )
                return out
            if self.process_name == 'linear_spectrum':
                return (batch - self.mean[0]) / self.std

        if self.norm_process == 'minmax':
            param = list(zip(self.min, self.max))
            if self.process_name == 'stft' or self.process_name == 'ifreq':
                out = tf.stack((
                    self.rescale(batch[:, :, :, 0], param[0]),
                    self.rescale(batch[:,:,:, 1], param[1])),
                    axis=-1)
                return out
            elif self.process_name == 'linear_spectrum':
                return self.rescale(batch, param[0])

    def rescale(self, batch, param):
        if self.norm_process == 'minmax':
            num = (batch - param[0]) * 2
            den = param[1] - param[0]
            return -1 + num/den
        if self.norm_process == 'standarize':
            return (batch - param[0]) / param[1]

    def rescale_back(self, batch, param):
        if self.norm_process == 'minmax':
            num = (batch + 1) * (param[1] - param[0])
            den = 2
            return num/den + param[0]
        if self.norm_process == 'standarize':
            return (batch - param[0]) / param[1]

    def denormalize(self, batch):
        if not self.normalize:
            return batch
        if self.norm_process == 'standarize':
            if self.process_name == 'stft' or self.process_name == 'ifreq':
                return tf.stack((
                    batch[:, :, :, 0] * self.std[0] + self.mean[0],
                    batch[:, :, :, 1] * self.std[1] + self.mean[1]),
                    axis=-1
                )
            if self.process_name == 'linear_spectrum':
                return batch * self.std[0] + self.mean[0]
        elif self.norm_process == 'minmax':
            param = list(zip(self.min, self.max))
            if self.process_name == 'stft' or self.process_name == 'ifreq':
                return tf.stack((
                    self.rescale_back(batch[:, :, :, 0], param[0]),
                    self.rescale_back(batch[:, :, :, 1], param[1])),
                    axis=-1
                    )
            if self.process_name == 'linear_spectrum':
                return self.rescale_back(batch, param[0])

    def process_data(self, batch):
        if self.process_name == 'ifreq':
            processed = self.waves_to_specgrams(tf.expand_dims(batch, -1))
        if self.process_name == 'linear_spectrum':
            processed = self.waves_to_lin_spectrum(tf.expand_dims(batch, -1))
        if self.process_name == 'stft':
            processed = self.waves_to_stfts(tf.expand_dims(batch, -1))
            processed = tf.concat((tf.math.abs(processed), tf.math.angle(processed)), -1)
        if self.process_name == 'phase2':
            stfts = self.waves_to_stfts(tf.expand_dims(batch, -1))
            processed = tf.concat((tf.math.abs(stfts), tf.math.angle(stfts)), -1)
        if self.process_name == 'phase':
            stfts = self.waves_to_stfts(tf.expand_dims(batch, -1))
            processed = tf.math.angle(stfts)
        if self.process_name == 'ifreq':
            processed = self.waves_to_specgrams(tf.expand_dims(batch, -1))
        if self.process_name == 'stft-real-im':
            stfts = self.waves_to_stfts(tf.expand_dims(batch, -1))
            processed = tf.concat((tf.math.real(stfts), tf.math.imag(stfts)), -1)
        return processed
    def __getitem__(self, item):
        """Give a batch of:
            ((spec1, spec3) spec2) where spec are formed from the audio sliced in three part"""
        audio_batch = next(self.dataset_ite).numpy()
        if self.frame_size is not None:
            idx1 = int(self.frame_size[0] * self.sr)
            idx2 = int(self.frame_size[1] * self.sr) + idx1
            idx3 = int(self.frame_size[2] * self.sr) + idx2
            audio1 = audio_batch[:, 0:idx1]
            audio2 = audio_batch[:, idx1:idx2]
            audio3 = audio_batch[:, idx2:idx3]
        else:
            audio1 = audio_batch[:, 0:self.audio_length]
            audio2 = audio_batch[:, self.audio_length: 2 * self.audio_length]
            audio3 = audio_batch[:, 2 * self.audio_length: 3 * self.audio_length]
        specs1 = self.process_data(audio1)
        specs3 = self.process_data(audio3)
        specs2 = self.process_data(audio2)
        if self.normalize:
            specs1 = self.normalize_data(specs1)
            specs2 = self.normalize_data(specs2)
            specs3 = self.normalize_data(specs3)
        self.spec1_shape = tf.shape(specs1)
        self.spec2_shape = tf.shape(specs2)
        self.spec3_shape = tf.shape(specs3)

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
        return self.length

    def fit_normalize(self):
        """Compute mean and standard deviation on the dataset"""
        if self.norm_process =='standarize':
            x = [0, 0]
            x_square = [0, 0]
            for i in range(self.length):
                if not self.prediction:
                    if self.get_audio:
                        (batch1, batch2), _, _ = self.__getitem__(i)
                    else:
                        (batch1, batch2), _ = self.__getitem__(i)
                    batch = tf.concat((batch1, batch2), 0)
                else:
                    if self.get_audio:
                        batch, _, _ = self.__getitem__(i)
                    else:
                        batch, _ = self.__getitem__(i)
                mag = batch[:, :, :, 0]
                x[0] += tf.math.reduce_mean(mag)
                x_square[0] += tf.math.reduce_mean(mag ** 2)
                if self.process_name == 'stft' or self.process_name == 'ifreq':
                    phase = batch[:, :, :, 1]
                    x[1] += tf.math.reduce_mean(phase)
                    x_square[1] += tf.math.reduce_mean(phase ** 2)
            x = np.asarray(x)
            x_square = np.asarray(x_square)
            x /= self.length
            x_square /= self.length
            mean = x
            std = [0, 0]
            std[0] = (x_square[0] - x[0] ** 2) ** 1 / 2
            std[1] = (x_square[1] - x[1] ** 2) ** 1 / 2
            print("std: " + str(std))
            print("mean: " + str(mean))
            self.normalize = True
            return mean, std
        if self.norm_process == "minmax":
            max = [-float('Inf'), -float('Inf')]
            min = [float('Inf'), float('Inf')]
            for i in range(self.length):
                if not self.prediction:
                    if self.get_audio:
                        (batch1, batch2), _, _ = self.__getitem__(i)
                    else:
                        (batch1, batch2), _ = self.__getitem__(i)
                    batch = tf.concat((batch1, batch2), 0)
                else:
                    if self.get_audio:
                        batch, _, _ = self.__getitem__(i)
                    else:
                        batch, _ = self.__getitem__(i)
                mag = batch[:, :, :, 0]
                batch_max = tf.math.reduce_max(mag)
                batch_min = tf.math.reduce_min(mag)
                if max[0] < batch_max:
                    max[0] = batch_max
                if min[0] > batch_min:
                    min[0] = batch_min
                if self.process_name == 'stft' or self.process_name == 'ifreq':
                    phase = batch[:, :, :, 1]
                    batch_max = tf.math.reduce_max(phase)
                    batch_min = tf.math.reduce_min(phase)
                    if max[1] < batch_max:
                        max[1] = batch_max
                    if min[1] > batch_min:
                        min[1] = batch_min
        return min, max

    def to_audio(self, spectrum_batch, hot_griffin=False):
        # print("spectrum batch shape: " + str(spectrum_batch.shape))
        if self.normalize:
            spectrum_batch = self.denormalize(spectrum_batch)
        if self.process_name == 'linear_spectrum':
            return self.lin_spectrum_to_waves(spectrum_batch)
        if self.process_name == 'stft':
            if not hot_griffin:
            # print("prediction shape: " + str(spectrum_batch.shape))
                stft = tf.multiply(tf.cast(spectrum_batch[:, :, :, 0], tf.complex64), tf.math.exp(1j * tf.cast(spectrum_batch[:, :, :, 1], tf.complex64)))
            # print("stft shape: " + str(tf.shape(stft)))
                return self.stfts_to_waves(tf.expand_dims(stft, -1)).numpy()
            else:
                mag = spectrum_batch[:, :, :, 0]
                phase = spectrum_batch[:, :, :, 1]
                phase = self.custom_griffinlim(mag, init=tf.expand_dims(phase, -1))
                stft = tf.cast(mag, tf.complex64) * tf.math.exp(1j * tf.cast(phase, tf.complex64))
            return self.stfts_to_waves(tf.expand_dims(stft, -1)).numpy()

        if self.process_name == 'ifreq':
            return self.specgrams_to_waves(spectrum_batch).numpy()

        if self.process_name == 'stft-real-im':
            print("spectrum batch: " + str(tf.shape(spectrum_batch)))
            stft = tf.complex(spectrum_batch[:, :, :, 0], spectrum_batch[:, :, :, 1])
            return self.stfts_to_waves(tf.expand_dims(stft, -1)).numpy()

    def snr(self, y_true, y_pred):
        return snr_batch(self.denormalize(y_true), self.denormalize(y_pred))


class MagSpec(ThreeSpecFrame, tf.keras.utils.Sequence):
    def __init__(self, sr, dataset, dataset_length, audio_length, window_size=0.040, overlap=0.75, get_audio=False,
                 process_name='linear_spectrum', normalize=False, normalization_process = 'standarize'):
        super().__init__(sr, dataset, dataset_length, audio_length, window_size=window_size, overlap=overlap,
                         get_audio=get_audio,
                         process_name=process_name, normalize=normalize, normalization_process=normalization_process)
        self.__getitem__(0)

    def __getitem__(self, item):
        if not self.prediction:
            if not self.get_audio:
                (specs1, specs3), specs2 = super().__getitem__(item)
                return tf.tuple((tf.concat((specs1, tf.zeros(tf.shape(specs2)), specs3), 1),
                                 tf.concat((specs1, specs2, specs3), 1)))
            else:
                (specs1, specs3), specs2, audio = super().__getitem__(item)
                return tf.tuple((tf.concat((specs1, tf.zeros(tf.shape(specs2)), specs3), 1),
                                 tf.concat((specs1, specs2, specs3), 1),
                                 audio))
        else:
            if not self.get_audio:
                specs1, specs2 = super().__getitem__(item)
                return tf.concat((specs1, tf.zeros(tf.shape(specs2))), 1), tf.concat((specs1, specs2), 1)
            else:
                specs1, specs2, audio = super().__getitem__(item)
                return tf.concat((specs1, tf.zeros(tf.shape(specs2))), 1), tf.concat((specs1, specs2), 1), audio

    def fit_normalize(self):
        """Compute mean and standard deviation on the dataset"""
        if self.norm_process == 'standarize':
            x = [0, 0]
            x_square = [0, 0]
            for i in range(self.length):
                if self.get_audio:
                    batch, _, _ = self.__getitem__(i)
                else:
                    batch, _ = self.__getitem__(i)
                mag = batch[:, :, :, 0]
                x[0] += tf.math.reduce_mean(mag)
                x_square[0] += tf.math.reduce_mean(mag ** 2)
                if self.process_name == 'stft' or self.process_name == 'ifreq':
                    phase = batch[:, :, :, 1]
                    x[1] += tf.math.reduce_mean(phase)
                    x_square[1] += tf.math.reduce_mean(phase ** 2)
            x = np.asarray(x)
            x_square = np.asarray(x_square)
            x /= self.length
            x_square /= self.length
            mean = x
            std = [0, 0]
            std[0] = (x_square[0] - x[0] ** 2) ** 1 / 2
            std[1] = (x_square[1] - x[1] ** 2) ** 1 / 2
            print("std: " + str(std))
            print("mean: " + str(mean))
            self.normalize = True
            return mean, std
        if self.norm_process == "minmax":
            max = [-float('Inf'), -float('Inf')]
            min = [float('Inf'), float('Inf')]
            for i in range(self.length):
                if self.get_audio:
                    batch, _, _ = self.__getitem__(i)
                else:
                    batch, _ = self.__getitem__(i)
                mag = batch[:, :, :, 0]
                batch_max = tf.math.reduce_max(mag)
                batch_min = tf.math.reduce_min(mag)
                if max[0] < batch_max:
                    max[0] = batch_max
                if min[0] > batch_min:
                    min[0] = batch_min
                if self.process_name == 'stft' or self.process_name == 'ifreq':
                    phase = batch[:, :, :, 1]
                    batch_max = tf.math.reduce_max(phase)
                    batch_min = tf.math.reduce_min(phase)
                    if max[1] < batch_max:
                        max[1] = batch_max
                    if min[1] > batch_min:
                        min[1] = batch_min
        return min, max

    def slice(self, batch):
        return batch[:, self.spec1_shape[1]:self.spec1_shape[1] + self.spec2_shape[1], :, :]

    def snr(self, y_true, y_pred):
        y_true = self.slice(y_true)
        y_pred = self.slice(y_pred)
        return super().snr(y_true, y_pred)

    def mse(self, y_true, y_pred):
        y_true = self.slice(y_true)
        y_pred = self.slice(y_pred)
        return tf.keras.losses.mean_absolute_error(y_true, y_pred)

    def scaled_mse(self, y_true, y_pred):
        y_true = self.denormalize(y_true)
        y_pred = self.denormalize(y_pred)
        return self.mse(y_true, y_pred)

    def to_audio(self, spectrum_batch):
        return super().to_audio(spectrum_batch[:, self.spec1_shape[1]:self.spec1_shape[1] + self.spec2_shape[1], :, :])

class MagSpec2channel(ThreeSpecFrame, tf.keras.utils.Sequence):
    def __init__(self, sr, dataset, dataset_length, audio_length, window_size=0.040, overlap=0.75, get_audio=False,
                 process_name='linear_spectrum', normalize=False, normalization_process = 'standarize'):
        super().__init__(sr, dataset, dataset_length, audio_length, window_size=window_size, overlap=overlap,
                         get_audio=get_audio,
                         process_name=process_name, normalize=normalize, normalization_process=normalization_process)
        self.__getitem__(0)

    def __getitem__(self, item):
        (specs1, specs3), specs2 = super().__getitem__(item)
        if not self.get_audio:
            return tf.concat((specs1, specs3), -1), specs2
        else:
            (specs1, specs3), specs2, audio = super().__getitem__(item)
            return tf.concat((specs1, specs3), -1), specs2, audio



class PhaseGen(SpecgramsHelper, tf.keras.utils.Sequence):
    def __init__(self, sr, dataset, dataset_length, audio_length, window_size=0.040, overlap=0.75, get_audio=False):
        """
        process name: linear_spectrum ifreq or stft
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
        self.normalize = False
        self.audio_length = int(audio_length)
        print("Gap length: " + str(self.audio_length))
        super().__init__(window_size, hop_size, overlap=overlap,
                         sample_rate=sr, ifreq=False)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        audio_batch = next(self.dataset_ite).numpy()
        audio1 = audio_batch[:, 0:self.audio_length]
        audio2 = audio_batch[:, self.audio_length: 2 * self.audio_length]
        audio3 = audio_batch[:, 2 * self.audio_length: 3 * self.audio_length]

        phase1 = tf.math.angle(self.waves_to_stfts(tf.expand_dims(audio1, -1)))
        phase2 = tf.math.angle(self.waves_to_stfts(tf.expand_dims(audio2, -1)))
        phase3 = tf.math.angle(self.waves_to_stfts(tf.expand_dims(audio3, -1)))
        phase1 /= 2 * math.pi
        phase2 /= 2 * math.pi
        phase3 /= 2 * math.pi

        if not self.get_audio:
            return (phase1, phase3), phase2
        else:
            return (phase1, phase3), phase2, audio2

    def to_audio(self, spectrum_batch):
        stft = tf.multiply(tf.cast(spectrum_batch[:, :, :, 0], tf.complex64),
                           tf.math.exp(1j * tf.cast(spectrum_batch[:, :, :, 1] * 2 * math.pi, tf.complex64)))
        return self.stfts_to_waves(tf.expand_dims(stft, -1)).numpy()




def count_elem_in_dataset(dataset):
    dataset = dataset.repeat(1)
    count = 0
    for _ in dataset:
        count += 1
    dataset = dataset.repeat()
    return count
