"""Helper object for transforming audio to spectrum.
"""
#Inspired from https://github.com/magenta/magenta/blob/master/magenta/models/gansynth/lib/specgrams_helper.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

from librosa import ParameterError
import numpy as np
import tensorflow.compat.v1 as tf
from plot_tool import save_audio

def polar2rect(mag, phase_angle):
    """Convert polar-form complex number to its rectangular form."""
    mag = tf.complex(mag, tf.convert_to_tensor(0.0, dtype=mag.dtype))
    phase = tf.complex(tf.cos(phase_angle), tf.sin(phase_angle))
    return mag * phase


class SpecgramsHelper(object):
    """Helper functions to compute specgrams."""

    def __init__(self, n_fft, n_hop, overlap,
                 sample_rate, discard_dc=True):
        """
        Input:
            - Int n_fft: analysis windows size in sample of the stft.
            - Int n_hop: Hop size when performing the stft
            - Float overlap: Overlap between two consecutive analysis windows.
            - Int sample_rate: sample_rate of the audio to process
            - Boolean discard_dc: whether to discard dc.
        """
        self._overlap = overlap
        self._sample_rate = sample_rate
        self._discard_dc = discard_dc
        self._nfft = n_fft
        self._nhop = n_hop
        self._pad_l, self._pad_r = self._get_padding()
        self._eps = 1.0e-6

    def _safe_log(self, x):
        return tf.log(x + self._eps)

    def _get_padding(self):
        """Infer left and right padding for STFT."""
        padding_l = self._nfft // 2
        padding_r = self._nfft // 2
        return padding_l, padding_r

    def waves_to_stfts(self, waves):
        """
            Convert from waves to complex stfts.
        Inputs:
            - Tensor waves: Tensor of the waveform, shape [batch, time, 1].
        Outputs:
            - Tensor stfts: Complex64 tensor of stft, shape [batch, time, freq, 1]."""
        waves_padded = tf.pad(waves, [[0, 0], [self._pad_l, self._pad_r], [0, 0]])
        stfts = tf.signal.stft(
            waves_padded[:, :, 0],
            frame_length=self._nfft,
            frame_step=self._nhop,
            fft_length=self._nfft,
            window_fn=tf.signal.hann_window,
            pad_end=False)[:, :, :, tf.newaxis]
        stfts = stfts[:, :, 1:] if self._discard_dc else stfts[:, :, :-1]
        return stfts

    def stfts_to_waves(self, stfts):
        """
        Convert from complex stfts to waves.
        Inputs:
            - Tensor stfts: Complex64 tensor of stft, shape [batch, time, freq, 1].
        Outputs:
            - Tensor waves: Tensor of the waveform, shape [batch, time, 1]."""
        waves_resyn = tf.signal.inverse_stft(
            stfts=stfts[:, :, :, 0],
            frame_length=self._nfft,
            frame_step=self._nhop,
            fft_length=self._nfft,
            window_fn=tf.signal.inverse_stft_window_fn(
                frame_step=self._nhop, forward_window_fn=tf.signal.hann_window))[:, :, tf.newaxis]
        # Python does not allow rslice of -0
        if self._pad_r == 0:
            return waves_resyn[:, self._pad_l:]
        else:
            return waves_resyn[:, self._pad_l:-self._pad_r]

    def waves_to_lin_spectrum(self, waves):
        """Convert waves to lin-spectrum
        Inputs:
            - Tensor waves: float tensor of shape [batch, time]
        Output:
            - Tensor mag: Magnitude of the stft of waves. Tensor of shape [batch, time, freq, 1]
        """
        stfts = self.waves_to_stfts(waves)
        mag = tf.abs(stfts)
        return mag

    def lin_spectrum_to_waves(self, spectrum):
        """Convert Magnitude spectrum to waves. Estimate the phase with the Griffin-Lim algorithm.
        Inputs:
            - Tensor spectrum: float tensor of shape [batch, time, freq, 1]
        Outputs:
            - Tensor waves: float tensor of shape [batch, time, 1]"""
        spectrum = tf.convert_to_tensor(spectrum)
        S =  spectrum[:, :, :, 0]
        phase_angle = self.custom_griffinlim(S, 100)
        return self.stfts_to_waves(tf.expand_dims(polar2rect(S, phase_angle), -1)).numpy()

    # Adapted from: https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py
    def griffinlim(self, S, n_iter=100, momentum=0.99, init=None, random_state=None):

        '''Approximate magnitude spectrogram inversion using the "fast" Griffin-Lim algorithm [1]_ [2]_.

        Given a short-time Fourier transform magnitude matrix (`S`), the algorithm randomly
        initializes phase estimates, and then alternates forward- and inverse-STFT
        operations.
        Note that this assumes reconstruction of a real-valued time-domain signal, and
        that `S` contains only the non-negative frequencies (as computed by
        `core.stft`).

        .. [1] Perraudin, N., Balazs, P., & Søndergaard, P. L.
            "A fast Griffin-Lim algorithm,"
            IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (pp. 1-4),
            Oct. 2013.

        .. [2] D. W. Griffin and J. S. Lim,
            "Signal estimation from modified short-time Fourier transform,"
            IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.

        Parameters
        ----------
        S : tf.tensor [shape=(n_fft / 2 + 1, t), non-negative]
            An array of short-time Fourier transform magnitudes as produced by
            `core.stft`.

        n_iter : int > 0
            The number of iterations to run

        momentum : number >= 0
            The momentum parameter for fast Griffin-Lim.
            Setting this to 0 recovers the original Griffin-Lim method [1]_.
            Values near 1 can lead to faster convergence, but above 1 may not converge.

        init : None [default] or a Tensor
            If 'None' (the default), then phase values are initialized randomly
            according to `random_state`.  This is recommended when the input `S` is
            a magnitude spectrogram with no initial phase estimates.

            If it is a tensor, then the phase is initialized from init. This is useful when
            an initial guess for phase can be provided, or when you want to resume
            Griffin-Lim from a previous output.

        random_state : None, int, or np.random.RandomState
            If int, random_state is the seed used by the random number generator
            for phase initialization.

            If `np.random.RandomState` instance, the random number
            generator itself.

            If `None`, defaults to the current `np.random` object.


        Returns
        -------
        y : tf.tensor [shape=(S.shape)]
            phase reconstructed from S

        '''

        if random_state is None:
            rng = np.random
        elif isinstance(random_state, int):
            rng = np.random.RandomState(seed=random_state)
        elif isinstance(random_state, np.random.RandomState):
            rng = random_state

        if momentum > 1:
            warnings.warn('Griffin-Lim with momentum={} > 1 can be unstable. '
                          'Proceed with caution!'.format(momentum))
        elif momentum < 0:
            raise ParameterError('griffinlim() called with momentum={} < 0'.format(momentum))
        S = tf.expand_dims(S, -1)
        S = S.numpy()

        # using complex64 will keep the result to minimal necessary precision
        angles = np.empty(S.shape, dtype=np.complex64)
        if init is None:
            # randomly initialize the phase
            angles[:] = np.exp(2j * np.pi * rng.rand(*S.shape))
        else:
            angles = np.exp(1j * tf.cast(init, tf.complex64))

        # And initialize the previous iterate to 0
        rebuilt = 0.
        angles = tf.Variable(initial_value=tf.convert_to_tensor(angles))
        S = tf.convert_to_tensor(S, dtype=tf.complex64)

        for _ in range(n_iter):
            # Store the previous iterate
            tprev = rebuilt

            # Invert with our current estimate of the phases
            inverse = self.stfts_to_waves(S * angles)

            # Rebuild the spectrogram
            rebuilt = self.waves_to_stfts(inverse)

            # Update our phase estimates
            angles.assign(rebuilt - (momentum / (1 + momentum)) * tprev)
            angles.assign(angles / tf.cast(tf.math.abs(angles) + 1e-16, dtype=tf.complex64))

        # Return the final phase estimates
        return tf.math.angle(angles)[:, :, :, 0]