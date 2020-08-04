import tensorflow as tf


def log(x, base, eps=1.0e-6):
    """Compute the log of x in base base
    Input:
        - Tensor x: Tensor on wich the log is computed element wise
        - Int base: base of the logarithm
    Output:
        Tensor log"""
    numerator = tf.math.log(tf.cast(x + eps, dtype=tf.float32))
    denominator = tf.math.log(tf.constant(base + eps, dtype=numerator.dtype))
    return numerator / denominator


def snr(y_true, y_pred):
    """Compute signal to noise ratio between two signal y_true, y_pred
    Input:
        - Tensor y_true: the reference signal
        - Tensor y_pred: the signal under test
    Output:
        - Float snr: The SNR between y_true, y_pred"""
    num = tf.math.pow(tf.norm(y_true), 2)
    den = tf.math.pow(tf.norm(y_true - y_pred), 2)
    out = log(num / den, 10) * 10
    return out


def snr_batch(y_true, y_pred):
    """compute the average snr between a batch of spectrum of shape [batch, time, freq, 1]
    Input:
        - Tensor y_true: batch of reference signal
        - Tensor y_pred: batch of signal under test
    Output:
        - Tensor snr: Batch of snr between y_true, y_pred"""
    num = tf.math.pow(tf.norm(y_true, axis=(1, 2)), 2)
    den = tf.math.pow(tf.norm(y_true - y_pred, axis=(1, 2)), 2)
    value = log(num / den, 10) * 10
    return tf.math.reduce_mean(tf.boolean_mask(value, tf.math.is_finite(value)))


def snr_audio_batch(y_true, y_pred):
    """Compute the average snr between a batch of audio of shape [batch, time]
    Input:
        - Tensor y_true: batch of reference signal
        - Tensor y_pred: batch of signal under test
    Output:
        - Tensor snr: Batch of snr between y_true, y_pred"""

    num = tf.math.pow(tf.norm(y_true, axis=1), 2)
    den = tf.math.pow(tf.norm(y_true - y_pred, axis=1), 2)
    value = 10 * log(num / den, 10)
    return tf.math.reduce_mean(tf.boolean_mask(value, tf.math.is_finite(value)))


def mse(y_true, y_pred):
    """Compute the average mse over a batch of signal
    Input:
        - Tensor y_true: batch of reference signal
        - Tensor y_pred: batch of signal under test
    Output:
        - Tensor mse: Batch of mse between y_true, y_pred"""
    return tf.math.reduce_mean((y_true - y_pred) ** 2)
