import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy
import json

def count_elem_in_dataset(dataset):
    """
    Count the number of element in dataset by iterating over it.
    Inputs:
        - tf.data.Dataset dataset: the dataset
    """
    dataset = dataset.repeat(1)
    count = 0
    for _ in dataset:
        count += 1
    dataset = dataset.repeat()
    return count

class LoadData:

    def __init__(self, path, sr=16000, audio_frame_length=2, buffer_size=512, batch_size=32, repeat=True):
        """
        Inputs:
            - String path: path to the tfrecord file
            - Int sr: sample rate of the audio record in the tfrecord
            - Float audio_frame_length: Trim the audio record to audio_frame_length second
            - Int buffer_size: size of the buffer
            - Int batch_size: Size of the batch
            - Boolean repeat: whether to repead the record in the tfrecord
        """
        self.path = path
        self.sr = sr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.audio_frame_length = int(audio_frame_length * sr)
        self.length = 0
        self.repeat = repeat

    def parse(self, record):
        """Parse the tfrecord. The tfrecord must contain the waveform in the 'audio' feature
        Inputs:
            - Tensor of string record: serialized audio record
        Output:
            - Tensor of float : The parsed audio waveform"""
        features = {'audio': tf.io.FixedLenFeature([64000], dtype=tf.float32)}
        example = tf.io.parse_single_example(record, features)
        return example['audio']

    def cut_audio(self, audio):
        """Trim an audio to self.audio_frame_length
        Inputs:
            - Tensor audio: A batch of audio of shape [batch, time]"""
        print("audio length in load data: " + str(self.audio_frame_length))

        return tf.image.random_crop(audio, tf.constant([self.audio_frame_length]))

    def create_dataset(self):
        """
        Create a tf.Dataset from the tfrecord specify in self.path
        Outputs:
            - tf.data.Dataset dataset: the dataset containing the waveform stored in the tfrecord
        """
        dataset = tf.data.TFRecordDataset(self.path)
        dataset = dataset.map(self.parse)
        print("Couting element in dataset")
        elems = count_elem_in_dataset(dataset)
        self.length =  elems // self.batch_size
        print("There is: " + str(elems) + " record in the tfrecord file")
        # cut audio
        cutted_audio = dataset.map(self.cut_audio)
        cutted_audio = cutted_audio.batch(self.batch_size, drop_remainder=True)
        cutted_audio = cutted_audio.shuffle(250)
        if self.repeat:
            cutted_audio = cutted_audio.repeat()
        cutted_audio.prefetch(tf.data.experimental.AUTOTUNE)
        return cutted_audio