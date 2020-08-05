import os
import argparse
import numpy as np
import librosa
import tensorflow as tf
import json
import csv
import audioread
import soundfile
from train import mkdir

"""
Class to handle conversion from audios to tf record
"""
class Wav2record():

    def __init__(self, sr=16000, length=4.0):
        #Desired sample rate
        self.sample_rate = sr
        self.split_duration = length

    def convert(self, audio):
        """
        Serialize an audio waveform
        """
        if len(audio.tolist()) != 64000:
            print("get length of " + str(len(audio.tolist())))
        example = tf.train.Example(features=tf.train.Features(feature={
            'sample_rate': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[self.sample_rate])),
            'audio': tf.train.Feature(float_list=tf.train.FloatList(value=audio.tolist())),
        }))
        return example.SerializeToString()

    def load_split_audio(self, filepath):
        """
        Load an audio of arbitrary length and split into chunk of self.split_duration
        """
        try:
            audio, sr = librosa.load(filepath, self.sample_rate)
            split_duration = int(sr * self.split_duration)
            return np.array_split(audio, np.arange(split_duration, len(audio), split_duration))[:-1]
        except FileNotFoundError:
            print(FileNotFoundError)
            return []

    def write_audio(self, audio_path, writer):
        """
        Process the audio at audio path and write it into a tfrecord.
        """
        audio_list = self.load_split_audio(audio_path)
        to_write = list(map(self.convert, audio_list))
        for string in to_write:
            writer.write(string)

    def write_audios(self, audios_path, writer):
        """
        Process and serialized a batch of audio
        """
        for audio in audios_path:
            try:
                print("writing file: " + str(audio))
                self.write_audio(audio, writer)
            except Exception as e:
                print('skipping file' + str(audio))
                print(str(e))
                pass


def parse_maestro(maestro_meta, maestro_data):
    """
    Parse maestro metadata to know which file belong to which set
    """
    train_files = []
    val_files = []
    test_files = []
    print("Opening json")
    with open(maestro_meta) as json_file:
        data = json.load(json_file)
        for p in data:
            if p['split'] == 'train':
                train_files.append(maestro_data + p['audio_filename'])
            if p['split'] == 'validation':
                val_files.append(maestro_data + p['audio_filename'])
            if p['split'] == 'test':
                test_files.append(maestro_data + p['audio_filename'])
    return train_files, val_files, test_files

def parse_fma(fma_meta, fma_data):
    "Parse fma metadata to know which file belong to which set"
    train_file = []
    val_file = []
    test_file = []
    with open(fma_meta, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[32] == 'small':
                name = row[0].zfill(6)
                print("NAME:")
                print(name)
                file_path = fma_data + '/' + name[:3] + '/' + name + '.mp3'
                if row[31] == 'training':
                    train_file.append(file_path)
                elif row[31] == 'validation':
                    val_file.append(file_path)
                elif row[31] == 'test':
                    test_file.append(file_path)
    return train_file, val_file, test_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    help_ = "Path to dataset of audio"
    parser.add_argument('dataset', help=help_)
    help_ = "Path to dataset metadata"
    parser.add_argument('meta', help=help_)
    help_ = "Dataset name. Either fma or maestro "
    parser.add_argument('name', help=help_)
    help_ = "Path of the target directory"
    parser.add_argument('-t', '--target', help=help_)
    help_ = "Desired sampling rate"
    parser.add_argument("-sr", "--sampling_rate", type=int, help=help_)
    help_ = "Record duration in second"
    parser.add_argument("-d", "--duration", type=float, help=help_)

    args = parser.parse_args()

    duration = 4.0
    sr = 16000
    target = '../' + args.name  + '_dataset' + '/'
    if args.target:
        print("hello")
        target = args.target
    if args.sampling_rate:
        sr = args.sampling_rate
    if args.duration:
        duration = args.duration

    mkdir(target)
    train_writer = tf.io.TFRecordWriter(target + "/train.tfrecord")
    val_writer = tf.io.TFRecordWriter(target + "/val.tfrecord")
    test_writer = tf.io.TFRecordWriter(target + "/test.tfrecord")

    train_file, val_file, test_file = (None, None, None)
    worker = Wav2record(sr, duration)
    if args.name == 'fma':
        train_file, val_file, test_file = parse_fma(args.meta, args.dataset)
    if args.name == 'maestro':
        train_file, val_file, test_file = parse_maestro(args.meta, args.dataset)
    if train_file is not None:
        print("Writing validation set")
        worker.write_audios(val_file, val_writer)
        print("Writing test set")
        worker.write_audios(test_file, test_writer)
        print("Writing training set")
        worker.write_audios(train_file, train_writer)
    else:
        default_writer = tf.io.TFRecordWriter(args.target)
        worker.write_audios(args.daataset, default_writer)

