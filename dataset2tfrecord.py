import os

import numpy as np
import librosa
import tensorflow as tf
import json
import csv
import audioread
import soundfile
"""
Class to handle conversion from audios to tf record
"""
class Wav2record:
    #Desired sample rate
    target_sr = 16000
    #desired length of an audio sample in the tfrecord
    split_duration = 4.0

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
            audio, sr = librosa.load(filepath, self.target_sr)
            split_duration = int(sr * self.split_duration)
            return np.array_split(audio, np.arange(split_duration, len(audio), split_duration))[:-1]
        except FileNotFoundError:
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
                print("writing file: " + str(audios_path))
                self.write_audio(audio, writer)
            except:
                print('skipping file')
                pass


def parse_maestro(maestro_path):
    """
    Parse maestro metadata to know which file belong to which set
    """
    train_files = []
    val_files = []
    test_files = []
    print("Opening json")
    with open(maestro_path + '/maestro-v2.0.0.json') as json_file:
        data = json.load(json_file)
        for p in data:
            if p['split'] == 'train':
                train_files.append(maestro_path + p['audio_filename'])
            if p['split'] == 'validation':
                val_files.append(maestro_path + p['audio_filename'])
            if p['split'] == 'test':
                test_files.append(maestro_path + p['audio_filename'])
    return train_files, val_files, test_files

def parse_fma(fma_path):
    "Parse fma metadata to know which file belong to which set"
    train_file = []
    val_file = []
    test_file = []
    with open(fma_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[32] == 'small':
                print(row[32] + ', ' + row[31] + ' id: ' + row[0])
                name = row[0].zfill(6)
                file_path = '../fma_dataset/fma_dataset/' + name + '.wav'
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
    parser.add_argument('target', help=help_)
    help_ = "Desired sampling rate"
    parser.add_argument("-sr", "--sampling_rate", type=int, help=help_)
    help_ = "Record duration in second"
    parser.add_argument("-d", "--duration", type=float, help=help_)

    args = parser.parse_args()

    duration = 4.0
    sr = 16000
    if args.sampling_rate:
        sr = sampling_rate
    if args.duration:
        duration = args.duration

    train_writer = tf.io.TFRecordWriter(args.target + "/train.tfrecord")
    val_writer = tf.io.TFRecordWriter(args.target + "/val.tfrecord")
    test_writer = tf.io.TFRecordWriter(args.target + "/val.tfrecord")
    default_writer = tf.io.TFRecordWriter(args.target)

    train_file, val_file, test_file = None
    worker = Wav2record()
    if args.name == 'fma':
        train_file, val_file, test_file = parse_fma(args.meta)
    if args.name == 'maestro':
        train_file, val_file, test_file = parse_maestro(args.meta)
    if train_file is not None:
        worker.write_audios(val_file, val_writer)
        worker.write_audios(test_file, test_writer)
        worker.write_audios(train_file, train_writer)
    else:
        worker.write_audios(args.daataset, default_writer)

