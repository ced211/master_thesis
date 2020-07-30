import tensorflow as tf
import numpy as np
import os
import json
from plot_tool import *
from Preprocessing import *
from metrics import *
import matplotlib.pyplot as plt
import scipy


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return


def _eval(model, gen, folder, plot=False):
    result = model.evaluate(gen)
    old_get_audio = gen.get_audio
    if isinstance(result, list):
        out = dict(zip(model.metrics_names, map(str, result)))
    else:
        out = {model.metrics_names, str(result)}
    print(folder)
    with open(folder + 'loss_metrics' + '.json', 'w') as file:
        json.dump(out, file)
    gen.get_audio = True
    sr = 16000
    SNR_audio = []
    for i in range(len(gen)):
        x, y, true = gen.__getitem__(0)
        pred = model.predict_on_batch(x)
        rec = gen.to_audio(pred)
        print("writing batch " + str(i))
        rec_audio = gen.to_audio(pred)
        SNR_audio.append(snr_audio_batch(np.squeeze(true_audio), np.squeeze(rec_audio)))
        if plot:
            if gen.prediction:
                predicted = pred
                truth = y
            else:
                prev, next = x
                predicted = np.concatenate((prev, pred, next), axis=1)
                truth = np.concatenate((prev, y, next), axis=1)
            prefix = folder + 'batch_' + str(i) + '_'
            plot_spectrums(predicted, truth, prefix)
            plot_audios(rec_audio, y, prefix)
            write_audio_batch(rec_audio, y, prefix, gen.sr)
    SNR_audio = np.stack(SNR_audio)
    plt.hist(SNR_audio)
    plt.savefig(folder + "_snr_histogram.png")
    print("SNR mean: " + str(np.mean(SNR_audio)))
    print("SNR std: " + str(np.std(SNR_audio)))
    gen.get_audio = old_get_audio
    return

def evaluate(trained_model, config):
    # Set where to store the evaluation result
    result_directory = './'
    if 'result_directory' in config:
        result_directory = config['result_directory']

    metrics = None
    plot = True
    if 'plot' in config:
        plot = config['plot']

    if 'metrics' in config:
        metrics = config['metrics']
    trained_model.compile(optimizer="adam", loss='mse', metrics=metrics)

    if 'train_generator' in config:
        train_generator = config['train_generator']
        folder = result_directory + 'train/'
        mkdir(folder)
        _eval(trained_model, train_generator, folder, plot)

    if 'val_generator' in config:
        val_generator = config['val_generator']
        folder = result_directory + 'val/'
        mkdir(folder)
        _eval(trained_model, val_generator, folder, plot)

    if 'test_generator' in config:
        test_generator = config['test_generator']
        folder = result_directory + 'test/'
        mkdir(folder)
        _eval(trained_model, test_generator, folder, plot)

    return


