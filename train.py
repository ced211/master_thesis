import DataLoader
import tensorflow as tf
import argparse
import IGAN
import Processing
import os
import datetime


def mkdir(path):
    """Create a directory if it doesn't exist. Do nothing Otherwise
    Inputs:
        - String path: path of the directory to create"""
    try:
        # Create target Directory
        os.makedirs(path)
        print("Directory ", path, " Created ")
    except FileExistsError:
        print("Directory ", path, " already exists")
    return


def create_pipeline(data_path, batch_size, audio_frame_length):
    """Create a pipeline to pre and post process the data.
    Inputs:
        - String data_path: path to the tfrecord file
        - Int batch_size: size of a batch
        - Float audio_frame_length: Length to which the audio will be trimmed in second
    Outputs:
        - Processeur pipeline: the pipeline to perform pre and post processing
        - Int sr: The sampling rate of the audio record in data_path"""

    data_loader = DataLoader.LoadData(data_path, sr=16000, batch_size=batch_size, audio_frame_length=audio_frame_length)
    dataset = data_loader.create_dataset()
    pipeline = Processing.Processeur(data_loader.sr, dataset, data_loader.length, audio_frame_length, window_size=0.025,
                                     overlap=0.75)
    return pipeline, data_loader.sr


def set_option():
    """
    Parse the command and return the setting.
    Output:
        - String train_path: path to the tfrecord with the audio record of the training set
        - String val_path: path to the tfrecord with the audio record of the validation set
        - String test_path: path to the tfrecord with the audio record of the testing set
        - Float audio_length: Length to which the audio will be trimmed in second
        - Int batch_size: Size of a batch
        - String log_path: path to the folder where the log will be saved
        - Int epoch: umber of epoch for training the model
        - String model_name: Either igan for inpainting or pgan for prediction
        - String ckpt_path: path to the folder where the checkpoint will be saved

    """
    parser = argparse.ArgumentParser(description='Audio in painting')
    help_ = "Path to Checkpoint folder"
    parser.add_argument("-ckpt", "--ckpt", help=help_)
    help_ = "Model type. choose between igan or pgan"
    parser.add_argument("-m", "--model", help=help_)
    help_ = "path to training data"
    parser.add_argument("-tr", "--train", help=help_)
    help_ = "path to validation data"
    parser.add_argument("-v", "--val", help=help_)
    help_ = "path to test data"
    parser.add_argument("-te", "--test", help=help_)
    help_ = "Number of epoch to train"
    parser.add_argument("-e", "--epoch", help=help_)
    help_ = "Path to log folder"
    parser.add_argument("-l", "--log", help=help_)
    help_ = "size of an audio frame in second"
    parser.add_argument("-s", "--size", type=float, help=help_)
    help_ = "Batch size"
    parser.add_argument("-b", "--batch", type=int, help=help_)

    args = parser.parse_args()

    # default value
    model_name = 'igan'
    train_path = '../fma_dataset/train.tfrecord'
    val_path = '../fma_dataset/val.tfrecord'
    test_path = '../fma_dataset/test.tfrecord'
    audio_length = 3 * 0.064
    batch_size = 256
    epoch = 200
    log_path = './log/' + model_name + '/' + str(audio_length) + '/'
    ckpt_path = './ckpt/' + model_name + '/' + str(audio_length) + '/'
    mkdir(log_path)
    mkdir(ckpt_path)

    if args.train:
        train_path = args.train
    if args.val:
        val_path = args.val
    if args.test:
        test_path = args.test
    if args.size:
        audio_length = size
    if args.batch:
        batch_size = args.batch
    if args.log:
        log_path = args.log
    if args.epoch:
        epoch = epoch
    if args.ckpt:
        ckpt_path = args.ckpt
    return train_path, val_path, test_path, audio_length, batch_size, log_path, epoch, model_name, ckpt_path


def init_model(ckpt, pipeline, model_name, log_path=None):
    """
    Initialize the model
    Inputs:
        - String ckpt: Path to the folder where the checkpoint are saved
        - Processeur pipeline: Pre and post processeur
        - String model_name: Either 'igan' for inpainting or 'pgan' for prediction
    Outputs:
        - IGAN model: the model
    """
    fig_path = None
    model = None
    if log_path is not None:
        # create summary writer
        summary_writer = tf.summary.create_file_writer(
            log_path + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        fig_path = log_path + '/fig/'
        mkdir(fig_path)
    if model_name == 'igan':
        sample = pipeline.__getitem__(1)[0][0][0]
        model = IGAN.IGAN(input_shape=sample.get_shape(), checkpoint_dir=ckpt, summary_writer=summary_writer,
                          fig_path=fig_path)
        model.restore(ckpt)
    if model is None:
        print("Unkown model name")
    return model


if __name__ == "__main__":

    train_path, val_path, test_path, audio_length, batch_size, log_path, epoch, model_name, ckpt = set_option()

    # Create pipeline
    train_pipeline, sr = create_pipeline(train_path, batch_size, audio_length)
    val_pipeline, sr = create_pipeline(val_path, batch_size, audio_length)
    test_pipeline, sr = create_pipeline(test_path, batch_size, audio_length)
    model = init_model(ckpt, test_pipeline, model_name, log_path)
    if model is not None:
        model.fit(train_pipeline, val_pipeline, epoch, 0)
    else:
        print("Unknow model type. Exit")
