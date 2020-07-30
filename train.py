import DataLoader
import tensorflow as tf
import argparse
import IGAN
import Processing
import os
import datetime

def mkdir(path):
    try:
        # Create target Directory
        os.makedirs(path)
        print("Directory ", path, " Created ")
    except FileExistsError:
        print("Directory ", path, " already exists")
    return

def create_pipeline(data_path, batch_size, audio_frame_length):
    data_loader = DataLoader.LoadData(data_path,sr=16000, batch_size=batch_size, audio_frame_length=audio_frame_length)
    dataset = data_loader.create_dataset()
    pipeline = Processing.Processeur(data_loader.sr, dataset, data_loader.length, audio_frame_length, window_size=0.025,
                              overlap=0.75)
    return pipeline, data_loader.sr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio in painting')
    help_ = "Path to Checkpoint folder"
    parser.add_argument("-ckpt", "--cktpt_path", help=help_)
    help_ = "Model type. choose between igan or pgan"
    parser.add_argument("-m", "--model")
    help_ = "path to training data"
    parser.add_argument("-tr", "--train")
    help_ = "path to validation data"
    parser.add_argument("-v", "--val")
    help_ = "path to test data"
    parser.add_argument("-te", "--test")
    help_ = "Number of epoch to train"
    parser.add_argument("-e", "--epoch")
    help_ = "Path to log folder"
    parser.add_argument("-l", "--log")
    help_ = "size of an audio frame in second"
    parser.add_argument("-s", "--size", type=float)
    help_ = "Batch size"
    parser.add_argument("-b", "--batch", type=int)

    args = parser.parse_args()

    #default value
    model_name = 'igan'
    train_path = '../fma_dataset/train.tfrecord'
    val_path = '../fma_dataset/val.tfrecord'
    test_path = '../fma_dataset/test.tfrecord'
    audio_length = 3 * 0.064
    batch_size = 256
    epoch = 3
    log_path = './log/' + model_name + '/' + str(audio_length)
    ckpt_path = './ckpt/' + model_name + '/' + str(audio_length)
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

    #Create pipeline
    train_pipeline, sr = create_pipeline(train_path, batch_size, audio_length)
    val_pipeline, sr = create_pipeline(val_path, batch_size, audio_length)
    test_pipeline, sr = create_pipeline(test_path, batch_size, audio_length)

    #create summary writer
    summary_writer = tf.summary.create_file_writer(
        log_path + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    fig_path = log_path + '/fig/'
    mkdir(fig_path)
    model = None
    if model_name == 'igan':
        sample = test_pipeline.__getitem__(1)[0][0][0]
        print(sample.shape)
        model = IGAN.IGAN(input_shape=sample.get_shape(), summary_writer=summary_writer, checkpoint_dir=ckpt_path, fig_path=fig_path)
        model.restore(ckpt_path)
    if model is not None:
        model.fit(train_pipeline, val_pipeline, epoch, 0)
    else:
        print("Unknow model type. Exit")


