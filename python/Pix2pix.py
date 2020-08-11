import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
from Processing import *
import librosa
import numpy as np
from train import create_pipeline

class Pix2Pix:

    def __init__(self, input_shape, target_shape, checkpoint_dir, batch_size, summary_writer=None, fig_path='', kernel_size=(4, 6), stride=(1, 2),
                 loss_object=tf.keras.losses.BinaryCrossentropy(from_logits=True), base=32):
        self.kernel_size = kernel_size
        self.stride = stride
        self.loss_object = loss_object
        self.base = base
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator = self.Discriminator(input_shape, target_shape, batch_size=batch_size)
        self.generator = self.Generator(input_shape, batch_size)
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)
        self.checkpoint_dir = checkpoint_dir
        self.summary_writer = summary_writer
        self.fig_path = fig_path

    def downsample(self, filters, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, self.kernel_size, strides=self.stride, padding='same',
                                   kernel_initializer=initializer, use_bias=False))
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU())
        return result

    def upsample(self, filters, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, self.kernel_size, strides=self.stride,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())
        return result

    def Generator(self, input_shape, batch_size):
        prev_frame = tf.keras.layers.Input(shape=input_shape, name="prev_frame", batch_size=batch_size)
        next_frame = tf.keras.layers.Input(shape=input_shape, name="next_frame", batch_size=batch_size)
        base = 32
        down_stack = [
            self.downsample(self.base, apply_batchnorm=False),
            self.downsample(2 * self.base),
            self.downsample(4 * self.base),
            self.downsample(8 * self.base),
        ]

        up_stack = [
            self.upsample(8 * self.base, apply_dropout=True),
            self.upsample(4 * self.base, apply_dropout=True),
            self.upsample(2 * self.base, apply_dropout=True),
            self.upsample(base)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(1, self.kernel_size, strides=(1, 2),
                                               padding='same',
                                               kernel_initializer=initializer,
                                               activation='relu')

        x = tf.concat((prev_frame, tf.zeros(prev_frame.shape), next_frame), axis=1)
        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])
        x = last(x)
        time_axis = x.shape[1]
        return tf.keras.Model(inputs=[prev_frame, next_frame], outputs=x[:, time_axis//3 : 2*time_axis//3])

    def generator_loss(self, disc_generated_output, gen_output, target):
        LAMBDA = 100
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        # mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (LAMBDA * l1_loss)
        return total_gen_loss, gan_loss, l1_loss

    def Discriminator(self, input_shape, target_shape, batch_size):
        initializer = tf.random_normal_initializer(0., 0.02)
        prev_frame = tf.keras.layers.Input(shape=input_shape, name='prev_image', batch_size=batch_size)
        next_frame = tf.keras.layers.Input(shape=input_shape, name='next_image', batch_size=batch_size)
        gap_frame = tf.keras.layers.Input(shape=target_shape, name='target_image', batch_size=batch_size)


        x = tf.keras.layers.concatenate([prev_frame, gap_frame, next_frame], axis=1)
        base = 32
        down1 = self.downsample(base, False)(x)
        down2 = self.downsample(2 * base)(down1)
        down3 = self.downsample(4 * base)(down2)
        conv = tf.keras.layers.Conv2D(8 * base, self.kernel_size, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False, padding='same')(down3)
        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
        last = tf.keras.layers.Conv2D(1, self.kernel_size, strides=1,
                                      kernel_initializer=initializer, padding='same')(leaky_relu)
        return tf.keras.Model(inputs=[prev_frame, gap_frame, next_frame], outputs=last, name="DISCR")

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    @tf.function
    def train_step(self, prev_frame, gap_frame, next_frame, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator([prev_frame, next_frame], training=True)
            disc_real_output = self.discriminator([ prev_frame, gap_frame, next_frame], training=True)
            disc_generated_output = self.discriminator([ prev_frame, gen_output, next_frame], training=True)
            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, gap_frame)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                         self.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)

    def fit(self, train_ds, epochs, test_ds, starting_epoch = 0):
        for epoch in range(epochs):
            epoch = epoch + starting_epoch
            start = time.time()
            print("Epoch: ", epoch)
            # Train
            for i in range(len(train_ds)):
                (prev_frame, next_frame), gap_frame = train_ds.__getitem__(i)
                self.train_step(prev_frame, gap_frame, next_frame, epoch)
                print(str(i) + ' batch done out of' + str(len(train_ds)))
                print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                                   time.time() - start))
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
    def restore(self):
            self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))


if __name__ == '__main__':
    import datetime
    from DataLoader import *
    from evaluate import *
    from train import mkdir

    model_name = 'pix2pix'

    train_path = 'nsynth-test.tfrecord'
    val_path = 'nsynth-test.tfrecord'
    test_path = 'nsynth-test.tfrecord'
    audio_length = 0.064 * 3
    batch_size = 256
    epoch = 100
    log_path = '../log/' + model_name + '/' + str(audio_length) + '/'
    ckpt_path = '../ckpt/' + model_name + '/' + str(audio_length) + '/'
    fig_path = log_path + 'fig/'
    mkdir(log_path)
    mkdir(ckpt_path)
    mkdir(fig_path)
    summary_writer = tf.summary.create_file_writer(
        log_path + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_pipeline, sr = create_pipeline(train_path, batch_size, audio_length, False)
    val_pipeline, sr = create_pipeline(val_path, batch_size, audio_length, False)
    test_pipeline, sr = create_pipeline(test_path, batch_size, audio_length, False)

    (prev_frame, next_frame), y = test_pipeline.__getitem__(1)
    pix2pix = Pix2Pix(list(tf.shape(y)[1:]), list(tf.shape(y)[1:]), ckpt_path, batch_size, summary_writer, fig_path=fig_path, base=32)
    tf.keras.utils.plot_model(pix2pix.generator, 'pix2pix_gen.png', expand_nested=True, show_shapes=True)
    tf.keras.utils.plot_model(pix2pix.discriminator, 'pix2pix_discr.png', expand_nested=True, show_shapes=True)

    #Training
    pix2pix.restore()
    #pix2pix.fit(train_pipeline, 100, val_pipeline, 0)
    res_dir = '../reconstruction/' + model_name + '/'
    mkdir(res_dir)

    #Evaluate the GAN
    config = {
        'test_generator': test_pipeline,
        'metrics': [ tf.keras.losses.mean_squared_error,  snr_batch],
        'result_directory': res_dir,
        'loss': tf.keras.losses.mean_squared_error,
        'sr': sr
    }
    evaluate(pix2pix.generator, config)

