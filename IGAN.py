import tensorflow as tf


class IGAN:
    """"
    Class for inpainting magnitude spectrum. The previous frame, gap frame and subsequent frame MUST have the same dimension.
    """
    def __init__(self, input_shape, summary_writer, checkpoint_dir, fig_path):
        """Build the GAN model
        Input: - (Int, Int, Int) input_shape: dimension of the spectrum
               - tf.Summary.SummaryWriter summary_writer: writer for the training log
               - String checpoint_dir: path to the directory where the checkpoint are saved
               - String fig_path: path to the directory where a generated sample will be saved at each epoch during the training"""
        self.input_shape = input_shape
        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam()
        self.discriminator_optimizer = tf.keras.optimizers.Adam()
        self.summary_writer = summary_writer
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)
        self.fig_path = fig_path

    def create_encoder(self, base=32, kernel_size=(4, 6), name="encoder"):
        """Build the encoder part of the generator
        Input: - Int base: Parameter related to the number of the filter used in the convolution layers
               - (Int, Int) or Int kernel_size: Specify the kernel dimension of the filter used in the convolution layers
               - String name : Name of the encoder.
        Output: tf.keras.Sequential encoder"""
        encoder = Sequential([
            Conv2D(base, input_shape=self.input_shape, strides=(1, 1), kernel_size=kernel_size,
                   data_format="channels_last",
                   activation="relu", padding='same'),
            Conv2D(2 * base, strides=(1, 2), padding='same', kernel_size=kernel_size, data_format="channels_last",
                   activation="relu"),
            Conv2D(4 * base, strides=(1, 2), kernel_size=kernel_size, padding='same', data_format="channels_last",
                   activation="relu"),
            Conv2D(8 * base, strides=(1, 2), kernel_size=kernel_size, padding='same', data_format="channels_last",
                   activation="relu"),
        ],
            name=name
        )
        tf.keras.utils.plot_model(encoder, show_shapes=True, expand_nested=True, to_file='encoder.png')
        return encoder

    def create_decoder(self, input_shape, base=32, kernel_size=(4, 6)):
        """Build the decoder part of the generator
        Input: - base must be an integer > 0. Parameter related to the number of the filter used in the convolution layers
               - kernel_size: int or (int, int). Specify the kernel dimension of the filter used in the convolution layers
               - name : string . Name of the encoder.
        Output: tf.keras.Sequential decoder"""
        decoder = Sequential([
            Conv2DTranspose(8 * base, input_shape=input_shape, strides=(1, 2), kernel_size=kernel_size, padding='same',
                            activation="relu", data_format="channels_last"),
            Conv2DTranspose(4 * base, strides=(1, 2), kernel_size=kernel_size, padding='same', activation="relu",
                            data_format="channels_last"),
            Conv2DTranspose(2 * base, strides=(1, 2), kernel_size=kernel_size, padding='same', activation="relu",
                            data_format="channels_last"),
            Conv2DTranspose(base, strides=(1, 1), kernel_size=kernel_size, padding='same', activation="relu",
                            data_format="channels_last"),
            Conv2DTranspose(1, strides=(1, 1), kernel_size=kernel_size, padding='same', activation="linear",
                            data_format="channels_last"),
        ],
            name='decoder')
        return decoder

    def create_generator(self):
        """Build the generator part of the GAN.
        Output: tf.keras.Model """
        encoder = self.create_encoder(name='encoder')
        spec1 = tf.keras.Input(shape=self.input_shape)
        spec3 = tf.keras.Input(shape=self.input_shape)

        spec1_encoded = encoder(spec1)
        spec3_encoded = encoder(spec3)

        dec_input = Concatenate()([spec1_encoded, spec3_encoded])
        decoder = self.create_decoder(dec_input.shape[1:])
        prediction = decoder(dec_input)
        return tf.keras.Model([spec1, spec3], prediction)

    def create_discriminator(self):
        """Build the discriminator part of the GAN.
        Output: tf.keras.Sequential """
        prev_frame = tf.keras.layers.Input(shape=self.input_shape, name='prev_frame')
        next_frame = tf.keras.layers.Input(shape=self.input_shape, name='next_frame')
        rec_frame = tf.keras.layers.Input(shape=self.generator.layers[-1].output_shape[1:], name='rec_frame')
        input = tf.concat([prev_frame, rec_frame, next_frame], axis=1)
        print(tf.shape(input))

        base = 32
        disc = tf.keras.Sequential([
            Conv2D(1 * base, strides=(2, 2), kernel_size=(3, 3), data_format="channels_last",
                   activation="relu", padding='same', input_shape=(15, 512, 1)),
            Conv2D(2 * base, strides=(2, 2), kernel_size=(3, 3), data_format="channels_last",
                   activation="relu", padding='same'),
            Conv2D(4 * base, strides=(2, 2), kernel_size=(3, 3), data_format="channels_last",
                   activation="relu", padding='same'),
            Conv2D(1, strides=(2, 2), kernel_size=(3, 3), data_format="channels_last",
                   activation="sigmoid", padding='same'),
        ], name="discriminator")
        last = disc(input)
        return tf.keras.Model(inputs=[prev_frame, rec_frame, next_frame], outputs=last, name="discriminator")

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        """"Compute the discriminator loss:
        Input:  - disc_real_output is a batch of discriminator output on real instance
                - disc_generated_output is a batch of discriminator output on fake instance
        Output:  tensor of float"""
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    def generator_loss(self, disc_generated_output, gen_output, target):
        """Compute the generator loss
        Input: - Tensor disc_generated_output is a batch of discriminator output on fake instance
               - Tensor gen_output is a batch of generator estimation
               - Tensor target is the ground truth
        Output:
               - Tensor total_gen_loss: generator loss, weighted average of gan_loss and l1_loss
               - Tensor gan_loss: adversarial loss
               - Tensor l1_loss: L1 loss between the generator prediction and ground truth"""

        LAMBDA = 10
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        # mean absolute error
        l1_loss = tf.abs(target - gen_output)
        total_gen_loss = gan_loss + (LAMBDA * l1_loss)
        return total_gen_loss, gan_loss, l1_loss

    def generate_images(self, prev_frame, gap_frame, next_frame, epoch):
        """
        Create and save sample image of the generator prediction and ground truth.
        Label the saved image with the epoch in the filename.

        Input: - Tensor prev_frame: batch of conditioning image, the frame just before the one to inpaint.
               - Tensor gap frame: the ground truth, the frame to be inpainted.
               - Tensor next frame: batch of conditioning image, the frame subsequent of the one to inpaint.
        """
        self.generator.summary()
        prediction = self.generator([prev_frame, next_frame], training=True)

        plt.figure()
        fig, axes = plt.subplots(ncols=2)
        plt.xlabel("Frequency")
        plt.ylabel("Time in sample")
        ax1, ax2 = axes
        ax1.set_title("reconstruction spectrum")
        librosa.display.specshow(librosa.amplitude_to_db(np.transpose(np.squeeze(prediction[0])), ref=np.max), ax=ax1,
                                 y_axis='log',
                                 x_axis='frames')
        ax2.set_title("true spectrum")
        librosa.display.specshow(librosa.amplitude_to_db(np.transpose(np.squeeze(gap_frame[0])), ref=np.max), ax=ax2,
                                 y_axis='log',
                                 x_axis='frames')
        plt.savefig(self.fig_path + '_' + str(epoch) + '.png')
        plt.close()

    @tf.function
    def train_step(self, prev_frame, gap_frame, next_frame, epoch):
        """Perform one epoch of training
        Input: - Tensor prev_frame: batch of conditioning image, the frame just before the one to inpaint.
               - Tensor gap frame: the ground truth, the frame to be inpainted.
               - Tensor next frame: batch of conditioning image, the frame subsequent of the one to inpaint.
               - Int epoch: the current epoch"""

        #compute gradient
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator([prev_frame, next_frame], training=True)
            disc_real_output = self.discriminator([prev_frame, gap_frame, next_frame], training=True)
            disc_generated_output = self.discriminator([prev_frame, gen_output, next_frame], training=True)
            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output,
                                                                            gap_frame)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        #Apply gradient
        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                         self.discriminator.trainable_variables))

        #Write log
        with self.summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)

    def fit(self, train_ds, test_ds, epochs, starting_epoch=0):
        """train the network
        Input: - Processing train_ds: the training pipeline.
               - Int epochs: the number of epoch to train the network
               - Processing test_ds: the test pipeline
               - Int starting_epoch: the epoch from which the training is resumed """
        for epoch in range(epochs):
            epoch = epoch + starting_epoch
            start = time.time()
            (prev_frame, next_frame), gap_frame = test_ds.__getitem__(0)
            self.generate_images(prev_frame, gap_frame, next_frame, epoch)

            # Train
            for i in range(len(train_ds)):
                (prev_frame, next_frame), gap_frame = train_ds.__getitem__(i)
                self.distributed_train_step(prev_frame, gap_frame, next_frame, epoch)
                print(str(i) + ' batch done out of' + str(len(train_ds)))
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))

    def restore(self, checkpoint_dir=None):
        """
        Restore the latest checkpoint.
        Input: - String checkpoint_dir: path to the folder where the checkpoint are saved.
        """
        if self.checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


