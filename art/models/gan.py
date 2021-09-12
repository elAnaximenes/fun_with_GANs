import tensorflow as tf
import numpy as np
import time
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class Discriminator(tf.keras.Model):

    def __init__(self, inputShape):

        super(Discriminator, self).__init__()

        self.inputLayer = layers.InputLayer(inputShape)

        self.convLayer1 = layers.Conv2D(64, (5,5), strides=(2,2), padding='same',
                input_shape = [28,28,1])
        self.reluLayer1 = layers.LeakyReLU()
        self.dropoutLayer1 = layers.Dropout(0.3)

        self.convLayer2 = layers.Conv2D(128, (5,5), strides=(2,2), padding='same')
        self.reluLayer2 = layers.LeakyReLU()
        self.dropoutLayer2 = layers.Dropout(0.3)

        self.flattenLayer = layers.Flatten()
        self.outputLayer = layers.Dense(1)

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

    def call(self, x):

        x = self.inputLayer(x)
        x = self.convLayer1(x)
        x = self.reluLayer1(x)
        x = self.dropoutLayer1(x)
        x = self.convLayer2(x)
        x = self.reluLayer2(x)
        x = self.dropoutLayer2(x)
        x = self.flattenLayer(x)

        return self.outputLayer(x)

    def loss(self, real_output, fake_output):

        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss

        return total_loss

class Generator(tf.keras.Model):

    def __init__(self, inputShape): 

        super(Generator, self).__init__()

        self.inputLayer = layers.InputLayer(inputShape)
        self.dense = layers.Dense(7*7*256, use_bias=False, input_shape=(100,))

        self.batchNormLayer1 = layers.BatchNormalization()
        self.reluLayer1 = layers.LeakyReLU()
        self.reshape1 = layers.Reshape((7,7,256))
        self.convLayer1 = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)

        self.batchNormLayer2 = layers.BatchNormalization()
        self.reluLayer2 = layers.LeakyReLU()
        self.convLayer2 = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)

        self.batchNormLayer3 = layers.BatchNormalization()
        self.reluLayer3 = layers.LeakyReLU()

        self.convLayer3 = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')


        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

    def call(self, x, training=False):

        x = self.inputLayer(x)
        x = self.dense(x)
        x = self.batchNormLayer1(x, training=training) 
        x = self.reluLayer1(x)
        x = self.reshape1(x)
        x = self.convLayer1(x)
        x = self.batchNormLayer2(x, training=training)
        x = self.reluLayer2(x)
        x = self.convLayer2(x)
        x = self.batchNormLayer3(x, training=training)
        x = self.reluLayer3(x)

        return self.convLayer3(x)

    def loss(self, fake_output):

        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

class GAN:

    def __init__(self, generator, discriminator):

        self.generator = generator
        self.discriminator = discriminator

        self.noiseDim = 100
        self.numSamplesToGenerate = 16
        self.seed = tf.random.normal([self.numSamplesToGenerate, self.noiseDim])
        self.batchSize = 16 
        self.epochs = 10

    def _generate_images(self, epoch, testInput):

        predictions = self.generator(testInput)

        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):

            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.show()

    def _train_step(self, images):

        noise = tf.random.normal([self.batchSize, self.noiseDim])

        with tf.GradientTape() as genTape, tf.GradientTape() as discTape:

            generatedImages = self.generator(noise, training=True)

            realOutput = self.discriminator(images)
            fakeOutput = self.discriminator(generatedImages)

            genLoss = self.generator.loss(fakeOutput)
            discLoss = self.discriminator.loss(realOutput, fakeOutput)

        generatorGrads = genTape.gradient(genLoss, self.generator.trainable_variables)
        discriminatorGrads = discTape.gradient(discLoss, self.discriminator.trainable_variables)

        self.generator.optimizer.apply_gradients(zip(generatorGrads, self.generator.trainable_variables))
        self.discriminator.optimizer.apply_gradients(zip(discriminatorGrads, self.discriminator.trainable_variables))

    def train(self, dataset):

        for epoch in range(self.epochs):

            start = time.time()

            for imageBatch in dataset:

                self._train_step(imageBatch)

            self._generate_images(epoch + 1, self.seed)

        self._generate_images(self.epochs, self.seed)
