import tensorflow as tf
import numpy as np
import time 
import os
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class Discriminator(tf.keras.Model):

    def __init__(self, inputShape):

        super(Discriminator, self).__init__()

        self.inputLayer = layers.InputLayer(inputShape)

        self.convLayer1 = layers.Conv2D(32, (5,5), strides=(2,2), padding='same')
        self.reluLayer1 = layers.LeakyReLU()
        self.dropoutLayer1 = layers.Dropout(0.3)

        self.convLayer2 = layers.Conv2D(64, (5,5), strides=(2,2), padding='same')
        self.reluLayer2 = layers.LeakyReLU()
        self.dropoutLayer2 = layers.Dropout(0.3)

        self.convLayer3 = layers.Conv2D(128, (5,5), strides=(2,2), padding='same')
        self.reluLayer3 = layers.LeakyReLU()
        self.dropoutLayer3 = layers.Dropout(0.3)

        self.convLayer4 = layers.Conv2D(256, (5,5), strides=(2,2), padding='same')
        self.reluLayer4 = layers.LeakyReLU()
        self.dropoutLayer4 = layers.Dropout(0.3)

        self.convLayer5 = layers.Conv2D(512, (5,5), strides=(2,2), padding='same')
        self.reluLayer5 = layers.LeakyReLU()
        self.dropoutLayer5 = layers.Dropout(0.3)

        self.flattenLayer = layers.Flatten()
        self.outputLayer = layers.Dense(1)

        #self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.binaryCrossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

        print('Initialized Discriminator...',flush=True)

    def call(self, x):

        x = self.inputLayer(x)
        x = self.convLayer1(x)
        x = self.reluLayer1(x)
        x = self.dropoutLayer1(x)
        x = self.convLayer2(x)
        x = self.reluLayer2(x)
        x = self.dropoutLayer2(x)
        x = self.convLayer3(x)
        x = self.reluLayer3(x)
        x = self.dropoutLayer3(x)
        x = self.convLayer4(x)
        x = self.reluLayer4(x)
        x = self.dropoutLayer4(x)
        x = self.convLayer5(x)
        x = self.reluLayer5(x)
        x = self.dropoutLayer5(x)
        x = self.flattenLayer(x)

        return self.outputLayer(x)

    #def loss(self, real_output, fake_output):
    def loss(self, labels, predictions):

        #real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        #fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        #total_loss = real_loss + fake_loss

        return self.binaryCrossEntropy(labels, predictions)

class Generator(tf.keras.Model):

    def __init__(self, inputShape): 

        super(Generator, self).__init__()

        self.inputLayer = layers.InputLayer(inputShape)
        self.dense = layers.Dense(8*8*3, use_bias=False)

        self.batchNormLayer1 = layers.BatchNormalization()
        self.reluLayer1 = layers.LeakyReLU()
        self.reshape1 = layers.Reshape((8, 8, 3))
        self.convLayer1 = layers.Conv2DTranspose(1024, (5, 5), strides=(1, 1), padding='same', use_bias=False)

        self.batchNormLayer2 = layers.BatchNormalization()
        self.reluLayer2 = layers.LeakyReLU()
        self.convLayer2 = layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False)

        self.batchNormLayer3 = layers.BatchNormalization()
        self.reluLayer3 = layers.LeakyReLU()
        self.convLayer3 = layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False)

        self.batchNormLayer4 = layers.BatchNormalization()
        self.reluLayer4 = layers.LeakyReLU()
        self.convLayer4 = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        
        self.batchNormLayer5 = layers.BatchNormalization()
        self.reluLayer5 = layers.LeakyReLU()
        self.convLayer5 = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')

        self.batchNormLayer6 = layers.BatchNormalization()
        self.reluLayer6 = layers.LeakyReLU()
        self.convLayer6 = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')

        #self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.binaryCrossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

        print('Initialized Generator...',flush=True)

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
        x = self.convLayer3(x)
        x = self.batchNormLayer4(x, training=training)
        x = self.reluLayer4(x)
        x = self.convLayer4(x)
        x = self.batchNormLayer5(x, training=training)
        x = self.reluLayer5(x)
        x = self.convLayer5(x)
        x = self.batchNormLayer6(x, training=training)
        x = self.reluLayer6(x)
        x = self.convLayer6(x)

        return x

    #def loss(self, fake_output):

    def loss(self, labels, predictions):
        return self.binaryCrossEntropy(labels, predictions)
        #loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        #return self.cross_entropy(tf.ones_like(fake_output), fake_output)

class ConditionalGAN(tf.keras.Model):

    def __init__(self, generator, discriminator, resume=False, latentDim = 128):

        super(ConditionalGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.resume = resume
        self.latentDim = latentDim

        self.channels = 3 
        self.imageWidth = 200 
        self.imageHeight = 200 
        self.numSamplesToGenerate = 16

        self.batchSize = 16 
        self.epochs = 10000 

        self.weightsDir = './weights/'
        if not os.path.exists(self.weightsDir):
            os.mkdir(self.weightsDir)

        self.generatorWeights = self.weightsDir + 'generator_weights'
        self.discriminatorWeights = self.weightsDir + 'discriminator_weights'
        
        """
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator.optimizer,
                                                discriminator_optimizer=self.discriminator.optimizer,
                                                generator=self.generator,
                                                discriminator=self.discriminator)
        """

    def _get_random_noise(self):

        return tf.random.normal([self.numSamplesToGenerate, 100])

    def _save(self):

        self.generator.save_weights(self.generatorWeights)
        self.generator.save_weights(self.discriminatorWeights)

    def _resume_training(self):

        self.generator.load_weights(self.generatorWeights)
        self.generator.load_weights(self.discriminatorWeights)

    def _generate_images(self, epoch):

        testInput = self._get_random_noise()

        predictions = self.generator(testInput)

        fig = plt.figure()
        plt.imshow((predictions[0, :, :, :] * 127.5 + 127.5).numpy().astype(int))
        plt.axis('off')
        plt.savefig('./samples/sample_{}.jpg'.format(epoch))
        
        if epoch % 1000 == 0:
            plt.show()
        else:
            plt.close(fig)

    def _train_step(self, batch):

        realImages, oneHots = batch 
        #noise = self.seed 
        imageSize = 256
        numClasses = 3


        imageOneHots = oneHots[:, :]
        imageOneHots = tf.repeat(imageOneHots, repeats=[imageSize*imageSize])
        imageOneHots = tf.reshape(imageOneHots, (-1, imageSize, imageSize, numClasses))

        batchSize = tf.shape(realImages)[0]
        #latentNoise = tf.random.normal(shape=(batchSize, self.latentDim))
        latentNoise = self._get_random_noise() 
        #randomLabels = tf.concat([latentNoise, oneHots], axis=1)

        generatedImages = self.generator(latentNoise, training=True)

        #print("one hots shape", imageOneHots.shape)
        #print("real images shape", realImages.shape)
        fakeImagesAndLabels = tf.concat([generatedImages, imageOneHots], -1)
        realImagesAndLabels = tf.concat([realImages, imageOneHots], -1)
        combinedImagesAndLabels= tf.concat([fakeImagesAndLabels, realImagesAndLabels], axis=0)

        labels = tf.concat([tf.ones((batchSize, 1)), tf.zeros((batchSize, 1))], axis = 0)

        with tf.GradientTape() as discTape:

            """
            realOutput = self.discriminator(realImagesAndLabels)
            fakeOutput = self.discriminator(fakeImagesAndLabels)

            genLoss = self.generator.loss(fakeOutput)
            discLoss = self.discriminator.loss(realOutput, fakeOutput)
            """
            predictions = self.discriminator(combinedImagesAndLabels)
            discLoss = self.discriminator.loss(labels, predictions)

        discriminatorGrads = discTape.gradient(discLoss, self.discriminator.trainable_variables)
        self.discriminator.optimizer.apply_gradients(zip(discriminatorGrads, self.discriminator.trainable_variables))

        #latentNoise = tf.random.normal(shape=(batchSize, self.latentDim))
        latentNoise = self._get_random_noise() 
        #randomLabels = tf.concat([latentNoise, oneHots], axis=1)

        wrongLabels = tf.zeros((batchSize, 1))

        with tf.GradientTape() as genTape:

            fakeImages = self.generator(latentNoise)
            fakeImagesAndLabels = tf.concat([fakeImages, imageOneHots], -1)

            predictions = self.discriminator(fakeImagesAndLabels)
            genLoss = self.generator.loss(wrongLabels, predictions)

        generatorGrads = genTape.gradient(genLoss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(generatorGrads, self.generator.trainable_variables))

        return genLoss, discLoss

    def train(self, dataset):

        if self.resume:
            self._resume_training()

        #samples, labels = dataset
        #print(samples.shape)
        #print(labels.shape)
        #dataset = zip(samples, labels)

        for epoch in range(self.epochs):

            print('Epoch Number {}'.format(epoch))

            start = time.time()

            for batch in dataset:

                genLoss, discLoss = self._train_step(batch)

            self._generate_images(epoch + 1)

            #if epoch %10 == 0:
                #self._save()

            print('generator loss', genLoss)
            print('discriminator loss', discLoss, flush=True)

        self._generate_images(self.epochs)
