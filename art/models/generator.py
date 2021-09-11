import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class Generator(tf.keras.Model):

    def __init__(self, inputShape): 

        super(Discriminator, self).__init__()

        self.inputLayer = layers.InputLayer(inputShape)
        self.dense = layers.Dense(7*7*256, use_bias=False, input_shape=(100,))

        self.batchNormLayer1 = layers.BatchNormalization()
        self.reluLayer1 = layers.leakyReLU()
        self.reshape1 = layers.Reshape((7,7,256))
        self.convLayer1 = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)

        self.batchNormLayer2 = layers.BatchNormalization()
        self.reluLayer2 = layers.leakyReLU()
        self.convLayer2 = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)

        self.batchNormLayer3 = layers.BatchNormalization()
        self.reluLayer3 = layers.leakyReLU()

        self.convLayer3 = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')

    def call(x):

        x = self.inputLayer(x)
        x = self.dense(x)
        x = self.batchNormLayer1(x) 
        x = self.reluLayer1(x)
        x = self.reshape1(x)
        x = self.convLayer1(x)
        x = sefl.batchNormLayer2(x)
        x = sefl.reluLayer2(x)
        x = sefl.convLayer2(x)
        x = sefl.batchNormLayer3(x)
        x = sefl.reluLayer3(x)

        return sefl.convLayer3(x)




