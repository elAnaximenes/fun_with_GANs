import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


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
        self.output = layers.Dense(1)

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

    def call(x):

        x = self.inputLayer(x)
        x = self.convLayer1(x)
        x = self.reluLayer1(x)
        x = self.dropoutLayer1(x)
        x = self.convLayer2(x)
        x = self.reluLayer2(x)
        x = self.dropoutLayer2(x)
        x = self.flattenLayer(1)

        return self.output(x)

    def discriminator_loss(real_output, fake_output):

        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss

        return total_loss



if __name__ == "__main__":

    print("hello")
