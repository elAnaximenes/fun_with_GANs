import mnist_conditional_gan
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import PIL
import tensorflow as tf
from tensorflow.keras import layers
import time
import csv
import tqdm

def get_dataset(datasetName):

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    digits = np.concatenate([x_train, x_test])
    labels = np.concatenate([y_train, y_test])

    digits = digits.astype("float32") / 255.0
    digits = np.reshape(digits, (-1, 28, 28, 1))
    labels = tf.keras.utils.to_categorical(labels, 10)

    return digits[:-48], labels[:-48]

def main(datasetName, resume):

    trainSamples, trainLabels = get_dataset(datasetName)
    print(trainSamples.shape, flush=True)
    print(trainLabels.shape, flush=True)

    LATENT_DIMENSIONS = 128
    CHANNELS = 1
    CLASSES = 10
    IMAGE_SIZE = 28

    # Batch and shuffle the data
    BUFFER_SIZE = 1024 
    BATCH_SIZE = 64 
    #trainLabels = tf.keras.utils.to_categorical(trainLabels, 3)
    trainDataset = tf.data.Dataset.from_tensor_slices((trainSamples, trainLabels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    print(trainDataset)


    discriminatorInputShape = (28, 28, CHANNELS + CLASSES)
    discriminator = conditional_gan.Discriminator(discriminatorInputShape)

    generatorInputShape = (LATENT_DIMENSIONS + CLASSES,)
    generator = conditional_gan.Generator(generatorInputShape)

    adversarialPair = conditional_gan.ConditionalGAN(generator, discriminator, resume)

    #trainDataset = trainSamples, trainLabels

    adversarialPair.train(trainDataset)

if __name__ == '__main__':

    resume = False
    if len(sys.argv) > 1:
        resume = True

    datasetName = 'animals'
    main(datasetName, resume)
