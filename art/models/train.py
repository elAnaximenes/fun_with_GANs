import gan
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow.keras import layers
import time

def load_images():

    imagesDir = "../data/images/images/"
    artists = os.listdir(imagesDir)

    dataset = []

    for artist in artists:

        artistFolder = os.path.join(imagesDir, artist)
        images = os.listdir(artistFolder)

        for imageName in images:

            imageName = os.path.join(artistFolder, imageName)

            img = PIL.Image.open(imageName)
            shape = img.size + (3,)

            data = np.array(list(img.getdata())).reshape(shape)
            dataset.append(data)
            print(data.shape, flush=True)

            img.close()

        print(np.array(dataset).shape, flush=True)


"""
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
"""

load_images()
exit(1)
BUFFER_SIZE = 60000
BATCH_SIZE = 256
# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

inputShape = train_images.shape[1:]

discriminator = gan.Discriminator(inputShape)
generator = gan.Generator(inputShape)
gan = gan.GAN(generator, discriminator)

gan.train(train_dataset)

