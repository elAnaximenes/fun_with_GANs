import gan
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

def convert_row_to_image_data(row):

    image = []
    pixelValue = ''

    for item in row:

        for char in item:

            if char == '[' or char == ' ':
                continue
            elif char == ',' or char == ']':
                if len(pixelValue) > 0:
                    image.append(float(pixelValue))
                    pixelValue = ''
                continue
            else:
                pixelValue += char

    image = np.array(image).reshape((200,200,3))

    fig = plt.figure()
    plt.imshow(image)
    plt.show()


    return image

def get_artist_paintings(artistName):

    paintingsFileName = os.path.join('../data/artists/', '{}.csv'.format(artistName))
    print('reading artist paintings {}'.format(artistName), flush=True)

    images = []

    with open(paintingsFileName, 'r') as f:

        reader = csv.reader(f)

        imgNum = 0
        for row in reader:

            imgNum += 1
            print('reading image {}'.format(imgNum))

            image = convert_row_to_image_data(row) 
            images.append(image)
            #if imgNum == 16:
                #break

    return np.array(images)

"""
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
"""

def main(artistName):

    trainImages = get_artist_paintings(artistName)
    print(trainImages.shape, flush=True)

    # Batch and shuffle the data
    BUFFER_SIZE = 60000
    BATCH_SIZE = 16 
    trainDataset = tf.data.Dataset.from_tensor_slices(trainImages).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    inputShape = trainImages.shape[1:]

    discriminator = gan.Discriminator(inputShape)
    generator = gan.Generator(inputShape)
    adversarialPair = gan.GAN(generator, discriminator)

    adversarialPair.train(trainDataset)

if __name__ == '__main__':

    artistName = 'Vincent_van_Gogh'
    main(artistName)

