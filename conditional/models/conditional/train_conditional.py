import conditional_gan
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

    image = np.array(image).reshape((256,256,3))

    return image

def get_dataset(datasetName):

    imagesFileName= os.path.join('{}.csv'.format(datasetName))
    labelsFileName= os.path.join('{}_labels.csv'.format(datasetName))
    print('reading images {}'.format(imagesFileName), flush=True)
    print('reading labels {}'.format(labelsFileName), flush=True)

    images = []

    with open(imagesFileName, 'r') as f:

        reader = csv.reader(f)

        imgNum = 0
        for row in reader:

            imgNum += 1
            print('reading image {}'.format(imgNum))

            image = convert_row_to_image_data(row) 
            images.append(image)

    labels = []

    with open(labelsFileName, 'r') as f:

        reader = csv.reader(f)
        
        for row in reader:
            labelNum = 0
            for label in row:
                labelNum+=1
                labels.append(int(label))

    return np.array(images).astype("float32"), np.array(labels).astype("float32")

"""
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
"""

def main(datasetName):

    resume = False
    trainSamples, trainLabels = get_dataset(datasetName)
    trainSamples = trainSamples[:-1]
    trainLabels = trainLabels[:-1]
    print(trainSamples.shape, flush=True)
    print(trainLabels.shape, flush=True)

    # Batch and shuffle the data
    BUFFER_SIZE = 60000
    BATCH_SIZE = 16 
    trainLabels = tf.keras.utils.to_categorical(trainLabels, 3)
    trainDataset = tf.data.Dataset.from_tensor_slices((trainSamples, trainLabels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    print(trainDataset)

    inputShape = trainSamples.shape[1:]

    discriminator = conditional_gan.Discriminator(inputShape)
    generator = conditional_gan.Generator(inputShape)
    adversarialPair = conditional_gan.ConditionalGAN(generator, discriminator, resume)

    #trainDataset = trainSamples, trainLabels

    adversarialPair.train(trainDataset)

if __name__ == '__main__':


    if len(sys.argv) < 2:
        print("Usage: {} dataset_name".format(sys.argv[0]))
        exit(2)

    datasetName = sys.argv[1] 
    main(datasetName)



