from PIL import Image
import time
import os
import imageio
import numpy as np
import csv

def save_dataset_to_csv(dataset, datasetName=None):

    print("writing outfile", flush=True)

    outFileName = "./{}.csv".format(datasetName)

    with open(outFileName, 'w') as f:

        writer = csv.writer(f, delimiter=',')
        for i in range(dataset.shape[0]):

            writer.writerow(dataset[i].tolist())

def get_image_encoding(imageName):

    data = None 

    with Image.open(imageName) as img:

        img = img.resize((256,256))

        if img.mode == "L":
            print("grayscale image:", img.mode)
            return None 

        shape = img.size + (3,)

        data = np.array(list(img.getdata()))
        data = data.reshape(shape)
        data = (data - 127.5) / 127.5

    return data

def get_landscape_dataset(landscapesDir, imageNames):

    print("reading landscape dataset...", flush=True)
    dataset = []

    for imageName in imageNames:

        imageName = os.path.join(landscapesDir, imageName)
        data = get_image_encoding(imageName)

        if data is not None:
            dataset.append(data)

    dataset = np.array(dataset)

    save_dataset_to_csv(dataset, 'landscapes')

    return dataset

landscapesDir = './datasets/landscape_set/'
landscapes = os.listdir(landscapesDir)
get_landscape_dataset(landscapesDir, landscapes)
