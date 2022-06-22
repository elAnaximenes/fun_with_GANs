from PIL import Image
import time
import os
import imageio
import numpy as np
import csv

def save_labelset_to_csv(labelset, labelsetName):

    print("writing labelset to file...", flush=True)

    outFileName = "./{}.csv".format(labelsetName)

    with open(outFileName, "w") as f:

        writer = csv.writer(f, delimiter=',')
        writer.writerow(labelset.tolist())

def save_dataset_to_csv(dataset, datasetName):

    print("writing dataset to file...", flush=True)

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

def get_single_animal_data(imageDir, imageNames, animalName):

    animalClasses = {"cats":0, "dogs":1, "panda":2}
    print("reading {} {} images...".format(len(imageNames), animalName), flush=True)
    labels = [animalClasses[animalName]] * len(imageNames)
    dataset = []

    for imageName in imageNames:
        #print(imageName, flush=True)

        imageName = os.path.join(imageDir, imageName)
        data = get_image_encoding(imageName)

        if data is not None:
            dataset.append(data)

    return dataset, labels

def get_animals_dataset(animalsDataDir):

    dataset = []
    labelset = []

    for animalName in os.listdir(animalsDataDir):

        animalDir = "./animals/" + animalName
        imageNames = os.listdir(animalDir)

        data, labels = (get_single_animal_data(animalDir, imageNames, animalName))

        for d, l in zip(data, labels):
            dataset.append(d)
            labelset.append(l)

    dataset = np.array(dataset)
    labelset = np.array(labelset)

    print("dataset shape:", dataset.shape)
    print("labelset shape:", labelset.shape)

    #save_dataset_to_csv(dataset, 'animals')
    save_labelset_to_csv(labelset, 'animals_labels')

animalsDir = './animals/'
get_animals_dataset(animalsDir)
