#Contains utility functions for extracting the COVID datasets from the paths specified
from PIL import Image, ImageOps
import random
import numpy as np
from matplotlib import pyplot as plt
import os
import math
import time

image_height, image_width, image_mode = 224, 224, 3

#Generates a dataset of Covid, Normal and Pneumonia images
def genDataset(covid_path, normal_path, bacterial_path, viral_path):
    files = genDatasetFiles(covid_path, normal_path, bacterial_path, viral_path)
    x_data = np.zeros(shape=(len(files), image_height, image_width, image_mode))
    y_data = np.zeros(shape=(len(files)))

    prog_index = 0
    print('Starting retrieval')
    for i in range(len(files)):
        image_path = files[i][0]
        img = Image.open(image_path)
        img_arr = np.asarray(img)

        x_data[i] = img_arr
        y_data[i] = files[i][1]

        prog_index = prog_index+1
        if (prog_index%100)==0:
            print(prog_index)
    
    return x_data, y_data

#Generates a dataset of Covid, Normal and Pneumonia images
def genDataset3D(covid_path, normal_path, bacterial_path, viral_path):
    files = genDatasetFiles(covid_path, normal_path, bacterial_path, viral_path)
    x_data = np.zeros(shape=(len(files), image_height, image_width, 3))
    y_data = np.zeros(shape=(len(files)))

    prog_index = 0
    for i in range(len(files)):
        image_path = files[i][0]
        img = Image.open(image_path).convert('RGB')
        img_arr = np.asarray(img)

        x_data[i] = img_arr
        y_data[i] = files[i][1]

        prog_index = prog_index+1
        if (prog_index%100)==0:
            print(prog_index)
    
    return x_data, y_data

#Generates a dataset of Covid, Normal and Pneumonia images, and saves the names as well
def genDataset3DWithNames(covid_path, normal_path, bacterial_path, viral_path):
    files = genDatasetFiles(covid_path, normal_path, bacterial_path, viral_path)
    x_data = np.zeros(shape=(len(files), image_height, image_width, 3))
    y_data = np.zeros(shape=(len(files)))
    path_data = list()

    prog_index = 0
    for i in range(len(files)):
        image_path = files[i][0]
        img = Image.open(image_path).convert('RGB')
        img_arr = np.asarray(img)

        x_data[i] = img_arr
        y_data[i] = files[i][1]
        path_data.append(files[i][0])

        prog_index = prog_index+1
        if (prog_index%100)==0:
            print(prog_index)
    
    return x_data, y_data, path_data

#Generates a files dataset of Covid, Normal and Pneumonia images
#Randomnly
#New Classes would be: NORMAL: 0, COVID: 1, PNEUMONIA: 2
def genDatasetFiles(covid_path, normal_path, bacterial_path, viral_path):
    true_class = 0

    if covid_path != '':
        covid_files, covid_data = retrieveDataFiles(covid_path, true_class)
        true_class = true_class + 1
    else:
        covid_files, covid_data = [], []
    
    if normal_path != '':
        normal_files, normal_data = retrieveDataFiles(normal_path, true_class)
        true_class = true_class + 1
    else:
        normal_files, normal_data = [], []
    
    if bacterial_path != '':
        bacterial_files, bacterial_data = retrieveDataFiles(bacterial_path, true_class)
        true_class = true_class + 1
    else:
        bacterial_files, bacterial_data = [], []
    
    if viral_path != '':
        viral_files, viral_data = retrieveDataFiles(viral_path, true_class)
        true_class = true_class + 1
    else:
        viral_files, viral_data = [], []

    random.seed(0)
    np.random.seed(0)
    files = covid_files+normal_files+bacterial_files+viral_files
    #data = np.concatenate((covid_data, normal_data, bacterial_data, viral_data), axis=0)

    #Randomize the dataset
    random.shuffle(files)
    #np.random.shuffle(data)

    return files #, data

#Retrieve data from the path specified and insert into numpy array
def retrieveDataFiles(path, dis_class):
    files = os.listdir(path)[0: 1144]
    for i in range(len(files)):
        files[i] = (os.path.join(path, files[i]), dis_class)
    #Each image shape is 227x227 (Change according to preference)
    data = np.zeros(shape=(len(files), image_height, image_width))
    index = 0

    #for filename in files:
    #    image_path = filename[0] #os.path.join(path, filename)
    #    #Image is already standard sized and grayscale, no need of any extra operations
    #    img = Image.open(image_path)
    #    img_arr = np.asarray(img)
        
    #    data[index] = img_arr
    #    index = index+1
    print('Retrieved data', len(files))
    return files, data
