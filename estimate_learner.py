#Main Estimate Learner for WavstaNet
import os
import tensorflow as tf
import numpy as np
import math
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPool2D, LeakyReLU, Flatten, Dense, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import xception, vgg19, mobilenet, resnet50
import tensorflow.keras.backend as kb

from dataset import genDataset, genDataset3D, genDatasetFiles, image_height, image_width, image_mode
from stats import plot_confusion_matrix
from config import covid_dwt_train_path, normal_dwt_train_path, bacterial_dwt_train_path, viral_dwt_train_path 
from config import covid_dwt_test_path, normal_dwt_test_path, bacterial_dwt_test_path, viral_dwt_test_path
from config import network_dwt_save_path, network_dwt_load_path, network_dwt_best_load_path, network_dwt_best_save_path

from el_functions import plus_one_pad, initializer, alpha, createDarkNet, createMobileNetwork, createResNetwork, createVGGNetwork

def custom_loss(y_true, y_pred):
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = scce(y_true, y_pred)

    for index in range(len(y_true)):
        if y_true[index][0]==3:
            loss_1 = abs(1-y_pred[index][3])
            loss = loss + loss_1
    return loss

#Create Complete framework where sub-networks are untrainable and estimate network is trainable
def createEstimateLearner(res_model, vgg_model, mobile_model, dark_model):
    x1 = Dense(256, activation=LeakyReLU(alpha=0.3))(res_model.get_layer('res_weights').output)
    x2 = Dense(256, activation=LeakyReLU(alpha=0.3))(vgg_model.get_layer('vgg_weights').output)
    x3 = Dense(256, activation=LeakyReLU(alpha=0.3))(mobile_model.get_layer('mobile_weights').output)
    #x4 = Dense(256, activation=LeakyReLU(alpha=0.3))(dark_model.get_layer('dark_weights').output)
    x = Concatenate()([x1, x2, x3])
    x = Dense(4, activation='softmax')(x)
    model = Model([res_model.input, vgg_model.input, mobile_model.input], x)
    return model

if __name__ == '__main__':
    # The arguments list for genDataset are being deliberately flipped so that Covid has class 3 and Normal has class 0
    x_train, y_train = genDataset3D(normal_dwt_train_path, bacterial_dwt_train_path, viral_dwt_train_path, covid_dwt_train_path)
    x_train_res = resnet50.preprocess_input(x_train)
    x_train_vgg = vgg19.preprocess_input(x_train)
    x_train_mobile = mobilenet.preprocess_input(x_train)
    print(x_train.shape, y_train.shape)
    x_test, y_test = genDataset3D(normal_dwt_test_path, bacterial_dwt_test_path, viral_dwt_test_path, covid_dwt_test_path)
    x_test_res = resnet50.preprocess_input(x_test)
    x_test_vgg = vgg19.preprocess_input(x_test)
    x_test_mobile = mobilenet.preprocess_input(x_test)
    print(x_test.shape, y_test.shape)


    # model = createDarkNetwork()
    #Create Estimate Learner Function
    res_model = createResNetwork()
    res_model.trainable = False
    print(res_model.summary()) #dense_51

    vgg_model = createVGGNetwork()
    vgg_model.trainable = False
    print(vgg_model.summary()) #dense_53

    mobile_model = createMobileNetwork()
    mobile_model.trainable = False
    print(mobile_model.summary()) #dense_55

    dark_model = createDarkNet()
    dark_model.trainable = False
    print(dark_model.summary()) #dense_59

    estimate_model = createEstimateLearner(res_model, vgg_model, mobile_model, dark_model)
    estimate_model.compile(optimizer='adam',loss=custom_loss,metrics=['accuracy'])

    print(estimate_model.summary())

    estimate_model = createEstimateLearner(res_model, vgg_model, mobile_model, dark_model)
    estimate_model.compile(optimizer='adam',loss=custom_loss,metrics=['accuracy'])

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=network_dwt_save_path, save_weights_only=True, verbose=1)
    cp_best_callback = tf.keras.callbacks.ModelCheckpoint(filepath=network_dwt_best_save_path, save_weights_only=True, monitor='val_accuracy', save_best_only=True, verbose=1)
    r = estimate_model.fit([x_train, x_train, x_train], y_train, batch_size=10, validation_data=([x_test, x_test, x_test], y_test), epochs=40, callbacks=[cp_callback, cp_best_callback]) 

    # Plot confusion matrix
    # p_test = model.predict(x_test).argmax(axis=1)
    # cm = confusion_matrix(y_test, p_test)
    # plot_confusion_matrix(cm, list(range(10)))