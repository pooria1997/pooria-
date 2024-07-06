import tensorflow as tf
import math
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPool2D, LeakyReLU, Flatten, Dense, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import xception, vgg19, mobilenet, resnet50
import tensorflow.keras.backend as kb

from dataset import genDataset, genDataset3D, genDatasetFiles, image_height, image_width, image_mode

#For padding
def plus_one_pad(tensor, mode):
    return tf.pad(tensor, [[0, 0], [1, 1], [1, 1], [0, 0]], mode)

#For initialization
def initializer(length):
    return tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0) #1/math.sqrt(length/2))

alpha = 0.15

#DarkNet
def createDarkNet():
    i = Input(shape=(image_height, image_width, image_mode))
    x = Conv2D(8, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(i)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2), strides=2)(x)

    x = Conv2D(16, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2), strides=2)(x)

    x = Conv2D(32, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    #x = BatchNormalization()(x) #Dark 4
    x = Conv2D(16, (1, 1), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    #x = BatchNormalization()(x) #Dark 4
    x = Conv2D(32, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2), strides=2)(x)

    x = Conv2D(64, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    #x = BatchNormalization()(x) #Dark 4
    x = Conv2D(32, (1, 1), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    #x = BatchNormalization()(x) #Dark 4
    x = Conv2D(64, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    x = BatchNormalization()(x)

    #Filler pad layers for continuity (We are using 227x227 instead of 256x256)
    #x = plus_one_pad(x, "SYMMETRIC")
    #x = plus_one_pad(x, "SYMMETRIC")
    x = MaxPool2D((2, 2), strides=2)(x)
    #x = MaxPool2D((2, 2), strides=2)(x)

    x = Conv2D(128, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    #x = BatchNormalization()(x) #Dark 4
    x = Conv2D(64, (1, 1), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    #x = BatchNormalization()(x) #Dark 4
    x = Conv2D(128, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2), strides=2)(x)

    x = Flatten()(x)
    x = Dense(1568, activation=LeakyReLU(alpha=0.3))(x) #2304 = 12x12x64/4
    x = Dropout(0.3)(x)
    x = Dense(1568, activation=LeakyReLU(alpha=0.3))(x)
    x = Dropout(0.3)(x)
    x = Dense(224, name='dark_weights', activation=LeakyReLU(alpha=0.3))(x)

    x = Dense(4, activation='softmax')(x)

    #print(d1, d2, d3)
    model = Model(i, x)
    return model

#MobileNet
def createMobileNetwork():
    model = mobilenet.MobileNet(weights='imagenet')
    for layer in model.layers:
      layer._name = 'mobile_'+layer._name

    x = Flatten()(model.get_layer('mobile_reshape_2').output)
    x = Dropout(0.3)(x)
    x = Dense(256, name='mobile_weights', activation=LeakyReLU(alpha=0.3))(x)
    x = Dense(4, activation='softmax')(x)

    model = Model(model.input, x)
    return model

#ResNet
def createResNetwork():
    model = resnet50.ResNet50(weights='imagenet')
    for layer in model.layers:
      layer._name = 'res_'+layer._name

    x = Flatten()(model.get_layer('res_avg_pool').output)
    x = Dropout(0.3)(x)
    x = Dense(256, name='res_weights', activation=LeakyReLU(alpha=0.3))(x)
    x = Dense(4, activation='softmax')(x)

    model = Model(model.input, x)
    return model

#VGG19
def createVGGNetwork():
    model = vgg19.VGG19(weights='imagenet')
    for layer in model.layers:
      layer._name = 'vgg_'+layer._name

    x = Flatten()(model.get_layer('vgg_block5_pool').output)
    x = Dropout(0.3)(x)
    x = Dense(256, name='vgg_weights', activation=LeakyReLU(alpha=0.3))(x)
    x = Dense(4, activation='softmax')(x)

    model = Model(model.input, x)
    return model
