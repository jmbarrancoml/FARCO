from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation, Convolution2D, MaxPooling2D

import cv2
import numpy as np


def train_cnn():
    #Imagenes
    train_path = '/content/drive/MyDrive/FARCO/dataset/train'
    val_path = '/content/drive/MyDrive/FARCO/dataset/validation'

    #Parametros preprocesamiento
    altura_img, anchura_img = 32, 32


    #Parametros e hiperparámetros modelo
    epocas = 3
    iteraciones = 50
    it_validacion = 25
    batch_size = 25
    nfiltros_conv1 = 16
    nfiltros_conv2 = 32
    filtros_conv1 = (5, 5)
    filtros_conv2 = (3, 3)
    pooling_size = (2, 2)
    labels = 2
    eta = 1e-2
    
        #Transformaciones a realizar
    train_generator = ImageDataGenerator(
        rescale = 1.0/255.0,
        shear_range = 0.3,
        zoom_range = 0.3,
        horizontal_flip = True
    )

    val_generator = ImageDataGenerator(
        rescale = 1.0/255.0
    )

    #Lectura y transformación de las imagenes
    train_img = train_generator.flow_from_directory(
        train_path,
        target_size = (altura_img, anchura_img),
        batch_size = batch_size,
        class_mode = "categorical"
    )

    val_img = val_generator.flow_from_directory(
        val_path,
        target_size = (altura_img, anchura_img),
        batch_size = batch_size,
        class_mode = "categorical"
    )
    
    cnn = Sequential()
    cnn.add(Convolution2D(nfiltros_conv1, filtros_conv1, padding="same", input_shape = (altura_img, anchura_img, 3), activation="elu"))
    cnn.add(MaxPooling2D(pool_size = pooling_size))
    cnn.add(Convolution2D(nfiltros_conv2, filtros_conv2, padding="same", activation="elu"))
    cnn.add(MaxPooling2D(pool_size = pooling_size))
    cnn.add(Flatten())
    cnn.add(Dense(128, activation="sigmoid"))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(labels, activation="softmax"))
    
    cnn.compile(loss = "categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    cnn.fit(train_img, steps_per_epoch = iteraciones, epochs = epocas, validation_data = val_img, validation_steps = it_validacion)
    
    cnn.save('/content/drive/MyDrive/FARCO/classifiers/cnn_fvsd_9008_model.h5')
    cnn.save_weights('/content/drive/MyDrive/FARCO/classifiers/cnn_fvsd_9008_weights.h5')
    
    

if __name__ == '__main__':
    train_cnn()