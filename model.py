# -*- coding: utf-8 -*-
"""
Created on Sat May 12 15:08:40 2018

@author: Allay
"""
# import libraries
import csv
import pickle
import tensorflow as tf
import numpy as np
import cv2
from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
import os, sys
import settings
import matplotlib.pyplot as plt
import sklearn
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l2
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# Load file paths
def LoadRawData(path, path_to_img=""):
    with open(path+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        lines = [line for line in reader]
    # load data: images and measurements
    images = np.array([plt.imread(path+path_to_img+line[0].split('\\')[-1]) for line in lines])
    measurements = np.array([float(line[3]) for line in lines])
    return images, measurements

def LoadAllRawData(path, path_to_img):
    filenames = [filename for filename in os.listdir(settings.DATA_PATH)]
    images = []
    measurements = []
    for filename in filenames:
        #if filename == 'DATA_0-Original' or filename == 'DATA_1-CounterClockwise' or filename == 'DATA_3-turns' or filename == 'DATA_6_recovery' or filename == 'DATA_7_recovery'or filename == 'DATA_8_Turns' or filename == 'DATA_9_turns':
        if filename == 'data' or filename == 'DATA_3-turns':
            with open(path+filename+'\\'+'driving_log.csv') as csvfile:
                reader = csv.reader(csvfile)
                lines = [line for line in reader]
            # load data: images and measurements
            for line in lines:
                image = cv2.imread(path+filename+'\\'+path_to_img+line[0].split('/')[-1])
                try:
                    measurment = float(line[3])
                    images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    measurements.append(measurment)
                except:
                    print("unable to load")
            print(filename, ' Data_size: ',len(images))
        elif filename == 'DATA_3-turns':
            with open(path+filename+'\\'+'driving_log.csv') as csvfile:
                reader = csv.reader(csvfile)
                lines = [line for line in reader]
            # load data: images and measurements
            for line in lines:
                image = cv2.imread(path+filename+'\\'+path_to_img+line[0].split('/')[-1])
                try:
                    measurment = float(line[3])
                    images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    measurements.append(measurment)
                except:
                    print("unable to load")
            print(filename, ' Data_size: ',len(images))
    return np.array(images), np.array(measurements)
	
def LoadAllCameraRawData(path, path_to_img):
    filenames = [filename for filename in os.listdir(settings.DATA_PATH)]
    images = []
    measurements = []
    for filename in filenames:
        #if filename == 'DATA_0-Original' or filename == 'DATA_1-CounterClockwise' or filename == 'DATA_3-turns' or filename == 'DATA_6_recovery' or filename == 'DATA_7_recovery'or filename == 'DATA_8_Turns' or filename == 'DATA_9_turns':
        if filename == 'data':
            with open(path+filename+'/'+'driving_log.csv') as csvfile:
                reader = csv.reader(csvfile)
                lines = [line for line in reader]
            # load data: images and measurements
            for line in lines:
                for i in range(3):
                    if i == 0:
                        image = cv2.imread(path+filename+'/'+path_to_img+line[0].split('/')[-1])
                        try:
                            measurment = float(line[3])
                            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                            measurements.append(measurment)
                        except:
                            print("unable to load")
                    elif i == 1:
                        image = cv2.imread(path+filename+'/'+path_to_img+line[1].split('/')[-1])
                        try:
                            measurment = float(line[3]) + settings.STEERING_CORRECTION
                            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                            measurements.append(measurment)
                        except:
                            print("unable to load")
                    elif i == 2:
                        image = cv2.imread(path+filename+'/'+path_to_img+line[2].split('/')[-1])
                        try:
                            measurment = float(line[3]) - settings.STEERING_CORRECTION
                            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                            measurements.append(measurment)
                        except:
                            print("unable to load")
                
            print(filename, ' Data_size: ',len(images))
        elif filename == 'DATA_3-turns':
            with open(path+filename+'/'+'driving_log.csv') as csvfile:
                reader = csv.reader(csvfile)
                lines = [line for line in reader]
            # load data: images and measurements
            for line in lines:
                for i in range(3):
                    if i == 0:
                        image = cv2.imread(path+filename+'/'+path_to_img+line[0].split('\\')[-1])
                        try:
                            measurment = float(line[3])
                            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                            measurements.append(measurment)
                        except:
                            print("unable to load")
                    elif i == 1:
                        image = cv2.imread(path+filename+'/'+path_to_img+line[1].split('\\')[-1])
                        try:
                            measurment = float(line[3]) + settings.STEERING_CORRECTION
                            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                            measurements.append(measurment)
                        except:
                            print("unable to load")
                    elif i == 2:
                        image = cv2.imread(path+filename+'/'+path_to_img+line[2].split('\\')[-1])
                        try:
                            measurment = float(line[3]) - settings.STEERING_CORRECTION
                            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                            measurements.append(measurment)
                        except:
                            print("unable to load")
            print(filename, ' Data_size: ',len(images))
    return np.array(images), np.array(measurements)

def PrintStats():
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    #n_test = X_test.shape[0]
    image_shape = X_train.shape[1:]
    n_classes, counts = np.unique(y_train, return_counts=True)

    print("Number of training examples =", n_train)
    #print("Number of validation examples =", n_val) NOTE: 20% Split in the fit function
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Labels data shape =", y_train.shape)
    
def LoadFromPickleFile():
    print('Loading data from pickle file...')
    for filename in os.listdir(settings.DATA_PATH):
        if filename == settings.DATA_PICKLE_FILENAME:
            try:
                with open(settings.DATA_PATH + filename, 'rb') as f:
                    data = pickle.load(f)
                    return data['train_dataset'], data['train_labels'], data['test_dataset'], data['test_labels']
            except:
                print('Unable to load training data from pickle file')
                raise
            
    
def SaveDataToPickleFile():
    print('Saving data to pickle file...')
    try:
        with open(settings.DATA_PATH + settings.DATA_PICKLE_FILENAME, 'wb') as pfile:
            pickle.dump(
                {
                    'train_dataset': X_train,
                    'train_labels': y_train,
                    'test_dataset': X_test,
                    'test_labels': y_test,
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', settings.DATA_PICKLE_FILENAME, ':', e)
        raise
        
def DataAugmentation(images, measurements):
    # Flip images and negate steering
    aug_imgs, aug_measure = [], []
    for image, measurement in zip(images, measurements):
        aug_imgs.append(image)
        aug_measure.append(measurement)
        aug_imgs.append(cv2.flip(image,1))
        aug_measure.append(measurement * -1.0)
    return np.array(aug_imgs), np.array(aug_measure)
        
def CreateModel():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(6,5,5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84)) 
    model.add(Dense(1)) 
    return model
    
def CreateNvidiaModel():
    model = Sequential()
    model.add(Lambda((lambda x: x/255.0 - 0.5), input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((65,25), (0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), W_regularizer=l2(0.001), activation="elu"))
    model.add(Dropout(0.3))
    model.add(Convolution2D(36,5,5, subsample=(2,2), W_regularizer=l2(0.001), activation="elu"))
    model.add(Dropout(0.3))
    model.add(Convolution2D(48,5,5, subsample=(2,2), W_regularizer=l2(0.001), activation="elu"))
    model.add(Dropout(0.3))
    model.add(Convolution2D(64,3,3, W_regularizer=l2(0.001), activation="elu"))
    model.add(Dropout(0.3))
    model.add(Convolution2D(64,3,3, W_regularizer=l2(0.001), activation="elu"))
    model.add(Dropout(0.3))
    model.add(Flatten())
    
    model.add(Dense(100, W_regularizer=l2(0.001), activation="elu"))
    model.add(Dropout(0.3))
    model.add(Dense(50, W_regularizer=l2(0.001), activation="elu")) 
    model.add(Dropout(0.3))
    model.add(Dense(10, W_regularizer=l2(0.001), activation="elu")) 
    model.add(Dropout(0.3))
    model.add(Dense(1)) 
    return model
	
def CreateOrigNvidiaModel():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((60,25), (0,0))))
    model.add(Convolution2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Convolution2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    return model

	
###############################################
#------------------- MAIN --------------------#
###############################################

# LOAD DATA
#check if pickle file exists
if os.path.isfile(settings.DATA_PATH + settings.DATA_PICKLE_FILENAME):
    X_train, y_train, X_test, y_test = LoadFromPickleFile()
else:
    #load raw data
    X_train, y_train = LoadAllCameraRawData(settings.DATA_PATH, settings.PATH_TO_IMG)
    #shuffle
    sklearn.utils.shuffle(X_train, y_train)
    # Split data - NOTE: not splitting any test data, using the simulator to test instead
    #X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.15, random_state=0)
    X_test = np.array([])
    y_test = np.array([])
    # Create pickle file for future use
    SaveDataToPickleFile()
    print('Data cached in pickle file.')

# AUGMENT DATA
X_train, y_train = DataAugmentation(X_train, y_train)

PrintStats()


# DEFINE MODEL
model = CreateOrigNvidiaModel()

model.compile(loss='mse', optimizer='adam',  metrics=['accuracy'])

# TRAINING
# define callbacks
checkpoint = ModelCheckpoint(filepath=settings.MODEL_FILENAME,verbose=0, save_best_only=True, mode=min, monitor='val_loss')
earlyStopping = EarlyStopping(monitor='val_loss', patience=2,verbose=0,mode='min')
callbacks = [checkpoint, earlyStopping]

history = model.fit(X_train, y_train, batch_size = settings.BATCH_SIZE, validation_split=0.2, shuffle=True, epochs=settings.EPOCHS, callbacks=callbacks)

print('Saving model...')
model.save(settings.MODEL_FILENAME)

# Visualizing loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training_set', 'validation_set'], loc='upper right')
plt.savefig('loss_plot.png')
print('Plot Saved as loss_plot.png..')
plt.show()
