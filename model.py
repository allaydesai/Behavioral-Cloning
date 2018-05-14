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
from keras.layers import Input, Flatten, Dense, Lambda
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import os, sys
import settings

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# define flags:
#flags = tf.app.flags
#FLAGS = flags.FLAGS

#flags.DEFINE_string('data_path', '', 'Path to data folder')
#flags.DEFINE_string('training_file', '', 'Pickle file of training data')
#flags.DEFINE_string('validation_file', '', 'Pickle file of validation data')
#flags.DEFINE_integer('epochs', 1, 'Number of epochs')


# Load file paths
def LoadRawData(path):
    with open(path+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        lines = [line for line in reader]
    # load data: images and measurements
    images = np.array([cv2.imread(line[0]) for line in lines])
    measurements = np.array([float(line[3]) for line in lines])
    return images, measurements

def PrintStats():
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    #n_test = X_test.shape[0]
    image_shape = X_train.shape[1:]
    n_classes, counts = np.unique(y_train, return_counts=True)

    print("Number of training examples =", n_train)
    #print("Number of validation examples =", n_val)
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
    
#def main(_):
# load data
#check if pickle file exists
if os.path.isfile(settings.DATA_PATH + settings.DATA_PICKLE_FILENAME):
    X_train, y_train, X_test, y_test = LoadFromPickleFile()
else:
    X_train, y_train = LoadRawData(settings.DATA_PATH)#FLAGS.data_path)
    # Split data 
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.15, random_state=0)
    # Create pickle file for future use
    SaveDataToPickleFile()
    print('Data cached in pickle file.')

PrintStats()

# pre-process data

# define model
print('Defining Model')
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1)) 

model.compile(loss='mse', optimizer='adam',  metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs = settings.EPOCHS)

print('Saving model...')
model.save(settings.MODEL_FILENAME)

print("Testing...")
metrics = model.evaluate(X_test, y_test)

for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))
    
# parses flags and calls the `main` function above
#if __name__ == '__main__':
#    tf.app.run()