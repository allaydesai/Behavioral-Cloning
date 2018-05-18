# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview

---

This project uses deep neural networks and convolutional neural networks to clone driving behavior. I trained, validated and tested a model using Keras. The model outputs a steering angle to an autonomous vehicle.

I used a simulator where I steered a car around a track for data collection. I used the collected image data and steering angles to train a neural network and then used this model to drive the car autonomously around the track.

**Project Goals**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

**Project Files**

The repository consists of the following files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car)
* model.h5 (a trained Keras model)
* a README writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

**Dependencies**

The project requires the libraries desicribed in the environment.yml file. The environment can be setup using the file in anaconda. 

---

## Dataset

The dataset was created by me performing laps around the test track in the simluated environment. The simulator had two tracks avialbale to collect data. The data collection was divided as follows:
- Track1: Clockwise 4 laps
- Track1: Counterclockwise 2 laps
- Each of the turns twice
- Offroad recovery
- Track2: Clockwise 1 lap

Total examples = 25902

Number of training examples = 20721 

Number of testing examples = 0 (tested in the simulator environment)

Number of validation examples = 5181

Image data shape = (160, 320, 3)

Labels data shape = (25902,)


## Model Architecture and Training Strategy

For the purpose of this project I chose the model architecture proposed by the autonomus team at nvidia in the following paper:
https://devblogs.nvidia.com/deep-learning-self-driving-cars/

I chose this as it has been tested previosuly on a similar appication. The model consists of a normalization layer followed by 5 convolutional layers followed by 4 fully-connected layers. The network uses convolutional strides to change image dimensions through the various layers.  

To avoid overfitting, I added L2 regularization of weights on most layers and dropout between the fully connected layers. This should help the model avoid learning the training data. 

The network architecture is shown below:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Lambda         		| Normalization: x/255 - 0  							| 
| Cropping2D         		| Crop image by (65,25), (0,0)   							| 
| Convolution 5x5     	| 24 filters, 2x2 stride, VALID padding, L2 regularize weights 	|
| RELU					|			Activation Function									|
| Convolution 5x5     	| 36 filters, 2x2 stride, VALID padding, L2 regularize weights 	|
| RELU					|			Activation Function									|
| Convolution 5x5     	| 48 filters, 2x2 stride, VALID padding, L2 regularize weights 	|
| RELU					|			Activation Function									|
| Convolution 3x3     	| 64 filters, 1x1 stride, VALID padding, L2 regularize weights 	|
| RELU					|			Activation Function									|
| Convolution 3x3     	| 64 filters, 1x1 stride, VALID padding, L2 regularize weights 	|
| RELU					|			Activation Function									|
| Flatten	    |       									|
| Fully connected		| 100 outputs, L2 regularize weights,RELU activation        									|
| Dropout					|			0.3 - Regularization									|
| Fully connected		| 50 outputs, L2 regularize weights, RELU activation        									|
| Dropout					|			0.3	- Regularization								|
| Fully connected		| 10 outputs, L2 regularize weights, RELU activation        									|
| Dropout					|			0.3 - Regularization									|
| Fully connected				| 1 output, Linear activation        									|

**Training**

Epochs: 30 (Model stopped training after 8-12 epochs due to early stopping)

Batch size: 128

Loss Function: Mean Square Error

Optimizer: Adam

Metrics: Loss & Accuracy

I used callback functions to ensure saving the best snapshot of the training process. This is the point when validation loss is lowest. 
Further, I added early stopping which monitors the validation loss and stops the training process if there is no change to loss value for defined amount of time. 

Finally, I used the fit function to train the network. I chose the size of my validation dataset to be 20% of the training dataset with the option to shuffle the dataset. 

## Model Evaluation

Upon completion of training, I evaluated the model based on the following principles:

- Underfitting – Validation and training error high
- Overfitting – Validation error is high, training error low
- Good fit – Validation error low, slightly higher than the training error
- Unknown fit - Validation error low, training error 'high'

Plot showing the loss through the training process: 

![alt text][image1]

If I found the model to be overfitting, I added more regularization and if I found it to be underfitting I added more data where the model tends to fail. 

The plot above shows that the model is a good fit for the data.

Next, using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

I recoded two runs around track 1 in autonomus mode using the following command:

```sh
python drive.py model.h5 run1
```
The data was then converted to a movie using:

```sh
python video.py run1
```

They can be viewed in the run1.mp4 and run2.mp4 files.


