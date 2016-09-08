#! /usr/bin/env python

import cPickle, gzip
import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils
from matplotlib.pyplot import imshow
from utils import *

# Load the dataset
num_classes = 18
X_train, y_train, X_test, y_test = getMNISTData()

## Categorize the labels
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

#################################
## Model specification

## Start from an empty sequential model where we can stack layers
model = Sequential()

## Add a fully-connected layer with 128 neurons. The input dim is 784 which is the size of the pixels in one image
model.add(Dense(output_dim=128, input_dim=784))

## Add rectifier activation function to each neuron
model.add(Activation("relu"))

## Add another fully-connected layer with 10 neurons, one for each class of labels
model.add(Dense(output_dim=18))

## Add a softmax layer to force the 10 outputs to sum up to one so that we have a probability representation over the labels.
model.add(Activation("softmax"))

##################################

## Compile the model with categorical_crossentrotry as the loss, and stochastic gradient descent (learning rate=0.001, momentum=0.5,as the optimizer)
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.055, momentum=0.8))

## Fit the model (10% of training data used as validation set)
model.fit(X_train, y_train, nb_epoch=10, batch_size=40, validation_split=0.1, show_accuracy=True)

## Evaluate the model on test data
objective_score = model.evaluate(X_test, y_test, show_accuracy=True, batch_size=32)
# objective_score is a tuple containing the loss as well as the accuracy
print 'Accuracy on test set: ' + str(objective_score[1])
