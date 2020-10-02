#!/usr/bin/env python
# coding: utf-8

# # First Steps with Keras Convolutional Neural Networks - Nature Conservancy Image Recognition Kaggle

# **Keras CNN that will yield 95% accuracy on its training data, if you use the full data set, 25+ epochs.** (Here we use a subset, and only a few epochs, for sake of speed.) More training epochs and more + better data --> more accuracy.
# 
# This CNN was created using public tutorials, but updated to work on the data set for the current project.

#  **For sources, see:**
# 
#  - [http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/][1]
#  - [https://medium.com/@waleedka/traffic-sign-recognition-with-tensorflow-629dffc391a6#.v471kaepx][2]
# 
# 
#   [1]: http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/
#   [2]: https://medium.com/@waleedka/traffic-sign-recognition-with-tensorflow-629dffc391a6#.v471kaepx

# ## Goals
# This Notebook is posted to give new users a start on using Keras in this competition. 

# ## Results
# This CNN will predict output with high level of accuracy, BUT: all outputs for predictions will be either 1.0 or 0.0
# 
# Because of the way this Kaggle is scored, the incorrect data points will lead to a large loss. 

# ## Next steps
# Need to find a network model/architecture that will distribute its predictions over the full set of classes, not simply a 0/1 binary prediction. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# imports needed for CNN
import csv
import cv2
import os, glob
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import time
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Load the data
def load_data(data_dir):
    """
    From: https://medium.com/@waleedka/traffic-sign-recognition-with-tensorflow-629dffc391a6#.v471kaepx
    """
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []

    category = 0
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".jpg")]
        
        # adding an early stop for sake of speed
        stop = 0
        for f in file_names:
            img = cv2.imread(f)
            imresize = cv2.resize(img, (200, 125))
            #plt.imshow(imresize)
            images.append(imresize)
            labels.append(category)
            # remove this to use full data set
            if stop > 30:
                break
            stop += 1
            # end early stop
            
        category += 1

    return images, labels

data_dir = "../input"
images, labels = load_data(data_dir)

# confirm that we have the data
print(images[0:10])
print(labels[0:10])


# Cross validate the data, so we can use a test set to check accuracy, before submitting.

# In[ ]:


def cross_validate(Xs, ys):
    X_train, X_test, y_train, y_test = train_test_split(
            Xs, ys, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = cross_validate(images, labels)

# confirm we got our data
print(y_test[0:10])


# Normalize the data and hot encode outputs

# In[ ]:


# normalize inputs from 0-255 and 0.0-1.0
X_train = np.array(X_train).astype('float32')
X_test = np.array(X_test).astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
print("Data normalized and hot encoded.")


# ### The code below creates and fits the CNN. It will take a while to load, even with 2 epochs.
# Please ensure the code below runs, before testing the final section, which will save your file. 

# Create our CNN Model

# In[ ]:


def createCNNModel(num_classes):
    """ Adapted from: # http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/
# """
    # Create the model
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(125, 200, 3), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    epochs = 3  # >>> should be 25+
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())
    return model, epochs

# create our CNN model
model, epochs = createCNNModel(num_classes)
print("CNN Model created.")


# Fit and run the model

# In[ ]:


# fit and run our model
seed = 7
np.random.seed(seed)
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

print("done")


# Save output for upload 
# 
# **Make sure to run the above code first, otherwise the model will not be defined**

# In[ ]:


from os import listdir
from os.path import isfile, join

prediction_output_list = []  # list of lists, containing logistic regression for each file
fnames = [f for f in listdir("../input/test_stg1/") if isfile(join("../input/test_stg1/", f))]
print("Testing File Names:")
print(fnames)

# early stoppage...
# only do 10
i = 0
for f in fnames:
    file_name = "../input/test_stg1/" + f
    print("---Evaluating File at: " + file_name)
    img = cv2.imread(file_name)  
    imresize = cv2.resize(img, (200, 125))  # resize so we're always comparing same-sized images
    imlist = np.array([imresize])
    print("Neural Net Prediction:")
    cnn_prediction = model.predict_proba(imlist)
    print(cnn_prediction)

    # format list for csv output
    csv_output_list = []
    csv_output_list.append(f)
    for elem in cnn_prediction:
        for value in elem:
            csv_output_list.append(value)

    # append filename to make sure we have right format to write to csv
    print("CSV Output List Formatted:")
    print(csv_output_list)


    # and append this file to the output_list (of lists)
    prediction_output_list.append(csv_output_list)
    
    ############## STOP EARLY TO SAVE TIME #################
    if i > 10:
        break
    i += 1
    #####  REMOVE TO RUN AGAINST FULL TEST SET ########

# Write to csv
"""
#  Commented out for Kaggle, but you can use this to write to a CSV on your own computer.
try:
    with open("cnn_predictions.csv", "wb") as f:
        writer = csv.writer(f)
        headers = ['image', 'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
        writer.writerow(headers)
        writer.writerows(prediction_output_list)
finally:
    f.close()
  """

print("done")

