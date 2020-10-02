#!/usr/bin/env python
# coding: utf-8

# # Simple Convolution Network to Classify MNIST Digits

# **In this notebook you will learn how to read the data, display some of the input images, create a very basic CNN model. Also, you will learn how to visualize how the model makes the decision**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

seed = 42
np.random.RandomState(seed)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# In[ ]:


# Kers modules
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, History
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D, GlobalMaxPool2D


# In[ ]:


import cv2
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.rcParams['figure.figsize'] = (16, 10)


# In[ ]:


dir_path = "/kaggle/input"


# # Load & Prepare data

# ## Explore the data

# In[ ]:


# size of the image
IMG_SIZE = 28


# In[ ]:


# read the input data
train_df = pd.read_csv(os.path.join(dir_path, "train.csv"))

# print the dimension of the data
print("Shape of the input data: ", train_df.shape)

# display 5 records
train_df.head(5)


# In[ ]:


def display_images(images, true_labels, pred_labels=None, cam_act=None):
    """
        Function to display the provided images along with their labels and if provided, predicted labels.
        Also, if provided the class activations will be overlayed on the original image.
    """
    
    # get number of images
    num_images = len(images)
    
    # we will display only 5 images in a row, so, we will calculate number of rows
    columns = 5
    rows = math.ceil(num_images/columns)
    
    # prepare the title of each image which will be True label and Predicted labels
    if pred_labels != None:
        titles = ["True-{} | Pred-{}".format(true, pred) for true, pred in zip(true_labels, pred_labels)]
    else:
        titles = list(map(lambda x: "True-{}".format(x), true_labels))
        
    # also specify the color of title if prediction was accurate and when not
    if pred_labels != None:
        colors = ['green' if true==pred else 'red' for true, pred in zip(true_labels, pred_labels)]
    else:
        # if pred lables do not exists put default color
        colors = ['black' for i in range(num_images)]
        
    # now plot the images
    fig, axes = plt.subplots(rows, columns, sharex=True, sharey=True)
    
    for r in range(rows):
        for c in range(columns):
            
            # calculate the index/position of an image
            img_index = (r*columns + c)
            
            # if we have displayed required number of images, then break the loop
            if (img_index + 1) > num_images:
                break
            
            # display the image
            axes[r, c].imshow(images[img_index], alpha=0.6 if cam_act else 1)
            
            # display the CAM if provided
            if cam_act:
                axes[r, c].imshow(cam_act[img_index], cmap='jet', alpha=0.4)
                
            # display the true and predicted labels
            axes[r, c].set_title(titles[img_index], fontdict=dict(color=colors[img_index]))
                
    pass


# In[ ]:


# filter out some images to display along with their labels
sample_images = train_df.iloc[:15, 1:].values.reshape(-1, 28, 28)
true_labels = train_df.iloc[:15, 0].values

# display using the provided function
display_images(sample_images, true_labels)


# ## Prepare the dataset for training

# In[ ]:


# for tutorial only, so that it trains during the course
train_df = train_df.sample(frac=.10)


# In[ ]:


# get the image features and normalize the values
X = train_df.values[:, 1:] / 255.0

# also, CNN expects the image to be of 3D, so, we will reshape the image
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# get and One-hot encode the labels
Y = to_categorical(train_df.values[:, 0])

# print the dimensions of the data
print("X shape-{}\tY shape-{}".format(X.shape, Y.shape))


# In[ ]:


# split the data into train, validation and test datasets
TEST_SPLIT = 0.3
VALIDATION_SPLIT = .2

# split into train and test datasets first
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=TEST_SPLIT, random_state=seed)

# now, split the train data into train and validation data
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, stratify=Y_train, test_size=VALIDATION_SPLIT, random_state=seed)

print("Train:      \tX shape-{}\tY shape-{}\n".format(X_train.shape, Y_train.shape))
print("Validation: \tX shape-{}\tY shape-{}\n".format(X_val.shape, Y_val.shape))
print("Test:       \tX shape-{}\tY shape-{}\n".format(X_test.shape, Y_test.shape))


# # Build the Convolution Neural Network

# In[ ]:


# means that the layers will be executed in sequential manner i.e. one after the other
model = Sequential()

# here we will add a convolution layer, which Conv2D takes multiple inputs, which are 
# filters: number of times to make the convolutions, can be imagined as number of dimensions in the convolved image
# input_shape: shape of an input image
# kernel_size: size of the kernel which is the area of an image to focus on
# padding: whether to keep the size of the image after convolution or not
# strides: number of steps to shift the kernel, with stride of 1image size will remain same, but it will be halved with stride as 2
# activation: the activation function
model.add(Conv2D(filters=5, input_shape=(28, 28, 1), kernel_size=3, padding='same', strides=1, activation='relu'))

# we also add a pooling layer, which takes average or maximum of the pixel values in the provided window/pool_size
# for pool_size of 2, we will take max/average of all the values in size a 2x2 window and will result into a single value
# padding: here has the same purpose as the Conv2D
# after this step the size of the image will be halved
model.add(MaxPool2D(pool_size=2, padding="same"))

# adding another Conv2D layer, but here we increase the number of filters
model.add(Conv2D(filters=10, kernel_size=3, padding='same', strides=1, activation='relu'))
model.add(Conv2D(filters=10, kernel_size=3, padding='same', strides=1, activation='relu'))

# another pooling layer
model.add(MaxPool2D(pool_size=2, padding="same"))

model.add(GlobalMaxPool2D())

# flatten the previous layer, i.e. align all the nodes in a single layer
# model.add(Flatten())

# add the dropout layer, which will randomly turn off x% of the nodes while training
# model.add(Dropout(rate=0.2))

# add a dense layer containing 2048 nodes
# model.add(Dense(2048, activation='relu'))

# finally a soft-max layer which will give the probabilities of predicting each of the class
model.add(Dense(10, activation='softmax'))

# we will compile the model and use the Adam optimizer, we can also use SGD in place of that
# the model will try to minimize the loss given by 'categorical_crossentropy'
# also, the model will track the accuracy during training
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()


# In[ ]:


# to stop training the model if no incremental benefit is gained in subsequent EPOCHS
early_stopping_monitor = EarlyStopping(patience=3, min_delta=1e-3)


# In[ ]:


# train the model
model.fit(X_train, Y_train, epochs=20, callbacks=[early_stopping_monitor], validation_data=(X_val, Y_val), verbose=True)


# In[ ]:


# get the loss and accuracy for each epoch using history
history = model.history.history

# visualize the loss for each epochs
plt.figure(figsize=(20, 4))
sns.lineplot(x=range(len(history["loss"])),y=history["loss"], label="loss")
sns.lineplot(x=range(len(history["val_loss"])),y=history["val_loss"], label="val_loss")


# In[ ]:


print("Train accuracy:      ", model.evaluate(X_train, Y_train, verbose=0)[1])
print("Validation accuracy: ", model.evaluate(X_val, Y_val, verbose=0)[1])
print("Test accuracy:       ", model.evaluate(X_test, Y_test, verbose=0)[1])


# ## Display some of the images with predicted labels

# In[ ]:


# filter out some images to display along with their labels

# get the raw images
sample_images = X_test[:20, ]

# get their true labels
true_labels = np.argmax(Y_test[:20,], axis=1)

# get their pred labels by predicting from model
pred_labels = np.argmax(model.predict(sample_images), axis=1).tolist()

# display using the provided function
display_images(sample_images.reshape(-1, IMG_SIZE, IMG_SIZE), true_labels, pred_labels)


# # Class Activation Maps

# In[ ]:


def get_cam(model, images):
    """
        Function returns the Class Activation Maps for the provided images.
    """
    
    global pred_label_weights, img_features
    
    # get the weights to the last layer which is softmax layer
    # also, the weights contains a pair of input weights and bias weights and we need only the non bias weights
    weights = model.layers[-1].get_weights()[0]
    
    # create a model to get the outputs from 'x'th layer and the predicted values
    cam_model = Model(inputs = model.input, outputs = (model.layers[-3].output, model.layers[-1].output))
    
    # get the outputs for the provided images
    features, predictions = cam_model.predict(images)
    
    # to store the activation overlapped images
    images_w_cam = []
    
    # iterate over each provided input image
    for idx, img in enumerate(images):
        
        # predict the label
        pred_label = np.argmax(predictions[idx])
        
        # get the weights of the predicted label
        pred_label_weights = weights[:, pred_label]
        #print(pred_label_weights.shape)
        
        # get the features of the image
        img_features = features[idx]
        #print(img_features.shape)
        
        # get the size of the feature
        F_SIZE = img_features.shape[0]
        #print(F_SIZE)
        
        # take a dot product of weights and img features
        cam_activation = np.dot(img_features, pred_label_weights)
        #print(cam_activation.shape)
        
        # map the feature map to the original size
        height_roomout = IMG_SIZE * 1.0 / F_SIZE
        width_roomout = IMG_SIZE *1.0 / F_SIZE
        
        # zoom in/out the cam_activation to the original image size
        cam_activation = sp.ndimage.zoom(cam_activation, (height_roomout, width_roomout), order=2).astype('float64')
        
        # append the overlayed_img
        images_w_cam.append(cam_activation)
        
    return images_w_cam


# In[ ]:


# get the CAM for the sample images using the above function
overlayed_imgs = get_cam(model, sample_images)

# display using the provided function
display_images(sample_images.reshape(-1, IMG_SIZE, IMG_SIZE), true_labels, pred_labels, overlayed_imgs)


# # Make Predictions

# In[ ]:


# load the prediction dataset
predict_df = pd.read_csv(os.path.join(dir_path, "test.csv"))
print("Shape of the test dataset: ", predict_df.shape)
predict_df.head(2)


# In[ ]:


# extract the features and normalize it
X_predict = predict_df.values / 255.0
X_predict = X_predict.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# In[ ]:


# make the predictions
predictions = model.predict_classes(X_predict)
print("Shape of the predictions: ", predictions.shape)

print("\nSome predicted outputs: ")
print(predictions[:10])


# In[ ]:


## create output final in the required format
output = pd.DataFrame()

output["ImageId"] = [i for i in range(1, predictions.shape[0]+1)]
output["Label"] = predictions

output.to_csv("predictions.csv", index=False)

