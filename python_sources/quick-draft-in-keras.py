#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


images_dir_name = '../input/stage_1_test_images'
input_dir = '../input/'


# In[ ]:


# retrieve all the labels and store those into a collection
classes_trainable = pd.read_csv(input_dir+'classes-trainable.csv')
all_labels = classes_trainable['label_code']
print ('The number of unique labels is {}'.format(len(all_labels)))


# In[ ]:


# set the number of labels which will be used as an output layer size for a model
num_labels = len(all_labels)

# build the index dictionary based on the labels collection
labels_index = {label:idx for idx, label in enumerate(all_labels)}


# In[ ]:


# retrieve the list of train images (in our case we'll be using the test images just to get the model up and running)
# this will be changed to the train data set in the future.
train_image_names = [img_name[:-4] for img_name in os.listdir(images_dir_name)]
print (train_image_names[0])
print ("number of training images is {}".format(len(train_image_names)))


# In[ ]:


# retrieve the list of train labels (machine labels for now; need to work on replacing the machine labels with human ones if available)
# for now I'll be using tuning labels
labels = pd.read_csv('../input/tuning_labels.csv')
labels.head()
train_images = []
train_labels_raw = []
for index, row in labels.iterrows():
    train_images.append(row[0])
    labels_raw = row[1].split(' ')
    train_labels_raw.append([labels_index[label] for label in labels_raw])


# In[ ]:


# do the multi-hot encoding
def multi_hot_encode(x, num_classes):
    encoded = []
    for labels in x:
        labels_encoded = np.zeros(num_classes)
        
        for item in labels:
            labels_encoded[item] = 1
            
        encoded.append(labels_encoded)
        
    encoded = np.array(encoded)
    
    return encoded



# In[ ]:


train_labels = multi_hot_encode(train_labels_raw, num_labels)
print (train_labels)


# In[ ]:


from sklearn.utils import shuffle
#import tensorflow as tf
import cv2


# In[ ]:


# define the normalization logic for an image data
def normalize(x):
    return (x.astype(float) - 128)/128


# In[ ]:


# define the dimensions of the processed image
x_dim = 100
y_dim = 100
n_channels = 3


# In[ ]:


# define scaling for logic for an image data
def scale(x):
    return cv2.resize(x, (x_dim, y_dim))


# In[ ]:


# read and pre-process image
def preprocess(image_name):
    img = cv2.imread(image_name)
    scaled = scale(img)
    normalized = normalize(scaled)
    
    return np.array(normalized)


# In[ ]:


# prepare the collection of labels
def get_labels(image_name):
    labels = []
    
    # todo implement
    
    return labels


# In[ ]:


# build the generator for training
def generator(samples, sample_labels, batch_size=32):
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples, sample_labels)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_labels = sample_labels[offset:offset+batch_size]

            images = []
            labels = []

            for i, batch_sample in enumerate(batch_samples):

                image = preprocess(images_dir_name+'/'+batch_sample+'.jpg')

                # this will be needed later once get the real data
                #image_labels = get_labels(batch_sample)

                images.append(image)
                labels.append(batch_labels[i])

            X_train = np.array(images)
            y_train = np.array(labels)
            yield sklearn.utils.shuffle(X_train, y_train)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, BatchNormalization, MaxPooling2D, Lambda, Dropout, Flatten, Cropping2D
from keras.callbacks import ModelCheckpoint, EarlyStopping


# In[ ]:


def build_model(num_classes):
    model = Sequential()
    
    # convolutions with maxpooling and batchnorm
    model.add(Conv2D(24, kernel_size=(5,5), strides=(1,1), padding='same', input_shape=(x_dim, y_dim, n_channels)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(36, kernel_size=(5,5), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(48, kernel_size=(5,5), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    # flatten and add fully connected layers
    model.add(Flatten())
    model.add(Dense(num_classes*2))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(num_classes*2))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))
    
    # compile with Adam optimizer and mean squared error as the loss function
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
    
    return model


# In[ ]:


from sklearn.model_selection import train_test_split
Xtrain, Xvalid, ytrain, yvalid = train_test_split(train_images, train_labels, test_size=0.1)


# In[ ]:


# define the number of epochs
epochs=5
batch_size = 32


# In[ ]:


# trains the model
# defined 2 callbacks: early stopping and checkpoint to save the model if the validation loss has been improved
def train_model(model, train_generator, validation_generator, epochs=3):
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=1)
    checkpoint_callback = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    
    model.fit_generator(train_generator, steps_per_epoch=len(ytrain)//batch_size, validation_data=validation_generator, validation_steps=len(yvalid)//batch_size, epochs=epochs, callbacks=[early_stopping_callback, checkpoint_callback])


# In[ ]:


# compile and train the model using the generator function
train_generator = generator(Xtrain, ytrain, batch_size=batch_size)
validation_generator = generator(Xvalid, yvalid, batch_size=batch_size)

model = build_model(num_labels)

train_model(model, train_generator, validation_generator, epochs)


# In[ ]:


# predict one label
def predict(model, image_name, threshold=0.5):
    image = preprocess(image_name)
    image = np.reshape(image,[1,x_dim, y_dim, n_channels])
    prediction = model.predict(image)[0]
    
    print (prediction)
    
    indices = np.argwhere(prediction >= threshold).flatten()
    print (indices)
    
    labels = all_labels[indices]
    
    return labels


# In[ ]:


img_name = images_dir_name+'/'+Xtrain[0]+'.jpg'

print (predict(model, img_name))


# In[ ]:




