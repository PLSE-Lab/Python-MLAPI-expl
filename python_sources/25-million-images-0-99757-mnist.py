#!/usr/bin/env python
# coding: utf-8

# # Accuracy=99.75% using 25 Million Training Images!!
# It's amazing that convolutional neural networks can classify handwritten digits so accurately. In this kernel, we witness an ensemble of 15 CNNs classify Kaggle's MNIST digits after training on Kaggle's 42,000 images in "train.csv" plus 25 million more images created by rotating, scaling, and shifting Kaggle's images. Learning from 25,042,000 images, this ensemble of CNNs achieves 99.75% classification accuracy. This kernel uses ideas from the best published models found on the internet. Advanced techniques include data augmentation, nonlinear convolution layers, learnable pooling layers, ReLU activation, ensembling, bagging, decaying learning rates, dropout, batch normalization, and adam optimization. 

# In[1]:


# LOAD LIBRARIES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler


# # Load Kaggle's 42,000 training images

# In[2]:


# LOAD THE DATA
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[3]:


# PREPARE DATA FOR NEURAL NETWORK
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)
X_train = X_train / 255.0
X_test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, num_classes = 10)


# # Generate 25 million more images!!
# by randomly rotating, scaling, and shifting Kaggle's 42,000 images.

# In[4]:


# CREATE MORE IMAGES VIA DATA AUGMENTATION
datagen = ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1)


# # Build 15 Convolutional Neural Networks!

# In[5]:


# BUILD CONVOLUTIONAL NEURAL NETWORKS

model= Sequential()

model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size = 4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

    # COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# In[9]:


model = Sequential()
    
    # Multiple convolution operations to detect features in the images
model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))
model.add(BatchNormalization())

model.add(Conv2D(32,kernel_size=3,activation='relu')) # no need to specify shape as there is a layer before
model.add(BatchNormalization())

model.add(Conv2D(48,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5)) # reduce overfitting

model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5)) # reduce overfitting
    
    # Flattening and classification by standard ANN
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
    
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()


# # Train 15 CNNs

# In[10]:



epochs = 45

X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.1)
model.fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64),epochs = epochs, steps_per_epoch = X_train2.shape[0]//64,  
        validation_data = (X_val2,Y_val2), verbose=1)


# In[11]:


model.save('lexnet.model')


# In[12]:


test_data = pd.read_csv( data_dir + test_file)
test_X = test_data.loc[:, test_data.columns !=  'label']
test_X = scaler.transform(test_X)
test_X_mat = test_X.reshape(28000, 28, 28, 1)


# In[13]:


class_prediction = model.predict_classes(X_test)
result = pd.DataFrame(class_prediction)
result['Label'] = result[0]
result['ImageId'] = result.index + 1


# In[14]:


result.head()


# In[15]:


result.to_csv(index=False, 
              path_or_buf='submission_result.csv',
             columns = ['ImageId', 'Label'])


# # Ensemble 15 CNN predictions and submit
