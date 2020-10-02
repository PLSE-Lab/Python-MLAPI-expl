#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import gc

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import mobilenet
from keras.layers import Dense
from keras.models import Model
from keras import optimizers

# Any results you write to the current directory are saved as output.

# Version 21 Oversampling and testing kernel_regularizer=regularizers.l2(0.01) kernel_regularizer=regularizers.l1(0.01) and kernel_initializer='random_uniform',
# with loss 'categorical_crossentropy' and focal loss where gamma = 2 and alpha = 0.25
# and keeping the entire original MobileNet while adding a new output Dense layer that is trained with those regularizers and loss functions


# In[ ]:


generalDir = ["../input/fruits-360_dataset/fruits-360/Training", "../input/fruits-360_dataset/fruits-360/Test"]

appleDirs = []
lemonDirs = []
orangeDirs = []
pearDirs = []
for tDir in generalDir:
    appleDirs += [tDir +"/{}".format(i) for i in os.listdir(tDir) if 'Apple' in i]
    lemonDirs += [tDir +"/{}".format(i) for i in os.listdir(tDir) if 'Lemon' in i]
    orangeDirs += [tDir +"/{}".format(i) for i in os.listdir(tDir) if 'Orange' in i]
    pearDirs += [tDir +"/{}".format(i) for i in os.listdir(tDir) if 'Pear' in i]

    
print(appleDirs)
print(lemonDirs)
print(orangeDirs)
print(pearDirs)

appleImgs = []
lemonImgs = []
orangeImgs = []
pearImgs = []
#all images of apples
for appleDir in appleDirs:
    appleImgs += [appleDir + "/{}".format(j) for j in os.listdir(appleDir)]
#all images of lemons
for lemonDir in lemonDirs:
    lemonImgs += [lemonDir + "/{}".format(j) for j in os.listdir(lemonDir)]
#all images of oranges
for orangeDir in orangeDirs:
    orangeImgs += [orangeDir + "/{}".format(j) for j in os.listdir(orangeDir)]
#all images of pears
for pearDir in pearDirs:
    pearImgs += [pearDir + "/{}".format(j) for j in os.listdir(pearDir)]
    
del appleDirs
del lemonDirs
del orangeDirs
del pearDirs
del generalDir
gc.collect() #save memory
#print()
#print("Apple Num " + str(len(appleImgs)))   #Apple Num 8554
#print("Lemon Num " + str(len(lemonImgs)))   #Lemon Num 1312
#print("Orange Num " + str(len(orangeImgs))) #Orange Num 639
#print("Pear Num " + str(len(pearImgs)))     #Pear Num 3914

#Over Sampling
appleImgs = appleImgs
lemonImgs = lemonImgs * int(len(appleImgs) / len(lemonImgs))
orangeImgs = orangeImgs * int(len(appleImgs) / len(orangeImgs))
pearImgs = pearImgs * int(len(appleImgs) / len(pearImgs))

#Under Sampling
#appleImgs = appleImgs[:639]
#lemonImgs = lemonImgs[:639]
#orangeImgs = orangeImgs
#pearImgs = pearImgs[:639]



#image dimensions
rows = columns = 224

#auxiliary function
def processImages(imgs, label):
    X= [] #array of resized images
    y = [] #labels 0 for Apple; 1 for Lemon; 2 for Orange; 3 for Pear
    
    for img in imgs:
        #read and resize image
        X.append(cv2.resize(cv2.imread(img, cv2.IMREAD_COLOR), (rows, columns), interpolation = cv2.INTER_CUBIC))
        y.append(label)
    
    return X, y

X0, y0 = processImages(appleImgs, 0)
X1, y1 = processImages(lemonImgs, 1)
X2, y2 = processImages(orangeImgs, 2)
X3, y3 = processImages(pearImgs, 3)

del appleImgs, lemonImgs, orangeImgs, pearImgs
gc.collect()

X = X0 + X1 + X2 + X3
y = y0 + y1 + y2 + y3

del X0, y0, X1, y1, X2, y2, X3, y3
gc.collect()

X = np.array(X)
y = np.array(y)

from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=0) #shuffle the images (and at the same time the labels)


# In[ ]:


from keras import utils as np_utils
y = np_utils.to_categorical(y)


# In[ ]:


#split all the data into train, test and validation
from sklearn.model_selection import train_test_split

#60% for training, 20% for testing, 20% for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

del X
del y
gc.collect()


# In[ ]:


batch_size = 10
#train test and validation batches
trainDataGen = ImageDataGenerator(preprocessing_function = mobilenet.preprocess_input, rescale=1./255).flow(X_train, y_train, batch_size = batch_size)
testDataGen = ImageDataGenerator(preprocessing_function = mobilenet.preprocess_input, rescale=1./255).flow(X_test, y_test, batch_size = batch_size)
validationDataGen = ImageDataGenerator(preprocessing_function = mobilenet.preprocess_input, rescale=1./255).flow(X_val, y_val, batch_size = batch_size)


# In[ ]:


#mobile net
mobile = mobilenet.MobileNet()
mobile.summary()


# In[ ]:


import keras.regularizers as regularizers
output = mobile.output #layers[-1].output
print(output)
predictions = Dense(4, activation = 'softmax',kernel_regularizer=regularizers.l2(0.01), kernel_initializer='random_uniform')(output) #4 classes

model = Model(inputs = mobile.input, outputs = predictions)
model.summary()


# In[ ]:


#how many layers to train (last 23 layers)
for layer in model.layers[:-1]:
    layer.trainable = False


# In[ ]:



import tensorflow as tf
from keras import backend as K

def focal_loss(y_true, y_pred):
    g = 2.0
    a = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(a * K.pow(1. - pt_1, g) * K.log(pt_1))-K.sum((1-a) * K.pow( pt_0, g) * K.log(1. - pt_0))


# In[ ]:


#compile and train the model
#model.compile(optimizers.Adam(lr=.0001), loss = 'categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizers.Adam(lr=.0001), loss = [focal_loss], metrics=['accuracy'])
history = model.fit_generator(trainDataGen, steps_per_epoch = 5, validation_data=validationDataGen, validation_steps=3, epochs=60, verbose=2)
print("Test Score:" + str(model.evaluate(X_test, y_test)))


# In[ ]:


import keras.regularizers as regularizers
output = mobile.layers[-1].output
predictions = Dense(4, activation = 'softmax',kernel_regularizer=regularizers.l2(0.01), kernel_initializer='random_uniform')(output) #4 classes

model = Model(inputs = mobile.input, outputs = predictions)
#how many layers to train (last 23 layers)
for layer in model.layers[:-1]:
    layer.trainable = False
#compile and train the model
model.compile(optimizers.Adam(lr=.0001), loss = 'categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizers.Adam(lr=.0001), loss = [focal_loss], metrics=['accuracy'])
history2 = model.fit_generator(trainDataGen, steps_per_epoch = 5, validation_data=validationDataGen, validation_steps=3, epochs=60, verbose=2)
print("Test Score:" + str(model.evaluate(X_test, y_test)))


# In[ ]:


output = mobile.layers[-1].output
predictions = Dense(4, activation = 'softmax',kernel_regularizer=regularizers.l1(0.01), kernel_initializer='random_uniform')(output) #4 classes

model = Model(inputs = mobile.input, outputs = predictions)
#how many layers to train (last 23 layers)
for layer in model.layers[:-1]:
    layer.trainable = False

#compile and train the model
#model.compile(optimizers.Adam(lr=.0001), loss = 'categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizers.Adam(lr=.0001), loss = [focal_loss], metrics=['accuracy'])
history3 = model.fit_generator(trainDataGen, steps_per_epoch = 5, validation_data=validationDataGen, validation_steps=3, epochs=60, verbose=2)
print("Test Score:" + str(model.evaluate(X_test, y_test)))


# In[ ]:


output = mobile.layers[-1].output
predictions = Dense(4, activation = 'softmax',kernel_regularizer=regularizers.l1(0.01), kernel_initializer='random_uniform')(output) #4 classes

model = Model(inputs = mobile.input, outputs = predictions)
#how many layers to train (last 23 layers)
for layer in model.layers[:-1]:
    layer.trainable = False

#compile and train the model
model.compile(optimizers.Adam(lr=.0001), loss = 'categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizers.Adam(lr=.0001), loss = [focal_loss], metrics=['accuracy'])
history4 = model.fit_generator(trainDataGen, steps_per_epoch = 5, validation_data=validationDataGen, validation_steps=3, epochs=60, verbose=2)
print("Test Score:" + str(model.evaluate(X_test, y_test)))


# In[ ]:


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'g', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'g', label = 'Training loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
print("activation = 'softmax',kernel_regularizer=regularizers.l2(0.01), kernel_initializer='random_uniform' , loss = 'focal_loss'")


# In[ ]:




acc = history2.history['acc']
val_acc = history2.history['val_acc']
loss = history2.history['loss']
val_loss = history2.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'g', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'g', label = 'Training loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
print("activation = 'softmax',kernel_regularizer=regularizers.l2(0.01), kernel_initializer='random_uniform' , loss = 'categorical_crossentropy'")


# In[ ]:


acc = history3.history['acc']
val_acc = history3.history['val_acc']
loss = history3.history['loss']
val_loss = history3.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'g', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'g', label = 'Training loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
print("activation = 'softmax',kernel_regularizer=regularizers.l1(0.01), kernel_initializer='random_uniform' , loss = 'focal_loss'")


# In[ ]:


acc = history4.history['acc']
val_acc = history4.history['val_acc']
loss = history4.history['loss']
val_loss = history4.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'g', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
plt.title('Training and Validation accuracy ')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'g', label = 'Training loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
print("activation = 'softmax',kernel_regularizer=regularizers.l1(0.01), kernel_initializer='random_uniform' , loss = 'categorical_crossentropy'")

