# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:55:19 2018

@author: Diwas.Tiwari
"""

import numpy as np
import os
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
## ReadLabels and Prepare data ##
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv('../input/labels.csv')
df.head(n = 10)
df.describe()
## One Hot Encoding for the for the multiclass labels ##
encoder = OneHotEncoder()
breed_encoded = pd.get_dummies(df.breed)
breed_encoded.info()
breed_labels = np.asarray(breed_encoded)

rows = 100
cols = 100
#n_channels = 1 ## Depends on the type of imange you will consider to use ##

import cv2 as cv
## Storing all the features of train and test in an array ##

x_train = []
y_train = []
i = 0
j = 0

import glob

for filename in glob.glob('../input/train/*.jpg'):
    image = cv.imread(filename)
    label = breed_labels[i]
    x_train.append(cv.resize(image,(rows,cols)))
    y_train.append(label)
    i = i+1
    
x_train_res = np.array(x_train,np.float32)/255
x_train_res.shape
y_train_res = np.array(y_train,np.uint8)
y_train_res.shape    
#x_test_res.shape

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x_train_res, y_train_res, train_size = 0.8,
                                                 random_state = 5)    
(x_train.shape,y_train.shape)    
(x_test.shape,y_test.shape)    

## Time to Fit in our Deep Neural Network, i.e Extended_LeNET (Because of GPU constraints)##
import keras
#from keras.applications.resnet50 import ResNet50
from keras.models import Model,Sequential
from keras.layers import Dense,Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D,Conv2D
import keras.backend as k
from keras.applications.resnet50 import ResNet50

k.set_image_data_format("channels_last")

        ###REMARKS###
### Using Extended_LeNet For Classification (Due to "GPU" Constraints of the kernel we are using this network and
# the ACCURACY of the results will also be not good) ###

input_shape = (rows,cols,3)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(120, activation='softmax'))
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])

## *** Epoch set to 5 due to memory constraint of the Kernel, therefore the accuracy is also less ##
model.fit(y_train, x_train,batch_size=200,epochs=5,verbose=1,validation_data=(x_test, y_test)) 
score = model.evaluate(x_test, y_test, verbose=0)

#trainable_model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
#callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]

submission_file = pd.read_csv("../input/sample_submission.csv")
test_img = submission_file['id']

##Reshape the test data same as Tain data ##
x_test_file = []
for filename in glob.glob('../input/test/*.jpg'):
    image = cv.imread(filename)
    x_test_file.append(cv.resize(image,(rows,cols)))
x_test_res = np.array(x_test_file,np.float32)/255

##Prediction on Test File ##

res = model.predict(x_test_res)
predicted_res = pd.DataFrame(res)

##Saving the results to the Format Specified ##

colomns = breed_encoded.columns.values
predicted_res.columns = colomns
predicted_res.insert(0, 'id', submission_file['id'])
submission_resnet = predicted_res
submission_resnet.to_csv('submission_resnet.csv', index=False)
## training the Model ##
## Due to GPU comstratint on the Kaggle Platform ##
       
    
    
    
    