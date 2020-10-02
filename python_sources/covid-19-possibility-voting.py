#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation,MaxPooling2D
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
import numpy as np
import os
from keras.layers import Input

from keras.optimizers import SGD 
from keras.callbacks import LearningRateScheduler
from keras.optimizers import *
from keras.models import Model,Sequential
from keras.layers import *
from keras.activations import *
from keras.callbacks import *
import numpy as np
import pandas as pd 
from numpy import zeros, newaxis
import cv2 
import matplotlib.pyplot as plt


# In[ ]:


metaData = pd.read_csv("../input/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv")
metaData


# # Normal & Abnormal Level  metadata extraction

# In[ ]:


normalMetaData=metaData.loc[metaData['Label'] == 'Normal']
normalMetaData


# In[ ]:


PnemoniaMetaData=metaData.loc[metaData['Label'] == 'Pnemonia']
PnemoniaMetaData


# # Virus & Bacteria pnemonia data extraction

# In[ ]:


VirusPnemoniaMetaData=PnemoniaMetaData.loc[PnemoniaMetaData['Label_1_Virus_category'] == 'Virus']
VirusPnemoniaMetaData


# In[ ]:


BacteriaPnemoniaMetaData=PnemoniaMetaData.loc[PnemoniaMetaData['Label_1_Virus_category'] == 'bacteria']
BacteriaPnemoniaMetaData


# # level 1 data extraction

# In[ ]:


normalMetaDataTrain=normalMetaData.loc[normalMetaData['Dataset_type'] == 'TRAIN']
normalMetaDataTrain


# In[ ]:


normalMetaDataTest=normalMetaData.loc[normalMetaData['Dataset_type'] == 'TEST']
normalMetaDataTest


# In[ ]:


PnemoniaMetaDataTrain=PnemoniaMetaData.loc[PnemoniaMetaData['Dataset_type'] == 'TRAIN']
PnemoniaMetaDataTrain


# In[ ]:


PnemoniaMetaDataTest=PnemoniaMetaData.loc[PnemoniaMetaData['Dataset_type'] == 'TEST']
PnemoniaMetaDataTest


# In[ ]:


X_Test_level1 = []
Y_Test_level1= []
for i in range (0,390):
    path='../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/'+ str(PnemoniaMetaDataTest.iloc[i]['X_ray_image_name'])
    reshapedimage =cv2.resize(cv2.imread(path, 1), (224,224))
    X_Test_level1.append(reshapedimage)
    Y_Test_level1.append(1.0)
for i in range (0,234):
    path='../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/'+ str(normalMetaDataTest.iloc[i]['X_ray_image_name'])
    reshapedimage =cv2.resize(cv2.imread(path, 1), (224,224))
    X_Test_level1.append(reshapedimage)
    Y_Test_level1.append(0.0)


X_Test_level1=np.array(X_Test_level1)
Y_Test_level1=np.array(Y_Test_level1)

X_Train_level1 = []
Y_Train_level1= []
for i in range (0,1500):
    path='../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/'+ str(PnemoniaMetaDataTrain.iloc[i]['X_ray_image_name'])
    reshapedimage =cv2.resize(cv2.imread(path, 1), (224,224))
    X_Train_level1.append(reshapedimage)
    Y_Train_level1.append(1.0)
for i in range (0,1342):
    path='../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/'+ str(normalMetaDataTrain.iloc[i]['X_ray_image_name'])
    reshapedimage =cv2.resize(cv2.imread(path, 1), (224,224))
    X_Train_level1.append(reshapedimage)
    Y_Train_level1.append(0.0)


X_Train_level1=np.array(X_Train_level1)
Y_Train_level1=np.array(Y_Train_level1)


# # class weights intialization

# In[ ]:


from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(Y_Train_level1),
                                                 Y_Train_level1)


# # Model intialization and parameters used

# In[ ]:


InputShape=(224,224,3)
act='relu'
def model():
    inputs = Input(shape=InputShape)

# First conv block
    x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

# Second conv block
    x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

# Third conv block
    x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

# Fourth conv block
    x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.5)(x)

# Fifth conv block
    x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same',name='CAM')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.5)(x)

# FC layer
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    x = Dropout(rate=0.7)(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dropout(rate=0.3)(x)

# Output layer
    output = Dense(units=1, activation='sigmoid')(x)

# Creating model and compiling
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss=keras.losses.binary_crossentropy,
              optimizer='SGD',metrics=["accuracy"])
    checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', save_best_only=True, save_weights_only=True)
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min')

    return model


# In[ ]:





# In[ ]:


model_level1=model()

hist1=model_level1.fit(X_Train_level1, Y_Train_level1, epochs=50, batch_size=32,class_weight=class_weights)
print(model_level1.evaluate(X_Test_level1, Y_Test_level1, verbose=2))
model_level1.save_weights('model_level1.h5')


# # Level 2 classifier viral / bacterial infection

# In[ ]:


PnemoniaViralMetaDataTrain=VirusPnemoniaMetaData.loc[VirusPnemoniaMetaData['Dataset_type'] == 'TRAIN']
PnemoniaViralMetaDataTrain


# In[ ]:


PnemoniaMetaViralDataTest=VirusPnemoniaMetaData.loc[VirusPnemoniaMetaData['Dataset_type'] == 'TEST']
PnemoniaMetaViralDataTest


# In[ ]:


PnemoniaBacteriaMetaDataTrain=BacteriaPnemoniaMetaData.loc[BacteriaPnemoniaMetaData['Dataset_type'] == 'TRAIN']
PnemoniaBacteriaMetaDataTrain


# In[ ]:


PnemoniaMetaBacterialDataTest=BacteriaPnemoniaMetaData.loc[BacteriaPnemoniaMetaData['Dataset_type'] == 'TEST']
PnemoniaMetaBacterialDataTest


# # Reading data for Level 2 Classifier

# In[ ]:


X_Test_level2 = []
Y_Test_level2= []
for i in range (0,148):
    path='../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/'+ str(PnemoniaMetaViralDataTest.iloc[i]['X_ray_image_name'])
    reshapedimage =cv2.resize(cv2.imread(path, 1), (224,224))
    X_Test_level2.append(reshapedimage)
    Y_Test_level2.append(1.0)
for i in range (0,242):
    path='../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/'+ str(PnemoniaMetaBacterialDataTest.iloc[i]['X_ray_image_name'])
    reshapedimage =cv2.resize(cv2.imread(path, 1), (224,224))
    X_Test_level2.append(reshapedimage)
    Y_Test_level2.append(0.0)


X_Test_level2=np.array(X_Test_level2)
Y_Test_level2=np.array(Y_Test_level2)

X_Train_level2 = []
Y_Train_level2= []
for i in range (0,1407):
    path='../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/'+ str(PnemoniaViralMetaDataTrain.iloc[i]['X_ray_image_name'])
    reshapedimage =cv2.resize(cv2.imread(path, 1), (224,224))
    X_Train_level2.append(reshapedimage)
    Y_Train_level2.append(1.0)
for i in range (0,1500):
    path='../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/'+ str(PnemoniaBacteriaMetaDataTrain.iloc[i]['X_ray_image_name'])
    reshapedimage =cv2.resize(cv2.imread(path, 1), (224,224))
    X_Train_level2.append(reshapedimage)
    Y_Train_level2.append(0.0)


X_Train_level2=np.array(X_Train_level2)
Y_Train_level2=np.array(Y_Train_level2)


# # Class weights

# In[ ]:





# In[ ]:


class_weights2 = class_weight.compute_class_weight('balanced',
                                                 np.unique(Y_Train_level2),
                                                 Y_Train_level2)


# In[ ]:


model_level2=model()

hist2=model_level2.fit(X_Train_level2, Y_Train_level2, epochs=100, batch_size=32,class_weight=class_weights)
print(model_level2.evaluate(X_Test_level2, Y_Test_level2, verbose=2))
model_level2.save_weights('level2.h5')


# In[ ]:


model_level2=model()

model_level2.load_weights('model_level2.h5')

model_level2.evaluate(X_Test_level2, Y_Test_level2, verbose=2)


# In[ ]:


preds2 = model_level2.predict(X_Test_level2)

acc = accuracy_score(Y_Test_level2, np.round(preds2))*100
cm = confusion_matrix(Y_Test_level2, np.round(preds2))
tn, fp, fn, tp = cm.ravel()

print('CONFUSION MATRIX ------------------')
print(cm)

print('\nTEST METRICS ----------------------')
precision = tp/(tp+fp)*100
recall = tp/(tp+fn)*100
print('Accuracy: {}%'.format(acc))
print('Precision: {}%'.format(precision))
print('Recall: {}%'.format(recall))
print('F1-score: {}'.format(2*precision*recall/(precision+recall)))

print('\nTRAIN METRIC ----------------------')
print('Train acc: {}'.format(np.round((hist2.history['accuracy'][-1])*100, 2)))


# In[ ]:


model_level1.save_weights('weights_level2.h5')


# In[ ]:




