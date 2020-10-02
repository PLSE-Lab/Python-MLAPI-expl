#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import cv2 as cv
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
import cv2 
from tqdm import tqdm

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
      #for filename in filenames:
       #print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


traindf=pd.read_csv("/kaggle/input/detect-emotions-of-your-favorite-toons/96714c94-6-Dataset/Dataset/Train.csv")
print(traindf.shape)
testdf=pd.read_csv("/kaggle/input/detect-emotions-of-your-favorite-toons/96714c94-6-Dataset/Dataset/Test.csv")
print(testdf.shape)


# In[ ]:


traindf.head()


# In[ ]:


'''
import cv2
vidcap = cv2.VideoCapture('detect-emotions-of-your-favorite-toons/96714c94-6-Dataset/Dataset/Train Tom and jerry.mp4')

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("/kaggle/input/detect-emotions-of-your-favorite-toons/frames/train_frames/"+str(count)+".jpg", image) # save frame as JPG file
    return hasFrames,image
train_images =[]
IMAGE_SIZE = (150,150)
sec = 1
frameRate = 1 #//it will capture image in each 0.5 second
count=1
success,image = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)
'''


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x='Emotion', data=traindf,
                   order=list(traindf['Emotion'].value_counts().sort_index().index) ,
                   color='cyan')


# In[ ]:


traindf.loc[traindf['Frame_ID'] == 'frame0.jpg']['Emotion']
class_names =np.unique(traindf['Emotion'])
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)
print(class_names_label)


# In[ ]:


"""
import cv2
vidcap = cv2.VideoCapture('detect-emotions-of-your-favorite-toons/96714c94-6-Dataset/Dataset/Test Tom and jerry.mp4')

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("/kaggle/input/detect-emotions-of-your-favorite-toons/frames/train_frames/"+str(count)+".jpg", image) # save frame as JPG file
    return hasFrames,image
train_images =[]
IMAGE_SIZE = (150,150)
sec = 1
frameRate = 1 #//it will capture image in each 0.5 second
count=1
success,image = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)
    
"""


# In[ ]:


IMAGE_SIZE = (256, 256)
dataset = '/kaggle/input/detect-emotions-of-your-favorite-toons/frames/train_frames'
output = []
train_images = []
train_labels = []
for files in tqdm(os.listdir(dataset)):
    try:
        label=class_names_label[traindf.loc[traindf['Frame_ID'] == files]['Emotion'].values[0]]
    except:
        #do nothing
        a=1
    img_path=os.path.join(dataset, files)
    # Open and resize the img
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE) 
    # Append the image and its corresponding label to the output
    train_images.append(image)
    train_labels.append(label)


# In[ ]:


train_images = np.array(train_images, dtype = 'float32')/255
train_labels = np.array(train_labels, dtype = 'int32') 


# In[ ]:


IMAGE_SIZE = (256, 256)
dataset = '/kaggle/input/detect-emotions-of-your-favorite-toons/frames/test_frames'
output = []
test_images = []
for files in tqdm(os.listdir(dataset)):
    img_path=os.path.join(dataset, files)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE) 
    # Append the image and its corresponding label to the output
    test_images.append(image)
test_images = np.array(test_images, dtype = 'float32')/255


# In[ ]:


plt.imshow(test_images[25])


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val=train_test_split(train_images,train_labels,test_size=0.3)


# In[ ]:


input_shape = x_train.shape[1:]
# Normalize data.
x_train = x_train.astype('float32') 
y_train = y_train.astype('float32') 
print(input_shape)


# In[ ]:


print(y_train.shape)
print(y_val.shape)


# In[ ]:


print(x_train.shape)
print(x_val.shape)


# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.callbacks import TensorBoard


# In[ ]:


classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32,(3,3), input_shape = (256,256, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Adding more layers 
classifier.add(Conv2D(32,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Adding more layers 

classifier.add(Conv2D(32,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(32,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 - Flattening
classifier.add(Dropout(0.5))
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 300, activation = 'relu'))
classifier.add(Dense(output_dim = 100, activation = 'relu'))
classifier.add(Dense(output_dim = 5, activation = 'softmax'))

# Compiling the CNN
from keras import optimizers

#o = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
classifier.compile(optimizer = Adam(lr=0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
epochs=50
classifier.fit(x_train,
             y_train,
             batch_size=32,
             nb_epoch=epochs,
             verbose=1,
             validation_data=(x_val,y_val))


# In[ ]:


classifier.summary()


# In[ ]:


import pydot
keras.utils.plot_model(classifier)


# In[ ]:


scores = classifier.evaluate(x_val, y_val, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# In[ ]:


y_test_prob = classifier.predict(test_images)
y_test_prob.round(2)


# In[ ]:


y_test_pred = classifier.predict_classes(test_images)
y_test_pred


# In[ ]:


elo = np.array(class_names)[y_test_pred]
elo


# In[ ]:


f=95
print(plt.imshow(test_images[f]),elo[f])


# In[ ]:


type(elo)


# In[ ]:


df2 = testdf.assign(Emotion = elo) 


# In[ ]:


df2.head()


# In[ ]:


df2.to_csv('submission2.csv', header=True, index=False)


# In[ ]:




