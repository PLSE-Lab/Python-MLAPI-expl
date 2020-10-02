#!/usr/bin/env python
# coding: utf-8

# <h1> Important imports </h1>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 #opencv library
import glob
import matplotlib.pyplot as plt  #plotting library
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import tensorflow
import random
from keras.callbacks import EarlyStopping
from PIL import Image
import h5py
import os
print(os.listdir("../input"))


# <h1> Setting directory paths </h1>

# In[ ]:


directory = '../input/state-farm-distracted-driver-detection/train'
test_directory = '../input/state-farm-distracted-driver-detection/test/'
random_test = '../input/driver/'
classes = ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']


# <h1> Create training / test data </h1>

# In[ ]:


training_data = []
testing_data = []


# In[ ]:


def create_training_data():
    for category in classes:
        path = os.path.join(directory,category)
        class_num = classes.index(category)
        
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            new_img = cv2.resize(img_array,(240,240))
            training_data.append([
                new_img,class_num])


# In[ ]:


def create_testing_data():        
    for img in os.listdir(test_directory):
        img_array = cv2.imread(os.path.join(test_directory,img),cv2.IMREAD_GRAYSCALE)
        new_img = cv2.resize(img_array,(240,240))
        testing_data.append([img,
            new_img])


# In[ ]:


for i in classes:
    path = os.path.join(directory,i)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_COLOR)
        plt.imshow(img_array, cmap='gray')
        plt.show()
        break
    break


# In[ ]:


create_training_data()
create_testing_data()


# <h1> Shuffling data </h1>

# In[ ]:


random.shuffle(training_data)
x = []
y = []
for features, label in training_data:
    x.append(features)
    y.append(label)


# In[ ]:


x[0].shape


# In[ ]:


y[0:20]


# <h2> Creating dummies for target </h2>

# In[ ]:


from keras.utils import np_utils
y_cat = np_utils.to_categorical(y,num_classes=10)


# In[ ]:


y_cat[0:10]


#  <h3> Reshaping the image to fit the batch size (batch count,h,w,c)</h3>

# In[ ]:


X = np.array(x).reshape(-1,240,240,1)
X[0].shape


# In[ ]:


X.shape


# <h1> Split into Train/test sets using train_test_split</h1>

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y_cat,test_size=0.3,random_state=50)


# In[ ]:


print("Shape of train images is:", X_train.shape)
print("Shape of validation images is:", X_test.shape)
print("Shape of labels is:", y_train.shape)
print("Shape of labels is:", y_test.shape)


# <h1> Creating model architecture </h1>

# In[ ]:


batch_size = 128
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,BatchNormalization


# In[ ]:


model = models.Sequential()

## CNN 1
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(240,240,1)))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization(axis = 3))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.2))

## CNN 2
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization(axis = 3))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.3))

## CNN 3
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization(axis = 3))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.5))

## CNN 3
#model.add(Conv2D(256,(5,5),activation='relu',padding='same'))
#model.add(BatchNormalization(axis = 3))
#model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
#model.add(Dropout(0.5))

## Dense & Output
model.add(Flatten())
model.add(Dense(units = 512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units = 128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))


# In[ ]:


model.summary()


# <h2> Compile and fit model </h2>

# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


callbacks = [EarlyStopping(monitor='val_acc',patience=5)]


# In[ ]:


results = model.fit(X_train,y_train,batch_size=batch_size,epochs=12,verbose=1,validation_data=(X_test,y_test),callbacks=callbacks)


# In[ ]:


test_img = np.array(testing_data[1][1]).reshape(-1,240,240,1)


# In[ ]:


preds = model.predict(test_img)
class_idx = np.argmax(preds[0])
class_output = model.output[:, class_idx]


# In[ ]:


class_idx


# In[ ]:


#model.save('Project13.h5')


# In[ ]:


#model_json = model.to_json()
#with open("C:\\Users\\sidsu\\Desktop\\Algorithms\\model1.json", "w") as json_file:
#    json_file.write(model_json)


# <h1> Save Model and weights</h1>

# In[ ]:


# serialize model to JSON
model_json = model.to_json()
with open("Modelarc.json", "w") as json_file:
    json_file.write(model_json)


# In[ ]:


model.save_weights("Modelweigh.h5")

