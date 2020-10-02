#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image
import os
from random import shuffle
import matplotlib.pyplot as plt
import random


# In[ ]:


CLASS = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',
              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']


# In[ ]:


train_dir = '../input/train'
test_dir = '../input/test'

train_Blackgrass = ['../input/train/Black-grass/{}'.format(i) for i in os.listdir(os.path.join(train_dir, 'Black-grass'))]  #get Black-grass images
train_Charlock = ['../input/train/Charlock/{}'.format(i) for i in os.listdir(os.path.join(train_dir, 'Charlock'))]  #get Black-grass images
train_Cleavers = ['../input/train/Cleavers/{}'.format(i) for i in os.listdir(os.path.join(train_dir, 'Cleavers'))]  #get Black-grass images
train_CommonChickweed = ['../input/train/Common Chickweed/{}'.format(i) for i in os.listdir(os.path.join(train_dir, 'Common Chickweed'))]  #get Black-grass images
train_Commonwheat = ['../input/train/Common wheat/{}'.format(i) for i in os.listdir(os.path.join(train_dir, 'Common wheat'))]  #get Black-grass images
train_FatHen = ['../input/train/Fat Hen/{}'.format(i) for i in os.listdir(os.path.join(train_dir, 'Fat Hen'))]  #get Black-grass images
train_LooseSilkybent = ['../input/train/Loose Silky-bent/{}'.format(i) for i in os.listdir(os.path.join(train_dir, 'Loose Silky-bent'))]  #get Black-grass images
train_Maize = ['../input/train/Maize/{}'.format(i) for i in os.listdir(os.path.join(train_dir, 'Maize'))]  #get Black-grass images
train_ScentlessMayweed = ['../input/train/Scentless Mayweed/{}'.format(i) for i in os.listdir(os.path.join(train_dir, 'Scentless Mayweed'))]  #get Black-grass images
train_ShepherdsPurse = ['../input/train/Shepherds Purse/{}'.format(i) for i in os.listdir(os.path.join(train_dir, 'Shepherds Purse'))]  #get Black-grass images
train_SmallfloweredCranesbill = ['../input/train/Small-flowered Cranesbill/{}'.format(i) for i in os.listdir(os.path.join(train_dir, 'Small-flowered Cranesbill'))]  #get Black-grass images
train_Sugarbeet = ['../input/train/Sugar beet/{}'.format(i) for i in os.listdir(os.path.join(train_dir, 'Sugar beet'))]  #get Black-grass images

test_imgs = ['../input/test/{}'.format(i) for i in os.listdir(test_dir)] #get test images

train_imgs = train_Blackgrass + train_Charlock + train_Cleavers + train_CommonChickweed + train_Commonwheat + train_FatHen + train_LooseSilkybent + train_Maize + train_ScentlessMayweed + train_ShepherdsPurse + train_SmallfloweredCranesbill+ train_Sugarbeet
random.shuffle(train_imgs)  # shuffle it randomly

test_imgs


# In[ ]:


import cv2
from keras.utils import to_categorical

nrows = 150
ncolumns = 150
channels = 3  #change to 1 if you want to use grayscale image

def read_and_process_image(list_of_images):
    """
    Returns two arrays: 
        X is an array of resized images
        y is an array of labels
    """
    X = [] # images
    y = [] # labels
    
    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))  #Read the image
        #get the labels
        if image.split('/')[3] == 'Black-grass':
            y.append(0)
        elif image.split('/')[3] == 'Charlock':
            y.append(1)
        elif image.split('/')[3] == 'Cleavers':
            y.append(2)
        elif image.split('/')[3] == 'Common Chickweed':
            y.append(3)
        elif image.split('/')[3] == 'Common wheat':
            y.append(4)
        elif image.split('/')[3] == 'Fat Hen':
            y.append(5)
        elif image.split('/')[3] == 'Loose Silky-bent':
            y.append(6)
        elif image.split('/')[3] == 'Maize':
            y.append(7)
        elif image.split('/')[3] == 'Scentless Mayweed':
            y.append(8)
        elif image.split('/')[3] == 'Shepherds Purse':
            y.append(9)
        elif image.split('/')[3] == 'Small-flowered Cranesbill':
            y.append(10)
        elif image.split('/')[3] == 'Sugar beet':
            y.append(11)
    
    return X, y


# In[ ]:


X, y = read_and_process_image(train_imgs)


# In[ ]:


plt.figure(figsize=(20,10))
columns = 5
for i in range(columns):
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.imshow(X[i])


# In[ ]:


import seaborn as sns

#Convert list to numpy array
X = np.array(X)
y = np.array(y)

sns.countplot(y)
plt.title('Labels for seedlings')


# In[ ]:


print("Shape of train images is:", X.shape)
print("Shape of labels is:", y.shape)


# In[ ]:


#Lets split the data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)

print("Shape of train images is:", X_train.shape)
print("Shape of validation images is:", X_val.shape)
print("Shape of labels is:", y_train.shape)
print("Shape of labels is:", y_val.shape)

# convert the labels from integers to vectors
y_train = to_categorical(y_train, num_classes=12)
y_val = to_categorical(y_val, num_classes=12)
print(y_train)
print(y_val.shape)


# In[ ]:


#get the length of the train and validation data
ntrain = len(X_train)
nval = len(X_val)

#We will use a batch size of 32. Note: batch size should be a factor of 2.***4,8,16,32,64...***
batch_size = 32


# In[ ]:


from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization
from keras.optimizers import RMSprop

model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(12, activation = 'softmax'))


# In[ ]:


#Lets see our model
model.summary()


# In[ ]:


model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])


# In[ ]:


train_datagen = ImageDataGenerator(rescale=1./255,   #Scale the image between 0 and 1
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,)

val_datagen = ImageDataGenerator(rescale=1./255)  #We do not augment validation data. we only perform rescale


# In[ ]:


train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

print(ntrain)
print(nval)


# In[ ]:


history = model.fit_generator(train_generator,
                              steps_per_epoch=ntrain // batch_size,
                              epochs=1024,
                              validation_data=val_generator,
                              validation_steps=nval // batch_size)


# In[ ]:


#Save the model
model.save_weights('model_wieghts.h5')
model.save('model_keras.h5')


# In[ ]:


#lets plot the train and val curve
#get the details form the history object
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()


# In[ ]:


#Now lets predict test set
X_test, y_test = read_and_process_image(test_imgs[10:20]) #Y_test in this case will be empty.
x = np.array(X_test)
test_datagen = ImageDataGenerator(rescale=1./255)


# In[ ]:


i = 0
text_labels = []
plt.figure(figsize=(30,20))
for batch in test_datagen.flow(x, batch_size=1):
    pred = model.predict(batch)
    result = np.argmax(pred)
    if result == 0:
        text_labels.append('Black-grass')
    elif result == 1:
        text_labels.append('Charlock')
    elif result == 2:
        text_labels.append('Cleavers')
    elif result == 3:
        text_labels.append('Common Chickweed')
    elif result == 4:
        text_labels.append('Common wheat')
    elif result == 5:
        text_labels.append('Fat Hen')
    elif result == 6:
        text_labels.append('Loose Silky-bent')
    elif result == 7:
        text_labels.append('Maize')
    elif result == 8:
        text_labels.append('Scentless Mayweed')
    elif result == 9:
        text_labels.append('Shepherds Purse')
    elif result == 10:
        text_labels.append('Small-flowered Cranesbill')
    elif result == 11:
        text_labels.append('Sugar beet')
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.title('This is a ' + text_labels[i])
    imgplot = plt.imshow(batch[0])
    i += 1
    if i % 10 == 0:
        break
plt.show()


# In[ ]:


def read_and_process_test_image(list_of_images):
    """
    Returns two arrays: 
        X is an array of resized images
        y is an array of labels
    """
    X = [] # images
    y = [] # filenames
    
    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))  #Read the image
        #get the names
        y.append(image.split('/')[3])
    
    return X, y

#Now lets predict test set
X_test, y_test = read_and_process_test_image(test_imgs)
x = np.array(X_test)
test_datagen = ImageDataGenerator(rescale=1./255)


# [[](http://)](http://)

# In[ ]:


n = 0
dic = {
    'file': [],
    'species': [],
}
for batch in test_datagen.flow(x, batch_size=1, shuffle=False):
    #idx = (test_datagen.batch_index - 1) * test_datagen.batch_size
    #print(test_datagen.filenames[idx : idx + test_datagen.batch_size][0].split('/')[1])
    pred = model.predict(batch)
    result = np.argmax(pred)
    print(result)
    print(np.array(y_test)[n])
    
    if result == 0:
        dic['species'].append('Black-grass')
        dic['file'].append(np.array(y_test)[n])
    elif result == 1:
        dic['species'].append('Charlock')
        dic['file'].append(np.array(y_test)[n])
    elif result == 2:
        dic['species'].append('Cleavers')
        dic['file'].append(np.array(y_test)[n])
    elif result == 3:
        dic['species'].append('Common Chickweed')
        dic['file'].append(np.array(y_test)[n])
    elif result == 4:
        dic['species'].append('Common wheat')
        dic['file'].append(np.array(y_test)[n])
    elif result == 5:
        dic['species'].append('Fat Hen')
        dic['file'].append(np.array(y_test)[n])
    elif result == 6:
        dic['species'].append('Loose Silky-bent')
        dic['file'].append(np.array(y_test)[n])
    elif result == 7:
        dic['species'].append('Maize')
        dic['file'].append(np.array(y_test)[n])
    elif result == 8:
        dic['species'].append('Scentless Mayweed')
        dic['file'].append(np.array(y_test)[n])
    elif result == 9:
        dic['species'].append('Shepherds Purse')
        dic['file'].append(np.array(y_test)[n])
    elif result == 10:
        dic['species'].append('Small-flowered Cranesbill')
        dic['file'].append(np.array(y_test)[n])
    elif result == 11:
        dic['species'].append('Sugar beet')
        dic['file'].append(np.array(y_test)[n])
    n += 1 
    if n == len(x):
        break
        
submission = pd.DataFrame(dic)
submission.to_csv("submission.csv", index=False, header=True)

