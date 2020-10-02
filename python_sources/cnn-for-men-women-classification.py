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
# for dirname, _, filenames in os.walk('../input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#the imports
import random
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Flatten, Convolution2D, MaxPooling2D, Dropout, Activation, AveragePooling2D
from keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


men = []
women = []
img_size = 300
MEN_IMGS_PATH = '../input/men-women-classification/men'
WOMEN_IMGS_PATH = '../input/men-women-classification/women'
DIRS = [(0, MEN_IMGS_PATH), (1, WOMEN_IMGS_PATH)]


# In[ ]:


train_images = []
labels = []
for num, _dir in DIRS:
    _dir = _dir + '/'
    count = 0
    for file in os.listdir(_dir):
        if count >= 1400:
            break
        img = image.load_img(_dir + str(file), target_size=(img_size, img_size))
        img = image.img_to_array(img)
        img = img/255
        train_images.append(img)
        labels.append(num)
        count += 1


# In[ ]:


train_images[1].shape


# In[ ]:


plt.imshow(train_images[1])


# In[ ]:


plt.imshow(train_images[1501])


# In[ ]:


len(train_images)


# In[ ]:


X = np.array(train_images)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=101)


# In[ ]:


len(X_train)


# In[ ]:


len(X_test)


# In[ ]:


y_train_labels = to_categorical(y_train)


# In[ ]:


def build(width, height, depth, classes):
    #initialize the model along with the input shape
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1
    
    if K.image_data_format() == 'channels_first':
        inputShape = (depth, height, width)
        chanDim = 1
        
    # CONV -> RELU -> MAXPOOL
    model.add(Convolution2D(64, (3,3), padding='same', input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))
    
    # (CONV -> RELU)*2 -> AVGPOOL
    model.add(Convolution2D(128, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Convolution2D(128, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(AveragePooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))
    
    # CONV -> RELU -> MAXPOOL
    model.add(Convolution2D(256, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))
    
    # CONV -> RELU -> AVGPOOL
    model.add(Convolution2D(512, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(AveragePooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))
    
    # DENSE -> RELU
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    # DENSE -> RELU
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    # sigmoid -> just to check the accuracy with this (softmax would work too)
    model.add(Dense(classes))
    model.add(Activation('sigmoid'))
    
    return model


# In[ ]:


model = build(img_size, img_size, 3, 2)


# In[ ]:


model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


history = model.fit(X_train, y_train_labels, batch_size=32, epochs=100, validation_split=0.2)


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# Accuracy in training is quite good, but the accuracy on the validation set is not good.
# 
# Let's see the loss in here.

# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# Clearly the validation loss doesn't looks good. We need to do some tweaking in the model.
# 
# Let's compare the predictions now.

# In[ ]:


predictions = model.predict_classes(X_test)


# In[ ]:


print(confusion_matrix(predictions, y_test))


# In[ ]:


print(classification_report(predictions, y_test))


# The classification report look good, 76% is kind of good in accuracy, but there are scopes for improvement.

# Let's try to see some predictions on random images from the testing set.

# In[ ]:


random_indices = [random.randint(0, 280) for i in range(9)]


# In[ ]:


plt.figure(figsize=(10,10))
for i, index in enumerate(random_indices):
    pred = predictions[index]
    pred = 'man' if pred==0 else 'woman'
    actual = 'man' if y_test[index]==0 else 'woman'
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[index], cmap='gray', interpolation='none')
    plt.title(f"Predicted: {pred}, \n Class: {actual}")
    plt.tight_layout()


# The results seems quite good.
# There can be improvement w.r.t model and image processing part (p.s: haven't focused much on the later).
# 
# 
# 
# Please feel free to provide constructive criticism/comments in case you didn't like any part or want me to improve things on some areas.
# 
# Also, upvote if you like the kernel.
# 
# 
# 
# 

# In[ ]:




