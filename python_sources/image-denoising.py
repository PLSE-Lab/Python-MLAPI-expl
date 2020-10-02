#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('pip install python-resize-image')


# In[ ]:


import cv2
import os
import numpy as np
from PIL import Image
from resizeimage import resizeimage

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename == "train":
            continue
        if filename == "test":
            continue
        if filename == "train_cleaned":
            continue
        img = cv2.imread(os.path.join(folder,filename))
        img = np.array(img)
        s = img.shape
        s = np.array(s)
        if  s[0] == 258:
            img1 = Image.open(os.path.join(folder,filename))
            new1 = resizeimage.resize_contain(img1, [540, 420, 3])
            new1 = np.array(new1, dtype='uint8')
            images.append(new1)
        else:
            img1 = Image.open(os.path.join(folder,filename))
            images.append(img)
    return images

train = load_images_from_folder("../input/train")
test = load_images_from_folder("../input/test")
train_cleaned = load_images_from_folder("../input/train_cleaned")


# In[ ]:


train = np.array(train)
test = np.array(test)
train_cleaned = np.array(train_cleaned)

train = train.astype('float32') / 255
test = test.astype('float32') / 255
train_cleaned = train_cleaned.astype('float32') / 255


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding
from keras.layers import SpatialDropout1D, Conv2D, MaxPooling2D, UpSampling2D


# In[ ]:



model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(420, 540, 3,))) 
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

model.summary() 

model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=["accuracy"])


# In[ ]:


model.fit(train, train_cleaned, epochs=500, batch_size=52, shuffle=True, validation_data=(train, train_cleaned))


# In[ ]:


y_hat = model.predict_proba(test)


# In[ ]:


y_hat = np.array(y_hat)
print(y_hat.shape)

    


# In[ ]:


import matplotlib.pyplot as plt
for ima in y_hat:
    plt.figure()
    plt.imshow(ima)


# In[ ]:




