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


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import random
import gc

train1_dir = '../input/clothset/Undamaged'
train2_dir = '../input/clothset/damaged'
test_dir = '../input/clothset/test'
train_undam = ['../input/clothset/Undamaged/{}'.format(i) for i in os.listdir(train1_dir) if 'um' in i ]
train_dam = ['../input/clothset/damaged/{}'.format(i) for i in os.listdir(train2_dir) if 'dm' in i ]
test_imgs = ['../input/clothset/test/{}'.format(i) for i in os.listdir(test_dir) if 'ts' in i ]

train_imgs = train_undam[:15] + train_dam[:15]
random.shuffle(train_imgs)
del train_undam
del train_dam
gc.collect()

import matplotlib.image as mpimg
for ima in train_imgs[:10] :
    img = mpimg.imread(ima)
    img_plot = plt.imshow(img)
    print (ima)
    plt.show()


# In[ ]:


nrows = 150
ncol = 150
channels = 3


# In[ ]:


def read_and_process_image(list_of_imgs):
    X=[]
    y=[]
    
    for image in list_of_imgs :
        X.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),(nrows,ncol),interpolation=cv2.INTER_CUBIC))
        if 'um' in image:
            y.append(1)
        elif 'dm' in image:
            y.append(0)
    return X,y


# In[ ]:


X, y = read_and_process_image(train_imgs)


# In[ ]:


X[0]


# In[ ]:


y


# In[ ]:


import seaborn as sns
del train_imgs
gc.collect()

X=np.array(X)
y=np.array(y)
sns.countplot(y)
plt.title('Undamaged and damaged')


# In[ ]:


print(X.shape)
print(y.shape)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.20, random_state = 2)
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)


# In[ ]:


del X
del y
gc.collect()

ntrain=len(X_train)
nval=len(X_val)


batch_size = 32


# In[ ]:


from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator 
from keras.preprocessing.image import img_to_array,load_img


# In[ ]:


model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])


# In[ ]:


train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range = 40,
                                    width_shift_range = 0.2,
                                    height_shift_range = 0.2,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True,)
val_datagen = ImageDataGenerator(rescale=1./255)


# In[ ]:


train_gen = train_datagen.flow(X_train,y_train,batch_size=batch_size)
val_gen = val_datagen.flow(X_val,y_val,batch_size=batch_size)


# In[ ]:


history = model.fit_generator(train_gen,
                                steps_per_epoch=ntrain//2,
                                epochs = 5,
                                validation_data = val_gen,
                                 validation_steps = nval//2)


# In[ ]:


model.save_weights('model_weights.h5')
model.save('model_keras.h5')


# In[ ]:



test1_imgs = ['../input/clothset/test/ts2.jpg']
print(test1_imgs)

#X_test,y_test = read_and_process_image(test1_imgs)
X_test1=[]
for image in test1_imgs :
        X_test1.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),(nrows,ncol),interpolation=cv2.INTER_CUBIC))
x = np.array(X_test1)
test_datagen = ImageDataGenerator(rescale=1./255)
print(X_test1)


# In[ ]:


i=0
text_labels = []
plt.figure(figsize=(30,20))
for batch in test_datagen.flow(x,batch_size=1) :
    pred = model.predict(batch)
    if pred>0.5 :
        text_labels.append('damaged')
    else :
        text_labels.append('Undamaged')
    plt.subplot(2,5,i+1)
    plt.title('this is ' + text_labels[i])
    imgplot = plt.imshow(batch[0])
    i += 1
    if i%2==0:
        break
plt.show()
    


# In[ ]:




