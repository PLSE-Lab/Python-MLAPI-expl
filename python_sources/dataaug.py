#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import glob
filelist = glob.glob('../input/diabetic-retinopathy-detection/*.jpeg')
np.size(filelist)


# In[ ]:


# don't run this cell, used for augmentation tests
#os.mkdir('/kaggle/working/preview')


# In[ ]:


# don't run this cell, used for augmentation tests
'''
img=mpimg.imread(filelist[1])
imgplot = plt.imshow(img)
plt.show()
'''


# In[ ]:


# don't run this cell, used for augmentation tests
'''
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        fill_mode = "nearest"
)

file = filelist[1]

fnm = file.replace("../input/","")
fnm = fnm.replace(".jpeg","")
fnm

img = load_img(file)  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='/kaggle/working/preview',save_prefix= fnm, save_format='jpeg'):
    i += 1
    if i > 10:
        break  # otherwise the generator would loop indefinitely
'''


# In[ ]:


# don't run this cell, used for augmentation tests
'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
aug = glob.glob('/kaggle/working/preview/*.jpeg')
for file in aug:
    img=mpimg.imread(file)
    imgplot = plt.imshow(img)
    plt.show()
'''


# In[3]:


trainLabels = pd.read_csv("../input/labels/trainLabels.csv")
print(trainLabels.head())


# In[ ]:


# don't run this cell, used for rescale check
'''
file = filelist[4]
tmp = cv2.imread(file)
tmp = np.array(tmp)
tmp = cv2.resize(tmp,(512, 512), interpolation = cv2.INTER_CUBIC)
from PIL import Image
img = Image.fromarray(tmp, 'RGB')
img.save('new.png')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread('new.png')
imgplot = plt.imshow(img)
plt.show()
'''


# In[ ]:


# all dataset
'''
import cv2
img_data = []
img_label = []
for file in filelist:
    tmp = cv2.imread(file)
    tmp = cv2.resize(tmp,(512, 512), interpolation = cv2.INTER_CUBIC)
    #tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    #img_data.append(np.array(tmp).flatten())
    img_data.append(np.array(tmp))
    #img_data.append(file)
    tmpfn = file
    ##tmpfn = tmpfn.replace("../input/bloodvessel/bloodvesselextraction/BloodVesselExtraction/","")
    tmpfn = tmpfn.replace("../input/diabetic-retinopathy-detection/","")
    tmpfn = tmpfn.replace(".jpeg","")
    img_label.append(trainLabels.loc[trainLabels.image==tmpfn, 'level'].values[0])
'''


# In[ ]:


# 0 and 2 excluded
'''
import cv2
img_data = []
img_label = []
for file in filelist:
    tmpfn = file
    ##tmpfn = tmpfn.replace("../input/bloodvessel/bloodvesselextraction/BloodVesselExtraction/","")
    tmpfn = tmpfn.replace("../input/diabetic-retinopathy-detection/","")
    tmpfn = tmpfn.replace(".jpeg","")
    label = trainLabels.loc[trainLabels.image==tmpfn, 'level'].values[0]
    if label != 0 and label != 2:
        tmp = cv2.imread(file)
        tmp = cv2.resize(tmp,(512, 512), interpolation = cv2.INTER_CUBIC)
        #tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        #img_data.append(np.array(tmp).flatten())
        img_data.append(np.array(tmp))
        if label == 1:
            img_label.append(0)
        if label == 3:
            img_label.append(1)
        if label == 4:
            img_label.append(2)
'''   


# In[4]:


# find count of each label
lab = []
for file in filelist:
    tmpfn = file
    tmpfn = tmpfn.replace("../input/diabetic-retinopathy-detection/","")
    tmpfn = tmpfn.replace(".jpeg","")
    lab.append(trainLabels.loc[trainLabels.image==tmpfn, 'level'].values[0])  
    
print(lab.count(0))
print(lab.count(1))
print(lab.count(2))
print(lab.count(3))
print(lab.count(4))


# In[5]:


# balanced dataset
import cv2
img_data = []
img_label = []
zero = 0
one = 0
two = 0
three = 0
four = 0
limit = 26
for file in filelist:
    tmpfn = file
    tmpfn = tmpfn.replace("../input/diabetic-retinopathy-detection/","")
    tmpfn = tmpfn.replace(".jpeg","")
    label = trainLabels.loc[trainLabels.image==tmpfn, 'level'].values[0]
    if label == 0 and zero < limit:
        tmp = cv2.imread(file)
        tmp = cv2.resize(tmp,(512, 512), interpolation = cv2.INTER_CUBIC)
        img_data.append(np.array(tmp))
        img_label.append(label)
        zero+=1
    elif label == 1 and one < limit:
        tmp = cv2.imread(file)
        tmp = cv2.resize(tmp,(512, 512), interpolation = cv2.INTER_CUBIC)
        img_data.append(np.array(tmp))
        img_label.append(label)
        one+=1
    elif label == 2 and two < limit:
        tmp = cv2.imread(file)
        tmp = cv2.resize(tmp,(512, 512), interpolation = cv2.INTER_CUBIC)
        img_data.append(np.array(tmp))
        img_label.append(label)
        two+=1
    elif label == 3 and three < limit:
        tmp = cv2.imread(file)
        tmp = cv2.resize(tmp,(512, 512), interpolation = cv2.INTER_CUBIC)
        img_data.append(np.array(tmp))
        img_label.append(label)
        three+=1
    elif label == 4 and four < limit:
        tmp = cv2.imread(file)
        tmp = cv2.resize(tmp,(512, 512), interpolation = cv2.INTER_CUBIC)
        img_data.append(np.array(tmp))
        img_label.append(label)
        four+=1


# In[6]:


print(len(img_data))
set(img_label)


# In[7]:


whole_data = np.array(img_data)
whole_data.shape


# In[8]:


import keras
num_classes = 5
whole_labels = keras.utils.to_categorical(img_label, num_classes)


# In[9]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(whole_data, whole_labels, test_size = 0.2, random_state = 0)


# In[10]:


import keras
from keras.layers.core import Layer
import keras.backend as K
import tensorflow as tf

from keras.models import Model
from keras.layers import Conv2D, MaxPool2D,      Dropout, Dense, Input, concatenate,          GlobalAveragePooling2D, AveragePooling2D,    Flatten

import cv2 
import numpy as np 
from keras.datasets import cifar10 
from keras import backend as K 
from keras.utils import np_utils

import math 
from keras.optimizers import SGD 
from keras.callbacks import LearningRateScheduler

def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    
    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    
    return output

kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)

input_layer = Input(shape=(512, 512, 3))

x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=64,
                     filters_3x3_reduce=96,
                     filters_3x3=128,
                     filters_5x5_reduce=16,
                     filters_5x5=32,
                     filters_pool_proj=32,
                     name='inception_3a')

x = inception_module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=128,
                     filters_3x3=192,
                     filters_5x5_reduce=32,
                     filters_5x5=96,
                     filters_pool_proj=64,
                     name='inception_3b')

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=192,
                     filters_3x3_reduce=96,
                     filters_3x3=208,
                     filters_5x5_reduce=16,
                     filters_5x5=48,
                     filters_pool_proj=64,
                     name='inception_4a')


'''
x1 = AveragePooling2D((5, 5), strides=3)(x)
x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
x1 = Flatten()(x1)
x1 = Dense(6400, activation='relu')(x1)
x1 = Dense(3200, activation='relu')(x1)
x1 = Dense(1600, activation='relu')(x1)
x1 = Dense(1024, activation='relu')(x1)
x1 = Dense(800, activation='relu')(x1)
x1 = Dense(400, activation='relu')(x1)
x1 = Dense(200, activation='relu')(x1)
x1 = Dropout(0.7)(x1)
x1 = Dense(5, activation='softmax', name='auxilliary_output_1')(x1)
'''

x = inception_module(x,
                     filters_1x1=160,
                     filters_3x3_reduce=112,
                     filters_3x3=224,
                     filters_5x5_reduce=24,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4b')

x = inception_module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=128,
                     filters_3x3=256,
                     filters_5x5_reduce=24,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4c')

x = inception_module(x,
                     filters_1x1=112,
                     filters_3x3_reduce=144,
                     filters_3x3=288,
                     filters_5x5_reduce=32,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4d')

'''
x2 = AveragePooling2D((5, 5), strides=3)(x)
x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
x2 = Flatten()(x2)
x2 = Dense(6400, activation='relu')(x2)
x2 = Dense(3200, activation='relu')(x2)
x2 = Dense(1600, activation='relu')(x2)
x2 = Dense(1024, activation='relu')(x2)
x2 = Dense(800, activation='relu')(x2)
x2 = Dense(400, activation='relu')(x2)
x2 = Dense(200, activation='relu')(x2)
x2 = Dropout(0.7)(x2)
x2 = Dense(5, activation='softmax', name='auxilliary_output_2')(x2)
'''

x = inception_module(x,
                     filters_1x1=256,
                     filters_3x3_reduce=160,
                     filters_3x3=320,
                     filters_5x5_reduce=32,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_4e')

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=256,
                     filters_3x3_reduce=160,
                     filters_3x3=320,
                     filters_5x5_reduce=32,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_5a')

x = inception_module(x,
                     filters_1x1=384,
                     filters_3x3_reduce=192,
                     filters_3x3=384,
                     filters_5x5_reduce=48,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_5b')

x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)
x = Dense(6400, activation='relu')(x)
x = Dense(3200, activation='relu')(x)
x = Dense(1600, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(800, activation='relu')(x)
x = Dense(400, activation='relu')(x)
x = Dense(200, activation='relu')(x)
x = Dropout(0.4)(x)

x = Dense(num_classes, activation='softmax', name='output')(x)

model = Model(input_layer, x, name='inception_v1')
print(model.summary())


# In[11]:


sgd = SGD(lr = 0.01, momentum=0.9, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# In[12]:


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# In[13]:


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        fill_mode = "nearest"
)

datagen.fit(x_train)

batch_size = 20
epochs = 10
steps_per_epoch = 40 #800 / 20
# Fit the model on the batches generated by datagen.flow().
model.fit_generator(datagen.flow(x_train, y_train,
                                 batch_size=batch_size),
                    epochs=epochs,
                    steps_per_epoch = 30,
                    validation_data=(x_test, y_test),
                    workers=4,
                    verbose=1)

save_dir = "kaggle/working/saved_models"
model_name = 'data_aug.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)


# In[ ]:


'''
n_model = Model(input_layer, x, name='inception_v1')
n_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
n_model.load_weights(model_path)
scores = n_model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
'''


# In[ ]:


'''
n_model.fit_generator(datagen.flow(x_train, y_train,
                                 batch_size=batch_size),
                    epochs=90,
                    steps_per_epoch = 20,
                    validation_data=(x_test, y_test),
                    workers=4,
                    verbose=1)
n_model.save(model_path)
'''


# In[14]:


scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# In[16]:


pred = model.predict(x_test,verbose=1)


# In[27]:


'''
def onehot2int(li):
    n_li = []
    for row in li:
        r = []
        for lab in row:
            if lab > 0.5:
                r.append(1)
            else:
                r.append(0)
        n_li.append(r)
        
    print(n_li)

    a = np.array(n_li)
    n_li = np.where(a==1)[1]
    n_li = n_li.tolist()
    return n_li
'''


# In[ ]:


'''
from sklearn.metrics import classification_report
print(classification_report(onehot2int(y_test),onehot2int(pred)))
'''

