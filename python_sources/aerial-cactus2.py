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





# In[ ]:


import pandas as pd
sample_submission = pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv")
train = pd.read_csv("../input/aerial-cactus-identification/train.csv")


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import os
import cv2
import csv
import shutil
from glob import glob
from PIL import Image
from IPython.display import FileLink


# In[ ]:


print(os.listdir("../input/aerial-cactus-identification/"))


# In[ ]:


get_ipython().system('unzip -q ../input/aerial-cactus-identification/train.zip')


# In[ ]:


dataset = pd.read_csv("../input/aerial-cactus-identification/train.csv")
dataset.head()


# In[ ]:


grouped_dataset = dataset.groupby("has_cactus")
grouped_dataset.count()


# In[ ]:


category=dataset.has_cactus.unique()


# In[ ]:


category


# In[ ]:


os.mkdir('train/0')


# In[ ]:


os.mkdir('valid')


# In[ ]:


os.mkdir('valid/0')


# In[ ]:


os.mkdir('train/1')
os.mkdir('valid/1')


# In[ ]:


dataset.values


# In[ ]:


for rec in dataset.values:
    print(rec) 
    break


# 

# In[ ]:


for rec in dataset.values:
    img=rec[0]
    cat=rec[1]
    print(cat)
    print(img)
    break
    


# In[ ]:


for rec in dataset.values:
    img=rec[0]
    cat=rec[1]
    shutil.move('train/'+img,'train/'+str(cat))


# In[ ]:


get_ipython().system('ls train/0 | wc -l')


# In[ ]:


avalid = glob('train/0/*.jpg')
shuf = np.random.permutation(avalid)

for i in range(int(len(avalid) / 10)): shutil.move(shuf[i], 'valid/0/')


# In[ ]:


bvalid = glob('train/1/*.jpg')
shuf = np.random.permutation(bvalid)

for i in range(int(len(bvalid) / 10)): shutil.move(shuf[i], 'valid/1/')


# In[ ]:


import tensorflow as tf
import tensorflow.keras as keras
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D, Conv2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image
from keras.models import model_from_json


# In[ ]:


datagen = image.ImageDataGenerator()
trn_batches = datagen.flow_from_directory('train/', target_size = (224, 224), class_mode = 'categorical', shuffle = True, batch_size = 128)

val_batches = datagen.flow_from_directory('valid/', target_size = (224, 224), class_mode = 'categorical', shuffle = True, batch_size = 128)


# In[ ]:


get_ipython().system('wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5')


# In[ ]:


get_ipython().system('mkdir weights/')
get_ipython().system('mv vgg16_weights_tf_dim_ordering_tf_kernels.h5 weights/')


# In[ ]:


def vgg_preprocess(x):
    """
        Subtracts the mean RGB value, and transposes RGB to BGR.
        The mean RGB was computed on the image set used to train the VGG model.
        Args: 
            x: Image array (height x width x channels)
        Returns:
            Image array (height x width x transposed_channels)
    """
    vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb->bgr


# In[ ]:


model = Sequential()
model.add(Lambda(vgg_preprocess, input_shape = (224, 224, 3), output_shape = (224, 224, 3)))

model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D((2, 2), strides = (2, 2)))

model.add(Conv2D(128, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(128, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D((2, 2), strides = (2, 2)))

model.add(Conv2D(256, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(256, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(256, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D((2, 2), strides = (2, 2)))

model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D((2, 2), strides = (2, 2)))

model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D((2, 2), strides = (2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation = 'softmax'))

model.compile(optimizer = Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

fname = 'weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
model.load_weights(fname)


# In[ ]:


model.pop()
for layer in model.layers[:]: layer.trainable = False


# In[ ]:


model.add(Dense(2, activation = 'softmax'))
model.compile(optimizer = Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


batch_size = 128


# In[ ]:





# In[ ]:



model.fit_generator(trn_batches, steps_per_epoch = trn_batches.n / batch_size, epochs = 3, validation_data = val_batches, 
                    validation_steps = val_batches.n / batch_size)


# In[ ]:


model.save_weights('model_weight.h5')


# In[ ]:


df = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')
df.head()


# In[ ]:


get_ipython().system('unzip -q ../input/aerial-cactus-identification/test.zip')


# In[ ]:


get_ipython().system('mkdir test/unk/')


# In[ ]:


get_ipython().system('mv test/*.jpg test/unk/')


# In[ ]:


test = datagen.flow_from_directory('test/', target_size=(224, 224), batch_size=64)


# In[ ]:


results = model.predict_generator(test)


# In[ ]:


results


# In[ ]:


filenames = [file.split('/')[1].replace('.jpg', '') for file in test.filenames]
filenames[:2]


# In[ ]:


results_df = pd.DataFrame(data={'id': filenames, 'label': list(results[:, 1])})
results_df.head()


# In[ ]:


results_df.to_csv("submissions.csv", index=False)


# In[ ]:


from IPython.display import FileLink

FileLink('submissions.csv')


# In[ ]:




