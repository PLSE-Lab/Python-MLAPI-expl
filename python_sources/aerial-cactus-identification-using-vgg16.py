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
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
get_ipython().system('unzip ../input/test.zip')
get_ipython().system('unzip ../input/train.zip')


# ## Data preparation

# To curate the data and make it easier for training, will follow below steps.
# 1. Read the `train.csv` file
# 2. Read the labels from `train.csv` which are the `positive` (containing cactus) and `negative` (not containing cactus) images
# 3. Create two different directories for positive and negative images
# 4. Move positive images to positive dir and negative images to negative dir
# 5. Train with `ImageDataGenerator`

# ### Making directories
# 1. `has_cactus`: Positive samples
# 2. `has_no_cactus`: Negative samples

# In[ ]:


get_ipython().system('mkdir has_cactus has_no_cactus ')


# ## Loading the `train.csv`

# In[ ]:


df = pd.read_csv('../input/train.csv')


# In[ ]:


df.head()


# In[ ]:


import shutil
images_having_cactus = []
images_having_no_cactus = []

for i in df[df['has_cactus'] == 1]['id']:
    p = os.path.join('./train/', i)
    images_having_cactus.append(p)

for i in df[df['has_cactus'] == 0]['id']:
    p = os.path.join('./train/', i)
    images_having_no_cactus.append(p)

# Copying the images actually
for i in images_having_cactus:
    shutil.copy(i, './has_cactus/')
for i in images_having_no_cactus:
    shutil.copy(i, './has_no_cactus/')


# ## Analysis

# Now, let's see how many training images are present per class

# In[ ]:


print('Has Cactus: {}'.format(df[df['has_cactus'] == 1]['id'].count()))
print('Has No Cactus: {}'.format(df[df['has_cactus'] == 0]['id'].count()))


# This means, we need `8772` more samples in training for `has_no_cactus` category.
# The solution is [Data Augumentation](https://towardsdatascience.com/data-augmentation-experimentation-3e274504f04b)

# ## Data Augumentation
# 1. We'll be adding the more images to the existing dataset
# 2. After iterating over the images under no_cactus, we'll
# * Flip the image horizontally and save
# * Flip the image vertically and save
# 3. We won't add it to the data frame since we are going to use ImageDataGenerator from keras

# In[ ]:


def augument_data(
    directory,              # Directory where augumentation is needed. Same dir will have sample images
    number_of_images_to_add # Image count to add 
):
    print('Images to add: {}'.format(number_of_images_to_add))
    import cv2
    from glob import glob
    l = glob(directory + '/*.jpg')
    for image in l:
        if number_of_images_to_add == 0:
            break
        img = cv2.imread(image)
        h_img = cv2.flip(img, 0)
        v_img = cv2.flip(img, 1)
        cv2.imwrite(directory + '/h_img_{}.jpg'.format(number_of_images_to_add), h_img)
        number_of_images_to_add -= 1
        cv2.imwrite(directory + '/v_img_{}.jpg'.format(number_of_images_to_add), v_img)
        number_of_images_to_add -= 1


# In[ ]:


augument_data('./has_no_cactus/', df[df['has_cactus'] == 1]['id'].count() - df[df['has_cactus'] == 0]['id'].count())


# * Moved the curated data to `curated_data` directory
# * Created `validation_data` and `test_data` under curated data along with `train_data`

# In[ ]:


get_ipython().system('mkdir -p curated_data/train_data curated_data/validation_data/has_cactus')
get_ipython().system('mkdir -p curated_data/validation_data/has_no_cactus')
get_ipython().system('mv has_cactus has_no_cactus curated_data/train_data')


# In[ ]:


from glob import glob
import shutil
l = glob('curated_data/train_data/has_cactus/*.jpg')
for i in range(300):
    shutil.move(l[i], 'curated_data/validation_data/has_cactus')

l = glob('curated_data/train_data/has_no_cactus/*.jpg')
for i in range(300):
    shutil.move(l[i], 'curated_data/validation_data/has_no_cactus')


# Now, the number of images in both the directories is equal.
# 
# 
# | Class  | Number of images   |
# |---|---|
# |  `has_cactus` | 13136  |
# |  `has_no_cactus` | 13136  |

# ## Creating Data Generator

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


datagen = ImageDataGenerator(
    featurewise_std_normalization=True,
    samplewise_std_normalization=True,
    horizontal_flip=True,
    vertical_flip=True
)


# In[ ]:


train_data = datagen.flow_from_directory(
    'curated_data/train_data/',
    class_mode='categorical'
)

validation_data = datagen.flow_from_directory(
    'curated_data/validation_data/',
    class_mode='categorical'
)


# ## Model Definition

# We'll try to use pre-trained model named `VGG16` with `imagenet` weights.

# In[ ]:


vgg16_model = VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(256, 256, 3)
)


# In[ ]:


vgg16_model.summary()


# In[ ]:


for layer in vgg16_model.layers[:5]:
    layer.trainable = False


# In[ ]:


x = vgg16_model.output
x = Flatten()(x)
x = Dense(1024)(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

# creating the final model 
model = Model(inputs= vgg16_model.input, outputs= predictions)

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# compile the model 
model.compile(loss = "binary_crossentropy", optimizer = SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])


# In[ ]:


model.fit_generator(
    train_data,
    epochs=20,
    validation_data=validation_data,
    callbacks=[early_stop]
)


# In[ ]:


hist = pd.DataFrame(model.history.history)
hist.plot()


# In[ ]:


import cv2
from glob import glob
test_images = glob('../input/test/test/*.jpg')
df = pd.DataFrame(columns=['id', 'has_cactus'])
df.index.name = 'id'
for img in test_images:
    i = cv2.imread(img)
    i.resize(256, 256, 3)
    pred = model.predict(i.reshape(1, 256, 256, 3))
    tempDf = pd.DataFrame({
        'id': [img.split('/')[-1]],
        'has_cactus': [pred[0][0]]
    })
    df = df.append(tempDf)


# In[ ]:


# # Removing the new directories created locally
get_ipython().system('rm -rf curated_data/')


# In[ ]:


df = df.set_index('id')
df.to_csv('submission.csv')


# ### Download `submission.csv` from here
# <a href='submission.csv'>Download Submission</a>

# In[ ]:




