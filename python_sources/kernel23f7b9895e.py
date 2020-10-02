#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
from spectral import *
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

pal = sns.color_palette()
sns.set_style("whitegrid")


# In[ ]:


#PATH = '../input/'
PATH = '../input/planet-understanding-the-amazon-from-space'


# In[ ]:


ls {PATH}


# In[ ]:


os.listdir()


# In[ ]:


train_df = pd.read_csv("../input/planet-understanding-the-amazon-from-space/train_v2.csv/train_v2.csv" )
train_df.head()


# In[ ]:


train_df.describe(include='all')


# In[ ]:


print(train_df.shape)


# In[ ]:


print(train_df.nunique())


# In[ ]:


train_df.isnull().values.any()


# In[ ]:


train_df.isnull().sum()


# In[ ]:


train_df.info()


# In[ ]:


all_tags = [item for sublist in list(train_df['tags'].apply(lambda row: row.split(" ")).values) for item in sublist]
all_tags


# In[ ]:


tags_counted_and_sorted = pd.DataFrame({'tag': all_tags}).groupby('tag').size().reset_index().sort_values(0, ascending=False)
tags_counted_and_sorted


# In[ ]:


tags_counted_and_sorted.plot(kind='bar', figsize=(12,8),x='tag',y=0)


# In[ ]:


import cv2

new_style = {'grid': False}
plt.rc('axes', **new_style)
_, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(20, 20))
i = 0
for f, l in train_df[:9].values:
    img = cv2.imread('../input/dataset/train-jpg/train-jpg/{}.jpg'.format(f))
    ax[i // 3, i % 3].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[i // 3, i % 3].set_title('{} - {}'.format(f, l))
    #ax[i // 4, i % 4].show()
    i += 1
    


# In[ ]:



    # Plot the intensities of the 3 bands
    fig, axes = plt.subplots(2,2, figsize=(7, 8))
    ax = axes.ravel()

    ax[0] = plt.subplot(2, 2, 1, adjustable='box-forced')
    ax[0].imshow(img[:,:,0], cmap='nipy_spectral')
    for i in range(2):
        ax[i+1] = plt.subplot(2, 2, i+2, sharex=ax[0], sharey=ax[0], adjustable='box-forced')
        ax[i+1].imshow(img[:,:,i+1], cmap='nipy_spectral')

    ax[0].set_title('Blue')
    ax[1].set_title('Green')
    ax[2].set_title('Red')


# In[ ]:


df = train_df


# In[ ]:


df['tag_set'] = df['tags'].map(lambda s: set(s.split(' ')))

tags = set()
for t in df['tags']:
    s = set(t.split(' '))
    tags = tags | s

tag_list = list(tags)
tag_list.sort()
tag_columns = ['tag_' + t for t in tag_list]
for t in tag_list:
    df['tag_' + t] = df['tag_set'].map(lambda x: 1 if t in x else  0)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
import matplotlib.image as mpimg

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# In[ ]:


X = df['image_name'].values
y = df[tag_columns].values

n_features = 1
n_classes = y.shape[1]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

print('We\'ve got {} feature rows and {} labels'.format(len(X_train), len(y_train)))
print('Each row has {} features'.format(n_features))
print('and we have {} classes'.format(n_classes))
assert(len(y_train) == len(X_train))
print('We use {} rows for training and {} rows for validation'.format(len(X_train), len(X_valid)))


# In[ ]:


print('Memory usage (train) kB', X_train.nbytes//(1024))
print('Memory usage (valid) kB', X_valid.nbytes//(1024))


# In[ ]:


def load_image(filename, resize=True, folder='train-jpg'):
    img = mpimg.imread('../input/dataset/train-jpg/{}/{}.jpg'.format(folder, filename))
    if resize:
        img = cv2.resize(img, (64, 64))
    return np.array(img)


# In[ ]:


samples = df.sample(32)
sample_images = [load_image(fn) for fn in samples['image_name']]
INPUT_SHAPE = sample_images[1].shape
print(INPUT_SHAPE)


# In[ ]:


# Initialising the ANN
model = Sequential()


# In[ ]:


model.add(Conv2D(48, (8, 8), strides=(2, 2), input_shape=(64,64,4), activation='elu'))
model.add(BatchNormalization())


# In[ ]:


model.add(Conv2D(64, (8, 8), strides=(2, 2), activation='elu'))
model.add(BatchNormalization())


# In[ ]:


model.add(Conv2D(96, (5, 5), strides=(2, 2), activation='elu'))
model.add(BatchNormalization())


# In[ ]:


model.add(Conv2D(96, (3, 3), activation='elu'))
model.add(BatchNormalization())


# In[ ]:


model.add(Flatten())
model.add(Dropout(0.3))


# In[ ]:


model.add(Dense(256, activation='elu'))
model.add(BatchNormalization())


# In[ ]:


model.add(Dense(64, activation='elu'))
model.add(BatchNormalization())


# In[ ]:


model.add(Dense(n_classes, activation='sigmoid'))


# In[ ]:


# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


def normalize(img):
    return img / 127.5 - 1


# In[ ]:


def preprocess(img):
    img = normalize(img)
    return img


# In[ ]:


def generator(X, y, batch_size=32):
    X_copy, y_copy = X, y
    while True:
        for i in range(0, len(X_copy), batch_size):
            X_result, y_result = [], []
            for x, y in zip(X_copy[i:i+batch_size], y_copy[i:i+batch_size]):
                rx, ry = [load_image(x)], [y]
                rx = np.array([preprocess(x) for x in rx])
                ry = np.array(ry)
                X_result.append(rx)
                y_result.append(ry)
            X_result, y_result = np.concatenate(X_result), np.concatenate(y_result)
            yield shuffle(X_result, y_result)
        X_copy, y_copy = shuffle(X_copy, y_copy)


# In[ ]:


EPOCHS = 6
BATCH = 32
PER_EPOCH = 256

X_train, y_train = X_train, y_train
X_valid, y_valid = X_valid, y_valid

filepath="weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_fbeta', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit_generator(
    generator(X_train, y_train, batch_size=BATCH),
    steps_per_epoch=PER_EPOCH,
    epochs=EPOCHS,
    validation_data=generator(X_valid, y_valid, batch_size=BATCH),
    validation_steps=len(y_valid)//(4*BATCH),
    callbacks=callbacks_list)

