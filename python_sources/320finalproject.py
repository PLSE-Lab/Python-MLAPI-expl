#!/usr/bin/env python
# coding: utf-8

# # Who Is Your Celebrity Look-a-like?
# CSCI 320 Final Project
# 
# Zara Saldanha and Jean Leong
# 
# ## Introduction
# 
# ## About the Dataset
# 
# The dataset contains 202,600 images that are 178x218 pixels each.

# In[ ]:


import matplotlib.pyplot as plt
import cv2  
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img, array_to_img
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
from keras.optimizers import SGD
from IPython.core.display import display, HTML
from PIL import Image
from io import BytesIO
import base64

import os
# Any results you write to the current directory are saved as output.w
tf.__version__


# In[ ]:


df_attr = pd.read_csv('/kaggle/input/celeba-dataset/list_attr_celeba.csv')
df_attr


# In[ ]:


for i, j in enumerate(df_attr.columns):
    print(i, j)


# In[ ]:


df_attr.replace(to_replace=-1, value=0, inplace=True)


# In[ ]:


df_attr.shape


# 40 columns of attributes
# 
# 202599 images

# In[ ]:


image_src = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/'
image_id = '121410.jpg'
img = load_img(image_src+image_id)
plt.grid(False)
plt.imshow(img)
df_attr.loc[df_attr['image_id'] == image_id]


# In[ ]:


image_id = '121123.jpg'
img = load_img(image_src+image_id)
plt.grid(False)
plt.imshow(img)
df_attr.loc[df_attr['image_id'] == image_id]


# In[ ]:


image_id = '122001.jpg'
img = load_img(image_src+image_id)
plt.grid(False)
plt.imshow(img)
df_attr.loc[df_attr['image_id'] == image_id]


# In[ ]:


image_id = '121399.jpg'
img = load_img(image_src+image_id)
plt.grid(False)
plt.imshow(img)
df_attr.loc[df_attr['image_id'] == image_id]


# In[ ]:


image_id = '114511.jpg'
img = load_img(image_src+image_id)
plt.grid(False)
plt.imshow(img)
df_attr.loc[df_attr['image_id'] == image_id]


# In[ ]:


image_id = '121420.jpg'
img = load_img(image_src+image_id)
plt.grid(False)
plt.imshow(img)
df_attr.loc[df_attr['image_id'] == image_id]


# We noticed that there is an attribute in the dataset called "Attractive". The attractiveness of a person is very subjective and is not a valid attribute. We're interested in finding out what exactly makes a person "attractive" to the creator of this dataset. We're assuming that it might just be certain attributes that have a value of 1 that contribute to "total attractiveness".

# ## Dataset Distributions

# In[ ]:


# Young or Old?
sns.set(style="darkgrid")
plt.title('Young or Old?')
sns.countplot(y='Young', data=df_attr, color="b")
plt.show()


# In[ ]:


# Attractive
plt.title('Attractive or Not')
sns.countplot(y='Attractive', data=df_attr, color="g")
plt.show()


# In[ ]:


# Gender
plt.title('Female or Male')
sns.countplot(y='Male', data=df_attr, color="c")
plt.show()


# In[ ]:


# Gender
plt.title('Black Hair')
sns.countplot(y='Black_Hair', data=df_attr, color="r")
plt.show()


# ## Aside: What Makes a Person "Attractive"?

# In[ ]:





# ## Preprocess Data
# 
# We will split the data into two sets: training and testing.
# The training set will contain 80% of the data and the testing set will have the remaining 20%.
# In total, there are 202599 observations. So, the training set will have 162080 rows and the testing set will have 40519 rows.
# 
# The dataset has a file with intended partitions, but since we want more we need to adjust some of the values for the partitions to get the 80/20 ratio.

# In[ ]:


num_train = 10000
num_val = 2000
num_test = 2000
img_x = 178
img_y = 218
BATCH_SIZE = 16
NUM_EPOCHS = 5


# In[ ]:


df_partition = pd.read_csv('/kaggle/input/celeba-dataset/list_eval_partition.csv')
df_partition.head()


# In[ ]:


df_partition['partition'].value_counts().sort_index()


# In[ ]:


df_partition.set_index('image_id', inplace=True)
df_attr.set_index('image_id', inplace=True)


# In[ ]:


df_full = df_partition.join(df_attr, how='inner')
df_full.head()


#  Now, we have the right number of training and testing rows.

# In[ ]:


img_x = 178
img_y = 218

def load_reshape_img(fname):
    img = load_img(fname)
    x = img_to_array(img)/255.
    x = x.reshape((1,) + x.shape)

    return x

def train_test_split(partition, attr, num_samples):
    
    df_ = df_full[(df_full['partition'] == partition) & (df_full[attr] == 0)]
    #.sample(int(num_samples/2))
    
    df_ = pd.concat([df_,df_full[(df_full['partition'] == partition) & (df_full[attr] == 1)].sample(int(num_samples/2),replace=True)])

    # for Train and Validation
    if partition != 2:
        print("starting !2")
        x_ = np.array([load_reshape_img(image_src + image) for image in df_.index])
        print("end x_")
        x_ = x_.reshape(x_.shape[0], 218, 178, 3)
        print("end x_ reshape")
        y_ = np_utils.to_categorical(df_[attr],2)
        print("end y_")
    # for Test
    else:
        x_ = []
        y_ = []

        for index, target in df_.iterrows():
            print("else")
            im = cv2.imread(image_src + index)
            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
            im = np.expand_dims(im, axis =0)
            x_.append(im)
            y_.append(target[attr])

    return x_, y_


# In[ ]:


df_full['partition'].value_counts()


# ## Build Model

# In[ ]:


# Generate image generator for data augmentation
datagen =  ImageDataGenerator(
  #preprocessing_function=preprocess_input,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True
)

# load one image and reshape
img = load_img(image_src+image_id)
x = img_to_array(img)/255.
x = x.reshape((1,) + x.shape)

# plot 10 augmented images of the loaded iamge
plt.figure(figsize=(20,10))
plt.suptitle('Data Augmentation', fontsize=28)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.subplot(3, 5, i+1)
    plt.grid(False)
    plt.imshow( batch.reshape(218, 178, 3))
    
    if i == 9:
        break
    i += 1
    
plt.show()


# In[ ]:


num_train = 10000
num_test = 2000
num_val = 2000
TEST_SAMPLES = 2000
BATCH_SIZE = 16
NUM_EPOCHS = 20


# In[ ]:


inception_v3 = InceptionV3(weights='imagenet',
                        include_top=False,
                        input_shape=(178, 218, 3))

inception_v3.layers


# In[ ]:


inception_v3.summary()


# In[ ]:


len(inception_v3.layers)


# In[ ]:


x = inception_v3.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)


# In[ ]:


model = Model(inputs=inception_v3.input, outputs=predictions)


# In[ ]:


for layer in model.layers[:52]:
    layer.trainable = False


# In[ ]:


model.compile(optimizer=SGD(lr=0.0001, momentum=0.9)
                    , loss='categorical_crossentropy'
                    , metrics=['accuracy'])


# ## Train Model

# In[ ]:


X_train, Y_train = train_test_split(0, 'Male', num_train)


# In[ ]:


X_valid, Y_valid = train_test_split(1, 'Male', num_val)


# In[ ]:


X_train


# In[ ]:


Y_train


# In[ ]:


# Train - Data Preparation - Data Augmentation with generators
train_datagen =  ImageDataGenerator(
  preprocessing_function=preprocess_input,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
)


# In[ ]:


train_datagen.fit(X_train)


# In[ ]:


X_train.shape


# In[ ]:


train_generator = train_datagen.flow(
X_train, Y_train,
batch_size=BATCH_SIZE,
)


# In[ ]:


checkpointer = ModelCheckpoint(filepath='weights.best.inc.hair.hdf5', 
                               verbose=1, save_best_only=True)


# In[ ]:


hist = model.fit_generator(train_generator
                     , validation_data = (X_valid, Y_valid)
                      , steps_per_epoch= TRAINING_SAMPLES/BATCH_SIZE
                      , epochs= 5
                      , callbacks=[checkpointer]
                      , verbose=1
                    )


# ## Evaluate and Test Model

# In[ ]:





# ## Using our Own Images

# ## Conclusion

# 
