#!/usr/bin/env python
# coding: utf-8

# Reference:
# 
# https://www.kaggle.com/bmarcos/image-recognition-gender-detection-inceptionv3/notebook

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


# Dataset information
# 
# Content
# 
# Overall 202,599 number of face images of various celebrities 10,177 unique identities, but names of identities are not given 40 binary attribute annotations per image 5 landmark locations
# 
# Data Files
# 
# img_align_celeba.zip: 
# All the face images, cropped and aligned
# 
# list_eval_partition.csv: 
# Recommended partitioning of images into training, validation, testing sets. Images 1-162770 are training, 162771-182637 are validation, 182638-202599 are testing
# 
# list_bbox_celeba.csv: 
# Bounding box information for each image. "x_1" and "y_1" represent the upper left point coordinate of bounding box. "width" and "height" represent the width and height of bounding box
# 
# list_landmarks_align_celeba.csv: 
# Image landmarks and their respective coordinates. There are 5 landmarks: left eye, right eye, nose, left mouth, right mouth
# 
# list_attr_celeba.csv: 
# Attribute labels for each image. There are 40 attributes. "1" represents positive while "-1" represents negative

# In[ ]:


import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Dense, Flatten, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
from keras.optimizers import SGD


# In[ ]:


from IPython.core.display import display, HTML
from PIL import Image
from io import BytesIO
import base64
plt.style.use("ggplot")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import tensorflow as tf
print(tf.__version__)


# Step 1: Data Exploration
# 
# We will be using the CelebA Dataset, which includes images of 178 x 218 px. Below is an example of how the pictures looks like.

# In[ ]:


main_folder = "../input/celeba-dataset/"
images_folder = main_folder + "img_align_celeba/img_align_celeba/"

example_pic = images_folder + "000506.jpg"

training_sample = 10000
validation_sample = 2000
test_sample = 2000
img_width = 178
img_height = 218
batch_size = 16
num_epochs = 5


# Load the attributes of every picture
# 
# File: list_attr_celeba.csv

# In[ ]:


df_attr = pd.read_csv(main_folder + 'list_attr_celeba.csv')
df_attr.set_index('image_id', inplace=True)
df_attr.replace(to_replace=-1, value=0, inplace=True)


# In[ ]:


df_attr.head(5)


# In[ ]:


df_attr.describe()


# In[ ]:


df_attr.columns


# In[ ]:


df_attr.isnull().sum()


# In[ ]:


df_attr.shape


# In[ ]:


for i,j in enumerate(df_attr.columns):
    print(i+1, j)


# In[ ]:


# load a example image

img = load_img(example_pic)
plt.grid(False)
plt.imshow(img)
df_attr.loc[example_pic.split('/')[-1]][['Smiling','Male',"Young"]]


# Distribution of the Attribute
# 
# As specified before, this Notebook is an image recognition project of the Gender. There are more Female gender than Male gender in the data set. This give us some insight about the need to balance the data in next steps.

# In[ ]:


sns.countplot(df_attr["Male"])
plt.show()


# Step 2: Split Dataset into Training, Validation and Test
# 
# The recommended partitioning of images into training, validation, testing of the data set is:
# 
# 1-162770 are training
# 162771-182637 are validation
# 182638-202599 are testing
# 
# The partition is in file list_eval_partition.csv
# 
# Due time execution, by now we will be using a reduced number of images:
# 
# Training 20000 images
# Validation 5000 images
# Test 5000 Images

# In[ ]:


df_partition = pd.read_csv(main_folder + "list_eval_partition.csv")
df_partition.head(5)


# In[ ]:


df_partition.sample(100)


# 0 =====> training
# 
# 1 =====> validation
# 
# 2 =====> testing

# In[ ]:


df_partition["partition"].value_counts().sort_index()


# Join the partition and the attributes in the same data frame

# In[ ]:


df_partition.set_index('image_id', inplace=True)
df_par_attr = df_partition.join(df_attr["Male"], how="inner")

df_par_attr.head(5)


# In[ ]:


df_par_attr.shape


# Generate Partitions (Train, Validation, Test)
# 
# Number of images need to be balanced in order to get a good performance for the model, each model will have its own folder of training, validation and test balanced data.
# 
# On this step we will create functions that will help us to create each partition.

# In[ ]:


def load_reshape_img(fname):
    img = load_img(fname)
    x = img_to_array(img)/255.
    x = x.reshape((1,)+x.shape)
    return x


# In[ ]:


def generate_df(partition, attr, num_samples):
    
    df_ = df_par_attr[(df_par_attr['partition'] == partition) 
                           & (df_par_attr[attr] == 0)].sample(int(num_samples/2))
    df_ = pd.concat([df_,
                      df_par_attr[(df_par_attr['partition'] == partition) 
                                  & (df_par_attr[attr] == 1)].sample(int(num_samples/2))])

    # for Train and Validation
    if partition != 2:
        x_ = np.array([load_reshape_img(images_folder + fname) for fname in df_.index])
        x_ = x_.reshape(x_.shape[0], 218, 178, 3)
        y_ = np_utils.to_categorical(df_[attr],2)
        
    # for Test
    else:
        x_ = []
        y_ = []

        for index, target in df_.iterrows():
            im = cv2.imread(images_folder + index)
            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (img_width, img_height)).astype(np.float32) / 255.0
            im = np.expand_dims(im, axis =0)
            x_.append(im)
            y_.append(target[attr])

    return x_, y_


# Pre-processing Images: Data Augmentation
# 
# Generates Data Augmentation for images.
# 
# Data Augmentation allows to generate images with modifications to the original ones. The model will learn from these variations (changing angle, size and position), being able to predict better never seen images that could have the same variations in position, size and position.
# 
# Let's start with an example: Data Augmentation
# 
# This is how an image will look like after data augmentation (based in the giving parameters below).

# In[ ]:


# generate image generator for data augmentation

datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# load one image and reshape

img = load_img(example_pic)
x = img_to_array(img)/255.
x = x.reshape((1,) + x.shape)

# plot 10 augmented images of the loaded image

plt.figure(figsize=(20,10))
plt.suptitle("Data augmentation", fontsize=28)

i = 0

for batch in datagen.flow(x, batch_size=1):
    plt.subplot(3,5,i+1)
    plt.grid(False)
    plt.imshow(batch.reshape(218,178, 3))
    
    if i==9:
        break
    i = i+1
    
plt.show()


# The result is a new set of images with modifications from the original one, that allows to the model to learn from these variations in order to take this kind of images during the learning process and predict better never seen images.

# In[ ]:


# build data generators

# train data

x_train, y_train = generate_df(0, "Male", training_sample)

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

train_datagen.fit(x_train)

train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)


# In[ ]:


# validation data

x_valid, y_valid = generate_df(1, "Male", validation_sample)


# With the data generator created and data for validation, we are ready to start modeling.
# 
# Build the Model - Gender Recognition
# 
# Set the Model

# In[ ]:


# import inceptionv3 model

inc_model = InceptionV3(weights="../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5", include_top=False, input_shape=(img_height,img_width,3))

print("number of layers in the model : ", len(inc_model.layers))


# The top layers (including classification) are not included. These layers will be replaced for the following layers:
# 
# adding custom layers

# In[ ]:


x = inc_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)


# In[ ]:


# creating the final model

model_ = Model(inputs=inc_model.input, outputs=predictions)

# lock initial layers to not to be trained

for layer in model_.layers[:52]:
    layer.trainable = False
    
# compile the model

model_.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


# train the model

checkpointer = ModelCheckpoint(filepath='weights.best.inc.male.hdf5', verbose=1, save_best_only=True)


# In[ ]:


hist = model_.fit_generator(train_generator, validation_data=(x_valid, y_valid), steps_per_epoch=training_sample/batch_size, epochs=num_epochs, callbacks=[checkpointer], verbose=1)


# In[ ]:


# plot loss with epochs

plt.figure(figsize=(18,4))
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='validation')
plt.legend()
plt.title('loss function')
plt.show()


# In[ ]:


# Plot accuracy through epochs
plt.figure(figsize=(18, 4))
plt.plot(hist.history['acc'], label = 'train')
plt.plot(hist.history['val_acc'], label = 'valid')
plt.legend()
plt.title('Accuracy')
plt.show()


# model evaluation

# In[ ]:


# load the best model

model_.load_weights('weights.best.inc.male.hdf5')


# In[ ]:


# test data

x_test, y_test = generate_df(2, 'Male', test_sample)

# generate predictions

model_prediction = [np.argmax(model_.predict(feature)) for feature in x_test]

# report test accuracy

test_accuracy = 100 * (np.sum(np.array(model_prediction)==y_test)/len(model_prediction))
print('model evaluation')
print("test accuracy : ", test_accuracy)
print('f1 score : ', f1_score(y_test, model_prediction))


# In[ ]:




