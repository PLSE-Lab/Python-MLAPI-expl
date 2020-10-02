#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Final-Project-Submission" data-toc-modified-id="Final-Project-Submission-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Final Project Submission</a></span></li><li><span><a href="#Import-necessary-libraries" data-toc-modified-id="Import-necessary-libraries-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Import necessary libraries</a></span></li><li><span><a href="#Dataset-Prep" data-toc-modified-id="Dataset-Prep-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Dataset Prep</a></span><ul class="toc-item"><li><span><a href="#Checking-for-invalid-images" data-toc-modified-id="Checking-for-invalid-images-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Checking for invalid images</a></span></li></ul></li><li><span><a href="#Read-in-Data" data-toc-modified-id="Read-in-Data-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Read in Data</a></span><ul class="toc-item"><li><span><a href="#What-is-the-distribution-across-the-categories?" data-toc-modified-id="What-is-the-distribution-across-the-categories?-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>What is the distribution across the categories?</a></span></li><li><span><a href="#Calculate-number-of-images-in-train,-test-and-validation" data-toc-modified-id="Calculate-number-of-images-in-train,-test-and-validation-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Calculate number of images in train, test and validation</a></span></li></ul></li><li><span><a href="#Preprocessing" data-toc-modified-id="Preprocessing-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Preprocessing</a></span><ul class="toc-item"><li><span><a href="#Create-keras-model" data-toc-modified-id="Create-keras-model-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Create keras model</a></span></li><li><span><a href="#Save-model" data-toc-modified-id="Save-model-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Save model</a></span></li><li><span><a href="#Visualize-training-history" data-toc-modified-id="Visualize-training-history-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Visualize training history</a></span></li><li><span><a href="#Evalute-test-data" data-toc-modified-id="Evalute-test-data-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>Evalute test data</a></span></li></ul></li><li><span><a href="#Pre-Trained-Network-Part-1" data-toc-modified-id="Pre-Trained-Network-Part-1-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Pre-Trained Network Part 1</a></span><ul class="toc-item"><li><span><a href="#Fine-tuning-the-network" data-toc-modified-id="Fine-tuning-the-network-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Fine-tuning the network</a></span></li><li><span><a href="#Save-model" data-toc-modified-id="Save-model-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Save model</a></span></li><li><span><a href="#Visualize-training-history" data-toc-modified-id="Visualize-training-history-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>Visualize training history</a></span></li><li><span><a href="#Evaluate-test-data" data-toc-modified-id="Evaluate-test-data-6.4"><span class="toc-item-num">6.4&nbsp;&nbsp;</span>Evaluate test data</a></span></li></ul></li><li><span><a href="#Pre-Trained-Network-Part-2-(Experimental)" data-toc-modified-id="Pre-Trained-Network-Part-2-(Experimental)-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Pre-Trained Network Part 2 (Experimental)</a></span><ul class="toc-item"><li><span><a href="#Visualize-training-history" data-toc-modified-id="Visualize-training-history-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Visualize training history</a></span></li><li><span><a href="#Evaluate-test-data" data-toc-modified-id="Evaluate-test-data-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>Evaluate test data</a></span></li><li><span><a href="#TODO" data-toc-modified-id="TODO-7.3"><span class="toc-item-num">7.3&nbsp;&nbsp;</span>TODO</a></span></li></ul></li><li><span><a href="#train_test_split-approach" data-toc-modified-id="train_test_split-approach-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>train_test_split approach</a></span><ul class="toc-item"><li><span><a href="#Creating-a-tuple-of-all-images-and-its-category" data-toc-modified-id="Creating-a-tuple-of-all-images-and-its-category-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>Creating a tuple of all images and its category</a></span></li><li><span><a href="#Process-data" data-toc-modified-id="Process-data-8.2"><span class="toc-item-num">8.2&nbsp;&nbsp;</span>Process data</a></span></li><li><span><a href="#Build-Sequential-Model-and-fit" data-toc-modified-id="Build-Sequential-Model-and-fit-8.3"><span class="toc-item-num">8.3&nbsp;&nbsp;</span>Build Sequential Model and fit</a></span></li><li><span><a href="#Visualize-Loss/Accuracy" data-toc-modified-id="Visualize-Loss/Accuracy-8.4"><span class="toc-item-num">8.4&nbsp;&nbsp;</span>Visualize Loss/Accuracy</a></span></li><li><span><a href="#Display-confusion-matrix" data-toc-modified-id="Display-confusion-matrix-8.5"><span class="toc-item-num">8.5&nbsp;&nbsp;</span>Display confusion matrix</a></span></li></ul></li><li><span><a href="#flow_from_dataframe-approach" data-toc-modified-id="flow_from_dataframe-approach-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>flow_from_dataframe approach</a></span><ul class="toc-item"><li><span><a href="#Create-dataframes-from-directories" data-toc-modified-id="Create-dataframes-from-directories-9.1"><span class="toc-item-num">9.1&nbsp;&nbsp;</span>Create dataframes from directories</a></span></li><li><span><a href="#Create-generators" data-toc-modified-id="Create-generators-9.2"><span class="toc-item-num">9.2&nbsp;&nbsp;</span>Create generators</a></span></li><li><span><a href="#Build-Sequential-Model-and-fit" data-toc-modified-id="Build-Sequential-Model-and-fit-9.3"><span class="toc-item-num">9.3&nbsp;&nbsp;</span>Build Sequential Model and fit</a></span></li><li><span><a href="#Predict-and-save-results-in-csv-file" data-toc-modified-id="Predict-and-save-results-in-csv-file-9.4"><span class="toc-item-num">9.4&nbsp;&nbsp;</span>Predict and save results in csv file</a></span></li><li><span><a href="#Multi-label-classification-with-a-Multi-Output-Model" data-toc-modified-id="Multi-label-classification-with-a-Multi-Output-Model-9.5"><span class="toc-item-num">9.5&nbsp;&nbsp;</span>Multi-label classification with a Multi-Output Model</a></span></li></ul></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Conclusion</a></span></li></ul></div>

# Blog post URL: https://stephanosterburg.github.io/deep_learning_art_images/

# # Import necessary libraries

# In[1]:


import sys, os, shutil, glob

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

# from tqdm import tqdm
from itertools import chain

# Printing models in ipynb
from PIL import Image
from scipy import ndimage
from keras.utils.vis_utils import model_to_dot, plot_model
from IPython.display import SVG

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize

import tensorflow as tf

import keras
from keras import regularizers, optimizers
from keras.applications import VGG16, VGG19
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array
from keras.preprocessing.image import load_img
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

import warnings
warnings.filterwarnings('ignore')
# OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

np.random.seed(42)


# In[2]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# # Dataset Prep

# The dataset contains two directories which can be compined. Also, there are some bad images (invalid) which we need to delete. While we are at it, we will re-orginize the directory structure a little bit and rename the images. This is just for us and not necessarily needed for keras. 
# 
# This is a one off and can be remove from the notebook. For the time being I will leave it in here for documentation. But shouldn't be executed again, otherwise we will just see errors.

# In[6]:


get_ipython().run_line_magic('ls', '../input/dataset/dataset_updated')


# In[7]:


get_ipython().run_line_magic('ls', '../input/dataset/dataset_updated/training_set/drawings/ | head -5')


# In[8]:


get_ipython().run_line_magic('ls', '../input/musemart/dataset_updated/training_set/drawings/ | head -5')


# In[9]:


imshow('../input/musemart/dataset_updated/training_set/drawings/1677_mainfoto_05.jpg');


# In[10]:


imshow('../input/dataset/dataset_updated/training_set/drawings/1677_mainfoto_05.jpg');


# Looks like that we have duplicates in dataset and musemart.

# In[11]:


src_dirs_0 = ['dataset', 'musemart']
src_dirs_1 = ['training_set', 'validation_set']
src_dirs_2 = ['sculpture', 'iconography', 'engraving', 'drawings', 'painting']


# In[ ]:


# copying files from musemart to dataset (merge data)
for sub_dir in src_dirs_1:
    for d in src_dirs_2:
        src_dir = src_dirs_0[1] + '/' + sub_dir + '/' + d
        files = os.listdir(src_dir)
        
        dst_dir = src_dirs_0[0] + '/' + sub_dir + '/' + d
        
        for file in files:
            shutil.copy(os.path.join(src_dir, file), os.path.join(dst_dir, file))

# Rename files to to something more simple
for sub_dir in src_dirs_1:
    for d in src_dirs_2:
        src_dir = src_dirs_0[0] + '/' + sub_dir + '/' + d
        files = os.listdir(src_dir)

        print('-' * 50)
        print('Renaming files in ' + src_dir + '\n')

        for i, file in enumerate(files):
            new_name = 'image.' + str(i) + '.jpg'
            os.rename(os.path.join(src_dir, file), os.path.join(src_dir, new_name))
#             print(os.path.join(src_dir, new_name))# How many images do we have
for sub_dir in src_dirs_1:
    for d in src_dirs_2:
        src_dir = src_dirs_0[0] + '/' + sub_dir + '/' + d
        files = os.listdir(src_dir)
        print("Number of images in {}: {}".format(src_dir, len(files)))# Create test set directory
os.mkdir('dataset/test/')
         
for d in src_dirs_2:
    os.mkdir('dataset/test/' + d)# Rename directories
shutil.move('dataset/training_set/', 'dataset/train')
shutil.move('dataset/validation_set/', 'dataset/validation')%ls dataset/*# Moving ~10% of the train data over to test
import math

for d in src_dirs_2:
    src_dir = 'dataset/train/' + d
    num = len(os.listdir(src_dir)) - math.floor(len(os.listdir(src_dir))*0.1)    
    images = [file for file in os.listdir(src_dir) if file.endswith('.jpg')]
    
    dst_dir = 'dataset/test/' + d
    test_img = images[num:]

    for file in test_img:
        shutil.copy(os.path.join(src_dir, file), os.path.join(dst_dir, file))
# ## Checking for invalid images
img_width, img_height = 128, 128
input_shape = (img_height, img_width, 1)

categories = ['drawings', 'engraving', 'iconography' ,'painting' ,'sculpture']

train_path = 'dataset/train/'
val_path = 'dataset/validation/'
test_path = 'dataset/test/'count = 0

for path in [train_path, test_path, val_path]:
    for i, cat in enumerate(categories):
        cat_path = os.path.join(path, cat)
        images = [file for file in os.listdir(cat_path)]

        for image in images:
            try:
                img = Image.open(os.path.join(cat_path, image))
            except:
                count += 1

    print("Total bad images in {}: {}".format(path, str(count)))
# Assumging that we have in train, test and validation bad images, we shall remove them.
count = 0

for path in [train_path, test_path, val_path]:
    for i, cat in enumerate(categories):
        cat_path = os.path.join(path, cat)
        images = [file for file in os.listdir(cat_path)]

        for image in images:
            try:
                img = Image.open(os.path.join(cat_path, image))
            except:
                os.remove(os.path.join(cat_path, image))
                count += 1
                
    print("Removed {} bad images from {}".format(str(count), path))
# # Read in Data

# Let's have a look at some of the images

# In[ ]:


img_width, img_height = 150, 150

categories = ['drawings', 'engraving', 'iconography' ,'painting' ,'sculpture']

train_path = 'dataset/train/'
valid_path = 'dataset/validation/'
test_path  = 'dataset/test/'


# In[ ]:


def show_images_for_art(art_type="drawings", num_pics=10):
    assert art_type in categories
    
    pic_dir = os.path.join(train_path, art_type)
    pic_files = [os.path.join(pic_dir, filename) for filename in os.listdir(pic_dir)]

    ncols = 5
    nrows = (num_pics - 1) // ncols + 1
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4))
    
    fig.set_size_inches((20, nrows * 5))
    ax = ax.ravel()
    
    for pic, ax in zip(pic_files[:num_pics], ax):
        img = imread(pic)
        ax.imshow(img, resample=True)
    
    plt.show();
    
show_images_for_art(art_type="drawings")


# In[ ]:


# Just have a look at the categories itself, one image shall be ok

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(16, 4))

for i, cat in enumerate(categories):
    cat_path = os.path.join(train_path, cat)
    img_name = os.listdir(cat_path)[0]
    
    img = imread(os.path.join(cat_path, img_name))
    img = resize(img, (img_width, img_height, 3), mode='reflect')
    
    ax[i].imshow(img, resample=True)
    ax[i].set_title(cat)
    
plt.show();


# ## What is the distribution across the categories?

# In[ ]:


n_imgs = []
for cat in categories:
    files = os.listdir(os.path.join(train_path, cat))
    n_imgs += [len(files)]

plt.figure(figsize=(16, 8))
plt.bar([_ for _ in range(5)], n_imgs, tick_label=categories)
plt.show();


# ## Calculate number of images in train, test and validation

# In[ ]:


num_train_sample = 0
for i, cat in enumerate(categories):
    cat_path = os.path.join(train_path, cat)
    num_train_sample += len(os.listdir(cat_path))
    
print('Total number of training samples: {}'.format(num_train_sample))


# In[ ]:


num_test_sample = 0
for i, cat in enumerate(categories):
    cat_path = os.path.join(test_path, cat)
    num_test_sample += len(os.listdir(cat_path))
    
print('Total number of test samples: {}'.format(num_test_sample))


# In[ ]:


num_validation_sample = 0
for i, cat in enumerate(categories):
    cat_path = os.path.join(valid_path, cat)
    num_validation_sample += len(os.listdir(cat_path))
    
print('Total number of validation samples: {}'.format(num_validation_sample))


# # Preprocessing

# Let's prepare the data using `flow_from_directory` to generate batches of image data (and labels) and resize all images to 128x128.
# 
# **Question**: should we preprocess the images for the pre-trained VGG network?

# In[ ]:


nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16


# In[ ]:


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

valid_generator = datagen.flow_from_directory(
        valid_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = datagen.flow_from_directory(
        test_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')


# ## Create keras model

# In[ ]:


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# In[ ]:


model = Sequential([
    Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(5, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


SVG(model_to_dot(model, show_layer_names=False, show_shapes=True).create(prog='dot', format='svg'))


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntrain_result = model.fit_generator(\n            train_generator,\n            steps_per_epoch=num_train_sample // batch_size,\n            epochs=epochs,\n            validation_data=valid_generator,\n            validation_steps=num_validation_sample // batch_size,\n            use_multiprocessing=True)')


# ## Save model

# In[ ]:


model.save('CNN_base_run1.h5')


# ## Visualize training history
# 
# Let's display Loss and Accuravy

# In[ ]:


sns.set(style="white", palette="muted", color_codes=True)

fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

ax[0].plot(train_result.history['loss'], label="Loss")
ax[0].plot(train_result.history['val_loss'], label="Validation loss")
ax[0].set_title('Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()

ax[1].plot(train_result.history['acc'], label="Accuracy")
ax[1].plot(train_result.history['val_acc'], label="Validation accuracy")
ax[1].set_title('Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].legend()
plt.tight_layout()

plt.show();


# ## Evalute test data

# In[ ]:


test_loss, test_acc = model.evaluate_generator(test_generator, steps=32)
y_hat_test = model.predict_generator(test_generator, steps=32)

print('Generated {} predictions'.format(len(y_hat_test)))
print('Test accuracy: {:.2f}%'.format(test_acc * 100))


# # Pre-Trained Network Part 1
# 
# We can leverage a pre-trained network like the VGG19 architecture, which is pre-trained on the ImageNet dataset. Even so the ImageNet dataset contains only "cats" and "dogs" it can be used for a more generalized problem like the Art Images.
# 
# Here we will only instantiate the convolutional part of the model, everything up to the fully-connected layers. In our case we will freeze the layers of the VGG19 model. And only fine-tune the added layers.
# 
# To further improve the model we could make the last five nodes trainable. But that is for another time.

# In[ ]:


# Load the VGG19 network
vgg_model = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
vgg_model.summary()


# ## Fine-tuning the network 

# In[ ]:


model = Sequential([
    vgg_model,
    Flatten(),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax')
])

vgg_model.trainable = False

# Check what layers are trainable
for layer in model.layers:
    print(layer.name, layer.trainable)
    
# model.summary()


# In[ ]:


SVG(model_to_dot(model, show_layer_names=False, show_shapes=True).create(prog='dot', format='svg'))


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Compilation\nmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])\n\n# Fitting the Model\ntrain_result = model.fit_generator(\n            train_generator,\n            steps_per_epoch=num_train_sample // batch_size,\n            epochs=epochs,\n            validation_data=valid_generator,\n            validation_steps=num_validation_sample // batch_size,\n            use_multiprocessing=True)")


# ## Save model

# In[ ]:


model.save('VGG19_Feature_Engineered.h5')


# ## Visualize training history

# In[ ]:


sns.set(style="white", palette="muted", color_codes=True)

fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

ax[0].plot(train_result.history['loss'], label="Loss")
ax[0].plot(train_result.history['val_loss'], label="Validation loss")
ax[0].set_title('Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()

ax[1].plot(train_result.history['acc'], label="Accuracy")
ax[1].plot(train_result.history['val_acc'], label="Validation accuracy")
ax[1].set_title('Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].legend()
plt.tight_layout()

plt.show();


# ## Evaluate test data

# In[ ]:


test_loss, test_acc = model.evaluate_generator(test_generator, steps=32, use_multiprocessing=True)
y_hat_test = model.predict_generator(test_generator, steps=32, use_multiprocessing=True)

print('Generated {} predictions'.format(len(y_hat_test)))
print('Test accuracy: {:.2f}%'.format(test_acc * 100))


# # Pre-Trained Network Part 2 (Experimental)

# As we did above we will use the pre-trained VGG19 network. Only this time we will run the model on our training and validation data once and record the output in two numpy arrays. Then we will train a small fully-connected model on top of the stored features.
# 
# The reason why we are storing the features offline rather than adding our fully-connected model directly on top of a frozen convolutional base and running the whole thing, is computational effiency. Running VGG16 is expensive, especially if you're working on CPU, and we want to only do it once. 
# 
# Note that this prevents us from using data augmentation.

# In[ ]:


# Load the VGG19 network
vgg_model = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
vgg_model.summary()


# We will set the `class_mode` to `None`. The generator will only yield batches of image data, which is useful to use with `model.predict_generator()`. This means that the generator will only have batches of data and no labels.

# In[ ]:


get_ipython().run_cell_magic('time', '', "datagen = ImageDataGenerator(rescale=1. / 255)\n\ngenerator = datagen.flow_from_directory(\n    train_path,\n    target_size=(img_width, img_height),\n    batch_size=batch_size,\n    class_mode='categorical',\n    shuffle=False)\n\nbottleneck_features_train = vgg_model.predict_generator(\n    generator, 500, use_multiprocessing=True)\n\n# Save the output as a numpy array\nnp.save(open('bottleneck_features_train.npy', 'wb'),\n        bottleneck_features_train)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "generator = datagen.flow_from_directory(\n    valid_path,\n    target_size=(img_width, img_height),\n    batch_size=batch_size,\n    class_mode='categorical',\n    shuffle=False)\n\nbottleneck_features_validation = vgg_model.predict_generator(\n    generator, 60, use_multiprocessing=True, verbose=1)\n\n# Save the output as a numpy array\nnp.save(open('bottleneck_features_validation.npy', 'wb'), \n        bottleneck_features_validation)")


# In[ ]:


bottleneck_features_train.shape, bottleneck_features_validation.shape


# In[ ]:


train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
# train_labels = np.array([0] * 4000 + [1] * 4000)
a = np.zeros(shape=(8000, 3))
b = np.ones(shape=(8000, 2))
train_labels = np.concatenate((a, b), axis=1)

validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
# validation_labels = np.array([0] * 480 + [1] * 480)
a = np.zeros(shape=(960, 3))
b = np.ones(shape=(960, 2))
validation_labels = np.concatenate((a, b), axis=1)


# In[ ]:


train_data.shape, train_labels.shape


# In[ ]:


model = Sequential([
    Flatten(input_shape=train_data.shape[1:]),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(5, activation='softmax')
])

SVG(model_to_dot(model, show_layer_names=False, show_shapes=True).create(prog='dot', format='svg'))
# In[ ]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_result = model.fit(train_data, train_labels,\n                         epochs=epochs,\n                         batch_size=batch_size,\n                         validation_data=(validation_data, validation_labels)\n                        )')


# In[ ]:


model.save_weights('bottleneck_fc_model.h5')


# ## Visualize training history

# In[ ]:


sns.set(style="white", palette="muted", color_codes=True)

fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

ax[0].plot(train_result.history['loss'], label="Loss")
ax[0].plot(train_result.history['val_loss'], label="Validation loss")
ax[0].set_title('Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()

ax[1].plot(train_result.history['acc'], label="Accuracy")
ax[1].plot(train_result.history['val_acc'], label="Validation accuracy")
ax[1].set_title('Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].legend()
plt.tight_layout()

plt.show();


# ## Evaluate test data
test_loss, test_acc = model.evaluate_generator(test_generator, steps=32)
y_hat_test = model.predict_generator(test_generator, steps=32)

print('Generated {} predictions'.format(len(y_hat_test)))
print('Test accuracy: {:.2f}%'.format(test_acc * 100))
# ## TODO
# 
# * Is the shape for the created label arrays correct? Doesn't look like it. Investigate.
# * Evaluate thoughts errors.

# # train_test_split approach

# In[ ]:


img_width, img_height = 128, 128
input_shape = (img_height, img_width, 3)

categories = ['drawings', 'engraving', 'iconography' ,'painting' ,'sculpture']

train_path = 'dataset/train/'
valid_path = 'dataset/validation/'
test_path = 'dataset/test/'


# In[ ]:


category_embeddings = {
    'drawings': 0,
    'engraving': 1,
    'iconography': 2,
    'painting': 3,
    'sculpture': 4
}


# ## Creating a tuple of all images and its category

# In[ ]:


train_data = [(file, cat) for cat in categories for file in glob.glob(train_path + cat + '/*')]
test_data = [(file, cat) for cat in categories for file in glob.glob(train_path + cat + '/*')]


# In[ ]:


train_data[:5]


# ## Process data
# 
# Read images and resize into memory

# In[ ]:


def load_dataset(tuples_list):
    indexes = np.arange(len(tuples_list))
    np.random.shuffle(indexes)
    
    X = []
    y = []
    
    cpt = 0
    for i in range(len(indexes)):
        t = tuples_list[indexes[i]]
        try:
            # skimage
            img = imread(t[0])
            img = resize(img, input_shape, mode='reflect')
            X += [img]
            
            y_tmp = [0 for _ in range(len(categories))]
            y_tmp[category_embeddings[t[1]]] = 1
            y += [y_tmp]
        except OSError:
            pass
        
        cpt += 1
        if cpt % 1000 == 0:
            print("Processed {} images".format(cpt))

    return np.array(X), np.array(y)


# In[ ]:


X_train, y_train = load_dataset(train_data)
X_valid, y_valid = load_dataset(test_data)


# ## Build Sequential Model and fit

# In[ ]:


train_datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.15, horizontal_flip=True)
train_datagen.fit(X_train)


# In[ ]:


# model = Sequential([
#     Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'),
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
    
#     Dropout(0.25),
#     Conv2D(64, (3, 3), padding='same', activation='relu'),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
    
#     Dropout(0.25),
#     Flatten(),
#     Dense(256, activation='relu'),
#     Dropout(0.5),
#     Dense(5, activation='sigmoid')
# ])
model = Sequential([
    Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
train_result = model.fit_generator(generator=train_generator, validation_data=(X_valid, y_valid),
                                  epochs=50, steps_per_epoch=len(X_train)/32, verbose=1, 
                                  use_multiprocessing=True)


# ## Visualize Loss/Accuracy

# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

ax[0].plot(train_result.history['loss'], label="Loss")
ax[0].plot(train_result.history['val_loss'], label="Validation loss")
ax[0].set_title('Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()

ax[1].plot(train_result.history['acc'], label="Accuracy")
ax[1].plot(train_result.history['val_acc'], label="Validation accuracy")
ax[1].set_title('Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].legend()

plt.tight_layout()
plt.show();


# In[ ]:


# Let's look at more metrics
from sklearn.metrics import classification_report

X_test = []
y_test = []
for t in test_data:
    try:
        img = skimage.io.imread(os.path.join(t[0]))
        img = skimage.transform.resize(img, input_shape, mode='reflect')
        X_test += [img]
        y_test += [category_embeddings[t[1]]]
    except OSError:
        pass

X_test = np.array(X_test)
y_test = np.array(y_test)

pred = model.predict(X_test, verbose=1)

y_pred = np.argmax(pred, axis=1)
print(classification_report(y_test, y_pred))


# ## Display confusion matrix

# In[ ]:


from sklearn.metrics import confusion_matrix

c_matrix = confusion_matrix(y_test, y_pred)
plt.imshow(c_matrix, cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
plt.show();

print(c_matrix)


# **Note**: The first two categories seem to be close together in style and therefore have some overlap.

# # flow_from_dataframe approach

# In[ ]:


categories = os.listdir("dataset/train/")
categories


# ## Create dataframes from directories

# In[ ]:


dataset = pd.DataFrame(columns=[categories])
dataset.insert(loc=0, column='filename', value=None)
dataset.head()


# In[ ]:


df1 = dataset.copy()
df2 = dataset.copy()
df3 = dataset.copy()
df4 = dataset.copy()
df5 = dataset.copy()
df5.head()


# In[ ]:


myList = [file.split("/", 1)[1] for file in glob.glob('dataset/train/sculpture/*')]
dataset = pd.DataFrame(data=myList, columns=['filename'])
dataset[categories] = pd.DataFrame([[1, 0, 0, 0, 0]], index=dataset.index)

myList = [file.split("/", 1)[1] for file in glob.glob('dataset/train/iconography/*')]
df2 = pd.DataFrame(data=myList, columns=['filename'])
df2[categories] = pd.DataFrame([[0, 1, 0, 0, 0]], index=df2.index)
dataset = dataset.append(df2)

myList = [file.split("/", 1)[1] for file in glob.glob('dataset/train/engraving/*')]
df3 = pd.DataFrame(data=myList, columns=['filename'])
df3[categories] = pd.DataFrame([[0, 0, 1, 0, 0]], index=df3.index)
dataset = dataset.append(df3)

myList = [file.split("/", 1)[1] for file in glob.glob('dataset/train/drawings/*')]
df4 = pd.DataFrame(data=myList, columns=['filename'])
df4[categories] = pd.DataFrame([[0, 0, 0, 1, 0]], index=df4.index)
dataset = dataset.append(df3)

myList = [file.split("/", 1)[1] for file in glob.glob('dataset/train/painting/*')]
df5 = pd.DataFrame(data=myList, columns=['filename'])
df5[categories] = pd.DataFrame([[0, 0, 0, 0, 1]], index=df5.index)
dataset = dataset.append(df5)


# In[ ]:


dataset.shape


# In[ ]:


dataset.reset_index(drop=True, inplace=True)


# In[ ]:


dataset.tail()


# In[ ]:


valid_dataset = pd.DataFrame(columns=[categories])
valid_dataset.insert(loc=0, column='filename', value=None)
valid_dataset.head()


# In[ ]:


del df1, df2, df3, df4, df5


# In[ ]:


df1 = valid_dataset.copy()
df2 = valid_dataset.copy()
df3 = valid_dataset.copy()
df4 = valid_dataset.copy()
df5 = valid_dataset.copy()
df5.head()


# In[ ]:


myList = [file.split("/", 1)[1] for file in glob.glob('dataset/validation/sculpture/*')]
valid_dataset = pd.DataFrame(data=myList, columns=['filename'])
valid_dataset[categories] = pd.DataFrame([[1, 0, 0, 0, 0]], index=dataset.index)

myList = [file.split("/", 1)[1] for file in glob.glob('dataset/validation/iconography/*')]
df2 = pd.DataFrame(data=myList, columns=['filename'])
df2[categories] = pd.DataFrame([[0, 1, 0, 0, 0]], index=df2.index)
valid_dataset = valid_dataset.append(df2)

myList = [file.split("/", 1)[1] for file in glob.glob('dataset/validation/engraving/*')]
df3 = pd.DataFrame(data=myList, columns=['filename'])
df3[categories] = pd.DataFrame([[0, 0, 1, 0, 0]], index=df3.index)
valid_dataset = valid_dataset.append(df3)

myList = [file.split("/", 1)[1] for file in glob.glob('dataset/validation/drawings/*')]
df4 = pd.DataFrame(data=myList, columns=['filename'])
df4[categories] = pd.DataFrame([[0, 0, 0, 1, 0]], index=df4.index)
valid_dataset = valid_dataset.append(df3)

myList = [file.split("/", 1)[1] for file in glob.glob('dataset/validation/painting/*')]
df5 = pd.DataFrame(data=myList, columns=['filename'])
df5[categories] = pd.DataFrame([[0, 0, 0, 0, 1]], index=df5.index)
valid_dataset = valid_dataset.append(df5)


# In[ ]:


valid_dataset.reset_index(drop=True, inplace=True)
valid_dataset.tail()


# In[ ]:


valid_dataset.shape


# In[ ]:


test_dataset = pd.DataFrame(columns=[categories])
test_dataset.insert(loc=0, column='filename', value=None)
test_dataset.head()


# In[ ]:


del df1, df2, df3, df4, df5

df1 = test_dataset.copy()
df2 = test_dataset.copy()
df3 = test_dataset.copy()
df4 = test_dataset.copy()
df5 = test_dataset.copy()
df5.head()


# In[ ]:


myList = [file.split("/", 1)[1] for file in glob.glob('dataset/test/sculpture/*')]
test_dataset = pd.DataFrame(data=myList, columns=['filename'])
test_dataset[categories] = pd.DataFrame([[1, 0, 0, 0, 0]], index=dataset.index)

myList = [file.split("/", 1)[1] for file in glob.glob('dataset/test/iconography/*')]
df2 = pd.DataFrame(data=myList, columns=['filename'])
df2[categories] = pd.DataFrame([[0, 1, 0, 0, 0]], index=df2.index)
test_dataset = test_dataset.append(df2)

myList = [file.split("/", 1)[1] for file in glob.glob('dataset/test/engraving/*')]
df3 = pd.DataFrame(data=myList, columns=['filename'])
df3[categories] = pd.DataFrame([[0, 0, 1, 0, 0]], index=df3.index)
test_dataset = test_dataset.append(df3)

myList = [file.split("/", 1)[1] for file in glob.glob('dataset/test/drawings/*')]
df4 = pd.DataFrame(data=myList, columns=['filename'])
df4[categories] = pd.DataFrame([[0, 0, 0, 1, 0]], index=df4.index)
test_dataset = test_dataset.append(df3)

myList = [file.split("/", 1)[1] for file in glob.glob('dataset/test/painting/*')]
df5 = pd.DataFrame(data=myList, columns=['filename'])
df5[categories] = pd.DataFrame([[0, 0, 0, 0, 1]], index=df5.index)
test_dataset = test_dataset.append(df5)


# In[ ]:


test_dataset.reset_index(drop=True, inplace=True)
test_dataset.tail()


# In[ ]:


test_dataset.shape


# ## Create generators

# In[ ]:


datagen=ImageDataGenerator(rescale=1./255.)

train_generator=datagen.flow_from_dataframe(
    dataframe=dataset,
    directory="dataset/",
    x_col="filename",
    y_col=categories,
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="other",
    target_size=(100,100))

valid_generator=datagen.flow_from_dataframe(
    dataframe=valid_dataset,
    directory="dataset/",
    x_col="filename",
    y_col=categories,
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="other",
    target_size=(100,100))

test_generator=datagen.flow_from_dataframe(
    dataframe=test_dataset,
    directory="dataset/",
    x_col="filename",
    batch_size=1,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(100,100))


# ## Build Sequential Model and fit

# In[ ]:


# model = Sequential([
#     Conv2D(32, (3, 3), padding='same', input_shape=(100,100,3), activation='relu'),
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
    
#     Dropout(0.25),
#     Conv2D(64, (3, 3), padding='same', activation='relu'),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
    
#     Dropout(0.25),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dropout(0.5),
#     Dense(5, activation='sigmoid')
# ])
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(100,100,3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(5, activation='softmax')
])

model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),
              loss="binary_crossentropy", metrics=["accuracy"])


# In[ ]:


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=50
)


# ## Predict and save results in csv file

# In[ ]:


test_generator.reset()
pred = model.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)


# In[ ]:


pred_bool = (pred >0.5)


# In[ ]:


predictions = pred_bool.astype(int)

results = pd.DataFrame(predictions, columns=categories)
results["filename"] = test_generator.filenames
ordered_cols = ["filename"] + categories

# To get the same column order
results = results[ordered_cols]
results.to_csv("results.csv", index=False)


# In[ ]:


results.head()


# ## Multi-label classification with a Multi-Output Model

# In[ ]:


input_ = Input(shape = (100,100,3))
x = Conv2D(32, (3, 3), padding = 'same')(input_)
x = Activation('relu')(x)
x = Conv2D(32, (3, 3))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size = (2, 2))(x)
x = Dropout(0.25)(x)
x = Conv2D(64, (3, 3), padding = 'same')(x)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size = (2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(512)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
output1 = Dense(1, activation = 'sigmoid')(x)
output2 = Dense(1, activation = 'sigmoid')(x)
output3 = Dense(1, activation = 'sigmoid')(x)
output4 = Dense(1, activation = 'sigmoid')(x)
output5 = Dense(1, activation = 'sigmoid')(x)

model = Model(input_, [output1, output2, output3, output4, output5])

model.compile(optimizers.rmsprop(lr = 0.0001, decay = 1e-6), 
              loss = ["binary_crossentropy", "binary_crossentropy", "binary_crossentropy",
                      "binary_crossentropy", "binary_crossentropy"], metrics = ["accuracy"])


# In[ ]:


def generator_wrapper(generator):
    for batch_x, batch_y in generator:
        yield (batch_x, [batch_y[:,i] for i in range(5)])


# In[ ]:


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

model.fit_generator(generator=generator_wrapper(train_generator),
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=generator_wrapper(valid_generator),
                    validation_steps=STEP_SIZE_VALID,
                    epochs=1, verbose=2)


# In[ ]:


test_generator.reset()
pred = model.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)


# In[ ]:


pred


# In[ ]:


print(type(pred[0]))


# # Conclusion
# 
# * For sure we want to use pre-trained networks where we can
# * Being able to use train_test_split seems to me a best option even so we are dumping everything into memory.
# * In regards to the images the *drawings* and *engraving* are close together in style and therefore have some overlap in results.

# In[ ]:




