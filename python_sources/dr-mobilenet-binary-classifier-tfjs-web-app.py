#!/usr/bin/env python
# coding: utf-8

# **Version 3**: 
# - Added the sklearn cohen_kappa_score metric (quadratic)
# - Added early stopping
# - Added a csv logger
# 
# <hr>

# **Diabetic Retinopathy Analyzer**<br>
# by Marsh [ @vbookshelf ]<br>
# 30 June 2019

# <img src="http://dr.test.woza.work/assets/blind.jpg" width="400"></img>

# ## Introduction

# Diabetic Retinopathy (DR) is the fastest growing cause of preventable blindness. All people with diabetes are at risk. They need to be screened once a year. 
# 
# This screening involves taking a picture of the back of the eye. The picture is called a fundus photo. It's taken using a special camera. An eye doctor then diagnoses this image. In many parts of the world there's a shortage of eye doctors. As a result, in India about 45% of people suffer some form of vision loss before the disease is detected.
# 
# It's now possible to take fundus photos using a cellphone camera.<br>
# https://www.jove.com/video/55958/smartphone-fundus-photography
# 
# Why not also use that same phone to automatically diagnose the photo?
# 
# The objective of this notebook is to build a binary classifier that can detect diabetic retinopathy on a fundus image. This model has been deployed online as a tensorflow.js web app. It can be easily accessed from anywhere where there's an internet connection.
# 
# Fundus images can be quite large, as can be seen by the size of the images in this competition. The good thing about tensorflow.js is that there's no need to upload images. The model runs in the browser and all processing is done locally on the user's computer or mobile phone. 
# 
# We will use a pre-trained MobileNet model. MobileNet was developed by Google to be small and fast. This makes it ideal for web use. Its performance metrics are close to larger models like Inception and VGG.
# 
# > Live Prototype Web App<br>
# > http://dr.test.woza.work/
# > 
# > Github<br>
# https://github.com/vbookshelf/Diabetic-Retinopathy-Analyzer
# 
# For best results please use the Chrome browser when accessing the app. In other browsers the app may freeze. The javascript, html and css code is available on github. 
# 
# Let's get started...

# <hr>

# In[ ]:


import pandas as pd
import numpy as np
import os

import cv2

from skimage.io import imread, imshow
from skimage.transform import resize

from PIL import Image

import tensorflow
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                        ModelCheckpoint, CSVLogger)
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Don't Show Warning Messages
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


os.listdir('../input')


# In[ ]:


IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3


# In[ ]:


df_train = pd.read_csv('../input/train.csv')


print(df_train.shape)


# ### Create a new column called file_name

# In[ ]:


# Add a file_name column to df_train and df_test

def create_fname(x):
    
    fname = str(x) + '.png'
    
    return fname

df_train['file_name'] = df_train['id_code'].apply(create_fname)


# In[ ]:


df_train.head()


# ### Check the target distribution

# In[ ]:


# Check the target distribution
df_train['diagnosis'].value_counts()


# ## Create Binary Targets

# In[ ]:


def binary_target(x):
    if x != 0:
        return 1
    else:
        return x
    
df_train['binary_target'] = df_train['diagnosis'].apply(binary_target)


# In[ ]:


df_train.head()


# In[ ]:


# Check the target distribution

df_train['binary_target'].value_counts()


# ## Balance the target distribution

# In[ ]:


df_0 = df_train[df_train['binary_target'] == 0]
df_1 = df_train[df_train['binary_target'] == 1].sample(len(df_0), random_state=101)


df_data = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)

df_data = shuffle(df_data)

print(df_data.shape)

df_data.head()


# In[ ]:


# Check the new target distribution

df_data['binary_target'].value_counts()


# ## Train Test Split

# In[ ]:


df_train, df_val = train_test_split(df_data, test_size=0.1, random_state=101)

print(df_train.shape)
print(df_val.shape)


# In[ ]:


# check the train set target distribution
df_train['binary_target'].value_counts()


# In[ ]:


# check the train set target distribution
df_val['binary_target'].value_counts()


# ## Create the directory structure

# In these folders we will store the resized images that will later be fed into the generators. Keras needs this directory structure in order for the generators to work.

# **Key**
# > 0 = No Diabetic Retinopathy<br>
# > 1 = Has Diabetic Retinopathy

# In[ ]:


# Create a new directory
base_dir = 'base_dir'
os.mkdir(base_dir)


#[CREATE FOLDERS INSIDE THE BASE DIRECTORY]

# now we create sub folders inside 'base_dir':

# train_dir
    # a_0
    # b_1

# val_dir
    # a_0
    # b_1


# create a path to 'base_dir' to which we will join the names of the new folders
# train_dir
train_dir = os.path.join(base_dir, 'train_dir')
os.mkdir(train_dir)

# val_dir
val_dir = os.path.join(base_dir, 'val_dir')
os.mkdir(val_dir)


# [CREATE FOLDERS INSIDE THE TRAIN, VALIDATION AND TEST FOLDERS]
# Inside each folder we create seperate folders for each class

# create new folders inside train_dir
a_0 = os.path.join(train_dir, 'a_0')
os.mkdir(a_0)
b_1 = os.path.join(train_dir, 'b_1')
os.mkdir(b_1)


# create new folders inside val_dir
a_0 = os.path.join(val_dir, 'a_0')
os.mkdir(a_0)
b_1 = os.path.join(val_dir, 'b_1')
os.mkdir(b_1)


# In[ ]:


# Check that the folders exist
os.listdir('base_dir')


# ### Transfer the Images into the Folders

# In[ ]:


df_train.head()


# In[ ]:


# Set the file_name as the index in df_data
df_data.set_index('file_name', inplace=True)


# In[ ]:


# Get a list of train and val images
train_list = list(df_train['file_name'])

# ============================
# Transfer the train images
# ============================

for fname in train_list:
    
    label = df_data.loc[fname,'binary_target']
    
    if label == 0:
        sub_folder = 'a_0'
        # source path to image
        src = os.path.join('../input/train_images', fname)
        # destination path to image
        dst = os.path.join(train_dir, sub_folder, fname)
        
        image = cv2.imread(src)
        image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        cv2.imwrite(dst, image)
        # save the image at the destination
        # save the image using PIL
        #result = Image.fromarray(image.astype(np.uint8))
        #result.save(dst)
        # copy the image from the source to the destination
        #shutil.copyfile(src, dst)
        
        
    if label == 1:
        sub_folder = 'b_1'
        # source path to image
        src = os.path.join('../input/train_images', fname)
        # destination path to image
        dst = os.path.join(train_dir, sub_folder, fname)
        
        image = cv2.imread(src)
        image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        cv2.imwrite(dst, image)


# In[ ]:


# ============================
# Transfer the val images
# ============================

# Get a list of train and val images
val_list = list(df_val['file_name'])

for fname in val_list:
    
    label = df_data.loc[fname,'binary_target']
    
    if label == 0:
        sub_folder = 'a_0'
        # source path to image
        src = os.path.join('../input/train_images', fname)
        # destination path to image
        dst = os.path.join(val_dir, sub_folder, fname)
        
        image = cv2.imread(src)
        image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        cv2.imwrite(dst, image)
        
        
    if label == 1:
        sub_folder = 'b_1'
        # source path to image
        src = os.path.join('../input/train_images', fname)
        # destination path to image
        dst = os.path.join(val_dir, sub_folder, fname)
        
        image = cv2.imread(src)
        image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        cv2.imwrite(dst, image)

    


# In[ ]:


# Check how many images are in the train sub folders

print(len(os.listdir('base_dir/train_dir/a_0')))
print(len(os.listdir('base_dir/train_dir/b_1')))


# In[ ]:


# Check how many images are in the val sub folders

print(len(os.listdir('base_dir/val_dir/a_0')))
print(len(os.listdir('base_dir/val_dir/b_1')))


# ## Set Up the Generators

# In[ ]:


train_path = 'base_dir/train_dir'
val_path = 'base_dir/val_dir'

num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 5
val_batch_size = 5

# Get the number of train and val steps
train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)


# In[ ]:


# Pre-process the input images in the same way as the ImageNet images 
# were pre-processed when they were used to train MobileNet.
datagen = ImageDataGenerator(
    preprocessing_function= \
    tensorflow.keras.applications.mobilenet.preprocess_input)

train_gen = datagen.flow_from_directory(train_path,
                                            target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
                                            batch_size=train_batch_size)

val_gen = datagen.flow_from_directory(val_path,
                                            target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
                                            batch_size=val_batch_size)

# Note: shuffle=False causes the test dataset to not be shuffled
# We are only going to use this to make a prediction on the val set. That's
# why the path is set as val_path
test_gen = datagen.flow_from_directory(val_path,
                                            target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
                                            batch_size=1,
                                            shuffle=False)


# ## MobileNet Pre-trained Model

# In[ ]:


# create a copy of a mobilenet model

mobile = tensorflow.keras.applications.mobilenet.MobileNet()


# In[ ]:


mobile.summary()


# In[ ]:


# The layers are set up as a list.

type(mobile.layers)


# In[ ]:


# How many layers does MobileNet have?
len(mobile.layers)


# In[ ]:


# CREATE THE MODEL ARCHITECTURE

# Exclude the last 5 layers of the above model.
# This will include all layers up to and including global_average_pooling2d_1
x = mobile.layers[-6].output

# Create a new dense layer for predictions
# 2 corresponds to the number of classes
x = Dropout(0.25)(x)
predictions = Dense(2, activation='softmax')(x)

# inputs=mobile.input selects the input layer, outputs=predictions refers to the
# dense layer we created above.

model = Model(inputs=mobile.input, outputs=predictions)


# In[ ]:


model.summary()


# In[ ]:


# We need to choose how many layers we actually want to be trained.

# Here we are freezing the weights of all layers except the
# last 23 layers in the new model.
# The last 23 layers of the model will be trained.

for layer in model.layers[:-23]:
    layer.trainable = False


# ## Train the Model

# In[ ]:


# Get the labels that are associated with each index
print(val_gen.class_indices)


# In[ ]:


# Add weights to try to make the model more sensitive to some classes.
# The dictionary is ordered as per the above output.

# Here the weights are set to 1 so this is not affecting the model.
# These weights can be changed later, if needed.

class_weights={
    0: 1.0, # Class 0
    1: 1.0, # Class 1
}


# In[ ]:


model.compile(Adam(lr=0.01), loss='categorical_crossentropy', 
              metrics=[categorical_accuracy])


filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, 
                             save_best_only=True, mode='max')


reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)

early_stopper = EarlyStopping(monitor="val_categorical_accuracy", 
                      mode="max", 
                      patience=7)

csv_logger = CSVLogger(filename='training_log.csv',
                       separator=',',
                       append=False)
                              
                              
callbacks_list = [checkpoint, reduce_lr, early_stopper, csv_logger]

history = model.fit_generator(train_gen, steps_per_epoch=train_steps, 
                              class_weight=class_weights,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=100, verbose=1,
                   callbacks=callbacks_list)


# ## Get the best epoch from the training log

# In[ ]:


# check that the training_log.csv file has been created
get_ipython().system('ls')


# In[ ]:


# load the training log
df = pd.read_csv('training_log.csv')

# we are monitoring val_loss
best_acc = df['val_categorical_accuracy'].max()

# display the row with the best accuracy
df[df['val_categorical_accuracy'] == best_acc]


# ## Evaluate the model using the val set

# In[ ]:


# get the metric names so we can use evaulate_generator
model.metrics_names


# In[ ]:


# Note: evaluate_generator appears to work when using tensorflow.keras but
# it gives wrong results when using ordinary Keras. This could be a bug.

# Here the best epoch will be used.
model.load_weights('model.h5')

val_loss, val_categorical_accuracy = model.evaluate_generator(test_gen, 
                        steps=len(df_val))

print('val_loss:', val_loss)
print('val_categorical_accuracy:', val_categorical_accuracy)


# ## Plot the Training Curves

# In[ ]:


# display the loss and accuracy curves

import matplotlib.pyplot as plt

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.plot(epochs, acc, 'bo', label='Training cat acc')
plt.plot(epochs, val_acc, 'b', label='Validation cat acc')
plt.title('Training and validation cat accuracy')
plt.legend()
plt.figure()



plt.show()


# ## Confusion Matrix

# In[ ]:


# Get the labels of the test images.

test_labels = test_gen.classes

# We need these to plot the confusion matrix.
test_labels


# In[ ]:


# Print the label associated with each class
test_gen.class_indices


# In[ ]:


# make a prediction on the val data
predictions = model.predict_generator(test_gen, steps=len(df_val), verbose=1)


# In[ ]:


predictions.shape


# In[ ]:


# Source: Scikit Learn website
# http://scikit-learn.org/stable/auto_examples/
# model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-
# selection-plot-confusion-matrix-py


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[ ]:


test_labels.shape


# In[ ]:


# argmax returns the index of the max value in a row
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))


# In[ ]:


test_gen.class_indices


# In[ ]:


# Define the labels of the class indices. These need to match the 
# order shown above.
cm_plot_labels = ['0', '1']

plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')


# ## Classification Report

# In[ ]:


# Get the index of the class with the highest probability score
y_pred = np.argmax(predictions, axis=1)

# Get the labels of the test images.
y_true = test_gen.classes


# In[ ]:


from sklearn.metrics import classification_report

# Generate a classification report
report = classification_report(y_true, y_pred, target_names=cm_plot_labels)

print(report)


# > **Recall** = Given a class, will the classifier be able to detect it?<br>
# > **Precision** = Given a class prediction from a classifier, how likely is it to be correct?<br>
# > **F1 Score** = The harmonic mean of the recall and precision. Essentially, it punishes extreme values.

# ## Cohen Kappa Score (Quadratic)

# In[ ]:


from sklearn.metrics import cohen_kappa_score

cohen_kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')

cohen_kappa


# ## Results
# 
# Based on the above scores performance looks surprisingly good. It appears that diabetic retinopathy has a distinctive pattern that the model can easily detect. Classifying the severity of diabetic retinopathy may prove to be more of a challenge but simply detecting that it exists is quite easy. 
# 
# Detection can be easily automated, as the app demonstrates. There is also a high level of confidence that the diagnosis that the app ouputs is in fact correct. It currently processes one image at a time but it can be modified to diagnose multiple images in seconds. This can help doctors to triage patients and help speed up their diagnostic workflows. 
# 
# Maybe this could also evolve into a free mobile phone based tool that people with diabetes can use at home - to check themselves for diabetic retinopathy in the same way that they monitor their blood glucose levels.

# In[ ]:





# In[ ]:


# Delete the image data directory we created to prevent a Kaggle error.
# Kaggle allows a max of 500 files to be saved.

shutil.rmtree('base_dir')


# In[ ]:





# ## Convert the Model to Tensorflow.js

# In[ ]:


# Install tensorflowjs.
# Don't use the latest version. Instead install version 1.1.2

# --ignore-installed is added to fix an error.

get_ipython().system('pip install tensorflowjs==1.1.2 --ignore-installed')


# In[ ]:


# Use the command line conversion tool to convert the model

get_ipython().system('tensorflowjs_converter --input_format keras model.h5 tfjs/model')


# In[ ]:


# check that the folder containing the tfjs model files has been created
get_ipython().system('ls')


# In[ ]:





# ## Helpful Resources
# 
# - Excellent tutorial series by deeplizard on how to use Mobilenet with Tensorflow.js<br>
# It explains how to build and deploy a tfjs web app.<br>
# https://www.youtube.com/watch?v=HEQDRWMK6yY
# 
# - Tensorflow.js gallery of projects<br>
# https://github.com/tensorflow/tfjs/blob/master/GALLERY.md
# 
# - Some practical tfjs related lessons I've learned are listed on the readme page of this repo:<br>
# https://github.com/vbookshelf/Skin-Lesion-Analyzer
# 
# - Google video discussing their work on diabetic retinopathy<br>
# https://www.youtube.com/watch?v=JzB7yS9t1YE&feature=youtu.be&t=261
# 
# - Video about taking fundus images using a mobile phone camera<br>
# https://www.jove.com/video/55958/smartphone-fundus-photography

# ## Citations
# 
# - MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications<br>
# Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam<br>
# https://arxiv.org/abs/1704.04861
# 
# - Image by cdd20 from Pixabay

# ## Conclusion
# 
# Thank you for reading.
