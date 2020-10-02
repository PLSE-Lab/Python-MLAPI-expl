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


# **Import Packages**

# In[ ]:


from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD,Adam,Adadelta
from keras.callbacks import History 
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.models import model_from_json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import json 


# ### Help funtions

# In[ ]:


def summarize_diagnostics(history):
    # plot loss
    plt.figure(figsize=(18,10))
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    plt.legend()
    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['acc'], color='blue', label='train')
    plt.plot(history.history['val_acc'], color='orange', label='test')
    plt.legend()
    
def save_model(model,model_name):
    #Save the Model and Weights 
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(project_path,model_name+".json"), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(project_path,model_name+".h5"))
    print("Saved model to disk")


# ## Pre-Process Photos into Directories

# In[ ]:


get_ipython().system('rm -rf /kaggle/working/dataset_dogs_vs_cats')


# In[ ]:


#Copy files
get_ipython().system('cp -avr ../input/train/train/ /kaggle/working/dataset_dogs_vs_cats')


# In[ ]:


os.listdir("/kaggle/working/dataset_dogs_vs_cats/")


# In[ ]:


# organize dataset
from os import makedirs
from os import listdir
from shutil import copyfile
# create directories
dataset_home = '/kaggle/working/dataset_dogs_vs_cats/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
    # create label subdirectories
    labeldirs = ['dogs/', 'cats/']
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(newdir, exist_ok=True)


# In[ ]:


# seed random number generator
random.seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.25
# copy training dataset images into subdirectories
src_directory = '/kaggle/working/dataset_dogs_vs_cats/'
for file in listdir(src_directory):
    src = src_directory + file
    if random.random() < val_ratio:
        dst_dir = 'test/'     
    else:
        dst_dir = 'train/'
        
    if file.startswith('cat'):
        dst = dataset_home + dst_dir + 'cats/'  + file
        os.rename(src, dst)
    elif file.startswith('dog'):
        dst = dataset_home + dst_dir + 'dogs/'  + file
        os.rename(src, dst)


# In[ ]:


os.listdir("/kaggle/working/dataset_dogs_vs_cats/")


# In[ ]:


#Define train and test path
train_path = '/kaggle/working/dataset_dogs_vs_cats/train/'
test_path  = '/kaggle/working/dataset_dogs_vs_cats/test/'


# ### Plot Some images

# In[ ]:


# plot dog photos from the dogs vs cats dataset
plt.figure(figsize=(18,10))
# define location of dataset
folder = train_path+'dogs/'
# plot first few images
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # define filename
    filename = random.choice(os.listdir(folder))
    file = folder + filename
    # load image pixels
    image = plt.imread(file)
    # plot raw pixel data
    plt.imshow(image)
# show the figure
plt.show()


# In[ ]:


# plot dog photos from the dogs vs cats dataset
plt.figure(figsize=(18,10))
# define location of dataset
folder = train_path+'cats/'
# plot first few images
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # define filename
    filename = random.choice(os.listdir(folder))
    file = folder + filename
    # load image pixels
    image = plt.imread(file)
    # plot raw pixel data
    plt.imshow(image)
# show the figure
plt.show()


# ### Create a CNN Model based on Transfer Learning from Xception model
# [https://keras.io/applications/#xception](https://keras.io/applications/#xception)

# #### 1 - CNN Model - Optimizer:SGD Without Regularization

# In[ ]:


# define cnn model
def Model_CNN_SGD_No_Regularization():
    # load model
    model = Xception(include_top=False,weights='imagenet', input_shape=(299, 299, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# #### 2 - CNN Model - Optimizer:Adam Without Regularization

# In[ ]:


# define cnn model
def Model_CNN_ADAM_No_Regularization():
    # load model
    model = Xception(include_top=False,weights='imagenet', input_shape=(299, 299, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = Adam(lr=0.001, decay=0.0)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# #### 3 - CNN Model - Optimizer:Adadelta Without Regularization

# In[ ]:


# define cnn model
def Model_CNN_ADADELTA_No_Regularization():
    # load model
    model = Xception(include_top=False,weights='imagenet', input_shape=(299, 299, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = Adadelta(lr=0.001, decay=0.0)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ### Prepare the datasets to the Fit process

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator, load_img


#For this case, we'll use Data Augmentation
training_datagen = ImageDataGenerator(rescale=1./255
                                      #,validation_split=0.1
                                      ,data_format='channels_last'
        ,shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        featurewise_center=True)


test_datagen = ImageDataGenerator(rescale=1./255
                                   #,validation_split=0.1
                                  ,data_format='channels_last')


training_set = training_datagen.flow_from_directory(
        directory=train_path,
        target_size=(299, 299),
        batch_size=64,
        #classes=['Dog','Cat'],
        subset = "training",
        #save_to_dir = os.path.join(dataset_path,'train'),
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
       directory=test_path,
        target_size=(299, 299),
        batch_size=64,
        #classes=['Dog','Cat'],
        #subset = "validation",
        #save_to_dir = os.path.join(dataset_path,'test'),
        class_mode='binary')


# ### Let's train our models

# In[ ]:


#Define Models
Model_CNN_SGD_No_Regularization = Model_CNN_SGD_No_Regularization()
Model_CNN_ADAM_No_Regularization = Model_CNN_ADAM_No_Regularization()
Model_CNN_ADADELTA_No_Regularization = Model_CNN_ADADELTA_No_Regularization()


# In[ ]:


history_1 = History()
history_2 = History()
history_3 = History()
epochs = 10


# In[ ]:


Model_CNN_SGD_No_Regularization.fit_generator(training_set,steps_per_epoch=len(training_set),epochs=epochs,validation_data=test_set,validation_steps=len(test_set),callbacks=[history_1])


# In[ ]:


Model_CNN_ADAM_No_Regularization.fit_generator(training_set,steps_per_epoch=len(training_set),epochs=epochs,validation_data=test_set,validation_steps=len(test_set),callbacks=[history_2])


# In[ ]:


Model_CNN_ADADELTA_No_Regularization.fit_generator(training_set,steps_per_epoch=len(training_set),epochs=epochs,validation_data=test_set,validation_steps=len(test_set),callbacks=[history_3])


# ### Let's evaluate our models

# #### 1 - Model CNN with SGD optimizer without Regularization

# In[ ]:


_,acc1 = Model_CNN_SGD_No_Regularization.evaluate_generator(training_set,steps=len(training_set), verbose=0)
print('> %.3f' % (acc1 * 100.0))


# In[ ]:


summarize_diagnostics(history_1)


# In[ ]:


save_model(Model_CNN_SGD_No_Regularization,'Model_CNN_SGD_No_Regularization')


# #### 2 - Model CNN with Adam optimizer without Regularization

# In[ ]:


_,acc2 = Model_CNN_ADAM_No_Regularization.evaluate_generator(training_set,steps=len(training_set), verbose=0)
print('> %.3f' % (acc2 * 100.0))


# In[ ]:


summarize_diagnostics(history_2)


# #### 3 - Model CNN with Adam Delta optimizer without Regularization

# In[ ]:


_,acc3 = Model_CNN_ADADELTA_No_Regularization.evaluate_generator(training_set,steps=len(training_set), verbose=0)
print('> %.3f' % (acc3 * 100.0))


# In[ ]:


summarize_diagnostics(history_3)


# #### As we can see, we can try increase the number of epochs, and the model 3 looks with a best potetntial based on Loss and Accuracy plots. Based on this, we'll try training this model with all data in order to check the results
# 
# 

# In[ ]:


#Prepare the dataset
os.mkdir("/kaggle/working/dataset_dogs_vs_cats/final_train/")
os.mkdir("/kaggle/working/dataset_dogs_vs_cats/final_train/cats")
os.mkdir("/kaggle/working/dataset_dogs_vs_cats/final_train/dogs")


# In[ ]:


#Copy files
get_ipython().system('cp -avr ../input/train/train/ /kaggle/working/dataset_dogs_vs_cats/')


# In[ ]:


# copy training dataset images into subdirectories
src_directory = '/kaggle/working/dataset_dogs_vs_cats/'
for file in listdir(src_directory):
    src = src_directory + file
    dst_dir = 'final_train/'       
    if file.startswith('cat'):
        dst = dataset_home + dst_dir + 'cats/'  + file
        os.rename(src, dst)
    elif file.startswith('dog'):
        dst = dataset_home + dst_dir + 'dogs/'  + file
        os.rename(src, dst)


# In[ ]:


os.listdir("/kaggle/working/dataset_dogs_vs_cats/")

