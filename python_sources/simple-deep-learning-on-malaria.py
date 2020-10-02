#!/usr/bin/env python
# coding: utf-8

# # Getting Started with Transfer Learning

# In[ ]:


from IPython.display import YouTubeVideo

YouTubeVideo('mPFq5KMxKVw', width=800, height=450)


# # Beginner's intro to Malaria

# ![Malaria_cycle](https://i1.wp.com/www.malariasite.com/wp-content/uploads/2015/02/EID_lec17_slide8-large.jpg?resize=799%2C664&ssl=1)

# In[ ]:


YouTubeVideo('3_2TnCqBFcY', width=800, height=450)


# # Importing Prerequisite Libraries

# # Loading Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2, VGG19
from tensorflow.keras.models import Sequential
from keras import regularizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, InputLayer, Reshape, Conv1D, MaxPool1D, SeparableConv2D
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import cross_validate, train_test_split
import matplotlib.pyplot as plt
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import shutil
print(os.listdir("../input/cell-images-for-detecting-malaria/cell_images/"))

base_dir = '../input/cell-images-for-detecting-malaria/cell_images/'
work_dir = 'work/'
#os.mkdir(work_dir)

base_dir_A = '../input/cell-images-for-detecting-malaria/cell_images/Parasitized/' 
base_dir_B = '../input/cell-images-for-detecting-malaria/cell_images/Uninfected/'

work_dir_A = 'work/A/'
#os.mkdir(work_dir_A)
work_dir_B = 'work/B/'
#os.mkdir(work_dir_B)


# # Dataset preprocessing

# In[ ]:


train_dir = os.path.join(work_dir, 'train')
#os.mkdir(train_dir)

validation_dir = os.path.join(work_dir, 'validation')
#os.mkdir(validation_dir)

test_dir = os.path.join(work_dir, 'test')
#os.mkdir(test_dir)

print("New directories for train, validation, and test created")
train_pos_dir = os.path.join(train_dir, 'pos')
#os.mkdir(train_pos_dir)
train_neg_dir = os.path.join(train_dir, 'neg')
#os.mkdir(train_neg_dir)

validation_pos_dir = os.path.join(validation_dir, 'pos')
#os.mkdir(validation_pos_dir)
validation_neg_dir = os.path.join(validation_dir, 'neg')
#os.mkdir(validation_neg_dir)

test_pos_dir = os.path.join(test_dir, 'pos')
#os.mkdir(test_pos_dir)
test_neg_dir = os.path.join(test_dir, 'neg')
#os.mkdir(test_neg_dir)

print("Train, Validation, and Test folders made for both A and B datasets")


# In[ ]:


i = 0
      
for filename in os.listdir(base_dir_A): 
    dst ="pos" + str(i) + ".jpg"
    src =base_dir_A + filename 
    dst =work_dir_A + dst 
          
       # rename() function will 
       # rename all the files 
    shutil.copy(src, dst) 
    i += 1


j = 0

for filename in os.listdir(base_dir_B): 
    dst ="neg" + str(j) + ".jpg"
    src =base_dir_B + filename 
    dst =work_dir_B + dst 
          
    # rename() function will 
    # rename all the files 
    shutil.copy(src, dst) 
    j += 1       
        
print("Images for both categories have been copied to working directories, renamed to A & B + num")


# # Parasitized Images

# In[ ]:


fnames = ['pos{}.jpg'.format(i) for i in range(3000)]
for fname in fnames:
    src = os.path.join(work_dir_A, fname)
    dst = os.path.join(train_pos_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['pos{}.jpg'.format(i) for i in range(3000, 4000)]
for fname in fnames:
    src = os.path.join(work_dir_A, fname)
    dst = os.path.join(validation_pos_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['pos{}.jpg'.format(i) for i in range(4000, 4500)]
for fname in fnames:
    src = os.path.join(work_dir_A, fname)
    dst = os.path.join(test_pos_dir, fname)
    shutil.copyfile(src, dst)


# # Uninfected Images

# In[ ]:



fnames = ['neg{}.jpg'.format(i) for i in range(3000)]
for fname in fnames:
    src = os.path.join(work_dir_B, fname)
    dst = os.path.join(train_neg_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['neg{}.jpg'.format(i) for i in range(3000, 4000)]
for fname in fnames:
    src = os.path.join(work_dir_B, fname)
    dst = os.path.join(validation_neg_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['neg{}.jpg'.format(i) for i in range(4000, 4500)]
for fname in fnames:
    src = os.path.join(work_dir_B, fname)
    dst = os.path.join(test_neg_dir, fname)
    shutil.copyfile(src, dst)
    
print("Train, validation, and test datasets split and ready for use")
print('total training pos images:', len(os.listdir(train_pos_dir)))
print('total training neg images:', len(os.listdir(train_neg_dir)))
print('total validation pos images:', len(os.listdir(validation_pos_dir)))
print('total validation neg images:', len(os.listdir(validation_neg_dir)))
print('total test pos images:', len(os.listdir(test_pos_dir)))
print('total test neg images:', len(os.listdir(test_neg_dir)))


# # Image Augmentation

# In[ ]:


train_datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.33)


# In[ ]:


train_generator = train_datagen.flow_from_directory(directory= train_dir,             
                                                     target_size=(128, 128),
                                                     class_mode='binary',
                                                     subset='training',
                                                    shuffle=True,
                                                     batch_size=32
                                 )

valid_generator = train_datagen.flow_from_directory(directory= validation_dir,
                                                      target_size=(128, 128),
                                                     class_mode='binary',
                                                           shuffle = True,
                                                     subset='validation',
                                                     batch_size=32,
                                                    
                                                     )


classes = ['Parasitized', 'Uninfected']


# # Displaying The Images

# In[ ]:


sample_training_images, train_label = next(train_generator)


# In[ ]:


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout() 
    plt.show()


# In[ ]:


print('Random Display of Cell images')
plotImages(sample_training_images[:5])


# # Depth-Wise Separable CNN (DS-CNN)
# 
# This model is faster form of convolution model 
# You can understand more about this [here](https://www.youtube.com/watch?vT7o3xvJLuHk&t=12s)

# In[ ]:


input_length = 128,128,3

ds_model = Sequential()
ds_model.add(Conv2D(16,(3,3),activation='relu',input_shape=(128,128,3)))
ds_model.add(MaxPool2D(2,2))
ds_model.add(Dropout(0.2))

ds_model.add(Conv2D(32,(3,3),activation='relu'))
ds_model.add(MaxPool2D(2,2))
ds_model.add(Dropout(0.2))

ds_model.add(SeparableConv2D(64,(3,3),activation='relu'))
ds_model.add(MaxPool2D(2,2))
ds_model.add(Dropout(0.3))

ds_model.add(SeparableConv2D(128,(3,3),activation='relu'))
ds_model.add(MaxPool2D(2,2))
ds_model.add(Dropout(0.3))

ds_model.add(Flatten())
ds_model.add(Dense(64,activation='relu'))
ds_model.add(Dropout(0.5))

ds_model.add(Dense(1,activation='sigmoid'))

opt = tf.keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)
ds_model.compile(optimizer= opt, loss='binary_crossentropy', metrics=['accuracy'])
ds_model.summary()


# In[ ]:


early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)


# In[ ]:


history = ds_model.fit_generator(train_generator,
                              epochs=20,
                              steps_per_epoch= len(train_generator),
                              validation_data = (valid_generator),
                              callbacks = [early_stop]
                              #verbose=1
                              )


# # Metrics Plot

# In[ ]:


def visualize_training(history, lw = 3):
    plt.figure(figsize=(10,6))
    plt.plot(history.history['accuracy'], label = 'training', marker = '*', linewidth = lw)
    plt.plot(history.history['val_accuracy'], label = 'validation', marker = 'o', linewidth = lw)
    plt.title('Training Accuracy vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(fontsize = 'x-large')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label = 'training', marker = '*', linewidth = lw)
    plt.plot(history.history['val_loss'], label = 'validation', marker = 'o', linewidth = lw)
    plt.title('Training Loss vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize = 'x-large')
    plt.show()
visualize_training(history)


# In[ ]:


ds_model_name = 'dsmalaria_predsmodel.h5'
ds_model.save_weights(ds_model_name)


# # MobileNet Model Developement
# 
# ***Here I will be using MobileNetV2 architecture below I will be showing architecture of different MobileNet models***
# 
# ![mobilenet](https://miro.medium.com/max/1882/1*bqE59FvgpvoAQUMQ0WEoUA.png)

# In[ ]:


model = Sequential()
model.add(MobileNetV2(include_top=False, pooling='avg', weights='imagenet', input_shape=(128, 128, 3), classes=2))
model.add(Dense(1, activation='sigmoid'))
model.layers[0].trainable = False

opt = tf.keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)
model.compile(optimizer= opt, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# In[ ]:


history = model.fit_generator(train_generator,
                              epochs=20,
                              steps_per_epoch= len(train_generator),
                              validation_data = (valid_generator),
                              callbacks = [early_stop],
                              verbose=1
                              )


# # Model metrics plot

# In[ ]:


def visualize_training(history, lw = 3):
    plt.figure(figsize=(10,6))
    plt.plot(history.history['accuracy'], label = 'training', marker = '*', linewidth = lw)
    plt.plot(history.history['val_accuracy'], label = 'validation', marker = 'o', linewidth = lw)
    plt.title('Training Accuracy vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(fontsize = 'x-large')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label = 'training', marker = '*', linewidth = lw)
    plt.plot(history.history['val_loss'], label = 'validation', marker = 'o', linewidth = lw)
    plt.title('Training Loss vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize = 'x-large')
    plt.show()
visualize_training(history)


# # Save Model

# In[ ]:


model_name = 'malaria_predsmodel.h5'
model.save_weights(model_name)


# # VGG-19
# 
# ![vgg_architecture](https://www.researchgate.net/profile/Clifford_Yang/publication/325137356/figure/fig2/AS:670371271413777@1536840374533/llustration-of-the-network-architecture-of-VGG-19-model-conv-means-convolution-FC-means.jpg)

# In[ ]:


vgg_model = Sequential()
vgg_model.add(VGG19(include_top=False, pooling='avg', weights='imagenet', input_shape=(128, 128, 3), classes=2))
vgg_model.add(Flatten())
vgg_model.add(Dense(256,activation='relu'))
vgg_model.add(Dense(64,activation='relu'))
vgg_model.add(Dense(1,activation = 'sigmoid'))

vgg_model.layers[0].trainable = False

opt = tf.keras.optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999)
vgg_model.compile(optimizer= opt, loss='binary_crossentropy', metrics=['accuracy'])
vgg_model.summary()


# In[ ]:


vgg_history = vgg_model.fit_generator(train_generator,
                              steps_per_epoch = len(train_generator),
                              epochs=20,
                              validation_steps = len(valid_generator),
                                      validation_data = valid_generator,
                              callbacks = [early_stop],
                                      verbose=1
                                     )


# # Save VGG model

# In[ ]:


vgg_model_name = 'vgg_malaria_predsmodel.h5'
model.save_weights(vgg_model_name)


# In[ ]:


def visualize_training(vgg_history, lw = 3):
    plt.figure(figsize=(10,6))
    plt.plot(vgg_history.history['accuracy'], label = 'training', marker = '*', linewidth = lw)
    plt.plot(vgg_history.history['val_accuracy'], label = 'validation', marker = 'o', linewidth = lw)
    plt.title('Training Accuracy vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(fontsize = 'x-large')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(vgg_history.history['loss'], label = 'training', marker = '*', linewidth = lw)
    plt.plot(vgg_history.history['val_loss'], label = 'validation', marker = 'o', linewidth = lw)
    plt.title('Training Loss vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize = 'x-large')
    plt.show()
visualize_training(vgg_history)


# # InceptionV3
# 
# ![architecture](https://miro.medium.com/max/960/1*gqKM5V-uo2sMFFPDS84yJw.png)

# In[ ]:


inception_model = Sequential()
inception_model.add(tf.keras.applications.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_shape=(128, 128, 3), classes=2))
inception_model.add(Flatten())
inception_model.add(Dense(64,activation='relu'))
inception_model.add(Dense(1,activation = 'sigmoid'))

inception_model.layers[0].trainable = False

opt = tf.keras.optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999)

inception_model.compile(optimizer= opt, loss='binary_crossentropy', metrics=['accuracy'])
inception_model.summary()


# In[ ]:


inception_history = inception_model.fit_generator(train_generator,
                              steps_per_epoch = len(train_generator),
                              epochs=20,
                              validation_data=valid_generator,
                              callbacks = [early_stop],
                                                  verbose=1
                                     )


# # Save InceptionV3

# In[ ]:


inception_model_name = 'inceptionv3_malaria_predsmodel.h5'
model.save_weights(inception_model_name)


# # InceptionV3 metrics plot

# In[ ]:


def visualize_training(inception_history, lw = 3):
    plt.figure(figsize=(10,6))
    plt.plot(inception_history.history['accuracy'], label = 'training', marker = '*', linewidth = lw)
    plt.plot(inception_history.history['val_accuracy'], label = 'validation', marker = 'o', linewidth = lw)
    plt.title('Training Accuracy vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(fontsize = 'x-large')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(inception_history.history['loss'], label = 'training', marker = '*', linewidth = lw)
    plt.plot(inception_history.history['val_loss'], label = 'validation', marker = 'o', linewidth = lw)
    plt.title('Training Loss vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize = 'x-large')
    plt.show()
visualize_training(inception_history)


# # Checking Trained Data on Test Data

# # DS-CNN

# In[ ]:


eval_datagen = ImageDataGenerator(rescale=1./255)
eval_generator = eval_datagen.flow_from_directory(
        test_dir,target_size=(128, 128),
        batch_size=32,
        class_mode='binary')
eval_generator.reset()    
pred = ds_model.predict_generator(eval_generator,1000,verbose=1)
print("Predictions finished")
  
import matplotlib.image as mpimg
for index, probability in enumerate(pred):
    image_path = test_dir + "/" +eval_generator.filenames[index]
    img = mpimg.imread(image_path)
    
    plt.imshow(img)
    print(eval_generator.filenames[index])
    if probability > 0.5:
        plt.title("%.2f" % (probability[0]*100) + "% B")
    else:
        plt.title("%.2f" % ((1-probability[0])*100) + "% A")
    plt.show()


# # MobileNet

# In[ ]:


eval_datagen = ImageDataGenerator(rescale=1./255)
eval_generator = eval_datagen.flow_from_directory(
        test_dir,target_size=(128, 128),
        batch_size=32,
        class_mode='binary')
eval_generator.reset()    
pred = model.predict_generator(eval_generator,1000,verbose=1)
print("Predictions finished")
  
import matplotlib.image as mpimg
for index, probability in enumerate(pred):
    image_path = test_dir + "/" +eval_generator.filenames[index]
    img = mpimg.imread(image_path)
    
    plt.imshow(img)
    print(eval_generator.filenames[index])
    if probability > 0.5:
        plt.title("%.2f" % (probability[0]*100) + "% B")
    else:
        plt.title("%.2f" % ((1-probability[0])*100) + "% A")
    plt.show()


# # VGG-19

# In[ ]:


eval_datagen = ImageDataGenerator(rescale=1./255)
eval_generator = eval_datagen.flow_from_directory(
        test_dir,target_size=(128, 128),
        batch_size=32,
        class_mode='binary')
eval_generator.reset()    
pred = vgg_model.predict_generator(eval_generator,1000,verbose=1)
print("Predictions finished")
  
import matplotlib.image as mpimg
for index, probability in enumerate(pred):
    image_path = test_dir + "/" +eval_generator.filenames[index]
    img = mpimg.imread(image_path)
    
    plt.imshow(img)
    print(eval_generator.filenames[index])
    if probability > 0.5:
        plt.title("%.2f" % (probability[0]*100) + "% B")
    else:
        plt.title("%.2f" % ((1-probability[0])*100) + "% A")
    plt.show()


# # InceptionNet

# In[ ]:


eval_datagen = ImageDataGenerator(rescale=1./255)
eval_generator = eval_datagen.flow_from_directory(
        test_dir,target_size=(128, 128),
        batch_size=32,
        class_mode='binary')
eval_generator.reset()    
pred = inception_model.predict_generator(eval_generator,1000,verbose=1)
print("Predictions finished")
  
import matplotlib.image as mpimg
for index, probability in enumerate(pred):
    image_path = test_dir + "/" +eval_generator.filenames[index]
    img = mpimg.imread(image_path)
    
    plt.imshow(img)
    print(eval_generator.filenames[index])
    if probability > 0.5:
        plt.title("%.2f" % (probability[0]*100) + "% B")
    else:
        plt.title("%.2f" % ((1-probability[0])*100) + "% A")
    plt.show()


# ***Big thanks to Medium article of Adrian Yijie Xu for excellent [article](https://medium.com/gradientcrescent/building-a-malaria-classifier-with-keras-background-implementation-d55c32773afa), 
# Do check it out!!***
#  ***
