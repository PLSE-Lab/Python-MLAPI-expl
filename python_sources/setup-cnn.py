#!/usr/bin/env python
# coding: utf-8

# # Instruction to set up a baseline CNN to train an image classifier
# 
# In this tutorial, I will show you how to train a simple CNN on the beauty image subset. Here are what you need:
# - `keras` installed, with GPU.
# -  enough space to extract the dataset; ideally, it should be at least 90GB. FYI, the format `.tar.gz`, as in `beauty_image.tar.gz`, is in fact a file name extension indicating this is a compressed file, similar to `.zip`.  I used Hamster Zip Archiver software (Windows) to extract, even though it is easier to do so in Linux, i.e. type in the terminal
# ```sh
# tar -zxvf yourfile.tar.gz
# ```
# 
# You can then set up a simple CNN as following steps:
# 1. Reorganize directory tree
# 2. Train a CNN for beauty images.
# 
# > **NOTE**: ignore the warning error here and run it on your local machine. 
# 

# # 1. Reorganize directory tree
# 
# You should have directory tree with following structure:
# ```
# NDSC_project    
# |-- train.csv
# |-- test.csv
# |-- this_jupyter_notebook.ipynb
# |-- categories.json
# |-- beauty_image
# |   |-- 000004d60c92af4390399d71a305f64b.jpg
# |   |-- 0000113904d1bc15232ee3b8a5432254.jpg
# |   |-- ...
# |-- fashion_image
# |   |-- 000006831bfaf0e3b5ce3564a10e6dd4.jpg
# |   |-- 6819bac64db6f99d6dc37a897c0e9651.jpg
# |   |-- ...
# |-- mobile_image
# |   |-- 0000456f97a4805ba4960084ffc8c058.jpg
# |   |-- 454bdaf9438c099583957964914ffbe5.jpg
# |   |-- ...
# ```

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import os
import sys

# check the current directory
cwd = os.getcwd()
print ('Current directory: {}'.format(cwd))

# create new directories
new_folder_paths = ['Train',
                    os.path.join('Train','Beauty'),
                    os.path.join('Train','Fashion'),
                    os.path.join('Train','Mobile'),
                    'Test',
                    os.path.join('Test','Beauty'),
                    os.path.join('Test','Fashion'),
                    os.path.join('Test','Mobile')]

for folder_path in new_folder_paths:
    if (os.path.isdir(folder_path) is False):
        os.mkdir(folder_path)

'''
We now reorganize image files in training set
'''

train_data = pd.read_csv('train.csv')
n_labels = 58
folder_path_dict = {i:'Beauty' for i in range(17)}
folder_path_dict.update({i:'Fashion' for i in range(17, 31, 1)})
folder_path_dict.update({i:'Mobile' for i in range(31, 58, 1)})


for category in range(n_labels):
        
    category_img_paths = train_data[train_data['Category']==category]['image_path'].values.tolist()
    folder_path = os.path.join('Train', folder_path_dict[category], str(category))

    if (os.path.isdir(folder_path) is False):
        os.mkdir(folder_path)

    for img_path in category_img_paths:
        img_name = img_path.split('/')[1]
        
        # some image paths does not contain file extension
        if (img_name[-4:] != '.jpg'):
            img_name += '.jpg'
            img_path += '.jpg'
            
        # if there is no image found, just pass and we will have a look later on
        try:
            os.rename(img_path, os.path.join(folder_path, img_name))
        except FileNotFoundError:
            pass

test_data = pd.read_csv('test.csv')
category_img_paths = test_data['image_path'].values.tolist()
for img_path in category_img_paths:
    img_master_label, img_name = img_path.split('/')
    
    if (img_master_label == 'beauty_image'):
        folder_path = os.path.join('Test', 'Beauty')
    elif (img_master_label == 'fashion_image'):
        folder_path = os.path.join('Test', 'Fashion')
    else:
        folder_path = os.path.join('Test', 'Mobile')
        
    if (img_name[-4:] != '.jpg'):
            img_name += '.jpg'
            img_path += '.jpg'
      
    try:
        os.rename(img_path, os.path.join(folder_path, img_name))
    except FileNotFoundError:
        pass


# It may take up to 40 minutes to finish copying files. The goal of the script is to re-organize the whole image dataset in such a way that it is easy to load and train. You can find 2 news folders:
# ```
# NDSC_project    
# |-- Train
# |   |-- Beauty
# |   |   |-- 0
# |   |   |-- 1 
# |   |   |-- ...
# |   |-- Fashion
# |   |   |-- 17
# |   |   |-- 18 
# |   |   |-- ...
# |   |-- Mobile
# |   |   |-- 31
# |   |   |-- 32
# |   |   |-- ...
# |-- Test
# |   |-- Beauty
# |   |-- Fashion
# |   |-- Mobile
# ```

# # 2. Train a CNN for beauty images.

# In[ ]:


import json

with open('categories.json') as json_file:
    labels = json.load(json_file)
numerical2label = {}

for master_label in labels.keys():
    master_dict = labels[master_label]
    for item_name, item_idx in master_dict.items():
        numerical2label[item_idx] = item_name
        
label2numerical = {}
for item_idx, item_name in numerical2label.items():
    label2numerical[item_name] = item_idx


# In[ ]:


# Source: https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.2-using-convnets-with-small-datasets.ipynb
import os, shutil


# Directories for our training,
# validation and test splits
base_dir = os.path.join(os.getcwd(), 'Train', 'Beauty')
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validate')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# Directory with our training categories
n_labels = 17
for category_id in range(n_labels):
    category_name = numerical2label[category_id]
    train_category_dir = os.path.join(train_dir, category_name)
    if (os.path.isdir(train_category_dir) is False):
        os.mkdir(train_category_dir)

# Directory with our validation categories
for category_id in range(n_labels):
    category_name = numerical2label[category_id]
    validation_category_dir = os.path.join(validation_dir, category_name)
    if (os.path.isdir(validation_category_dir) is False):
        os.mkdir(validation_category_dir)

# Directory with our test categories
for category_id in range(n_labels):
    category_name = numerical2label[category_id]
    test_category_dir = os.path.join(test_dir, category_name)
    if (os.path.isdir(test_category_dir) is False):
        os.mkdir(test_category_dir)


# In[ ]:


for category in range(n_labels):
    print('Category {0}|{1} \t has {2} images.'.format(numerical2label[category],
                                                    category,
                                                    len(os.listdir(os.path.join(base_dir, str(category))))))


# In[ ]:


train_ratio = 0.7; validation_ratio = 0.1; test_ratio = 0.2

for category in range(n_labels):
    category_size = len(os.listdir(os.path.join(base_dir, str(category))))
    train_size = int(train_ratio * category_size)
    validation_size = int(validation_ratio * category_size)
    test_size = category_size - (train_size + validation_size)
    
    # Copy data from category_dir to create train set for category
    category_dir = os.path.join(base_dir, str(category))
    train_category_dir = os.path.join(train_dir, numerical2label[category])
    fnames = os.listdir(category_dir)[0:train_size]
    for fname in fnames:
        src = os.path.join(category_dir, fname)
        dst = os.path.join(train_category_dir, fname)
        shutil.copyfile(src, dst)
        
    # Copy data from category_dir to create validation set for category
    validation_category_dir = os.path.join(validation_dir, numerical2label[category])
    fnames = os.listdir(category_dir)[train_size:train_size+validation_size]
    for fname in fnames:
        src = os.path.join(category_dir, fname)
        dst = os.path.join(validation_category_dir, fname)
        shutil.copyfile(src, dst)

    # Copy data from category_dir to create test set for category
    test_category_dir = os.path.join(test_dir, numerical2label[category])
    fnames = os.listdir(category_dir)[train_size+validation_size:]
    for fname in fnames:
        src = os.path.join(category_dir, fname)
        dst = os.path.join(test_category_dir, fname)
        shutil.copyfile(src, dst)


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')


# In[ ]:


from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(17, activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


from keras import optimizers

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.adam(),
              metrics=['acc'])


# In[ ]:


history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)


# In[ ]:


model.save('cnn_baseline_beauty.h5')


# In[ ]:


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[ ]:


test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)

