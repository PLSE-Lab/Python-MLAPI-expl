#!/usr/bin/env python
# coding: utf-8

# # OREGON WILDLIFE - TENSORFLOW 2.0 + KERAS vs GAPCV
# 
# If you are wondering why I'm using [gapcv](https://gapml.github.io/CV/) instead of [flow_from_directory](https://keras.io/preprocessing/image/) from `tf.keras`. Let me show you in this notebook why.
# 
# I'm going to compare both libraries and their performance when we need to preprocess images to fit a model.

# ## install tensorboard and gapcv

# In[ ]:


get_ipython().run_cell_magic('capture', '', '# install tensorflow 2.0 alpha\n!pip install -q tensorflow-gpu==2.0.0-alpha0\n\n#install GapCV\n!pip install -q gapcv')


# ## import libraries

# In[ ]:


import os
import time
import gc
import shutil
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import regularizers

import gapcv
from gapcv.vision import Images
from gapcv.utils.img_tools import ImgUtils

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print('tensorflow version: ', tf.__version__)
print('keras version: ', tf.keras.__version__)
print('gapcv version: ', gapcv.__version__)


# In[ ]:


os.makedirs('model', exist_ok=True)
print(os.listdir('../input'))
print(os.listdir('./'))


# ## utils functions

# In[ ]:


def plot_sample(imgs_set, labels_set, img_size=(12,12), columns=4, rows=4, random=False):
    """
    Plot a sample of images
    """
    
    fig=plt.figure(figsize=img_size)
    
    for i in range(1, columns*rows + 1):
        
        if random:
            img_x = np.random.randint(0, len(imgs_set))
        else:
            img_x = i-1
        
        img = imgs_set[img_x]
        ax = fig.add_subplot(rows, columns, i)
        ax.set_title(str(labels_set[img_x]))
        plt.axis('off')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


# ## data sample
# 
# Let's get a sample of 5 classes from the data to reduce preprocesing and training times instead of using the whole data set.

# In[ ]:


wildlife_filter = ['black_bear', 'bald_eagle', 'cougar', 'elk', 'gray_wolf']

for folder in os.scandir('../input/oregon_wildlife/oregon_wildlife'):
    if folder.name in wildlife_filter:
        shutil.copytree(folder.path, './oregon_wildlife/{}'.format(folder.name))
        print('{} copied from main data set'.format(folder.name))


# In[ ]:


get_ipython().system('ls -l oregon_wildlife')


# ## Metadata
# 
# Let's compare apples with apples for this I'm going to have the same metadata for both trainings. 

# In[ ]:


data_set = 'wildlife'
data_set_folder = './oregon_wildlife'
img_height = 128 
img_width = 128
batch_size = 32
nb_epochs = 50


# ## Model definition
# 
# We are going to use the same model definition I used on a previous notebook into a function to call it later.

# In[ ]:


def model_def(img_height, img_width):
    return Sequential([
        layers.Conv2D(filters=128, kernel_size=(4, 4), activation='tanh', input_shape=(img_height, img_width, 3)),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.Dropout(0.22018745727040784),
        layers.Conv2D(filters=64, kernel_size=(4, 4), activation='relu'),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.Dropout(0.02990527559235584),
        layers.Conv2D(filters=32, kernel_size=(4, 4), activation='tanh'),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.Dropout(0.0015225556862044631),
        layers.Conv2D(filters=32, kernel_size=(4, 4), activation='tanh'),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.Dropout(0.1207251417283281),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4724418446300173),
        layers.Dense(len(wildlife_filter), activation='softmax')
    ])


# ## Keras Image Preprocessing
# 
# Let's start with keras defining `ImageDataGenerator` where we will get:
# 
# * normalization,
# * augmentation:
#     * zoom = 0.3
#     * horizontal flip = True
# * split the data set between train and validation by 20%
# 
# and get a couple generators for training and validation data to fit the model later

# In[ ]:


image_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 # set validation split
)

train_generator = image_datagen.flow_from_directory(
    data_set_folder,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training' # set as training data
)

validation_generator = image_datagen.flow_from_directory(
    data_set_folder, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size, # tried to set 703 manually to compare apples with apples but performance was way worse
    class_mode='categorical',
    subset='validation' # set as validation data
)


# In[ ]:


model = model_def(img_height, img_width)
model.summary()


# In[ ]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


get_ipython().system('free -m')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model.fit_generator(\n    train_generator,\n    steps_per_epoch=train_generator.samples // batch_size,\n    validation_data=validation_generator, \n    validation_steps=validation_generator.samples // batch_size,\n    epochs = nb_epochs\n)')


# As we can see the elapsed time using `flow_from_directory` from keras. We got over an hour to complete training of 50 epochs, with 88 steps to load each batch of 32 images per step. It's interesting to see how the time was between 75 to 88 sec per epoch. Out of the scope of this notebook but notice the accuracy and val_accuracy at the final epoch.

# ## GapCV image preprocessing
# 
# Now let's test `gapcv` `mini_batch` generator. The first thing that gap does is create a preprocessed `h5` file where we will find our data set already transform using numpy arrays to normalize the data.
# 
# With this line of code we will use three configurations:
# 
# 1. image resize
# 2. store: to save the information in a `h5` file
# 3. stream: allows the flow of preprocessed image directly to the `h5` file without save it in-memory (ideal if we have limitated CPU, GPU, or TPU resources)

# In[ ]:


images = Images(data_set, data_set_folder, config=['resize=({},{})'.format(img_height, img_width), 'store', 'stream'])
# explore directory to see if the h5 file is there
get_ipython().system('ls')


# Once the `h5` file is created we can call it again in `stream` mode and also apply image augmentation while the images are getting fit into the model. If we have a `mini_batch` of 32. `gapcv` does is split the data (1/2) and fit the model with half of the images augmented and the other half without augmentation. Even shuffles the data on each epoch so we have a complete new set of images to fit our model.

# In[ ]:


# stream from h5 file
images = Images(config=['stream'])
images.load(data_set, '../input')


# Let's split the data set as well as we did with keras 20% for validation and the rest for training. Then define the generator:

# In[ ]:


# split data set
images.split = 0.2
X_test, Y_test = images.test

# generator
images.minibatch = batch_size
gap_generator = images.minibatch


# Here get can get some information about our data set:

# In[ ]:


total_train_images = images.count - len(X_test)
n_classes = len(images.classes)

print('content:', os.listdir("./"))
print('time to preprocess the data set:', images.elapsed)
print('number of images in data set:', images.count)
print('classes:', images.classes)
print('data type:', images.dtype)


# ## Keras model definition
# 
# Now let's clean the previous session to start the `keras` model all over again

# In[ ]:


shutil.rmtree(data_set_folder) # delete images folder for kaggle kernel limitations
K.clear_session() # clean previous model
gc.collect()


# OK! Let's define and train our model!

# In[ ]:


model = model_def(img_height, img_width)
model.summary()


# In[ ]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


get_ipython().system('free -m')


# ## training

# In[ ]:


get_ipython().run_cell_magic('time', '', 'model.fit_generator(\n    generator=gap_generator,\n    validation_data=(X_test, Y_test),\n    steps_per_epoch=total_train_images // batch_size,\n    epochs=nb_epochs,\n    verbose=1\n)')


# Woah! What a difference almost 3 minutes to complete the whole traing of 50 epochs, with 88 steps to load each batch of 32 images per step. As we can see the time per epoch was constant 3 sec each (after the first one). Plus check the accuracy!
# 
# For `gapcv` was 2:40 minutes plus 15 minutes to preprocess the data but once we have the `h5` file. It is done, we don't have to create it again and it's way easy to share with the team.
# 
# Now if we are using some library for hyperparameters optimization such as hyperas or grid search from `sklearn`. Where we have to run several evaluations and train several models with diferent hyperparameters. We don't want to wait so long to get our results plus we can add callbacks and stop unusefull trainings. Even worse if we are paying cloud computing! We want to expend as low as posible and don't ruin our budgets.
