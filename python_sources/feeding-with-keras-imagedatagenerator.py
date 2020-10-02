#!/usr/bin/env python
# coding: utf-8

# # Using Keras ImageDataGenerator
# Here I'm using Keras [`ImageDataGenerator`](https://keras.io/preprocessing/image/) to read files from the `train` directory and feed a very simple CNN. I'm still not searching for accuracy here, just trying to simplify the pipeline. ImageDataGenerator has the ability to generate a flow of images to the CNN, applying:
# * resampling to a smaller size
# * changing to grayscale if needed
# * train/validation split (still not done here)
# * data augmentation
# 
# TODO:
# * resample to a different size
# * change the CNN to a more effective one (this one is not learning at all)
# * add the validation_generator and pass it to the model's `fit_generator` method
# * analyze source data
# * train on whole dataset
# * data augmentation
# * implement Mean Average Precision @ 5 for submission (see https://www.kaggle.com/pestipeti/explanation-of-map5-scoring-metric)
# 

# In[ ]:


import os
import math
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, ZeroPadding2D, BatchNormalization
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from scipy.misc import imresize
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

warnings.simplefilter("ignore", category=DeprecationWarning)

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("display.max_rows", 10)
np.random.seed(42)


# In[ ]:


os.listdir("../input/")


# ImageDataGenerator requires `filename` and `class` respectively for the column with all the file names and for the other with the classes. Here I change the columns names but I could have used:
# 
#     x_col: string, column in the dataframe that contains
#            the filenames of the target images.
#     y_col: string or list of strings,columns in
#            the dataframe that will be the target data.
# in `flow_from_dataframe` method to override the default Keras behaviour

# In[ ]:


dataset = pd.read_csv("../input/train.csv")
dataset.columns = ['filename', 'class'] # renaming to match ImageDataGenerator expectations
dataset.sample(5)


# In[ ]:


dataset.shape


# Here I use only a subset of all the 25k picture in order to be faster. Slicing the dataframe is enough.
# 
# * `batch_size` controls how many samples the generator sends to the network each step
# * `subset` is used to slice the source dataset and work on a smaller one when making experiments
# * `target_size` is the image shape to use in the training

# In[ ]:



batch_size = 128
subset = 500
target_size = (64, 64, 1) # set to grayscale

datagen = ImageDataGenerator(
    validation_split=.2,
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    horizontal_flip=False)

train_generator = datagen.flow_from_dataframe(
        dataframe=dataset.iloc[:subset],
        directory='../input/train',
        target_size=target_size[0:2],
        color_mode='grayscale', # this has to match the target_size parameter
        batch_size=batch_size,
        class_mode='categorical',
        interpolation='nearest')

num_classes = len(np.unique(train_generator.classes))


# The CNN. Notes:
# * `target_size` is passed to the first layer
# * optimizer set to Adam with default learning rate of .02 and a learning rate decay at each epoch

# In[ ]:


model = Sequential()
model.add(BatchNormalization(input_shape = target_size ))
model.add(Conv2D(filters=32, 
                 kernel_size=(7,7), 
                 activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(.25))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
opt = Adam(lr=0.02, decay=0.005)
model.compile(optimizer = opt , loss = "categorical_crossentropy", metrics=["categorical_accuracy"])
model.build()
model.summary()


# In[ ]:


epochs = 3

history = model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=subset//epochs)


# In[ ]:


plt.plot(history.history['categorical_accuracy'])
plt.title('Model categorical_accuracy')
plt.ylabel('categorical_accuracy')
plt.xlabel('epoch')
plt.show()


# ### TO BE CONTINUED
