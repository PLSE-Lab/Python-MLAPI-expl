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


get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
from scipy.stats import itemfreq
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns # visualizations

import random # for setting seed


# # Goal: Create a model that can identify an artist given a painting from the ["Painter By Numbers" dataset](https://www.kaggle.com/c/painter-by-numbers)
# 
# # Methods / Tools
# 1. [Keras](https://keras.io/)
# 2. [Tensorflow](https://www.tensorflow.org/)
# 3. [ResNet50](https://keras.io/applications/#resnet50)
#   * a Convolutional Neural Network (CNN) model instance in Keras
# 4. [Transfer Learning](https://www.kaggle.com/dansbecker/transfer-learning)
# 
# # Questions
# ## 1. Can a ResNet50 model be modified & retrained to classify *Picasso* vs *Not-Picasso*?
# * similar to [hot dogs or not hot dogs](https://www.youtube.com/watch?v=ACmydtFDTGs)
# 
# 

# In[ ]:


# #### miscellaneous questions
# >2. Is it possible to identify an artist from the patterns of pixels in an image of a painting using the methods/tools listed above? 
# >  * __If yes:__ perhaps a CNN can process simple visual patterns (like vertical lines or horizontal lines) and using that understanding, begin to process more complex visual patterns (like shapes) then finally understand even more abstract visual patterns (like people or [hot dogs or not hot dogs](https://www.youtube.com/watch?v=ACmydtFDTGs)) 
# >  * __If no:__ well... hmmm... see question #2
# >3. How does a CNN work to categorize images?
# >4. Considering that ResNet50 was trained on images from the [ImageNet database](http://www.image-net.org/), would the "low-level" visual patterns that it learned from those images still transfer over into the domain of paintings? Why or why not and to what degree? Could the model benefit from learning from scratch, rather than by transfer learning? 


# In[ ]:


import tensorflow
import keras


# # First and foremost set a random seed and get environment info for the sake of [reproducibility](https://www.kaggle.com/rtatman/reproducible-research-best-practices-jupytercon?utm_medium=blog&utm_source=wordpress&utm_campaign=reproducibility-guide)

# In[ ]:


my_seed = 42 # 480 could work too
random.seed(my_seed)
np.random.seed(my_seed)
tensorflow.set_random_seed(my_seed)


# In[ ]:


import IPython

# print system information (but not packages)
print(IPython.sys_info())

# get module information
get_ipython().system('pip freeze > frozen-requirements.txt')

# append system information to file
with open("frozen-requirements.txt", "a") as file:
    file.write(IPython.sys_info())


# # Info about [ResNet50](https://keras.io/applications/#resnet50) on Keras
# ### arguments:
# * "The default **input** size for this model is **224x224**."
# * "**input_tensor**: optional Keras tensor (i.e. output of layers.Input()) to use as image input for the model."
# * "**classes**: optional number of classes to classify images into, only to be specified if include_top is  True, and if no weights argument is specified."

# # let's have a look at [Picasso](https://www.wikiart.org/en/pablo-picasso/)
# ### He has 48 paintings inside train_2

# In[ ]:


filename = '2640.jpg'
data_dir = '../input/painter-by-numbers/'
train_dir = data_dir + 'train_2/'
img = cv2.imread(train_dir + filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

df = pd.read_csv(data_dir + 'train_info.csv')
print("there are " + str(df.shape[0]) + " paintings inside train_2") 
# get a dataframe that has rows referring to files starting with 2
# because we only have those files downloaded currently
# eg. '2.jpg' or '2640.jpg'
mask = (df['filename'].str.startswith('2'))
train_2_df = df[mask]

# string of just the artist's hash code
img_artist = train_2_df[(train_2_df['filename'] == filename)].artist.get_values()[0]

artist_data = train_2_df[(train_2_df['artist'] == img_artist)]
not_artist_data = train_2_df[(train_2_df['artist'] != img_artist)]

num_artist = len(artist_data)
print("Picasso has " + str(num_artist) + " paintings inside train_2")

artist_data.head(5)

# just some code to find 2640.jpg
#mask = [n[4:] == '.jpg' for n in df[(df['artist'] == '1950e9aa6ad878bc2a330880f77ae5a1')].filename]
#[n for n in df[(df['artist'] == '1950e9aa6ad878bc2a330880f77ae5a1')].filename.where(mask) if type(n) == type("string")]


# # Ok, now let's put *Picasso* images into a folder and *Not-Picasso* images into another folder

# In[ ]:


working_train_dir = "train/"
working_test_dir = "test/"
if (os.path.isdir(working_train_dir) == False):
    os.mkdir(working_train_dir)
    print("created " + working_train_dir)
else:
    print(working_train_dir + " exists")
if (os.path.isdir(working_test_dir) == False):
    os.mkdir(working_test_dir)
    print("created " + working_test_dir)
else:
    print(working_test_dir + " exists")

artist_dir = working_train_dir + 'picasso/'
not_artist_dir = working_train_dir + 'not-picasso/'
if (os.path.isdir(artist_dir) == False):
    os.mkdir(artist_dir)
    print("created " + artist_dir)
else:
    print(artist_dir + " exists")
if (os.path.isdir(not_artist_dir) == False):
    os.mkdir(not_artist_dir)
    print("created " + not_artist_dir)
else:
    print(not_artist_dir + " exists")

# same for test data 
test_artist_dir = working_test_dir + 'picasso/'
test_not_artist_dir = working_test_dir + 'not-picasso/'
if (os.path.isdir(test_artist_dir) == False):
    os.mkdir(test_artist_dir)
    print("created " + test_artist_dir)
else:
    print(test_artist_dir + " exists")
if (os.path.isdir(test_not_artist_dir) == False):
    os.mkdir(test_not_artist_dir)
    print("created " + test_not_artist_dir)
else:
    print(test_not_artist_dir + " exists")


# In[ ]:


from shutil import copy2

num_for_test = 10

i=0
num_artist_in_working_dir = len([name for name in os.listdir(artist_dir)])
for f in artist_data['filename']:
    len_dir = len([name for name in os.listdir(artist_dir)]) # to do make this more efficient
    len_test_dir = len([name for name in os.listdir(test_artist_dir)])
    if (len_dir >= num_artist - num_for_test):
        if (len_test_dir >= num_for_test):
            break
        if (os.path.exists(train_dir+f) and not os.path.exists(test_artist_dir+f)):
            copy2(train_dir+f, test_artist_dir)
            i+=1
    elif (os.path.exists(train_dir+f) and not os.path.exists(artist_dir+f)):
        copy2(train_dir+f, artist_dir)
        i+=1
    else:
        None #print(str(i), end=" ")

print("\ncopied artist_data " + str(i))

i=0
num_not_artist_in_working_dir = len([name for name in os.listdir(not_artist_dir)])
for f in not_artist_data['filename']:
    len_dir = len([name for name in os.listdir(not_artist_dir)]) # to do make this more efficient
    len_test_dir = len([name for name in os.listdir(test_not_artist_dir)])
    if (len_dir >= num_artist - num_for_test):
        if (len_test_dir >= num_for_test):
            break
        if (os.path.exists(train_dir+f) and not os.path.exists(test_not_artist_dir+f)):
            copy2(train_dir+f, test_not_artist_dir)
            i+=1
    elif (os.path.exists(train_dir+f) and not os.path.exists(not_artist_dir+f)):
        copy2(train_dir+f, not_artist_dir)
        i+=1
    else:
        None #print(str(i), end=" ")

print("\ncopied not_artist_data " + str(i))
# # # check if files are in the dirs
# # for f in artist_data['filename']:
# #     print(f+'\t', str(os.path.exists(train_dir+f))+'\t\t', os.path.exists('picasso/'+f))


# In[ ]:


print(artist_dir+'\t\t', len(os.listdir(artist_dir)))
print(test_artist_dir+'\t\t', len(os.listdir(test_artist_dir)))
print(not_artist_dir+'\t', len(os.listdir(not_artist_dir)))
print(test_not_artist_dir+'\t', len(os.listdir(test_not_artist_dir)))


# # Let's try following [this lesson on transfer learning](https://www.kaggle.com/dansbecker/transfer-learning)

# ## Specify Model

# In[ ]:


from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

num_classes = 2 # picasso or not picasso

# this file is the resnet50 model trained on ImageNet data...
# "notop" means the file does not include weights for the last layer (prediction layer)
# in order to allow for transfer learning
weights_notop_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

# declare new Sequential model
# meaning each layer is in sequence, one after the other. 
# apparently there can be non-sequential neural networks... wow!
model = Sequential()

# now let's set up the first layers
model.add(ResNet50(    # add a whole ResNet50 model
  include_top=False,          # without the last layer
  weights=weights_notop_path, # and with the "notop" weights file
  pooling='avg' # means collapse extra "channels" into 1D tensor by taking an avg across channels
))


# Now lets add a "Dense" layer to make predictions
model.add(Dense(
  num_classes, # this last layer just has 2 nodes
  activation='softmax' # apply softmax function to turn values of this layer into probabilities
))

# do not train the first layer
# because it is already smart
# it learned cool patterns from ImageNet
model.layers[0].trainable = False


# ## Compile Model
# ### TODO: learn what it means to "***compile***"

# In[ ]:


model.compile(
  optimizer='sgd', # stochastic gradient descent (how to update Dense connections during training)
  loss='categorical_crossentropy', # aka "log loss" -- the cost function to minimize 
  # so 'optimizer' algorithm will minimize 'loss' function
  metrics=['accuracy'] # ask it to report % of correct predictions
)


# ## Fitting the Model

# In[ ]:


from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224

data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator_no_aug = data_generator_no_aug.flow_from_directory(
        working_train_dir,
        target_size=(image_size, image_size),
        batch_size=24,
        class_mode='categorical')

validation_generator = data_generator_no_aug.flow_from_directory(
        working_test_dir,
        target_size=(image_size, image_size),
        class_mode='categorical')

print("\n\nmodel - train_generator_no_aug")

history = model.fit_generator(
        train_generator_no_aug,
        steps_per_epoch=3,
        validation_data=validation_generator,
        validation_steps=1)


# # This is how to [export the "model.h5"](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) file
# > **Click** "Output" on this Kernel and **then scroll** all the way down past the images until you see the ".h5" files

# In[ ]:


# this line is probably all that is needed
model.save('picasso_one_epoch.h5')

# I chose to save the weights to see the difference in filesize
model.save_weights('picasso_one_epoch_weights.h5')


# # Try to overfit the model

# In[ ]:


history_overfit = model.fit_generator(
        train_generator_no_aug,
        steps_per_epoch=3,
        epochs=8, # so... total of 9 epochs?
        validation_data=validation_generator,
        validation_steps=1)


# # A plot of the ~~```learning```~~ memorization process
# * It's hard to say the model "learned" a general concept of what makes a Picasso painting, given a small dataset of only 38 *Picasso* images and 38 *Not-Picasso* images to train on. 

# In[ ]:


# Plot training & validation accuracy values
plt.plot(history_overfit.history['acc'])
plt.plot(history_overfit.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history_overfit.history['loss'])
plt.plot(history_overfit.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# # Looking at the plot above
# * What would cause the orange graph of the loss function to ever spike or increase? 
#   * Loss function is expected to descrease over time with gradient descent
#   * perhaps because this is a very very small subset of the whole dataset
# * What would cause the model accuracy to ever decrease when overfitting?
# * What would it take to memorize the Picasso images with 100% accuracy? 

# # Save the overfit model too
# > **Click** "Output" on this Kernel and **then scroll** all the way down past the images until you see the ".h5" files

# In[ ]:


# this line is probably all that is needed
model.save('picasso_overfit.h5')

# I chose to save the weights to see the difference in filesize
model.save_weights('picasso_overfit_weights.h5')


# 

# 

# 

# 

# 

# 

# 

# 

# 

# ## Fitting a Model With [Data Augmentation](https://www.kaggle.com/dansbecker/data-augmentation/)
# ### seems like the validation accuracy is worse... maybe data augmentation is better for larger datasets
# ### after all Picasso only has 38 training and 10 test paintings
# ### and apparently there is "luck" involved with small datasets

# In[ ]:


data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   horizontal_flip=True,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2)

train_generator_with_aug = data_generator_with_aug.flow_from_directory(
        working_train_dir,
        target_size=(image_size, image_size),
        batch_size=24,
        class_mode='categorical')

train_generator_small_batch = data_generator_no_aug.flow_from_directory(
        working_train_dir,
        target_size=(image_size, image_size),
        batch_size=1,
        class_mode='categorical')


# ## Try different models with different hyperparameters

# In[ ]:


#--------------------
model2 = Sequential()
model3 = Sequential()
model4 = Sequential()
#--------------------


# ## setup

# In[ ]:


#--------------------
model2.add(ResNet50(    # add a whole ResNet50 model
  include_top=False,          # without the last layer
  weights=weights_notop_path, # and with the "notop" weights file
  pooling='avg' # means collapse extra "channels" into 1D tensor by taking an avg across channels
))
model3.add(ResNet50(    # add a whole ResNet50 model
  include_top=False,          # without the last layer
  weights=weights_notop_path, # and with the "notop" weights file
  pooling='avg' # means collapse extra "channels" into 1D tensor by taking an avg across channels
))
model4.add(ResNet50(    # add a whole ResNet50 model
  include_top=False,          # without the last layer
  weights=weights_notop_path, # and with the "notop" weights file
  pooling='avg' # means collapse extra "channels" into 1D tensor by taking an avg across channels
))
#--------------------
model2.add(Dense(
  num_classes, # this last layer just has 2 nodes
  activation='softmax' # apply softmax function to turn values of this layer into probabilities
))
model3.add(Dense(
  num_classes, # this last layer just has 2 nodes
  activation='softmax' # apply softmax function to turn values of this layer into probabilities
))
model4.add(Dense(
  num_classes, # this last layer just has 2 nodes
  activation='softmax' # apply softmax function to turn values of this layer into probabilities
))
#--------------------
model2.layers[0].trainable = False
model3.layers[0].trainable = False
model4.layers[0].trainable = False
#--------------------


# ## compile

# In[ ]:


model2.compile(
  optimizer='sgd', # stochastic gradient descent (how to update Dense connections during training)
  loss='categorical_crossentropy', # aka "log loss" -- the cost function to minimize 
  # so 'optimizer' algorithm will minimize 'loss' function
  metrics=['accuracy'] # ask it to report % of correct predictions
)

model3.compile(
  optimizer='sgd', # stochastic gradient descent (how to update Dense connections during training)
  loss='categorical_crossentropy', # aka "log loss" -- the cost function to minimize 
  # so 'optimizer' algorithm will minimize 'loss' function
  metrics=['accuracy'] # ask it to report % of correct predictions
)

model4.compile(
            optimizer='adam',
  loss='categorical_crossentropy', # aka "log loss" -- the cost function to minimize 
  # so 'optimizer' algorithm will minimize 'loss' function
  metrics=['accuracy'] # ask it to report % of correct predictions
)


# ## train

# In[ ]:


#--------------------
print("\n\nmodel2 - train_generator_with_aug")

model2.fit_generator(
        train_generator_with_aug,
        steps_per_epoch=3,
        epochs=2, # data aug allows more epochs with less worry of overfitting
        validation_data=validation_generator,
        validation_steps=1)

print("\n\nmodel3 - train_generator_small_batch")

model3.fit_generator(
        train_generator_small_batch,
        steps_per_epoch=3,
        validation_data=validation_generator,
        validation_steps=1)

print("\n\nmodel4 - optimizer=adam")

model4.fit_generator(
        train_generator_no_aug,
        steps_per_epoch=3,
        validation_data=validation_generator,
        validation_steps=1)
#--------------------
#--------------------
#--------------------

