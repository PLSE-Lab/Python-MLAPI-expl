#!/usr/bin/env python
# coding: utf-8

# # Transfer learning with Keras
# 
# Transfer learning is a very powerful and quick way to have a working prototype Neural Net. It leverages from NNs that were trained in huge datasets and allows us to only fine tune the top layers of our model to address a new problem.

# **NOTE:**
# This model relies on transfer learning and will not work on your kaggle environment.  You can, however, download the .ipynb file and run it locally. It is recommended to run it on a GPU as the inception v3 is a very deep NN. 

# In[ ]:


# Initialization step 1

import numpy as np
import pandas as pd

import os, cv2, random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from matplotlib import ticker
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Initialization step 2: Here we will define which model from Keras library we will use.

from keras.models import Sequential
from keras.layers import Dense, Activation, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

# First attempt will be with the inception v3 model
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import backend as K


# # Data loading and Preprocessing

# In[ ]:


""" 
Forking Jeff Delaney's great method to load and preprocess the inputs. Check his notebook at:

https://www.kaggle.com/jeffd23/the-nature-conservancy-fisheries-monitoring/deep-learning-in-the-deep-blue-lb-1-279

"""

TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test_stg1/'
FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

# Image dimentions for resizing:
ROWS = 299  # Default input of inception v3 
COLS = 299 # Default input of inception v3 
CHANNELS = 3


# In[ ]:


def get_images(fish):
    """Load files from train folder"""
    fish_dir = TRAIN_DIR+'{}'.format(fish)
    images = [fish+'/'+im for im in os.listdir(fish_dir)]
    return images

def read_image(src):
    """Read and resize individual images"""
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (COLS, ROWS), interpolation=cv2.INTER_CUBIC)
    return im


files = []
y_all = []

for fish in FISH_CLASSES:
    fish_files = get_images(fish)
    files.extend(fish_files)
    
    y_fish = np.tile(fish, len(fish_files))
    y_all.extend(y_fish)
    print("{0} photos of {1}".format(len(fish_files), fish))
    
y_all = np.array(y_all)


# In[ ]:


X_all = np.ndarray((len(files), ROWS, COLS, CHANNELS), dtype=np.uint8)

for i, im in enumerate(files): 
    X_all[i] = read_image(TRAIN_DIR+im)
    if i%1000 == 0: print('Processed {} of {}'.format(i, len(files)))

print(X_all.shape)


# In[ ]:


# One Hot Encoding Labels to use with keras
y_all = LabelEncoder().fit_transform(y_all)
y_all = np_utils.to_categorical(y_all)

X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, 
                                                    test_size=0.2, random_state=23, 
                                                    stratify=y_all)


# # Putting the inception model together
# 
# We will leverage from Keras imagenet weights and use transfer learning just to fine tune the final layer. The method below was "forked" from the Keras application examples page: https://keras.io/applications/
# 
# Please refer to the paper below for details on Inception v3:
# http://arxiv.org/abs/1512.00567

# In[ ]:


# Remember that as a part of the preprocessing we have already scaled down the images to fit
# the inception model input requirements.

# create the base pre-trained model from Keras library
base_model = InceptionV3(weights=None, include_top=False) #change weights to 'imagenet' on your local build


# In[ ]:


# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# and a logistic layer for the FISH_CLASSES = 8 we are trying to predict
predictions = Dense(len(FISH_CLASSES), activation='sigmoid')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)


# In[ ]:


# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


# In[ ]:


# train the model on the new data for a few epochs. 
# Uncommednt the code and change the batch_size and nb_epoch below:

"""
model.fit(X_train, y_train, batch_size=64, nb_epoch=1,
              validation_split=0.2, verbose=1, shuffle=True)
"""

# This will only train the recently created top layers
# while keeping the pretrained model weights constant


# In[ ]:


# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first N layers and unfreeze the rest:

N=172

for layer in model.layers[:N]:
    layer.trainable = False
for layer in model.layers[N:]:
    layer.trainable = True


# In[ ]:


# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')


# In[ ]:


# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers

"""
model.fit(X_train, y_train, batch_size=64, nb_epoch=5,
              validation_split=0.2, verbose=1, shuffle=True)
"""


# ## Applying the model to your validation set

# In[ ]:


#loss_and_metrics = model.evaluate(X_valid, y_valid, batch_size=64)
#print ("Validation Logloss: ", loss_and_metrics)


# ## DO NOT CONTINUE IF YOU DON'T LIKE YOUR LOSS ABOVE.
# 
# Using your test set to fine tune your model will significantly increase the chance of overfitting.

# After you are happy with the results obtained on the validation set above you should run it on the test set and submit your predictions

# # ... and now to the Test Set
# 
# Once again forking it from Jeff Delaney's NB. See link on the beggining of data loading and preprocessing.

# In[ ]:


test_files = [im for im in os.listdir(TEST_DIR)]
test = np.ndarray((len(test_files), ROWS, COLS, CHANNELS), dtype=np.uint8)

for i, im in enumerate(test_files): 
    test[i] = read_image(TEST_DIR+im)
    
#test_preds = model.predict(test, verbose=1)


# In[ ]:


submission = pd.DataFrame(test_preds, columns=FISH_CLASSES)
submission.insert(0, 'image', test_files)
submission.head()


# In[ ]:


OUT_DIR =  '../output/'
file_name = 'output.csv'
file_path_name = os.path.join(OUT_DIR, file_name)


# In[ ]:


# Uncommment to create a submission file:
#submission.to_csv(file_path_name, index=False)


# In[ ]:




