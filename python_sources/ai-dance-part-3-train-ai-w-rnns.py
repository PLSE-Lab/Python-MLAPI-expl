#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import Image
Image("../input/Dance_Robots_Comic.jpg")


# (This is part 3 of 3 of my How to Teach an AI to Dance. I originally made 3 separate notebooks for this task before compiling them into one later. The complete assembled notebook of all 3 parts can be found here: https://www.kaggle.com/valkling/how-to-teach-an-ai-to-dance)
# 
# # AI Dance Part 3: Train AI w/ RNNs
# 
# If you have read any of my text generating notebooks or know text generating AIs this next part will be familiar with you. If not, here is one of my related notebooks:
# 
# The Pythonic Python Script for Making Monty Python Scripts: https://www.kaggle.com/valkling/pythonicpythonscript4makingmontypythonscripts
# 
# For the dancing AI, the technique is pretty much the same. We will use our compressed pictures to make n length sequences as input that the model will use to predict the n+1 frame in the sequence. The differences are:
# 
# - The input/outputs will not be in one-hot encoding but rather an array of floats between 0 and 1
# 
# - We will need a larger brain for our model to make it work.
# 
# - We will need to decode the results after to turn them into a usable video.

# In[ ]:


import numpy as np
import pandas as pd
import keras as K
import random
import sqlite3
import cv2
import os

from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imread, imshow
import matplotlib.pyplot as plt

from keras.layers import Input, Dropout, Dense, concatenate, Embedding
from keras.layers import Flatten, Activation
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import np_utils

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM, CuDNNGRU, CuDNNLSTM
from keras.layers import MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

import warnings
warnings.filterwarnings('ignore')


# ## Read in Data
# 
# When processing this type of model on text data, each character is expressed in one hot arrays between ~50-100, (depending on the unique characters in the text to consider). Our data is in 128 numpy arrays, so it is not that much more load on our model to consider our compressed images over single characters of a text document.

# In[ ]:


Dance_Data = np.load('../input/Encoded_Dancer.npy')
Dance_Data.shape


# ## Create Compressed Dance Sequences
# 
# Our model will look at the last 70 frames and attemp to predict the 71st. As such, sur X variable will be an array of 70 (compressed) frames in sequence and our Y variable will be the 71st frame. This block chops our Dance_Data into such sequences of frames.

# In[ ]:


TRAIN_SIZE = Dance_Data.shape[0]
INPUT_SIZE = Dance_Data.shape[1]
SEQUENCE_LENGTH = 70
X_train = np.zeros((TRAIN_SIZE-SEQUENCE_LENGTH, SEQUENCE_LENGTH, INPUT_SIZE), dtype='float32')
Y_train = np.zeros((TRAIN_SIZE-SEQUENCE_LENGTH, INPUT_SIZE), dtype='float32')
for i in range(0, TRAIN_SIZE-SEQUENCE_LENGTH, 1 ): 
    X_train[i] = Dance_Data[i:i + SEQUENCE_LENGTH]
    Y_train[i] = Dance_Data[i + SEQUENCE_LENGTH]

print(X_train.shape)
print(Y_train.shape)


# ## Create the RNN Model
# 
# The model is simply 6 LSTM layers stacked on top of each other. While text data only needs around 2-4 LSTM layers to work, the dance data benifits from a few more as the result is not categorical this time and a large brain allows for more "creativity"(variation) on the AIs part. (Note: CuDNNLSTM layers are just LSTM layers that automatically optimize for the GPU. They run a lot faster than standard LSTM layers at the cost of customization options)

# In[ ]:


def get_model():
    inp = Input(shape=(SEQUENCE_LENGTH, INPUT_SIZE))
    x = CuDNNLSTM(512, return_sequences=True,)(inp)
    x = CuDNNLSTM(256, return_sequences=True,)(x)
    x = CuDNNLSTM(512, return_sequences=True,)(x)
    x = CuDNNLSTM(256, return_sequences=True,)(x)
    x = CuDNNLSTM(512, return_sequences=True,)(x)
    x = CuDNNLSTM(1024,)(x)
    x = Dense(512, activation="elu")(x)
    x = Dense(256, activation="elu")(x)
    outp = Dense(INPUT_SIZE, activation='sigmoid')(x)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='mse',
                  optimizer=Adam(lr=0.0002),
                  metrics=['accuracy'],
                 )

    return model

model = get_model()

model.summary()


# ## Callbacks

# In[ ]:


filepath="Ai_Dance_RNN_Model.hdf5"

checkpoint = ModelCheckpoint(filepath,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

early = EarlyStopping(monitor="loss",
                      mode="min",
                      patience=3,
                     restore_best_weights=True)


# ## Train RNN Model

# In[ ]:


model_callbacks = [checkpoint, early]
model.fit(X_train, Y_train,
          batch_size=64,
          epochs=60,
          verbose=2,
          callbacks = model_callbacks)


# In[ ]:


model.save(filepath)
model.save_weights('Ai_Dance_RNN_Weights.hdf5')


# ## Generate New Computer Generated Dances
# 
# This block generates new dance sequences in the style of the video of DANCE_LENGTH size in frames. It takes a random seed pattern from the training set, predicts the next frame, adds it to the end of the pattern and drops the first frame of the pattern and predicts on the new pattern and so forth. The default DANCE_LENGTH of 6000 frames is 5 minutes of video at 20 FPS.
# 
# Pretty much the AI will try to accurately duplicate the Dance video but inevitably makes errors, and those errors compound, but is still trained well enough that it ends up making similar, but not quite the same, dances.
# 
# The LOOPBREAKER is used to add noise to the prediction pattern, replacing a random frame in the pattern with a random frame in the Dance_Data after every LOOPBREAKER frames. This noise can be used to force the AI to change up what it is doing. This can stop undertrained models from looping or overtrained models from duplication the training data too closely. Setting it too low, on the other hand, can cause the results to distort more. It is worth playing around with this setting and is a quick and dirty way to adjust the dance output post training.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'DANCE_LENGTH  = 6000\nLOOPBREAKER = 4\n\nx = np.random.randint(0, X_train.shape[0]-1)\npattern = X_train[x]\noutp = np.zeros((DANCE_LENGTH, INPUT_SIZE), dtype=\'float32\')\nfor t in range(DANCE_LENGTH):\n#   if t % 500 == 0:\n#     print("%"+str((t/DANCE_LENGTH)*100)+" done")\n  \n    x = np.reshape(pattern, (1, pattern.shape[0], pattern.shape[1]))\n    pred = model.predict(x, verbose=0)\n    result = pred[0]\n    outp[t] = result\n    new_pattern = np.zeros((SEQUENCE_LENGTH, INPUT_SIZE), dtype=\'float32\') \n    new_pattern[0:SEQUENCE_LENGTH-1] = pattern[1:SEQUENCE_LENGTH]\n    new_pattern[-1] = result\n    pattern = np.copy(new_pattern)\n    ####loopbreaker####\n    if t % LOOPBREAKER == 1:\n        pattern[np.random.randint(0, SEQUENCE_LENGTH-10)] = Y_train[np.random.randint(0, Y_train.shape[0]-1)]')


# ## Output the Dance
# 
# Before we can save the video, we need to decode the frames back into images using the decoder we made in part 2.

# In[ ]:


Decoder = load_model('../input/Dancer_Decoder_Model.hdf5')
Decoder.load_weights('../input/Dancer_Decoder_Weights.hdf5') 


# In[ ]:


Dance_Output = Decoder.predict(outp)
Dance_Output.shape


# In[ ]:


IMG_HEIGHT = Dance_Output[0].shape[0]
IMG_WIDTH = Dance_Output[0].shape[1]

for row in Dance_Output[0:10]:
    imshow(row.reshape(64,96))
    plt.show()


# ## Save Video

# In[ ]:


video = cv2.VideoWriter('AI_Dance_Video.avi', cv2.VideoWriter_fourcc(*"XVID"), 20.0, (IMG_WIDTH, IMG_HEIGHT),False)

for img in Dance_Output:
    img = resize(img, (IMG_HEIGHT,IMG_WIDTH), mode='constant', preserve_range=True)
    img = img * 255
    img = img.astype('uint8')
    video.write(img)
    cv2.waitKey(50)
    
video.release()


# ## Part 3 Results
# 
# The results of the video are surprisingly crisp. Even small things like the swish of the skirt or swoop of the hair are caught in the video. Like in the youtube video, these results are pretty overfit and the computer is mostly duplicating the dances. However, there are some interesting variations and deformations in the video. The dancer will sometimes shrink and expand its arms or compress into a blob and reform. Playing with the model or the loopbreaker can lead to some interesting results.
# 
# ### Possible Improvements
# 
# - The RNN model could use more and varied dances to train on. A cheap way to do this is just take more frames from the video in part 1. There is also a 5 hour version of these dancing silhouettes. (However, only 3 hours are usable)
# 
# - I don't think that the model needs any more layers, it is large enough as is, but readjusting the shape might make it more efficient. On text data RNNs, I tried using 1D convolution layers with mixed results. (It speeds up the training time a lot but the model is more prone to looping) Might work here though.
