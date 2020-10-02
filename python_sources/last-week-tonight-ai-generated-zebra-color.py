#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import Image
Image("../input/Zebra_Test_Image.jpg")


# # Teaching an AI to "Zebra" (In Color!)
# 
# (Since the zebra AI worked so well in black and white, here is an attempt using color. The models are kept the same, except where it necessary, like the input/output shapes of the autoencoder)
# 
# Last Week Tonight released a green screen video of a zebra dancing and doing various other activities for viewers to edit into their own videos. This video is actually pretty good for video processing algorithms and AI video training. Let's try using this to create our own AI generated zebra dancing video.
# 
# This is a variation of my original notebook How to Teach an AI to Dance with new data: https://www.kaggle.com/valkling/how-to-teach-an-ai-to-dance
# 
# Watch a sample output from this notebook here: https://youtu.be/_Eq-u67ZJRI
# 
# NLP and image CNNs are all the rage right now, here we combine techniques from both to have a computer learn to make it's own dance videos. This notebook is a consolidation of 3 smaller notebooks I made for this project: 
# 
# Part 1-Video Preprocessing: We will take the frames from the zebra video, preprocess them to smaller and simpler, and add them to a zip file in sequence.
# 
# Part 2-Autoencoder Compression: To save even more memory for our model, we will compress these frames with an Autoencoder into a much smaller numpy array.
# 
# Part 3-Train AI w/ RNNs: We will put these compressed frames into sequences and train a model to create more.
# 
# 
# I based the original How to Teach an AI to Dance notebook off the project in this youtube video: https://www.youtube.com/watch?v=Sc7RiNgHHaE While he does not share his code, the steps expressed in the video were clear enough to piece together this project. Thanks to Kaggle's kernals GPU and some alterations, we can achieve even better results in less time than what is shown in the video. While still pretty computationally expensive for modern computing power, using these techniques for a dancing AI opens up the groundwork for AI to predict on and create all types of different videos.
# 
# 
# ### Skip Training
# 
# The results of the 3 parts are already recorded in the dataset and each part can work independently from each other by loading the pretrained data. Setting the following variables to *True* will skip the training for that part and just use the pretrained data instead. *False* will train through that step.
# 
# Top to bottom, this whole notebook takes around ~2 hours to train with all 3 skips set to *False*. 
# 

# In[ ]:


PART_1_SKIP = True
PART_2_SKIP = True
PART_3_SKIP = True


# # AI Zebra Part 1: Video Preprocessing (In Color!)
# 
# The video is ~23 minutes of a guy in a zebra outfit preforming a veriety of actions. This is ideal as most other green screen dancing videos are too short, loops, and/or messy for easy preprocessing. While it could be easily done, I will not take a binary shadow of the zebra (unlike the dancing silloette video I used in the first notebook). I am just curious how the AI handles the texture (and color). There is also little need to cut out the green screen but the simple continous background will help the computer focus on the zebra's movements and make it easier to compress. (The dancing video in the first notebook had a constanly changing background that needed to be removed for simple processing)
# 
# The original Last Week Tonight zebra video is here: https://youtu.be/-Z668Qc0P4Q

# In[ ]:


import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import skimage
from PIL import Image
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.util import crop, pad
from skimage.morphology import label
from skimage.color import rgb2gray, gray2rgb

import os

import zipfile
z = zipfile.ZipFile("Zebra_Color_Images.zip", "w")


# In[ ]:


cap = cv2.VideoCapture('../input/Dancing_Zebra_Footage.mp4')
print(cap.get(cv2.CAP_PROP_FPS))


# ## Preprocess the Video
# 
# In this step we will take each frame in the video and add them to a zip file in sequence. We will also preprocess the frames in the following way to save space and make it easier for are models to process them later:
# 
# - Only take every other frame: We don't need every frame and will mean that we won't need to use as many frames to look further back in time during with the RNN model later. 
# 
# - Resize the image to 64 by 96 pixels: Much smaller file size and 64 by 96 is easily divided which makes it easier to structure the autoencoder without data loss later.
# 
# - Round down the total frames in the video to 21,000 frames. we could get more but the title screen at the end is worthless to us.
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif PART_1_SKIP == False:\n    try:\n        if not os.path.exists(\'data\'):\n            os.makedirs(\'data\')\n    except OSError:\n        print (\'Error: Creating directory of data\')\n\n    currentFrame = 0\n    count = 0\n    TRAIN_SIZE = 21000\n    FRAME_SKIP = 2\n    IMG_WIDTH = 96\n    IMG_HEIGHT = 64\n    IMG_CHANNELS = 3\n    X_train = np.zeros((TRAIN_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=\'float32\')\n    \n    video = cv2.VideoWriter(\'Simple_Zebra_Dancer_Video_C.avi\',cv2.VideoWriter_fourcc(*"MJPG"), 30, (IMG_WIDTH, IMG_HEIGHT), True)\n\n    while(count < TRAIN_SIZE):\n        try:\n            ret, frame = cap.read()\n\n            if currentFrame % FRAME_SKIP == 0:\n                count += 1\n                if count % int(TRAIN_SIZE/10) == 0:\n                    print(str((count/TRAIN_SIZE)*100)+"% done")\n                # preprocess frames\n                img = frame\n                img = resize(img, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode=\'constant\', preserve_range=True)\n                img = (img).astype(\'uint8\')\n                # save frame to zip and new video sample\n                name = \'./data/frame\' + str(count) + \'.jpg\'\n                cv2.imwrite(name, img)\n                video.write(img)\n                z.write(name)\n                os.remove(name)\n                # save image to training set if training directly to part 2\n                img = img.astype(\'float32\') / 255.\n                X_train[count] = img\n        except:\n            print(\'Frame error\')\n            break\n        currentFrame += 1\n\n    print(str(count)+" Frames collected")\n    cap.release()\n    z.close()\n    video.release()')


# ## Part 1 Results
# 
# The zebra looks a bit too low res at 64 by 96. The shadow dancer did not look quite as bad due to less detail. 
# 
# ### Possible Improvments and Thoughts
# 
# - There are also title frames at the bigining that need to be cut out
# 
# - Taking every one of the 42,000 frames ends up being a problem because changes frame to frame are too small and the RNN will need a much longer sequence length to avoid the AI getting frozen. The zebra often moves slower or has subtle movements so taking every 3 frames might be a good idea.
# 
# - Zebra might need a larger size to really look good.
# 

# # AI Dance Part 2: Autoencoder Compression
# 
# Now that we have the preprocessed frames from the zebra video, we will still need to compress them much further to fit them into our RNN model. Among the many uses of autoencoders is making specialized compression models. In this section, we will train an autoencoder on our zebra images and use it to compress the images into a much smaller numpy array, saving the model so that we can decode the images later.

# In[ ]:


import os
import sys
import random
import warnings
from pylab import imshow, show, get_cmap

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
import skimage
from PIL import Image
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.util import crop, pad
from skimage.morphology import label

from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, UpSampling2D, Flatten, Reshape
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from keras import backend as K
import tensorflow as tf

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed


# ## Read in Images

# In[ ]:


get_ipython().run_cell_magic('time', '', 'if PART_1_SKIP:\n    IMG_WIDTH = 96\n    IMG_HEIGHT = 64\n    IMG_CHANNELS = 3\n    INPUT_SHAPE=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)\n    TRAIN_PATH = \'../input/zebra_color_images/data/\'\n#     train_ids = next(os.walk(TRAIN_PATH))[2]\n    X_train = np.zeros((21000, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=\'float32\')\n    missing_count = 0\n    print(\'Getting training images ... \')\n#     sys.stdout.flush()\n    for n, id_ in tqdm(enumerate(X_train), total=21000):\n        path = TRAIN_PATH +\'frame\'+ str(n+1) + \'.jpg\'\n        try:\n            img = imread(path)\n            img = img.astype(\'float32\') / 255.\n#             img = resize(img, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode=\'constant\', preserve_range=True)\n            X_train[n-missing_count] = img\n        except:\n            print(" Problem with: "+path)\n            missing_count += 1\n\n    print("Done! total missing: "+ str(missing_count))\n    X_train = X_train[75:]\nelse:\n    INPUT_SHAPE=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)\n    X_train = X_train[75:]')


# In[ ]:


for n in range(0,5):
    imshow(X_train[n])
    plt.show()


# ## Create the Models
# 
# In addition to the Autoencoder model, we will also prepare an encoder and decoder for later. It is important to give the layers the same unique names and shapes in all 3 as we will be using the keras load_weights by_name option to copy our trained Autoencoder weights to each respective layer later. I'm leaving these models the exact same shape as in the original AI dance notebook to see how it does in comparison.

# In[ ]:


def Encoder():
    inp = Input(shape=INPUT_SHAPE)
    x = Conv2D(128, (4, 4), activation='elu', padding='same',name='encode1')(inp)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',name='encode2')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',name='encode3')(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same',name='encode4')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',name='encode5')(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same',name='encode6')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',name='encode7')(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same',name='encode8')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same',name='encode9')(x)
    x = Flatten()(x)
    x = Dense(256, activation='elu',name='encode10')(x)
    encoded = Dense(128, activation='sigmoid',name='encode11')(x)
    return Model(inp, encoded)

encoder = Encoder()
encoder.summary()


# In[ ]:


D_INPUT_SHAPE=[128]
def Decoder():
    inp = Input(shape=D_INPUT_SHAPE, name='decoder')
    x = Dense(256, activation='elu', name='decode1')(inp)
    x = Dense(768, activation='elu', name='decode2')(x)
    x = Reshape((4, 6, 32))(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same', name='decode3')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same', name='decode4')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same', name='decode5')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same', name='decode6')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (2, 2), activation='elu', padding='same', name='decode7')(x)
    x = Conv2D(128, (3, 3), activation='elu', padding='same', name='decode8')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (2, 2), activation='elu', padding='same', name='decode9')(x)
    x = Conv2D(64, (4, 4), activation='elu', padding='same', name='decode10')(x)
    x = Conv2D(128, (3, 3), activation='elu', padding='same', name='decode11')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (4, 4), activation='elu', padding='same', name='decode12')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same', name='decode13')(x)
    x = Conv2D(16, (2, 2), activation='elu', padding='same', name='decode14')(x)
    decoded = Conv2D(IMG_CHANNELS, (2, 2), activation='sigmoid', padding='same', name='decode15')(x)
    return Model(inp, decoded)

decoder = Decoder()
decoder.summary()


# In[ ]:


def Autoencoder():
    inp = Input(shape=INPUT_SHAPE)
    x = Conv2D(128, (4, 4), activation='elu', padding='same',name='encode1')(inp)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',name='encode2')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',name='encode3')(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same',name='encode4')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',name='encode5')(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same',name='encode6')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',name='encode7')(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same',name='encode8')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same',name='encode9')(x)
    x = Flatten()(x)
    x = Dense(256, activation='elu',name='encode10')(x)
    encoded = Dense(128, activation='sigmoid',name='encode11')(x)
    x = Dense(256, activation='elu', name='decode1')(encoded)
    x = Dense(768, activation='elu', name='decode2')(x)
    x = Reshape((4, 6, 32))(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same', name='decode3')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same', name='decode4')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same', name='decode5')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same', name='decode6')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (2, 2), activation='elu', padding='same', name='decode7')(x)
    x = Conv2D(128, (3, 3), activation='elu', padding='same', name='decode8')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (2, 2), activation='elu', padding='same', name='decode9')(x)
    x = Conv2D(64, (4, 4), activation='elu', padding='same', name='decode10')(x)
    x = Conv2D(128, (3, 3), activation='elu', padding='same', name='decode11')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (4, 4), activation='elu', padding='same', name='decode12')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same', name='decode13')(x)
    x = Conv2D(16, (2, 2), activation='elu', padding='same', name='decode14')(x)
    decoded = Conv2D(IMG_CHANNELS, (2, 2), activation='sigmoid', padding='same', name='decode15')(x)
    return Model(inp, decoded)

model = Autoencoder()
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


# ## Callbacks

# In[ ]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=4, 
                                            verbose=1, 
                                            factor=0.5,
                                            min_lr=0.00001)

checkpoint = ModelCheckpoint("Zebra_Auto_Model_C.hdf5",
                             save_best_only=True,
                             monitor='val_loss',
                             mode='min')

early_stopping = EarlyStopping(monitor='val_loss',
                              patience=8,
                              verbose=1,
                              mode='min',
                              restore_best_weights=True)


# ### Custom Image Sample Callback
# 
# Here is a custom callback I made named ImgSample. It tests the result of the autoencoder after every epoch by desplaying an sample image. The goal is to have the dancer come into focus as clearly as possible.

# In[ ]:


class ImgSample(Callback):

    def __init__(self):
       super(Callback, self).__init__() 

    def on_epoch_end(self, epoch, logs={}):
        sample_img = X_train[50]
        sample_img = sample_img.reshape(1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
        sample_img = self.model.predict(sample_img)[0]
        imshow(sample_img)
        plt.show()


imgsample = ImgSample()
model_callbacks = [learning_rate_reduction, checkpoint, early_stopping, imgsample]
imshow(X_train[50])


# ## Train the Autoencoder

# In[ ]:


get_ipython().run_cell_magic('time', '', 'if PART_2_SKIP == False:\n    model.fit(X_train, X_train,\n              epochs=40, \n              batch_size=32,\n              verbose=2,\n              validation_split=0.05,\n            callbacks=model_callbacks)\nelse:\n    model = load_model(\'../input/Zebra_Auto_Model_C.hdf5\')\n    model.load_weights("../input/Zebra_Auto_Weights_C.hdf5")')


# ## Sample the Autoencoder Results
# 
# If the reconstructions look pretty close to the originals, then the autoencoder works.

# In[ ]:


decoded_imgs = model.predict(X_train)


# In[ ]:


plt.figure(figsize=(20, 4))
for i in range(5,10):
    # original
    plt.subplot(2, 10, i + 1)
    plt.imshow(X_train[i])
    plt.axis('off')
 
    # reconstruction
    plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(decoded_imgs[i])
    plt.axis('off')
 
plt.tight_layout()
plt.show()


# ## Save Models and Create Encoded Dataset

# In[ ]:


model.save('Zebra_Auto_Model_C.hdf5')
model.save_weights("Zebra_Auto_Weights_C.hdf5")


# In[ ]:


encoder = Encoder()
decoder = Decoder()

encoder.load_weights("Zebra_Auto_Weights_C.hdf5", by_name=True)
decoder.load_weights("Zebra_Auto_Weights_C.hdf5", by_name=True)

decoder.save('Zebra_Decoder_Model_C.hdf5') 
encoder.save('Zebra_Encoder_Model_C.hdf5')

decoder.save_weights("Zebra_Decoder_Weights_C.hdf5")
encoder.save_weights("Zebra_Encoder_Weights_C.hdf5")


# In[ ]:


encoder_imgs = encoder.predict(X_train)
print(encoder_imgs.shape)
np.save('Encoded_Zebra_C.npy',encoder_imgs)


# ## Decode a Sample to Double Check Results
# 
# If the encoder and decoder models are working correctly, the zebra should appear like in the reconstruction of the autoencoder above.

# In[ ]:


decoded_imgs = decoder.predict(encoder_imgs[0:11])

plt.figure(figsize=(20, 4))
for i in range(5,10):
    # reconstruction
    plt.subplot(1, 10, i + 1)
    plt.imshow(decoded_imgs[i])
    plt.axis('off')
 
plt.tight_layout()
plt.show()


# ## Part 2 Results
# 
# The results, like with the dancer video, are really good. Again there is only really a bit of blurriness around the hands after decoding.
# 
# ### Possible Improvements and Thoughts
# 
# - The autoencoder still works alright but is showing it's limits with color data.
# 
# - The Autoencoder could be used to make a much much larger training set. Even if the uncompressed images get to big for the memory limit, it is possible to just train the autoencoder on a subset of the images then compress the whole set after. A 128 array is not that big, I don't foresee resource exhaustion errors being an major issue, even for much larger datasets.
# 

# # AI Dance Part 3: Train AI w/ RNNs
# 
# If you have read any of my text generating notebooks or know text generating AIs this next part will be familiar with you. If not, here is one of my related notebooks: 
# 
# The Pythonic Python Script for Making Monty Python Scripts: https://www.kaggle.com/valkling/pythonicpythonscript4makingmontypythonscripts
# 
# For the zebra AI, the technique is pretty much the same. We will use our compressed pictures to make n length sequences as input that the model will use to predict the n+1 frame in the sequence. The differences are:
# 
# - The input/outputs will not be in one-hot encoding but rather an array of floats between 0 and 1.
# 
# - We will need a larger brain for our model to make it work.
# 
# - We will need to decode the results after to turn them into a usable video.
# 
# Once again, I will be sticking with the same structure as in the dance AI notebook. Keeping with it until it hits something it cannot handle.
# 

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


if PART_2_SKIP:
    Dance_Data = np.load('../input/Encoded_Zebra_C.npy')
else:
    Dance_Data = encoder_imgs

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


checkpoint = ModelCheckpoint("AI_Zebra_RNN_Model_C.hdf5",
                             monitor='loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

early = EarlyStopping(monitor="loss",
                      mode="min",
                      patience=3,
                     restore_best_weights=True)

model_callbacks = [checkpoint, early]


# ## Train RNN Model
# 
# One change I am making to this model over the black and white version is to cut back the epochs to 40. Not only will this save time but since the zebra was so overfit the last 40 epochs was just excessive. I hope that cutting back the epochs might yield more interesting results in the color version.

# In[ ]:


get_ipython().run_cell_magic('time', '', "if PART_3_SKIP == False:\n    model.fit(X_train, Y_train,\n              batch_size=64,\n              epochs=40,\n              verbose=2,\n              callbacks = model_callbacks)\nelse:\n    model = load_model('../input/AI_Zebra_RNN_Model_C.hdf5')\n    model.load_weights('../input/AI_Zebra_RNN_Weights_C.hdf5')")


# In[ ]:


model.save("AI_Zebra_RNN_Model_C.hdf5")
model.save_weights('AI_Zebra_RNN_Weights_C.hdf5')


# ## Generate New Computer Generated Dances
# 
# This block generates new video sequences in the style of the video of VIDEO_LENGTH size in frames. It takes a random seed pattern from the training set, predicts the next frame, adds it to the end of the pattern and drops the first frame of the pattern and predicts on the new pattern and so forth. The default VIDEO_LENGTH of 6000 frames is 5 minutes of video at 20 FPS.
# 
# Pretty much the AI will try to accurately duplicate the video but inevitably makes errors, and those errors compound, but is still trained well enough that it ends up making similar, but not quite the same, dances.
# 
# The LOOPBREAKER is used to add noise to the prediction pattern, replacing a random frame in the pattern with a random frame in the Dance_Data after every LOOPBREAKER frames. This noise can be used to force the AI to change up what it is doing. This can stop undertrained models from looping or overtrained models from duplication the training data too closely. Setting it too low, on the other hand, can cause the results to distort more. It is worth playing around with this setting and is a quick and dirty way to adjust the dance output post training.

# In[ ]:


get_ipython().run_cell_magic('time', '', "VIDEO_LENGTH = 6000\nLOOPBREAKER = 10\n\nx = np.random.randint(0, X_train.shape[0]-1)\npattern = X_train[x]\noutp = np.zeros((VIDEO_LENGTH, INPUT_SIZE), dtype='float32')\nfor t in range(VIDEO_LENGTH):\n    x = np.reshape(pattern, (1, pattern.shape[0], pattern.shape[1]))\n    pred = model.predict(x)\n    result = pred[0]\n    outp[t] = result\n    new_pattern = np.zeros((SEQUENCE_LENGTH, INPUT_SIZE), dtype='float32') \n    new_pattern[0:SEQUENCE_LENGTH-1] = pattern[1:SEQUENCE_LENGTH]\n    new_pattern[-1] = result\n    pattern = np.copy(new_pattern)\n    ####loopbreaker####\n    if t % LOOPBREAKER == 0:\n        pattern[np.random.randint(0, SEQUENCE_LENGTH-10)] = Y_train[np.random.randint(0, Y_train.shape[0]-1)]")


# ## Output the Dance
# 
# Before we can save the video, we need to decode the frames back into images using the decoder we made in part 2.

# In[ ]:


if PART_2_SKIP:
    Decoder = load_model('../input/Zebra_Decoder_Model_C.hdf5')
    Decoder.load_weights('../input/Zebra_Decoder_Weights_C.hdf5')
else:
    Decoder = load_model('Zebra_Decoder_Model_C.hdf5')
    Decoder.load_weights('Zebra_Decoder_Weights_C.hdf5')

AI_Output = Decoder.predict(outp)
AI_Output.shape


# In[ ]:


IMG_HEIGHT = AI_Output[0].shape[0]
IMG_WIDTH = AI_Output[0].shape[1]

for row in AI_Output[0:10]:
    imshow(row)
    plt.show()


# ## Save Video

# In[ ]:


video = cv2.VideoWriter('AI_Zebra_Video.avi', cv2.VideoWriter_fourcc(*"XVID"), 20.0, (IMG_WIDTH, IMG_HEIGHT),True)

for img in AI_Output:
    img = img * 255
    img = img.astype('uint8')
    video.write(img)
    cv2.waitKey(50)
    
video.release()


# # Part 3 Results
# 
# The zebra comes out pretty accurately. It's movements are slower and less interesting than the dance video. The arms disappear a bit when the zebra moves them quickly. You can see some freezing when the zebra stops moving for a bit, probably because the movements during those frames are so minute that the computer treats them as the same.
# 
# ### Possible Improvements and Thoughts
# 
# - The video is pretty overfit and could use more training data. However, the zebra video is pretty much tapped out at this point. With the often slow movements in the video, taking more frames would just lead to a much heavier demand on the models without any real gain.
# 
# - The autoencoder is showing it's limits with this color data. The arms are getting too blurry and the stopsign is a mild red blur. I think that now is the time to start encoding to a larger numpy array.
# 

# # Conclusion
# 
# The models still did a good job even with color channels. The results are boring like with the black and white data. I am a bit disappointed that there was not any "rainbow glitches" like with gans and other color image predictions. This was likely due to the video still being predominantly  tri-colored( black, white, and green). Using more complex color videos I hope will make some interesting "rainbow glitches" and larger training set to take care of overfitting.
# 
# If you enjoyed and learned something from this notebook, please like, comment, and check out some of my other coding projects on Kaggle and Github.

# In[ ]:


from IPython.display import Image
Image("../input/Zebra_Test_Image.jpg")

