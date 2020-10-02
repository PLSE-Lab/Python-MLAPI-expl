#!/usr/bin/env python
# coding: utf-8

# # Motivation

# Treating Audio Data as Image can have their own benefits, as image data can be used in Deep Learning and popular and effective methods like Convolutional Neural Networks can be applied to the audio data which is represented as an image. This kernel is basically a Data Processing Step which shows how to convert and Audio Data to image by extracting features. 
# 
# Once you have an image, you can apply various image based models and Convolutional Networks for various AI related tasks. One such use case is here is of classification of audio data.
# 

# # References and Credits:
# 
# The audio data analysis is motivated by the following resources
# https://www.kaggle.com/daisukelab/cnn-2d-basic-solution-powered-by-fast-ai
# http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
# 
# Python library **Librosa** is used for audio feature extraction.
# Data used for demonstration is from the freesound audio tagging competition. https://www.kaggle.com/c/freesound-audio-tagging-2019 
# 
# ### Spectrogram
# A spectrogram is a visual representation of the spectrum of frequencies of a signal as it varies with time. A spectrogram is a visual representation of the spectrum of frequencies in a sound or other signal as they vary with time or some other variable.
# https://en.wikipedia.org/wiki/Spectrogram
# 
# ### Convolutional Models
# Convolutional models are built with reference to deeeplearning.ai specialization on Coursera, taught by Prof. Andrew Ng. Thanks!

# # Section:1 Audio Data Extraction

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import os
import IPython
import IPython.display
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from PIL import Image               # to load images
from IPython.display import display
import time
print(os.listdir("../input"))


# In[ ]:


# Special thanks to https://github.com/makinacorpus/easydict/blob/master/easydict/__init__.py
class EasyDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(EasyDict, self).pop(k, d)


# Following are some of the parameters which are defined for audio data handling. 
# 
# Sampling Rate is perhaps the most important one which talks about how many times in a second, the data is collected (Hertz.) Usually this value is 44100 for audio data.

# In[ ]:


conf = EasyDict()
conf.sampling_rate = 44100
conf.duration = 2
conf.hop_length = 347 # to make time steps 128
conf.fmin = 20
conf.fmax = conf.sampling_rate // 2
conf.n_mels = 128
conf.n_fft = conf.n_mels * 20
conf.samples = conf.sampling_rate * conf.duration


# ## Functions to analyze Audio Data
# 
# Some of the functions developed for the task are
# 
# * read_audio - Returns the Y values of audio file.
# * audio_to_melspectrogram - Returning a mel spectrogram from raw audio.
# * mono_to_color

# In[ ]:


import librosa
import librosa.display
def read_audio(conf, pathname, trim_long_data):
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    # trim silence
    if 0 < len(y): # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y) # trim, top_db=default(60)
    # make it unified length to conf.samples
    if len(y) > conf.samples: # long enough
        if trim_long_data:
            y = y[0:0+conf.samples]
    else: # pad blank
        padding = conf.samples - len(y)    # add padding at both ends
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')
    return y


# ## Reading train_curated data files
# 
# As you have seen above, the list of curated files are available in the data file called train_curated.csv. Lets examine the contents of the file. It tells about the name of the file (fname) and labels attached to it. (Bark, Raindrop, etc.) The objective of this competition is to build a model which can predict the labels on unknown data.

# In[ ]:


start_time_data_processing = time.time()

DATA = Path('../input')
CSV_TRN_CURATED = DATA/'train_curated.csv'
CSV_TRN_NOISY = DATA/'train_noisy.csv'
CSV_SUBMISSION = DATA/'sample_submission.csv'
TRN_CURATED = DATA/'train_curated'
TRN_NOISY = DATA/'train_noisy'
TEST = DATA/'test'

WORK = Path('work')
IMG_TRN_CURATED = WORK/'image/trn_curated'
IMG_TRN_NOISY = WORK/'image/train_noisy'
IMG_TEST = WORK/'image/test'

df_train_curated = pd.read_csv(CSV_TRN_CURATED)
print(df_train_curated.head(10))
# Collecting various data frames for further processing.
df_bark = df_train_curated.loc[df_train_curated['labels'] == 'Bark'][1:5]
df_run = df_train_curated.loc[df_train_curated['labels'] == 'Run'][1:5]


# Following function gives the Y(amplitude) values of the wave. We can examine that when we take data for only two seconds, the length is 88200 which is 2 times sampling rate. Otherwise the entire length of soundclip is returned.

# In[ ]:


buzz_1_file =  DATA/'train_curated'/'02f54ef1.wav'
y_2_secs = read_audio(conf, buzz_1_file, trim_long_data = True)
y_full = read_audio(conf, buzz_1_file, trim_long_data = False)
print(len(y_full))
print(len(y_2_secs))
print(y_full.shape[0]/44100)


# ### Looking at shape of Data for a 'Bark' audio file.

# In[ ]:


bark_file = DATA/'train_curated'/'0006ae4e.wav'
y_bark = read_audio(conf, buzz_1_file, trim_long_data = False)
print(y_bark.shape[0]/ 44100)


# ## Feature Extraction using MFCC 
# 
# 
# **Mel Frequency Cepstral Coefficient (MFCC) **
# 
# The first step in speech analytics is to extract features i.e. identify the components of the audio signal that are good for identifying the linguistic content and discarding all the other stuff which carries information like background noise, emotion etc.
# 
# http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
# 
# **What is the Mel scale? **
# 
# The Mel scale relates perceived frequency, or pitch, of a pure tone to its actual measured frequency. Humans are much better at discerning small changes in pitch at low frequencies than they are at high frequencies. Incorporating this scale makes our features match more closely what humans hear.
# 
# The librosa library provides functions for getting the melspectrogram from the audio data.

# In[ ]:


def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(audio, 
                                                 sr=conf.sampling_rate,
                                                 n_mels=conf.n_mels,
                                                 hop_length=conf.hop_length,
                                                 n_fft=conf.n_fft,
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

def show_melspectrogram(conf, mels, title='Log-frequency power spectrogram'):
    librosa.display.specshow(mels, x_axis='time', y_axis='mel', 
                             sr=conf.sampling_rate, hop_length=conf.hop_length,
                            fmin=conf.fmin, fmax=conf.fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()

def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):
    x = read_audio(conf, pathname, trim_long_data)
    mels = audio_to_melspectrogram(conf, x)
    if debug_display:
        IPython.display.display(IPython.display.Audio(x, rate=conf.sampling_rate))
        show_melspectrogram(conf, mels)
    return mels


# ### Looking at data after reading mel features
# 
# The following block examines the various features of the audio clip. In this case, the first row which contains the barking clip. We are looking to derive 128 mel features or n_mels as supplied in the conf dict. The resulting data frame shape is defined as (n_mels, ). You can verify the first dimension is 128 in this case.

# In[ ]:


TRN_CURATED = DATA/'train_curated'
x = read_audio(conf, TRN_CURATED/'0006ae4e.wav', trim_long_data=False)
print(x.shape)
mels = audio_to_melspectrogram(conf, x)
print(mels.dtype)
print(mels.shape)
print(mels)


# The following chunk of code draws the spectrogram image as provided by the librosa library. The corresponding audio files are also embedded here. We have used the function *librosa.display.specshow* to display the spectrogram. By looking at the functions, you wull notice that a number of functions from librosa library is used to extract the mel features and display them in image form.
# 
# Lets look at how **bark** and **buzz** categories look like.

# In[ ]:


bark = read_as_melspectrogram(conf, TRN_CURATED/'0006ae4e.wav', trim_long_data=False, debug_display=True)
buzz = read_as_melspectrogram(conf, TRN_CURATED/'02f54ef1.wav', trim_long_data=False, debug_display=True)


# Printing shape of the two files. The second dimension vary, this is probably has to do with the length of the audio clip. First dimension always is the number of mels.

# In[ ]:


print(bark.shape)
print(buzz.shape)


# ### Converting numerical values to image
# 
# The array of numbers are converted into a image file. By simply stacking the X value in R,G and B dimensions,we can get the image. Notice that standardization of data is done here and later it is multiplied by 255.

# In[ ]:


def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    Xstd = (X - mean) / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Scale to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

def convert_wav_to_image(df, source, img_dest):
    X = []
    for i, row in tqdm_notebook(df.iterrows()):
        x = read_as_melspectrogram(conf, source/str(row.fname), trim_long_data=False)
        x_color = mono_to_color(x)
        X.append(x_color)
    return df, X

def convert_wav_to_cropped_image(df, source, img_dest):
    X = []
    for i, row in tqdm_notebook(df.iterrows()):
        x = read_as_melspectrogram(conf, source/str(row.fname), trim_long_data=False)
        x_color = mono_to_color(x)
        # - - - - - - - - - - #
        x_color = Image.fromarray(x_color)
        time_dim, base_dim = x_color.size
        crop_x = random.randint(0, time_dim - base_dim)
        x_cropped = x_color.crop([crop_x, 0, crop_x+base_dim, base_dim]) 
        # - - - - - - - - - - #
        X.append(x_cropped)
    return df, X


# ### Examining the shape of array before and after converting to image

# In[ ]:


print(bark.shape)
bark_image = mono_to_color(bark)
print(bark_image.shape)


# ### Cropping the image
# 
# We have used PIL library to manipulate and show the image. As in image data, we require dimensions of the image to be fixed, we are taking the number of mels to fix the other dimension. In this way, a cropped image can be produced as below. Feel free to run the following cell a few times to see that every time, the cropped image is a different one as it is pciked from the original image.

# In[ ]:


import random
x = Image.fromarray(bark_image)
display(x)
time_dim, base_dim = x.size
crop_x = random.randint(0, time_dim - base_dim)
x_cropped = x.crop([crop_x, 0, crop_x+base_dim, base_dim]) 
display(x_cropped)


# Putting it altogether with the help of **display_cropped_image** function. We will be looking at the various types of image of activities, e.g. Bark and Run respectively as below. You can observe the similarities/ dissimilaritites yourself.

# In[ ]:


def get_cropped_image(conf, path_of_image, display_image =True):
    mel_spec_gram = read_as_melspectrogram(conf, path_of_image, trim_long_data=False, debug_display=False)
    img_array = mono_to_color(mel_spec_gram)
    img = Image.fromarray(img_array)
    time_dim, base_dim = img.size
    cropped = random.randint(0, time_dim - base_dim)
    cropped_image = img.crop([cropped, 0, cropped+base_dim, base_dim]) 
    if display:
        display(cropped_image)
    return(cropped_image)


# ### Looking at various Bark files:

# In[ ]:


x1 = get_cropped_image(conf, TRN_CURATED/df_bark.iloc[0,0])
x2 = get_cropped_image(conf, TRN_CURATED/df_bark.iloc[1,0])
x3 = get_cropped_image(conf, TRN_CURATED/df_bark.iloc[2,0])


# ### Looking at various Run files:

# In[ ]:


get_cropped_image(conf, TRN_CURATED/df_run.iloc[0,0])
get_cropped_image(conf, TRN_CURATED/df_run.iloc[1,0])
get_cropped_image(conf, TRN_CURATED/df_run.iloc[2,0])


# ## Getting the Data Set for Training

# In[ ]:


df_test_bark, X_train_bark = convert_wav_to_cropped_image(df_bark, source=TRN_CURATED, img_dest=IMG_TRN_CURATED)
df_test_run, X_train_run = convert_wav_to_cropped_image(df_run, source=TRN_CURATED, img_dest=IMG_TRN_CURATED)


# ### Doing a quick dimension check

# In[ ]:


# Just a quick dimension check here.
test_np_bark = np.vstack(X_train_bark)
test_np_run = np.vstack(X_train_run)
print(test_np_bark.shape)
print(test_np_run.shape)
end_time_data_processing = time.time()


# # Section: 2 Building a convolutional model
# 
# Getting a small dataset for building a toy model with 100 samples.

# In[ ]:


df_train_curated = df_train_curated.sample(100) # taking a sample data set of 100 examples.
df_train_curated, X_train_curated = convert_wav_to_cropped_image(df_train_curated, source=TRN_CURATED, img_dest=IMG_TRN_CURATED)
np_train_curated = np.vstack(X_train_curated)
X_train_curated = np_train_curated.reshape(-1, 128, 128, 3) # Reshaping the training dataset.
print(X_train_curated.shape)


# In[ ]:


X_train = X_train_curated
df_train = df_train_curated


# You can also extract the **Noisy** Dataset and append to the training examples.

# In[ ]:


# Getting the labels required for submission, there are eighty of them.
df_test = pd.read_csv('../input/sample_submission.csv')
label_columns = list( df_test.columns[1:] )
label_mapping = dict((label, index) for index, label in enumerate(label_columns))
#label_mapping
def split_and_label(rows_labels):
    row_labels_list = []
    for row in rows_labels:
        row_labels = row.split(',')
        labels_array = np.zeros((80))
        
        for label in row_labels:
            index = label_mapping[label]
            labels_array[index] = 1
        
        row_labels_list.append(labels_array)
    
    return row_labels_list


# In[ ]:


train_curated_labels = split_and_label(df_train['labels'])
for f in label_columns:
    df_train[f] = 0.0 # This adds all the labels as column names.

df_train[label_columns] = train_curated_labels


# ## Train inputs and Response are available now for further processing.

# In[ ]:


Y_train = np.vstack(train_curated_labels)
print(Y_train.shape)
print(X_train.shape)


# ## Getting test Data

# In[ ]:


df_test, X_test = convert_wav_to_cropped_image(df_test, source=TEST, img_dest=IMG_TEST)
np_test_small = np.vstack(X_test)
print(np_test_small.shape)
X_test = np_test_small.reshape(-1, 128, 128, 3)
print(X_test.shape)


# In[ ]:


print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))


# ## Importing Keras and Other libraries for Building Convolutional Models

# In[ ]:


import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
#from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Building the convolutional model with 4 convolutional layers

# In[ ]:


def Audio2DConvModel(input_shape, classes):
    """
    Implementation of the Basic Model.
    Arguments:
    input_shape -- shape of the images of the dataset
    Returns:
    model -- a Model() instance in Keras
    """
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(64, (16, 16), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    #X = MaxPooling2D((2, 2), name='max_pool')(X)
    X = MaxPooling2D(pool_size=(2, 2), name = 'maxpool_0')(X)
    X = Dropout(rate=0.1)(X)
# -------------------------------------------------------------------------------------
    X = ZeroPadding2D((3, 3))(X)
    X = Conv2D(32, (8, 8), strides = (1, 1), name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), name = 'maxpool_1')(X)
    # ------------------------------------------------------------------------------------
    X = ZeroPadding2D((3, 3))(X)
    X = Conv2D(16, (4, 4), strides = (1, 1), name = 'conv2')(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='maxpool_2')(X)
    #------------------------------------------------------------
    X = ZeroPadding2D((3, 3))(X)
    X = Conv2D(16, (2, 2), strides = (1, 1), name = 'conv3')(X)
    X = BatchNormalization(axis = 3, name = 'bn32')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='maxpool_3')(X)
    #X = AveragePooling2D(pool_size=(2, 2), name = 'avg_pool_1')(X)
    # -------------------------------------------------------------------------------------    
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc_softmax' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    model = Model(inputs = X_input, outputs = X, name='Audio2DConvModel')
    return model


# ## Building Convolutional Model and looking at summary

# In[ ]:


basic_conv_model = Audio2DConvModel(X_train.shape[1:], 80)
basic_conv_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
basic_conv_model.fit(X_train/255, Y_train, epochs=1, batch_size=64)
basic_conv_model.summary()


# ## Generating predictions

# In[ ]:


y_hat = basic_conv_model.predict(X_test/255)
df_test[label_columns] = y_hat
print(df_test.head())


# That's it. Thanks for reading. Currently I have used only 100 observations of curated data. This can be enhanced with noisy dataset. Also deep layers can be embedded and regularization can be done as well. Stay tuned!
