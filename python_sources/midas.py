#!/usr/bin/env python
# coding: utf-8

# **Importing Libraries**

# In[ ]:


import numpy as np
import os
from os.path import isfile
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, Flatten, Conv2D, BatchNormalization, Lambda
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend
from keras.utils import np_utils
from keras.optimizers import Adam, RMSprop
from keras import regularizers
import librosa
import librosa.display
import matplotlib.pyplot as plt
import random
import numpy as np
np.random.seed(1001)
import os
import shutil
import IPython
import seaborn as sns
from scipy.io import wavfile
from tqdm import tqdm_notebook 
import IPython.display as ipd
import librosa
import numpy as np
import scipy
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.layers import (Convolution1D, Dense, Dropout, GlobalAveragePooling1D, 
                          GlobalMaxPool1D, Input, MaxPool1D, concatenate)
from keras.utils import Sequence, to_categorical
from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten,
                          GlobalMaxPool2D, MaxPool2D, concatenate, Activation)
from keras.utils import Sequence, to_categorical
from keras import backend as K


# In[ ]:


import sys
sys.stdout.write('hello')


# In[ ]:


train_folder = '../input/midasemotions/meld/train/'
val_folder = '../input/midasemotions/meld/val/'


# In[ ]:


test_folder = ''


# **Visualizations**

# In[ ]:


wav_files = os.listdir(train_folder+'happy/')
index = random.randint(0, len(wav_files)-1)
fname = train_folder+'happy/'+wav_files[index]
rate, data = wavfile.read(fname)
print("Sampling (frame) rate = ", rate)
print("Total samples (frames) = ", data.shape)
plt.plot(data, '-', )


# In[ ]:


plt.figure(figsize=(16, 4))
plt.plot(data[:500], '.'); plt.plot(data[:500], '-');


# In[ ]:


plt.figure(figsize=(10, 5))
librosa.display.specshow(data.T, y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Test Melspectogram')
plt.tight_layout()


# **Mapping of given emotions to constants**

# In[ ]:


emotions = {'happy':0, 'sad':1, 'disgust':2, 'neutral':3, 
               'fear':4}


reverse_emotions = {v: k for k, v in emotions.items()}
print(reverse_emotions)


# In[ ]:


class Config(object):
    def __init__(self,
                 sampling_rate=16000, audio_duration=2, n_classes=41,
                 use_mfcc=False, n_folds=10, learning_rate=0.0001, 
                 max_epochs=50, n_mfcc=20):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)
        else:
            self.dim = (self.audio_length, 1)


# **Normalization Function**

# In[ ]:


def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5


# **Lists for storing features corresponding to .wav files and the corresponding labels**

# In[ ]:


X=[]
Y=[]


# **Function for populating training data**

# > Due to non-uniform distribution of audio files corresponding to a given label, I have chosen to apply data augmentation which mainly includes mainly includes the following methods:
# >1. Noise Injection
# >2. Changing Pitch
# >3. Changing Speed

# In[ ]:


def prepare_train_data( config, data_dir):
   # X = np.empty(shape=(size, config.dim[0], config.dim[1], 1))
   # Y = np.zeros(shape = (size,5))
   # print(X.shape)
    input_length = config.audio_length
    index = 0
    for emotion_dir in os.listdir(data_dir):
        label = emotion_dir
        audio_dir = data_dir+ emotion_dir
        count  = 0
        if label == 'neutral':
            count = 0
        elif label == 'happy':
            count = 3
        else :
            count = 15
        for file in os.listdir(audio_dir):
            
            file_path = audio_dir+'//' + file
            data, _ = librosa.core.load(file_path, sr=config.sampling_rate, res_type="kaiser_fast")
            
            # Random offset / Padding
            if len(data) > input_length:
                max_offset = len(data) - input_length
                offset = np.random.randint(max_offset)
                data = data[offset:(input_length+offset)]
            else:
                if input_length > len(data):
                    max_offset = input_length - len(data)
                    offset = np.random.randint(max_offset)
                else:
                    offset = 0
                data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
            d = data
            data = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
            data = np.expand_dims(data, axis=-1)
            #X[index,] = data
            X.append(data)
            l = emotions[label]
            #Y[index][l] = 1
            z= np.zeros(5)
            z[l]=1
            Y.append(z)
            #print(Y[index])
            #print(index)
            #index+=1
            noise_examples = count/3 
            while noise_examples :
                noise_factor = random.randint(0,10)%100
                noise = np.random.randn(len(d))
                augmented_data = d + noise_factor * noise
                data = augmented_data
                if len(data) > input_length:
                    max_offset = len(data) - input_length
                    offset = np.random.randint(max_offset)
                    data = data[offset:(input_length+offset)]
                else:
                    if input_length > len(data):
                        max_offset = input_length - len(data)
                        offset = np.random.randint(max_offset)
                    else:
                        offset = 0
                data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
                data = librosa.feature.mfcc(augmented_data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
                data = np.expand_dims(data, axis=-1)
                X.append(data)
                Y.append(z)
                noise_examples-=1
             
            pitch_examples = count/3 
            while pitch_examples : 
                    pitch_factor = random.randint(0, 20)/10
                    data = librosa.effects.pitch_shift(d, config.sampling_rate, pitch_factor)
                    if len(data) > input_length:
                        max_offset = len(data) - input_length
                        offset = np.random.randint(max_offset)
                        data = data[offset:(input_length+offset)]
                    else:
                        if input_length > len(data):
                            max_offset = input_length - len(data)
                            offset = np.random.randint(max_offset)
                        else:
                            offset = 0
                        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
                    data = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
                    data = np.expand_dims(data, axis=-1)
                    X.append(data)
                    Y.append(z)
                    pitch_examples -= 1
             
            time_examples = count/3  
            while time_examples :
                speed_factor =random.randint(1,2)
                data = librosa.effects.time_stretch(d, speed_factor)
                if len(data) > input_length:
                    max_offset = len(data) - input_length
                    offset = np.random.randint(max_offset)
                    data = data[offset:(input_length+offset)]
                else:
                    if input_length > len(data):
                        max_offset = input_length - len(data)
                        offset = np.random.randint(max_offset)
                    else:
                        offset = 0
                data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
                data = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
                data = np.expand_dims(data, axis=-1)
                X.append(data)
                Y.append(z)
                #print(Y[len(Y)-1])
                time_examples -= 1  
           #print(X[len(X)-1].shape)    
    #return X,Y


# **Function for populating validation data**

# In[ ]:


def prepare_val_data( config, data_dir,size):
    X = np.empty(shape=(size, config.dim[0], config.dim[1], 1))
    Y = np.zeros(shape = (size,5))
    print(X.shape)
    input_length = config.audio_length
    index = 0
    for emotion_dir in os.listdir(data_dir):
        
        label = emotion_dir
        audio_dir = data_dir+ emotion_dir
        for file in os.listdir(audio_dir):
            file_path = audio_dir+'//' + file
            data, _ = librosa.core.load(file_path, sr=config.sampling_rate, res_type="kaiser_fast")
            
            # Random offset / Padding
            if len(data) > input_length:
                max_offset = len(data) - input_length
                offset = np.random.randint(max_offset)
                data = data[offset:(input_length+offset)]
            else:
                if input_length > len(data):
                    max_offset = input_length - len(data)
                    offset = np.random.randint(max_offset)
                else:
                    offset = 0
                data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
            d = data
            data = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
            data = np.expand_dims(data, axis=-1)
            X[index,] = data
            l = emotions[label]
            Y[index][l] = 1
    print('Done')        
    return X,Y       


# In[ ]:


config = Config(sampling_rate=16000, audio_duration=1, n_folds=10, 
                learning_rate=0.0001, use_mfcc=True, n_mfcc=50)


# In[ ]:


prepare_train_data(config,train_folder)


# In[ ]:


prepare_train_data(config,val_folder)


# In[ ]:


#Xval , Yval = prepare_val_data(config,val_folder,830)


# In[ ]:


Y_train = np.asarray(Y)


# In[ ]:


X_train = np.asarray(X)


# In[ ]:


mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean)/std


# In[ ]:


from sklearn.model_selection import train_test_split
Xtrain, Xval, Ytrain, Yval = train_test_split(X_train,Y_train, test_size = 0.3)


# In[ ]:





# In[ ]:





# **Model Definiton and Architecture**

# In[ ]:


def get_2d_dummy_model(config):
    
    nclass = config.n_classes
    
    inp = Input(shape=(config.dim[0],config.dim[1],1))
    x = GlobalMaxPool2D()(inp)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model


def get_2d_conv_model(config):

    inp = Input(shape=(config.dim[0],config.dim[1],1))
    x = Convolution2D(32, (3,3), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(64, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(128, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(256, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    out = Dense(5, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model


# In[ ]:


model = get_2d_conv_model(config)
    


# In[ ]:


print(model.summary())


# In[ ]:





# In[ ]:


history = model.fit(X_train, Y_train, validation_data=(Xval, Yval), epochs=10)


# In[ ]:


plt.plot(history.history['loss']) 


# In[ ]:


plt.plot(history.history['val_loss'])


# In[ ]:


model.save('model.h5')

