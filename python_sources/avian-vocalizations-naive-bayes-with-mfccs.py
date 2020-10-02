#!/usr/bin/env python
# coding: utf-8

# Trains a Naive Bayes classifier on [melspectrograms](https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html) and [Mel-frequency Cepstral Coefficients (MFCCs)](https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html) generated from the [Avian Vocalizations from CA & NV, USA](https://www.kaggle.com/samhiatt/xenocanto-avian-vocalizations-canv-usa) dataset, to be used as a benchmark classification model.
# 
# This version was forked from [Avian Vocalizations: Benchmark Naive Bayes](https://www.kaggle.com/samhiatt/avian-vocalizations-benchmark-naive-bayes?scriptVersionId=19012978) and adds MFCCs to the feature space.

# In[ ]:


""" `imageio_ffmpeg` contains a pre-built `ffmpeg` binary, needed for mp3 decoding by `librosa`. 
    It is installed as a custom package on Kaggle. If no `ffmpeg` binary is found in `/usr/local/bin` 
    then create a softlink to the `imageio_ffmpeg` binary. 
"""
import os
if not os.path.exists("/usr/local/bin/ffmpeg"): 
    import imageio_ffmpeg
    os.link(imageio_ffmpeg.get_ffmpeg_exe(), "/usr/local/bin/ffmpeg")


# In[ ]:


import pandas as pd
import numpy as np
import librosa as lr
# from librosa import feature
from librosa.display import specshow
from glob import glob
import os
from IPython.display import Audio
from matplotlib import pyplot as plt
# from zipfile import ZipFile
from tqdm import tqdm

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import keras
from keras.utils import to_categorical

import re
def parse_shape(shape_str):
    a,b = re.search('\((\d+), (\d+)\)',shape_str).groups()
    return int(a), int(b)

def log_clipped(a):
    return np.log(np.clip(a,.0000001,a.max()))

def get_full_path(sample): return os.path.join(sounds_dir, sample['file_name'])
sounds_dir = "../input/xenocanto-avian-vocalizations-canv-usa/xeno-canto-ca-nv/"
# sounds_dir = "../input/xeno-canto-ca-nv/"

df = pd.read_csv("../input/xenocanto-avian-vocalizations-canv-usa/xeno-canto_ca-nv_index.csv")
# df = pd.read_csv("../input/xeno-canto_ca-nv_index.csv")
files_list = glob(os.path.join(sounds_dir,"*.mp3"))
print("%i mp3 files in %s"%(len(files_list), sounds_dir))
print("%i samples in index."%len(df))
df.head(2)


# In[ ]:


melspec_dir = "../input/avian-vocalizations-melspectrograms-log-norm/"
melspec_features_dir = melspec_dir + "/melspectrograms_logscaled_normalized/features"

shapes_df = pd.read_csv("../input/avian-vocalizations-spectrograms-and-mfccs/feature_shapes.csv")
# shapes_df.head(2) 

label_encoder = LabelEncoder().fit(df['english_cname'] )
n_classes = len(label_encoder.classes_)
print("The dataset contains %i distinct species labels."%n_classes)
print("%i mp3s found in %s"%(len(glob(melspec_features_dir+"*.mp3")), melspec_features_dir))


# In[ ]:


train_df = pd.read_csv("../input/avian-vocalizations-partitioned-data/train_file_ids.csv",
                       index_col=0)
test_df = pd.read_csv("../input/avian-vocalizations-partitioned-data/test_file_ids.csv",
                      index_col=0)
X_train = np.array(train_df.index)
y_train = np.array(train_df.label)
print("Training data shape:",X_train.shape)
X_test = np.array(test_df.index)
y_test = np.array(test_df.label)
print("Test data shape:    ",X_test.shape)


# #### Create a Data Generator to generate fixed-length samples from random windows within clips

# In[ ]:


sg_dir = "../input/avian-vocalizations-spectrograms-and-mfccs/melspectrograms/features/"
sglog_dir = "../input/avian-vocalizations-melspectrograms-log-norm/melspectrograms_logscaled_normalized/features/"
mfcc_dir = "../input/avian-vocalizations-spectrograms-and-mfccs/mfccs/features/"

class AudioFeatureGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size, n_frames=128, n_channels=1,
                 n_classes=10, shuffle=False, seed=37):
        'Initialization'
        self.n_frames = n_frames
        self.dim = (128, self.n_frames)
        self.batch_size = batch_size
        self.labels = {list_IDs[i]:l for i,l in enumerate(labels)}
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.seed = seed
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Update indexes, to be called after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.seed(self.seed)
            self.seed = self.seed+1 # increment the seed so we get a different batch.
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        #X = np.empty((self.batch_size, *self.dim, self.n_channels))
        X = np.empty((self.batch_size, 128+20, self.dim[1], self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=int) # one-hot encoded labels

        for i, ID in enumerate(list_IDs_temp):
            sg_lognorm = np.memmap(sglog_dir+'XC%s_melspectrogram_logscaled_normalized.dat'%ID, 
                    shape=parse_shape(shapes_df[shapes_df.file_id==ID]['melspectrogram_shapes'].values[0]),  
                    dtype='float32', mode='readonly')
#             sg = np.memmap(sg_dir+'XC%s_melspectrogram.dat'%ID, 
#                     shape=parse_shape(shapes_df[shapes_df.file_id==file_id]['melspectrogram_shapes'].values[0]),  
#                     dtype='float32', mode='readonly')
            mfcc = np.memmap(mfcc_dir+'XC%s_mfcc.dat'%ID, 
                    shape=parse_shape(shapes_df[shapes_df.file_id==ID]['mfcc_shapes'].values[0]),  
                    dtype='float32', mode='readonly')
            # Normalize MFCCs
            mfcc = mfcc_scaler.transform(mfcc)
            
            # Filter out quiet frames, thanks to:
            # https://www.kaggle.com/fleanend/extract-features-with-librosa-predict-with-nb
            # Take mean amplitude M from frame with highest energy
#             m = sg[:,np.argmax(sg.mean(axis=0))].mean()
#             # Filter out all frames with energy less than 5% of M
#             mask = sg.mean(axis=0)>=m/20
#             sg = sg[:,mask]
#             sg_lognorm = sg_lognorm[:,mask]
#             mfcc = mfcc[:,mask]
            
            # Pick a random window from the sound file
            d_len = mfcc.shape[1] - self.dim[1]
            if d_len<0: # Clip is shorter than window, so pad with mean value.
                n = int(np.random.uniform(0, -d_len))
                pad_range = (n, -d_len-n) # pad with n values on the left, clip_length - n values on the right 
#                 sg_cropped = np.pad(sg, ((0,0), pad_range), 'constant', constant_values=sg.mean())
                sg_lognorm_cropped = np.pad(sg_lognorm, ((0,0), pad_range), 'constant', constant_values=0)
                mfcc_cropped = np.pad(mfcc, ((0,0), pad_range), 'constant', constant_values=0)
            else: # Clip is longer than window, so slice it up
                n = int(np.random.uniform(0, d_len))
#                 sg_cropped = sg[:, n:(n+self.dim[1])]
                sg_lognorm_cropped = sg_lognorm[:, n:(n+self.dim[1])]
                mfcc_cropped = mfcc[:, n:(n+self.dim[1])]
                
            # Stack the MFCCs and spectrograms to create a single array
            X[i,] = np.concatenate([sg_lognorm_cropped.reshape(1,128,self.dim[1],1), 
                                    mfcc_cropped.reshape(1,20,self.dim[1],1)], axis=1)
            #X[i,] = sg_lognorm_cropped.reshape(1,128,self.dim[1],1)
            # Overwrite the bottom of X with MFCCs (we don't need the low frequency bands anyway) 
            #X[i,:20] = mfcc_cropped.reshape(1,20,self.dim[1],1)
            y[i,] = to_categorical(self.labels[ID], num_classes=self.n_classes)

        return X, y
    
# MFCC statistics didn't get saved properly in pervious processing steps. Let's just fit a scaler to them here. 
from sklearn.preprocessing import StandardScaler
mfcc_scaler = StandardScaler()
for file_id in shapes_df.file_id:
    mfcc = np.memmap(mfcc_dir+'/XC%s_mfcc.dat'%file_id, 
        shape=parse_shape(shapes_df[shapes_df.file_id==file_id]['mfcc_shapes'].values[0]),  
        dtype='float32', mode='readonly')
    mfcc_scaler.partial_fit(mfcc.flatten().reshape(-1, 1))
print("MFCC scaler:",mfcc_scaler.mean_, mfcc_scaler.var_, np.sqrt(mfcc_scaler.var_))  


# In[ ]:


from itertools import islice
generator = AudioFeatureGenerator(X_train, y_train, batch_size=1, shuffle=True, seed=37, n_frames=128, n_classes=n_classes)
for g in islice(generator,0,4): # show a few examples
    for i,spec in enumerate(g[0]): 
        plt.figure(figsize=(10,4))
        spec_ax = specshow(spec.squeeze(), x_axis='time', y_axis='mel')
        plt.title(label_encoder.classes_[np.argmax(g[1][i])])
        plt.colorbar()
        plt.show()


# In[ ]:


def vis_learning_curve(learning):
    train_loss = learning.history['loss']
    train_acc = learning.history['acc']
    val_loss = learning.history['val_loss']
    val_acc = learning.history['val_acc']

    fig, axes = plt.subplots(1, 2, figsize=(20,4), subplot_kw={'xlabel':'epoch'} )
    axes[0].set_title("Accuracy")
    axes[0].plot(train_acc)
    axes[0].plot(val_acc)
    axes[0].legend(['training','validation'])
    axes[1].set_title("Loss")
    axes[1].plot(train_loss)
    axes[1].plot(val_loss)
    axes[1].legend(['training','validation'])

    best_training_epoc = val_loss.index(np.min(val_loss))
    axes[0].axvline(x=best_training_epoc, color='red')
    axes[1].axvline(x=best_training_epoc, color='red')
    fig.show()


# ## Benchmark Models

# In[ ]:


print("Accuracy by random guess: %.4f"%(1/len(df['english_cname'].unique())))


# In[ ]:


from sklearn.naive_bayes import GaussianNB
import warnings; warnings.simplefilter('ignore')

n_splits = 3
n_epochs = 50
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=1/4, random_state=37)
scores = []
n_classes = len(label_encoder.classes_)
params = {#'dim': (128,128),
          'n_frames': 128,
          'n_classes': n_classes,
          'n_channels': 1}
split_i=0
for cv_train_index, cv_val_index in sss.split(X_train, y_train):
    split_i+=1
    training_generator = AudioFeatureGenerator([X_train[i] for i in cv_train_index], [y_train[i] for i in cv_train_index], 
                                               batch_size=64, shuffle=True, seed=37, **params)
    validation_generator = AudioFeatureGenerator([X_train[i] for i in cv_val_index], [y_train[i] for i in cv_val_index], 
                                                 batch_size=len(cv_val_index), **params)
    nb = GaussianNB()
    epoch_scores = []
    print("Training split %i"%split_i)
    for epoch in range(n_epochs):
        n_batches = int(len(cv_train_index)/training_generator.batch_size)
        for X_batch, y_batch in training_generator:
            # Take the mean along the spectral band
            nb.partial_fit(X_batch.reshape(X_batch.shape[:-1]).mean(axis=2), 
                           [np.argmax(y) for y in y_batch], 
                           classes=range(n_classes))
        training_generator.on_epoch_end()
        # Test it out
        X_val_batch, y_val_batch = validation_generator[0]
        predictions = nb.predict(X_val_batch.reshape(X_val_batch.shape[:-1]).mean(axis=2)) 
        score = accuracy_score([np.argmax(y) for y in y_val_batch], predictions)
        epoch_scores.append(score)
        print("\rEpoch %i, score: %.5f, mean: %.5f"%(epoch+1, score, np.mean(epoch_scores)), end='')
        
    scores.append(np.mean(epoch_scores))
    plt.plot(epoch_scores)
    plt.title("Split %i learning curve, score: %.5f"%(len(scores), np.mean(epoch_scores)))
    plt.show()
print("Cross Validation Accuracy: %.4f"%(np.mean(scores)))


# In[ ]:


print("Mean Validation Accuracy: %.5f, std. dev.: %.5f"%(np.mean(scores), np.std(scores)))


# 5.41% accuracy. Now we're getting somewhere. The mean amplitude of each frequency band (a 128x1 array) seems to have some predictive power. I think we can do better still.

# What if we feed in all the pixels instead of flattening by taking the mean?

# In[ ]:


from sklearn.naive_bayes import GaussianNB
import warnings; warnings.simplefilter('ignore')

n_splits = 3
n_epochs = 50
n_classes = len(label_encoder.classes_)
params = {#'dim': (128,128),
          'n_frames': 128,
          'n_classes': n_classes,
          'n_channels': 1}
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=1/4, random_state=37)
scores = []
split_i=0
for cv_train_index, cv_val_index in sss.split(X_train, y_train):
    split_i+=1
    training_generator = AudioFeatureGenerator([X_train[i] for i in cv_train_index], [y_train[i] for i in cv_train_index], 
                                               batch_size=64, shuffle=True, seed=37, **params)
    validation_generator = AudioFeatureGenerator([X_train[i] for i in cv_val_index], [y_train[i] for i in cv_val_index], 
                                                 batch_size=len(cv_val_index), **params)
    nb = GaussianNB()
    epoch_scores = []
    for epoch in range(n_epochs):
        n_batches = int(len(cv_train_index)/training_generator.batch_size)
        for X_batch, y_batch in training_generator:
            nb.partial_fit(X_batch.reshape(X_batch.shape[0],X_batch.shape[1]*X_batch.shape[2]), 
                           [np.argmax(y) for y in y_batch], 
                           classes=range(n_classes),)
        training_generator.on_epoch_end()
        # Test it out
        X_val_batch, y_val_batch = validation_generator[0]
        predictions = nb.predict(X_val_batch.reshape(X_val_batch.shape[0],X_val_batch.shape[1]*X_val_batch.shape[2])) 
        score = accuracy_score([np.argmax(y) for y in y_val_batch], predictions)
        epoch_scores.append(score)
        print("\rEpoch %i, score: %.5f, mean: %.5f"%(epoch+1, score, np.mean(epoch_scores)), end='')
    scores.append(np.mean(epoch_scores))
    plt.plot(epoch_scores)
    plt.title("Split %i learning curve, score: %.5f"%(len(scores), np.mean(epoch_scores)))
    plt.show()
print("Cross Validation Accuracy: %.4f"%(np.mean(scores)))


# In[ ]:




