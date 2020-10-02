#!/usr/bin/env python
# coding: utf-8

# # Intro

# This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music. The classes are drawn from the urban sound taxonomy. For a detailed description of the dataset and how it was compiled please refer to our paper.
# All excerpts are taken from field recordings uploaded to www.freesound.org. The files are pre-sorted into ten folds (folders named fold1-fold10) to help in the reproduction of and comparison with the automatic classification results reported in the article above.
# 
# In addition to the sound excerpts, a CSV file containing metadata about each excerpt is also provided.

# ### Methodology

# 1. There are 3 basic methods to extract features from audio file :
#     a) Using the mffcs data of the audio files
#     b) Using a spectogram image of the audio and then converting the same to data points (As is done for images). This is easily done using mel_spectogram function of Librosa
#     c) Combining both features to build a better model. (Requires a lot of time to read and extract data).
# 2. I have chosen to use the second method.
# 3. The labels have been converted to categorical data for classification.
# 4. CNN has been used as the primary layer to classify data

# # Importing Necessary Libraries

# In[ ]:


# Basic Libraries

import pandas as pd
import numpy as np

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler


# In[ ]:


# Libraries for Classification and building Models

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical 

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# In[ ]:


# Project Specific Libraries

import os
import librosa
import librosa.display
import glob 
import skimage


# # Analysing Data Type and Format

# #### Analysing CSV Data

# In[ ]:


df = pd.read_csv("../input/urbansound8k/UrbanSound8K.csv")

'''We will extract classes from this metadata.'''

df.head()


# ##### Column Names
# 
# * slice_file_name: 
# The name of the audio file. The name takes the following format: [fsID]-[classID]-[occurrenceID]-[sliceID].wav, where:
# [fsID] = the Freesound ID of the recording from which this excerpt (slice) is taken
# [classID] = a numeric identifier of the sound class (see description of classID below for further details)
# [occurrenceID] = a numeric identifier to distinguish different occurrences of the sound within the original recording
# [sliceID] = a numeric identifier to distinguish different slices taken from the same occurrence
# 
# * fsID:
# The Freesound ID of the recording from which this excerpt (slice) is taken
# 
# * start
# The start time of the slice in the original Freesound recording
# 
# * end:
# The end time of slice in the original Freesound recording
# 
# * salience:
# A (subjective) salience rating of the sound. 1 = foreground, 2 = background.
# 
# * fold:
# The fold number (1-10) to which this file has been allocated.
# 
# * classID:
# A numeric identifier of the sound class:
# 0 = air_conditioner
# 1 = car_horn
# 2 = children_playing
# 3 = dog_bark
# 4 = drilling
# 5 = engine_idling
# 6 = gun_shot
# 7 = jackhammer
# 8 = siren
# 9 = street_music
# 
# * class:
# The class name: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, 
# siren, street_music.

# #### Using Librosa to analyse random sound sample - SPECTOGRAM

# In[ ]:


dat1, sampling_rate1 = librosa.load('../input/urbansound8k/fold5/100032-3-0-0.wav')
dat2, sampling_rate2 = librosa.load('../input/urbansound8k/fold5/100263-2-0-117.wav')


# In[ ]:


plt.figure(figsize=(20, 10))
D = librosa.amplitude_to_db(np.abs(librosa.stft(dat1)), ref=np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')


# In[ ]:


plt.figure(figsize=(20, 10))
D = librosa.amplitude_to_db(np.abs(librosa.stft(dat2)), ref=np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')


# In[ ]:


'''Using random samples to observe difference in waveforms.'''

arr = np.array(df["slice_file_name"])
fold = np.array(df["fold"])
cla = np.array(df["class"])

for i in range(192, 197, 2):
    path = '../input/urbansound8k/fold' + str(fold[i]) + '/' + arr[i]
    data, sampling_rate = librosa.load(path)
    plt.figure(figsize=(10, 5))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
    plt.subplot(4, 2, 1)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title(cla[i])


# # Feature Extraction and Database Building

# #### Method
# 
# 1. I have used Librosa to extract features.
# 2. To do so, I will go through each fold and extract the data for each file. Then I have used the mel_spectogram function of librosa to extract the spectogram data as a numpy array.
# 3. After reshaping and cleaning the data, 75-25 split has been performed.
# 4. Classes (Y) have been converted to Categorically Encoded Data usng Keras.utils
# 
# Note : Running the parser function may take upto 45 minutes depending on your system since it has to extract spectogram data for 8732 audio files

# In[ ]:


'''EXAMPLE'''

dat1, sampling_rate1 = librosa.load('../input/urbansound8k/fold5/100032-3-0-0.wav')
arr = librosa.feature.melspectrogram(y=dat1, sr=sampling_rate1)
arr.shape


# In[ ]:


feature = []
label = []

def parser(row):
    # Function to load files and extract features
    for i in range(8732):
        file_name = '../input/urbansound8k/fold' + str(df["fold"][i]) + '/' + df["slice_file_name"][i]
        # Here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        # We extract mfcc feature from data
        mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)        
        feature.append(mels)
        label.append(df["classID"][i])
    return [feature, label]


# In[ ]:


temp = parser(df)


# In[ ]:


temp = np.array(temp)
data = temp.transpose()


# In[ ]:


X_ = data[:, 0]
Y = data[:, 1]
print(X_.shape, Y.shape)
X = np.empty([8732, 128])


# In[ ]:


for i in range(8732):
    X[i] = (X_[i])


# In[ ]:


Y = to_categorical(Y)


# In[ ]:


'''Final Data'''
print(X.shape)
print(Y.shape)


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1)


# In[ ]:


X_train = X_train.reshape(6549, 16, 8, 1)
X_test = X_test.reshape(2183, 16, 8, 1)


# In[ ]:


input_dim = (16, 8, 1)


# # Creating Keras Model and Testing

# #### Model 1:
# 
# 1. CNN 2D with 64 units and tanh activation.
# 2. MaxPool2D with 2*2 window.
# 3. CNN 2D with 128 units and tanh activation.
# 4. MaxPool2D with 2*2 window.
# 5. Dropout Layer with 0.2 drop probability.
# 6. DL with 1024 units and tanh activation.
# 4. DL 10 units with softmax activation.
# 5. Adam optimizer with categorical_crossentropy loss function.
# 
# 90 epochs have been used.

# In[ ]:


model = Sequential()


# In[ ]:


model.add(Conv2D(64, (3, 3), padding = "same", activation = "tanh", input_shape = input_dim))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding = "same", activation = "tanh"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(1024, activation = "tanh"))
model.add(Dense(10, activation = "softmax"))


# In[ ]:


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


model.fit(X_train, Y_train, epochs = 90, batch_size = 50, validation_data = (X_test, Y_test))


# In[ ]:


model.summary()


# In[ ]:


predictions = model.predict(X_test)
score = model.evaluate(X_test, Y_test)
print(score)


# In[ ]:


preds = np.argmax(predictions, axis = 1)


# In[ ]:


result = pd.DataFrame(preds)
result.to_csv("UrbanSound8kResults.csv")

