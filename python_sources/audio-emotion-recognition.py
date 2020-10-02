#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
import matplotlib.pyplot as plt
# Any results you write to the current directory are saved as output.


# In[ ]:


###### import libraries
import librosa
import wave
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import rmsprop


# In[ ]:


def extract_mfcc(wav_file_name):
    '''This function extracts mfcc features and obtain the mean of each dimension
    Input : path_to_wav_file
    Output: mfcc_features'''
    y, sr = librosa.load(wav_file_name)
#     trimmed_data = np.zeros((160, 20))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
#     data = np.array(librosa.feature.mfcc(y = y, sr = sr, n_mfcc=40).T)
#     if data.shape[0] <= 160:
#         trimmed_data[:data.shape[0],0:] = data[:,0:]
#     else:
#         trimmed_data[0:,0:] = data[0:160,0:]
    return mfccs


# In[ ]:


### extract audio data from AV RAVDESS data
root_dir = "../input/ravdess-audiof-files-from-video/ravdess_av/Audio_only/" 

audio_only_data = [] ###stores the mfcc data
audio_only_labels = [] ###stores the labels
for subdirs, dirs, files in os.walk(root_dir):
    for file in files:
        y, sr = librosa.load(os.path.join(subdirs,file))
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
        audio_only_data.append(mfccs)
        audio_only_labels.append(int(file[7:8]) - 1)


# In[ ]:


#### convert data to array and make labels categorical
audio_only_data_array = np.array(audio_only_data)
audio_only_labels_array = np.array(audio_only_labels)
audio_only_data_array.shape


# In[ ]:


##### load data from savee dataset
#### although, we load the data here, it is not used in training or validation
root_dir = "../input/savee-emotion-recognition/audiodata/AudioData/"
# root_dir = "../input/audio_speech_actors_01-24/"
savee_data = []
savee_labels = []
for actor_dir in sorted(os.listdir(root_dir)):
    if actor_dir[-4:] == ".txt":
        continue
    for file_name in os.listdir(os.path.join(root_dir, actor_dir)):
        if file_name[0] == "c":
            continue
        wav_file_name = os.path.join(root_dir, actor_dir, file_name)
        savee_data.append(extract_mfcc(wav_file_name))
        if file_name[0] == "n":
            savee_labels.append(0)
        if file_name[0] == "a":
            savee_labels.append(4)
        if file_name[0] == "d":
            savee_labels.append(6)
        if file_name[0] == "f":
            savee_labels.append(5)
        if file_name[0] == "h":
            savee_labels.append(2)
        if file_name[:2] == "sa":
            savee_labels.append(3)
        if file_name[:2] == "su":
            savee_labels.append(7)


# In[ ]:


#### convert data to array and make labels categorical
savee_data_array = np.asarray(savee_data)
savee_label_array = np.array(savee_labels)
to_categorical(savee_label_array)[0].shape
# savee_data_array.shape


# In[ ]:


##### load radvess speech data #####
root_dir = "../input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"
# root_dir = "../input/audio_speech_actors_01-24/"
# actor_dir = os.listdir("../input/audio_speech_actors_01-24/")
radvess_speech_labels = []
ravdess_speech_data = []
for actor_dir in sorted(os.listdir(root_dir)):
    actor_name = os.path.join(root_dir, actor_dir)
    for file in os.listdir(actor_name):
        radvess_speech_labels.append(int(file[7:8]) - 1)
        wav_file_name = os.path.join(root_dir, actor_dir, file)
        ravdess_speech_data.append(extract_mfcc(wav_file_name))


# In[ ]:


#### convert data to array and make labels categorical
ravdess_speech_data_array = np.asarray(ravdess_speech_data)
ravdess_speech_label_array = np.array(radvess_speech_labels)
ravdess_speech_label_array.shape


# In[ ]:


### load RAVDESS song data
root_dir = "../input/ravdess-song-files/audio_song_actors_01-24/"
radvess_song_labels = []
ravdess_song_data = []
for actor_dir in sorted(os.listdir(root_dir)):
    actor_name = os.path.join(root_dir, actor_dir)
    for file in os.listdir(actor_name):
        radvess_song_labels.append(int(file[7:8]) - 1)
        wav_file_name = os.path.join(root_dir, actor_dir, file)
        ravdess_song_data.append(extract_mfcc(wav_file_name))


# In[ ]:


#### convert data to array and make labels categorical
ravdess_song_data_array = np.asarray(ravdess_song_data)
ravdess_song_label_array = np.array(radvess_song_labels)
ravdess_song_label_array.shape


# In[ ]:


# #### combine data
data = np.r_[audio_only_data_array, ravdess_speech_data_array, ravdess_song_data_array]
labels = np.r_[audio_only_labels_array, ravdess_speech_label_array, ravdess_song_label_array]
# data = ravdess_speech_data_array
# labels = ravdess_speech_label_array
labels.shape


# In[ ]:


### plot a histogram to understand the distribution of the data
import matplotlib.pyplot as plt
plt.hist(labels)
plt.show()


# In[ ]:


### make categorical labels
labels_categorical = to_categorical(labels)
data.shape
labels_categorical.shape


# In[ ]:


def create_model_LSTM():
    ### LSTM model, referred to the model A in the report
    model = Sequential()
    model.add(LSTM(128, return_sequences=False, input_shape=(40, 1)))
    model.add(Dense(64))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model

def create_model_CNN():
    ### CNN model, referred to the model B in the report
    model = Sequential()
    model.add(Conv1D(8, kernel_size = 3, input_shape=(40, 1)))
    model.add(Activation('relu'))
    model.add(Conv1D(16,kernel_size = 3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, kernel_size = 3))
    model.add(Activation('relu'))
    model.add(Conv1D(16, kernel_size = 3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model

def new_CNN():
    ### CNN model, referred to the model C in the report
    model = Sequential()
    model.add(Conv1D(8, 5,padding='same', input_shape=(40, 1)))
    model.add(Activation('relu'))
    model.add(Conv1D(16, 5,padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(32, 5,padding='same',))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(16, 5,padding='same',))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Activation('softmax'))
    opt = rmsprop(lr=0.00001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['accuracy'])
    return model

def train_CNN():
    ### CNN model, referred to the model D in the report
    model = Sequential()
    model.add(Conv1D(128, 5,padding='same',
                 input_shape=(40,1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5,padding='same',))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.00005, rho=0.9, epsilon=None, decay=0.0)
    

    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    return model


# In[ ]:


number_of_samples = data.shape[0]
training_samples = int(number_of_samples * 0.8)
validation_samples = int(number_of_samples * 0.1)
test_samples = int(number_of_samples * 0.1)


# In[ ]:


### train using model A
model_A = create_model_LSTM()
history = model_A.fit(np.expand_dims(data[:training_samples],-1), labels_categorical[:training_samples], validation_data=(np.expand_dims(data[training_samples:training_samples+validation_samples], -1), labels_categorical[training_samples:training_samples+validation_samples]), epochs=100, shuffle=True)


# In[ ]:


### loss plots using model A
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


### accuracy plots using model A
plt.clf()                                                

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'ro', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


### evaluate using model A
model_A.evaluate(np.expand_dims(data[training_samples + validation_samples:], -1), labels_categorical[training_samples + validation_samples:])
# model.evaluate(predictions, labels_categorical[training_samples + validation_samples:])


# In[ ]:


### train using model C
model_C = new_CNN()
history = model_C.fit(np.expand_dims(data[:training_samples],-1), labels_categorical[:training_samples], validation_data=(np.expand_dims(data[training_samples:training_samples+validation_samples], -1), labels_categorical[training_samples:training_samples+validation_samples]), epochs=100, shuffle=True)


# In[ ]:


### loss plots using model C
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


### accuracy plots using model C
plt.clf()                                                

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'ro', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


### evaluate using model C
model_C.evaluate(np.expand_dims(data[training_samples + validation_samples:], -1), labels_categorical[training_samples + validation_samples:])


# In[ ]:


### train using model B 
model_B = create_model_CNN()
history = model_B.fit(np.expand_dims(data[:training_samples],-1), labels_categorical[:training_samples], validation_data=(np.expand_dims(data[training_samples:training_samples+validation_samples], -1), labels_categorical[training_samples:training_samples+validation_samples]), epochs=100, shuffle=True)


# In[ ]:


### loss plots using model B
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


### accuracy plots using model B
plt.clf()                                                

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'ro', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


###  evaluate using model B
model_B.evaluate(np.expand_dims(data[training_samples + validation_samples:], -1), labels_categorical[training_samples + validation_samples:])


# In[ ]:


### train using model D
model_D = train_CNN()
history = model_D.fit(np.expand_dims(data[:training_samples],-1), labels_categorical[:training_samples], validation_data=(np.expand_dims(data[training_samples:training_samples+validation_samples], -1), labels_categorical[training_samples:training_samples+validation_samples]), epochs=100, shuffle=True)


# In[ ]:


### loss plots using model D
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


### accuracy plots using model D
plt.clf()                                                

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'ro', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


### evaluate using model D
model_D.evaluate(np.expand_dims(data[training_samples + validation_samples:], -1), labels_categorical[training_samples + validation_samples:])


# In[ ]:


model_B.summary()


# In[ ]:


import seaborn as sn
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(model_D.predict_classes(np.expand_dims(data[training_samples + validation_samples:], -1)), labels[training_samples + validation_samples:])
sn.set(font_scale=1.4)#for label size
sn.heatmap(cm, annot=True,annot_kws={"size": 16})# font size


# In[ ]:


model_A.save_weights("Model_A.h5")
model_B.save_weights("Model_B.h5")
model_C.save_weights("Model_C.h5")
model_D.save_weights("Model_D.h5")

