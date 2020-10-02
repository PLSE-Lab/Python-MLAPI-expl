#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')
import os
import pandas as pd
import librosa
import librosa.display
import IPython.display as ipd
import glob 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np
import pandas as pd
import wave
from scipy.io import wavfile
import os
import librosa
import warnings
from sklearn.utils import shuffle
import sklearn
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, LSTM, SimpleRNN

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# ## Background & References
# 
# **Problem Statement**
# The basic problem statement boils down to using Audio Clips of bird calls to identify the bird speceis. What we will try, in this notebook is try to implement sequential learning in the form of LSTM for this classification task.
# 
# This is my first attempt at using LSTM for sequential learning from audio recordings, please feel free to post any suggestions or questions
# 
# PS This is a work in progress, keep checking on this nb for future updates
# 
# 
# **Kernels Used for Reference**
# 
# The beauty of Kaggle is that a lot of work is already done, a lot of the work presented here borrows from the below Kernel:
# 
# [My Chen's Kernel on Using LSTM for Heart Sound Analysis](https://www.kaggle.com/mychen76/heart-sounds-analysis-and-classification-with-lstm)
# 
# [Francois Lemarchand's Notebook for this competition](https://www.kaggle.com/frlemarchand/bird-song-classification-using-an-lstm)
# 

# ## EDA
# 
# Training and Test Dataset summary:
# 

# In[ ]:


train_df = pd.read_csv('../input/birdsong-recognition/train.csv')
train_df.head()


# Whoa, there are so many columns, it would be fun to explore, but before that, let us have a look at the test dataset to see if all these columns are actually available

# In[ ]:


test = pd.read_csv('../input/birdsong-recognition/test.csv',)
test.head()


# Unlike the test data, where we have multiple additional columns available, test column has only the audio clip, hence this is purely a Ornithological Language Processing Task 

# In[ ]:


# Y variable - Ebird Code 

print('Number of Unique Birds in the the Dataset is: ' + str(train_df['ebird_code'].nunique()))

# Distribution of the labels

train_df['ebird_code'].value_counts().plot.bar()


# Great, an extremely legible label. We can see that about 50% of the birds have hundred recordings, the rest trailing off, with the minimum being a 100 recordings
# 
# Since the test set is purely based on audio recordings, we will be focussing on audio features rather than additional EDA on the test columns

# ## Audio Processing
# 
# **From Trigger Words Notebook (Sequence Model -Coursera)**
# 
# What really is an audio recording? 
# * A microphone records little variations in air pressure over time, and it is these little variations in air pressure that your ear also perceives as sound. 
# * You can think of an audio recording is a long list of numbers measuring the little air pressure changes detected by the microphone. 
# * We will use audio sampled at 44100 Hz (or 44100 Hertz). 
#     * This means the microphone gives us 44,100 numbers per second. 
#     * Thus, a 10 second audio clip is represented by 441,000 numbers (= $10 \times 44,100$). 
# 
# #### Spectrogram
# * It is quite difficult to figure out from this "raw" representation of audio whether the word "activate" was said. 
# * In  order to help your sequence model more easily learn to detect trigger words, we will compute a *spectrogram* of the audio. 
# * The spectrogram tells us how much different frequencies are present in an audio clip at any moment in time. 
# * If you've ever taken an advanced class on signal processing or on Fourier transforms:
#     * A spectrogram is computed by sliding a window over the raw audio signal, and calculating the most active frequencies in each window using a Fourier transform. 
#     * If you don't understand the previous sentence, don't worry about it.
# 
# Let's look at an example. 
#    

# In[ ]:


# Play the firt clip for an aldfly
aldfly = '../input/birdsong-recognition/train_audio/aldfly/XC134874.mp3'
y,sr = librosa.load(aldfly, sr=None)
ipd.Audio(aldfly) 


# **Visualising the Audio**
# 
# Audio Clips can be visualised as a sequence of waves, with Amplitudes, crests and troughs using librosa
# 

# In[ ]:


plt.figure(figsize=(14, 5))
librosa.display.waveplot(y,sr = sr)


# **Spectogram**

# In[ ]:


Y = librosa.stft(y)
Xdb = librosa.amplitude_to_db(abs(Y))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()


# The color in the spectrogram shows the degree to which different frequencies are present (loud) in the audio at different points in time.

# # LSTM

# Before we delve into the task of using LSTM for the task, we need to understand why LSTM works here
# 
# An easier analogy would be understanding how Echo uses LSTM to understand your voice command, we can later apply the same philosophy here.
# 
# When you use a regular ANN, every input is in iteself an isolated data point (following the definition of IID). Say you are building a Churn risk model, two customers would be independently evaluated on their Churn Risk, outcome of one doesn't affect the other at all. Or in other words, they have no memory of the previous outcome while evaluating the new outcome.
# 
# But by using LSTMs, we can harness the sequence and order into data and use it while making classifications. 
# 
# **Will Add more later**
# 
# [Andrew Ng explaining LSTM](https://www.youtube.com/watch?v=5wh4HWWfZIY)
# 

# In[ ]:


#Using Francois's code to extract the data/ run model


def get_sample(filename, bird, samples_df):
    wave_data, wave_rate = librosa.load(filename)
    data_point_per_second = 10
    
    #Take 10 data points every second
    prepared_sample = wave_data[0::int(wave_rate/data_point_per_second)]
    #We normalize each sample before extracting 5s samples from it
    normalized_sample = sklearn.preprocessing.minmax_scale(prepared_sample, axis=0)
    
    #only take 5s samples and add them to the dataframe
    song_sample = []
    sample_length = 5*data_point_per_second
    for idx in range(0,len(normalized_sample),sample_length): 
        song_sample = normalized_sample[idx:idx+sample_length]
        if len(song_sample)>=sample_length:
            samples_df = samples_df.append({"song_sample":np.asarray(song_sample).astype(np.float32),
                                            "bird":ebird_to_id[bird]}, 
                                           ignore_index=True)
    return samples_df



# In[ ]:


birds_selected = shuffle(train_df["ebird_code"].unique())
train_df = train_df.query("ebird_code in @birds_selected")

ebird_to_id = {}
id_to_ebird = {}
ebird_to_id["nocall"] = 0
id_to_ebird[0] = "nocall"
for idx, unique_ebird_code in enumerate(train_df.ebird_code.unique()):
    ebird_to_id[unique_ebird_code] = str(idx+1)
    id_to_ebird[idx+1] = str(unique_ebird_code)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'warnings.filterwarnings("ignore")\nsamples_df = pd.DataFrame(columns=["song_sample","bird"])\n\n#We limit the number of audio files being sampled to 5000 in this notebook to save time\n#However, we have already limited the number of bird species\nsample_limit = 5000\nwith tqdm(total=sample_limit) as pbar:\n    for idx, row in train_df[:sample_limit].iterrows():\n        pbar.update(1)\n        audio_file_path = "/kaggle/input/birdsong-recognition/train_audio/"\n        audio_file_path += row.ebird_code\n        samples_df = get_sample(\'{}/{}\'.format(audio_file_path, row.filename), row.ebird_code, samples_df)')


# In[ ]:


samples_df = shuffle(samples_df)
samples_df[:10]


# In[ ]:


sequence_length = 50
training_percentage = 0.9
training_item_count = int(len(samples_df)*training_percentage)
validation_item_count = len(samples_df)-int(len(samples_df)*training_percentage)
training_df = samples_df[:training_item_count]
validation_df = samples_df[training_item_count:]


# ### Define the Neural Network Model
# 
# If you are new to Keras, please refer to this [Keras Tutorial](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)
# 
# This is a multiclass classification problem, therefore we will be using the following architecture:
# 
# Output Layer: Softmax with 264 outputs (This will output 264 Probabilities, corresponding to each of the 264 birds
# 
# Optimiser: [Adam](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)
# 
# Loss: [Categorical Cross Entropy](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/categorical-crossentropy)
# 
# Metrics: Accuracy 

# In[ ]:


#Base Model, lots of room for improvement


model = Sequential()
model.add(LSTM(32, return_sequences=True, recurrent_dropout=0.2,input_shape=(None, sequence_length)))
model.add(LSTM(32,recurrent_dropout=0.2))
model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(len(ebird_to_id.keys()), activation="softmax"))

model.summary()

callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.7),
             EarlyStopping(monitor='val_loss', patience=10),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
model.compile(loss="categorical_crossentropy", optimizer='adam')


# In[ ]:


X_train = np.asarray(np.reshape(np.asarray([np.asarray(x) for x in training_df["song_sample"]]),(training_item_count,1,sequence_length))).astype(np.float32)
groundtruth = np.asarray([np.asarray(x) for x in training_df["bird"]]).astype(np.float32)
Y_train = to_categorical(
                groundtruth, num_classes=len(ebird_to_id.keys()), dtype='float32'
            )


X_validation = np.asarray(np.reshape(np.asarray([np.asarray(x) for x in validation_df["song_sample"]]),(validation_item_count,1,sequence_length))).astype(np.float32)
validation_groundtruth = np.asarray([np.asarray(x) for x in validation_df["bird"]]).astype(np.float32)
Y_validation = to_categorical(
                validation_groundtruth, num_classes=len(ebird_to_id.keys()), dtype='float32'
            )


# In[ ]:


history = model.fit(X_train, Y_train, 
          epochs = 100, 
          batch_size = 32, 
          validation_data=(X_validation, Y_validation), 
          callbacks=callbacks)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss over epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()


# ### Making Predictions

# In[ ]:


model.load_weights("best_model.h5")

def predict_submission(df, audio_file_path):
        
    loaded_audio_sample = []
    previous_filename = ""
    data_point_per_second = 10
    sample_length = 5*data_point_per_second
    wave_data = []
    wave_rate = None
    
    for idx,row in df.iterrows():
        if previous_filename == "" or previous_filename!=row.filename:
            filename = '{}/{}.mp3'.format(audio_file_path, row.filename)
            wave_data, wave_rate = librosa.load(filename)
            sample = wave_data[0::int(wave_rate/data_point_per_second)]
        previous_filename = row.filename
        
        #basically allows to check if we are running the examples or the test set.
        if "site" in df.columns:
            if row.site=="site_1" or row.site=="site_2":
                song_sample = np.array(sample[int(row.seconds-5)*data_point_per_second:int(row.seconds)*data_point_per_second])
            elif row.site=="site_3":
                #for now, I only take the first 5s of the samples from site_3 as they are groundtruthed at file level
                song_sample = np.array(sample[0:sample_length])
        else:
            #same as the first condition but I isolated it for later and it is for the example file
            song_sample = np.array(sample[int(row.seconds-5)*data_point_per_second:int(row.seconds)*data_point_per_second])

        input_data = np.reshape(np.asarray([song_sample]),(1,sequence_length)).astype(np.float32)
        prediction = model.predict(np.array([input_data]))
        predicted_bird = id_to_ebird[np.argmax(prediction)]

        df.at[idx,"birds"] = predicted_bird
    return df


# In[ ]:


audio_file_path = "/kaggle/input/birdsong-recognition/example_test_audio"
example_df = pd.read_csv("/kaggle/input/birdsong-recognition/example_test_audio_summary.csv")
example_df["filename"] = [ "BLKFR-10-CPL_20190611_093000.pt540" if filename=="BLKFR-10-CPL" else "ORANGE-7-CAP_20190606_093000.pt623" for filename in example_df["filename"]]


if os.path.exists(audio_file_path):
    example_df = predict_submission(example_df, audio_file_path)
example_df


# In[ ]:


audio_file_path = "/kaggle/input/birdsong-recognition/test_audio/"
test_df = pd.read_csv("/kaggle/input/birdsong-recognition/test.csv")
submission_df = pd.read_csv("/kaggle/input/birdsong-recognition/sample_submission.csv")

if os.path.exists(audio_file_path):
    submission_df = predict_submission(test_df, audio_file_path)


# In[ ]:


submission_df[["row_id","birds"]].to_csv('submission.csv', index=False)
submission_df.head()

