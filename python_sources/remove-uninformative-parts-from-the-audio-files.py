#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# In many of the audio files there are silent parts. My guess is that there is not much useful information.
# 
# In this kernel we will try to explore that assumption and see if it is a good idea to crop the silent parts.

# In[ ]:


import os
import numpy as np
from tqdm import tqdm
import IPython.display as ipd
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


TRAIN_PATH = '../input/audio_train/'
train_ids = next(os.walk(TRAIN_PATH))[2]


# We will look on one example

# In[ ]:


ipd.Audio(TRAIN_PATH + "31440023.wav")


# As you can hear, most of he time there is no sound at all.
# 
# Now lets look at signal in time domain

# In[ ]:


sample_rate, audio = wavfile.read(TRAIN_PATH + "31440023.wav")

plt.plot(audio);


# So the idea is to crop and segment the audio files and leave only the part that contain information.
# 
# First wi will normalize the data to be between -1 and 1:
# 
# Note that not all audio values are between -32768 and 32768

# In[ ]:


def normalize_audio(audio):
    #audio = (audio + 32768) / 65535
    audio = audio / max(np.abs(audio))
    return audio


# We will implement a sliding window that measures the power in each segment and based on that decides if this part is a noise.
# 
# Finally the function returns the start and stop of each segment in the audio.

# In[ ]:


def divide_audio(audio, resolution=100, window_duration=0.1, minimum_power=0.001, sample_rate=44100):    
    duration = len(audio) / sample_rate #in seconds
    itterations = int(duration * resolution)
    step = int(sample_rate / resolution)
    window_length = np.floor(sample_rate*window_duration)
    audio_power = np.square(normalize_audio(audio)) / window_length #Normalized power to window duration
    
    start = np.array([])
    stop = np.array([])
    is_started = False
    
    for n in range(itterations):
        power = np.sum(audio_power[n*step : int(n*step+window_length)])
        if not is_started and power > minimum_power:
            start = np.append(start, n*step+window_length/2)
            is_started = True
        elif is_started and (power <= minimum_power or n == itterations-1):
            stop = np.append(stop, n*step+window_length/2)
            is_started = False
    
    if start.size == 0:
        start = np.append(start, 0)
        stop = np.append(stop, len(audio))
        
    start = start.astype(int)
    stop = stop.astype(int)
    return start, stop


# In[ ]:


start, stop =  divide_audio(audio)
print(start)
print(stop)
plt.plot(audio[start[0]:stop[0]]);


# After some manual tunning it seems that the it works
# 
# Now lets briefly look on more examples:
# 
# Loop logic taken from: [https://www.kaggle.com/codename007/a-very-extensive-freesound-exploratory-analysis](http://www.kaggle.com/codename007/a-very-extensive-freesound-exploratory-analysis)

# In[ ]:


columns = ['File Name', 'Audio Duration', 'Segment Number']
audio_segments = pd.DataFrame(columns=columns)

fig, ax = plt.subplots(10, 4, figsize = (12, 16))
for i in tqdm(range(40), total=40):
    random_audio_idxs = np.random.randint(len(train_ids)+1, size=1)[0]
    _, tmp = wavfile.read(TRAIN_PATH + train_ids[random_audio_idxs])
    start, stop = divide_audio(tmp)
    
    audio_segments = audio_segments.append({'File Name': train_ids[random_audio_idxs],
                                            'Audio Duration': len(tmp)/sample_rate,
                                            'Segment Number': len(start)}, ignore_index=True)
    
    ax[i//4, i%4].plot(tmp)
    ax[i//4, i%4].set_title(train_ids[random_audio_idxs])
    ax[i//4, i%4].get_xaxis().set_ticks([])


# In[ ]:


audio_segments


# After some more manual tuning it seems that the code can sufficiently segment and remove the noise.
# 
# To create a new training set, run the following code:

# In[1]:


#train = pd.read_csv('../input/train.csv')
#new_train = pd.DataFrame(columns=train.columns)

#if not os.path.exists(TRAIN_PATH + 'segmented'):
#    os.makedirs(TRAIN_PATH + 'segmented')

#for n in tqdm(range(len(train_ids)), total=len(train_ids)):
#    _, tmp = wavfile.read(TRAIN_PATH + train_ids[n])
#    start, stop = divide_audio(tmp, window_duration=0.1, minimum_power=0.001)
#    new_path = TRAIN_PATH + 'segmented/' + train_ids[n]
    
#    if len(start) <= 1:
#        wavfile.write(new_path, sample_rate, tmp[start[0]:stop[0]])
#        new_train = new_train.append(train.iloc[n])
#    else:
#        for m in range(len(start)):
#            wavfile.write(new_path[:-4] + '_' + str(m) + '.wav', sample_rate, tmp[start[m]:stop[m]])
#            new_train = new_train.append({train.columns[0]: train.iloc[n][train.columns[0]][:-4] + '_' + str(m) + '.wav',
#                                          train.columns[1]: train.iloc[n][train.columns[1]],
#                                          train.columns[2]: train.iloc[n][train.columns[2]],}, ignore_index=True)
            
#new_train.to_csv(TRAIN_PATH + 'segmented/train.csv', index=False)


# I did not have the chance to check this new test set on a NN.
# 
# I will continue update here.
