#!/usr/bin/env python
# coding: utf-8

# # Removing Uninformative Parts of the Audio Files
# 
# *Initially forked [from ILM](https://www.kaggle.com/ilyamich/remove-uninformative-parts-from-the-audio-files). Updated and presented by [Matthew](http://www.kaggle.com/matthewa313).*
# 
# In this notebook, I will demonstrate the removal of empty or very quiet parts of audio files in the train and test audio set.  This new training corpus and testing set should hopefully increase accuracy and decrease the memory of models when employed.

# In[2]:


import os # kaggle OS
import numpy as np # linear algebra
import pandas as pd # data processing
from tqdm import tqdm_notebook as tqdm # progress bar
import IPython.display as ipd # .wav visualizations
from scipy.io import wavfile # return sample rate (in samples/sec) and data from a WAV file
import matplotlib.pyplot as plt # plots

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


train_ids = next(os.walk("../input/audio_train/"))[2]
test_ids = next(os.walk("../input/audio_test/"))[2]


# Here's an example from the train set of a mostly empty track:

# In[5]:


ipd.Audio("../input/audio_train/31440023.wav")


# > **wavfile.read**
# 
# Returns  the sample rate (samples per second) from a .wav file

# In[6]:


sample_rate, audio = wavfile.read("../input/audio_train/31440023.wav")

plt.plot(audio); # plot the audio chart with MatPlot


# Another competitor had the idea to crop and segment the audio files and leave only the part that contains information and a little bit around it.
# 
# First we'll define a function that normalizes the data to be between -1 and 1, as opposed to -32768 and 32768.

# In[7]:


def normalize_audio(audio):
    # audio = (audio + 32768) / 65535 (only if bits were correct)
    audio = audio / max(np.abs(audio))
    return audio


# We'll eliminate noise from the tracks by measuring the volume in each segment and eliminating very quiet segments.
# 
# This function will return the start and stop of the signal parts of each track.

# In[8]:


def divide_audio(audio, resolution=100, window_duration=0.1, minimum_power=0.001, sample_rate=44100):    
    duration = len(audio) / sample_rate # in samples/sec
    iterations = int(duration * resolution)
    step = int(sample_rate / resolution)
    window_length = np.floor(sample_rate * window_duration)
    audio_power = np.square(normalize_audio(audio)) / window_length #Normalized power to window duration

    start = np.array([])
    stop = np.array([])
    is_started = False
    
    for n in range(iterations):
        power = 10 * np.sum(audio_power[n * step : int(n * step + window_length)]) # sensitive
        if not is_started and power > minimum_power:
            start = np.append(start, n * step + window_length / 2)
            is_started = True
        elif is_started and (power <= minimum_power or n == iterations-1):
            stop = np.append(stop, n * step + window_length / 2)
            is_started = False
    
    if start.size == 0:
        start = np.append(start, 0)
        stop = np.append(stop, len(audio))
        
    start = start.astype(int)
    stop = stop.astype(int)
    
    # We don't want to eliminate EVERYTHING that's unnecessary
    # There should be a little boundary...
    # 200 frame buffer before and after
    
    # minus = ?
    if start[0] > 200:
        minus = 200
    else:
        minus = start[0]
        
    # plus = ?
    if (len(audio) - stop[0]) > 200:
        plus = 200
    else:
        plus = len(audio) - stop[0]
    
    return (start - minus), (stop + plus)


# In[9]:


# same wav file as before
start, stop =  divide_audio(audio)
print(start)
print(stop)
plt.plot(audio[start[0]:stop[0]]);


# After some manual tunning it seems that the it works
# 
# Now lets briefly look on more examples:
# 
# [Loop logic](http://www.kaggle.com/codename007/a-very-extensive-freesound-exploratory-analysis](http://www.kaggle.com/codename007/a-very-extensive-freesound-exploratory-analysis) taken from codename007.

# In[10]:


columns = ['File Name', 'Audio Duration', 'Segment Number']
audio_segments = pd.DataFrame(columns=columns)

fig, ax = plt.subplots(10, 4, figsize = (12, 16))
for i in tqdm(range(40), total=40):
    random_audio_idxs = np.random.randint(len(train_ids)+1, size=1)[0]
    _, tmp = wavfile.read("../input/audio_train/" + train_ids[random_audio_idxs])
    start, stop = divide_audio(tmp)
    
    audio_segments = audio_segments.append({'File Name': train_ids[random_audio_idxs],
                                            'Audio Duration': len(tmp)/sample_rate,
                                            'Segment Number': len(start)}, ignore_index=True)
    
    ax[i//4, i%4].plot(tmp)
    ax[i//4, i%4].set_title(train_ids[random_audio_idxs])
    ax[i//4, i%4].get_xaxis().set_ticks([])


# In[11]:


audio_segments


# ## Let's create a new training set.
# 
# We'll run out of memory if we run the for loop too many times, but if you want to, you can run it on your home computer (you will need to run it 9,473 times). In this example, I will only run 1the first 3 as a mere example.

# In[13]:


train     = pd.read_csv('../input/train.csv')
new_train = pd.DataFrame(columns=train.columns)

# we can only iterate threw the first ~2300 without losing memory
# change this value to 9473 to iterate through all of them
for n in tqdm(range(3)):
    _, tmp = wavfile.read('../input/audio_train/' + train_ids[n])
    start, stop = divide_audio(tmp, window_duration=0.1, minimum_power=0.001)
    new_path = 'segmented_' + train_ids[n]
    
    if len(start) <= 1:
        wavfile.write(new_path, sample_rate, tmp[start[0]:stop[0]])
        new_train = new_train.append(train.iloc[n])
    else:
        for m in range(len(start)):
            wavfile.write(new_path[:-4] + '_' + str(m) + '.wav', sample_rate, tmp[start[m]:stop[m]])
            new_train = new_train.append({train.columns[0]: train.iloc[n][train.columns[0]][:-4] + '_' + str(m) + '.wav',
                                          train.columns[1]: train.iloc[n][train.columns[1]],
                                          train.columns[2]: train.iloc[n][train.columns[2]],}, ignore_index=True)
            
new_train.to_csv('segmented_train.csv', index=False)


# ## Finally, let's create a new test set.
# 
# We'll run out of memory if we run the for loop too many times, but if you want to, you can run it on your home computer (you will need to run it 9,473 times). In this example, I will only run 1the first 3 as a mere example.

# In[18]:


test     = pd.read_csv('../input/sample_submission.csv')
new_test = pd.DataFrame(columns=train.columns)

print(len(test_ids))
# for n in tqdm(range(len(test_ids))):
#     _, tmp = wavfile.read('../input/audio_test/' + test_ids[n])
#     start, stop = divide_audio(tmp, window_duration=0.1, minimum_power=0.001)
#     new_path = 'segmented_' + test_ids[n]
    
#    if len(start) <= 1:
#        wavfile.write(new_path, sample_rate, tmp[start[0]:stop[0]])
#        new_train = new_train.append(train.iloc[n])
#    else:
#        for m in range(len(start)):
#            wavfile.write(new_path[:-4] + '_' + str(m) + '.wav', sample_rate, tmp[start[m]:stop[m]])
#            new_train = new_train.append({train.columns[0]: train.iloc[n][train.columns[0]][:-4] + '_' + str(m) + '.wav',
#                                         train.columns[1]: train.iloc[n][train.columns[1]],
# train.columns[2]: train.iloc[n][train.columns[2]],}, ignore_index=True)
            
# new_train.to_csv('segmented_test.csv', index=False)


#  I can't create a new full test set because we'll run out of memory on Kaggle. Instead, to create a new test set, you should run this locally.

# ## Future Works
# 
# The next step in this process is evaluating the merits of removing noisy (as in, not signal), parts of audio files for sound tagging. If you cite this in a research paper, please use:
# > Author(s) name: Matthew Anderson
# > Date: 04/07/2018 (April 7th)
# > Title: "Removing Uninformative Parts of Audio Files"
# > Version 1.1
# > Type: Program
# > Availability: Anyone is welcome to use it
