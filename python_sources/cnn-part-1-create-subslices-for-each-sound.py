#!/usr/bin/env python
# coding: utf-8

# This is the 1st part of my mini series relating to Detecting Respiratory Disease with the use of Respiratory Audio (breathing sounds). For this kernel, we're only going to slice each audio file into subslices which is defined by the txt files. 
# 
# - Part 2: [Split into train and test](https://www.kaggle.com/danaelisanicolas/cnn-part-2-split-to-train-and-test)
# - Part 3: [Create spectrogram images from audio](https://www.kaggle.com/danaelisanicolas/cnn-part-3-create-spectrogram-images)
# - Part 4: [Create and train a VGG16 model with the spec images](https://www.kaggle.com/danaelisanicolas/cnn-part-4-training-and-modelling-with-vgg16)
# 
# Let's start.
# 
# Here you'll see that i'm importing librosa and soundfile which are python packages that deals with audio files.
# * [Librosa](https://librosa.github.io/librosa/)
# * [Soundfile](https://pysoundfile.readthedocs.io/en/latest/)
# 
# Boxplot_stats will be used later to see outliers. More of this later

# In[ ]:


import pandas as pd
import numpy as np
import math

import librosa as lb # https://librosa.github.io/librosa/
import soundfile as sf # https://pysoundfile.readthedocs.io/en/latest/

import os

import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats


# I had to look first where the audio and text files are by using !ls

# In[ ]:


get_ipython().system('ls ../input/respiratory-sound-database/respiratory_sound_database/Respiratory_Sound_Database')


# Load the patient diagnosis file first and check all the unique diagnosis we have in our data.
# 
# This is necessary in how we will sort our output later.

# In[ ]:


#load patient diagnosis.csv

diag_csv = '../input/respiratory-sound-database/respiratory_sound_database/Respiratory_Sound_Database/patient_diagnosis.csv'
diagnosis = pd.read_csv(diag_csv, names=['pId', 'diagnosis'])
diagnosis.head()


# In[ ]:


ds = diagnosis['diagnosis'].unique()
ds


# Next we will need to read all the unique files in our dataset. This is done by using the os.listdir function with the condition of checking only .txt files.
# 
# Note: We can also use the condition to check .wav files. Eitherway we'll just check all unique files. Condition is needed because if we're checking all kinds of files, we may get replicates. Our dataset consists of .txt and its equivalent .wav files.

# In[ ]:


#get all text files
audio_text_loc = '../input/respiratory-sound-database/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files'
files = [s.split('.')[0] for s in os.listdir(path = audio_text_loc) if '.txt' in s]
files


# We know that our filenames have a certain meaning in them. We get all info (or tokens) by splitting the filename by using "_" as separators. We define a function to do this task.

# In[ ]:


def tokenize_file(filename):
    return filename.split('_')


# Now that we have our files list, we have to read each one to get the crackles and wheezes information--including when in the audio file it is recorded (start and end time denoted in seconds).
# 
# I've created the files_df to compile all these data (which includes the patient id and acquisition mode--stereo or mono--which will be used later).

# In[ ]:


#read each file

files_ = []
for f in files:
    df = pd.read_csv(audio_text_loc + '/' + f + '.txt', sep='\t', names=['start', 'end', 'crackles', 'wheezes'])
    df['filename'] = f
    #get filename features
    f_features = tokenize_file(f)
    df['pId'] = f_features[0]
    df['ac_mode'] = f_features[3]
    
    files_.append(df)
    
files_df = pd.concat(files_)
files_df.reset_index()
files_df.head()


# We want to combine the 2 dataframes we've made so far. However to do that, we have to make sure that the column where we'll combine them must have the same data type.
# 
# In this case I just changed the files_df pId column to float64 to be the same as the diagnosis pId dataframe. We can also use int32 to minimise the data allocation in our system however that'll not be our concern for now.
# 
# Once the 2 columns have the same data type, we'll use the pandas merge function to combine the 2 dataframes.

# In[ ]:


files_df.info()


# In[ ]:


diagnosis.info()


# In[ ]:


files_df['pId'] = files_df['pId'].astype('float64')
files_df.info()


# In[ ]:


files_df = pd.merge(files_df, diagnosis, on='pId')
files_df.head()


# We want to slice the wav file into subslices to get the pure breathing part of the audio file. Again, this is denoted by the start and end times mentioned in the txt files. 
# 
# I've defined the slice_data function to do this.

# In[ ]:


#code taken from eatmygoose https://www.kaggle.com/eatmygoose/cnn-detection-of-wheezes-and-crackles
def slice_data(start, end, raw_data,  sample_rate):
    max_ind = len(raw_data) 
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)
    return raw_data[start_ind: end_ind]


# So.. sure we can slice the audio files, but there are things we need to consider as well
# - We need to make sure that they will have the same length (this is in preparation for feeding them into the model for training later)
# - If they're not the same length, then we have to pad the audio with silent (or zeroes) sounds.
# - For the length, we have to know what is the optimal length of time we should use.
# 
# For the next part I tried to get the max length per slice that we currently have in our dataframe

# In[ ]:


files_df['len_per_slice'] = files_df['end'].sub(files_df['start'], axis = 0) 
max_len_per_slice = max(files_df['len_per_slice'])
max_len_per_slice


# 16 seconds long? Is there someone who can have 1 breathe as long as 16 seconds? No, obviously.
# 
# So we try and understand our data and check the outliers and the relative maximum of the dataset.

# In[ ]:


plt.scatter(files_df['len_per_slice'], y=files_df.index)


# In[ ]:


box = plt.boxplot(files_df['len_per_slice'])


# And define the forced maximum length force_max_len as round up of the relative max (5.48 sec round up to 6 sec) length of time.

# In[ ]:


force_max_len = math.ceil(boxplot_stats(files_df['len_per_slice'])[0]['whishi'])
force_max_len


# Next is we have to compute the length of the raw data of the slices. I defined a function compute_len to do this.
# 
# Remember we had to store the acquisition mode? It's because it has different computation for stereo and mono.
# - Stereo: (Sampling rate * time) * 2
# - Mono: (Sampling rate * time)

# In[ ]:


def compute_len(samp_rate=22050, time=force_max_len, acquisition_mode=0):
    '''Computes the supposed length of sliced data
        samp_size = sample size from the data
        samp_rate = sampling rate. by default since we're working on 24-bit files, we'll use 96kHz
        time = length of time for the audio file. by default we'll use the max we have which is 5.48
        acquisition_mode = either mono or stereo. 0 for mono, 1 for stereo
    '''
    comp_len = 0
    if acquisition_mode == 1: #ac mode is single channel which means it's 'mono'
        comp_len = samp_rate * time
    else: #stereo
        comp_len = (samp_rate * time) * 2

    return comp_len


# We're almost ready to create new wav files based on the slices. There's one more problem: we must prepare where to store these new wav files.
# 
# So we have to create our output directory and subfolders as defined by the unique diagnosis.

# In[ ]:


#create output path
os.makedirs('output')


# In[ ]:


for d in ds:
    path = os.path.join('output', d)
    os.makedirs(path)


# Which we can now check using !ls

# In[ ]:


get_ipython().system('ls')


# Alright time to roll! Now that everything is set, let's start processing the files now. 
# 
# The first thing you might notice is the declaration of the i = 0. This will be used in saving of files later. As you may know, each wav file will consist of different slices. When saving these slices, we'll add the ith number of slice. For example, 104_1b1_Ll_sc_Litt3200.wav might have 4 slices, so we'll save them as 
# - 104_1b1_Ll_sc_Litt3200_0, 
# - 104_1b1_Ll_sc_Litt3200_1, 
# - 104_1b1_Ll_sc_Litt3200_2, 
# - 104_1b1_Ll_sc_Litt3200_3
# 
# Next, for each file 
# - get the filename (which will be used for saving later)
# - get the start and end times for the processing of the sliced wav files
# - get the diagnosis so we'll know which subfolder we'll save the sliced wav file
# 
# If the slice is greater than force_max_len, force the slice to only be force_max_len long.
# 
# Using the librosa function load, load the wav file which returns raw data and the sampling rate.
# Then we feed the raw data to the compute_len function we defined earlier to get the expected length of data.
# Pad the sliced audio with zeroes using the librosa function util.pad_center.
# Then finally, save this file to our destination path which includes which subfolder we'll save it.
# 
# TLDR;
# - Read the audio file
# - Slice the audio file
# - Pad in case it's less than 6 sec
# - Save

# In[ ]:


i = 0 #iterator for file naming

for idx, row in files_df.iterrows():
    filename = row['filename']
    start = row['start']
    end = row['end']
    diag = row['diagnosis']
    
    #check len and force to 6 sec if more than that
    if force_max_len < end - start:
        end = start + force_max_len
    
    aud_loc = audio_text_loc + '/' + f + '.wav'
    
    if idx != 0:
        if files_df.iloc[idx-1]['filename'] == filename:
            i=i+1
        else:
            i=0
    n_filename = filename + '_' + str(i) + '.wav'
    path = 'output/' + diag + '/' + n_filename
    
    print('processing ' + n_filename + '...')

    data, samplingrate = lb.load(aud_loc)
    sliced_data = slice_data(start=start, end=end, raw_data=data, sample_rate=samplingrate)
    
    #pad audio if < forced_max_len
    a_len = compute_len(samp_rate=samplingrate, acquisition_mode=row['ac_mode']=='sc')
    padded_data = lb.util.pad_center(sliced_data, a_len)

    sf.write(file=path, data=padded_data, samplerate=samplingrate)


# Then it's done! We can check using !ls again

# In[ ]:


get_ipython().system('ls output/')


# In[ ]:




