#!/usr/bin/env python
# coding: utf-8

# ## Introduction

# This quick kernel provides the code to help you do the following:
#     
#     - Listen to an audio file in a jupyter notebook
#     - Read the files associated with this dataset
#     - Create a spectrogram
#     - Read an audio file as a numpy array
#     - Slice an audio file into sections

# <hr>

# In[ ]:


import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Don't Show Warning Messages
import warnings
warnings.filterwarnings('ignore')


# ## What files are available?

# In[ ]:


# demographic_info.txt was added in version 2

os.listdir('../input')


# In[ ]:


os.listdir('../input/respiratory_sound_database/Respiratory_Sound_Database')


# There are a total of 4 files and one folder.

# ## Listen to an Audio File

# In[ ]:


# Install the pydub library

# Check that kernel Internet is connected before running this cell
get_ipython().system(' pip install pydub')


# In[ ]:


# Play an audio file

from pydub import AudioSegment
import IPython

# We will listen to this file:
# 213_1p5_Pr_mc_AKGC417L.wav

audio_file = '213_1p5_Pr_mc_AKGC417L.wav'

path = '../input/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/' + audio_file

IPython.display.Audio(path)


# ## Read the Files

# ### a. Read the 'demographic_info.txt' file

# In[ ]:


path ='../input/demographic_info.txt'
col_names = ['patient_id', 'age', 'sex', 'adult_bmi', 'child_weight', 'child_height']

# Adult BMI (kg/m2)
# Child Weight (kg)
# Child Height (cm)

df_demo = pd.read_csv(path, sep=" ", header=None, names=col_names)

df_demo.head(10)


# ### b. Read the 'patient_diagnosis.csv' file

# In[ ]:


path = '../input/respiratory_sound_database/Respiratory_Sound_Database/patient_diagnosis.csv'

df_diag = pd.read_csv(path, header=None, names=['patient_id', 'diagnosis'])

df_diag.head(10)


# ### c. Read the 'filename_differences.txt' file

# In[ ]:


path = '../input/respiratory_sound_database/Respiratory_Sound_Database/filename_differences.txt'

df_diff = pd.read_csv(path, sep=" ", header=None, names=['file_names'])

df_diff.head(10)


# ### d. Read the 'filename_format.txt' file
# 
# The file naming format is described on the Kaggle description page for this dataset. That description reads much more clearly than the format displayed here.

# In[ ]:


path = '../input/respiratory_sound_database/Respiratory_Sound_Database/filename_format.txt'

data = open(path, 'r').read()

data


# ### e. List the files in the 'audio_and_txt_files' folder

# In[ ]:


path = '../input/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files'

os.listdir(path)


# ### f. Display the contents of one annotation .txt file

# In[ ]:


path = '../input/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/154_2b4_Al_mc_AKGC417L.txt'

col_names = ['Beginning_of_respiratory_cycle', 'End_of_respiratory_cycle', 'Presence/absence_of_crackles', 'Presence/absence_of_wheezes']

# Respiratory cycle column values are in 'seconds'.
# Presence = 1
# Absence = 0

df_annot = pd.read_csv(path, sep="\t", header=None, names=col_names)

df_annot.head(10)


# ##  Create an Audio  Spectrogram

# A microphone records small variations in air pressure (represented by changes in voltage) over time. The ear percieves these slight variations in air pressure as sound. The spectrogram tells us how much different frequencies are present (loudness) in an audio clip at a moment in time.

# We will use this audio file:<br>
# 154_2b4_Al_mc_AKGC417L.wav
# 
# First let's take a look at the annotation text file for this audio recording to see how many respiration cycles have been recorded.

# In[ ]:


path = '../input/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/154_2b4_Al_mc_AKGC417L.txt'

col_names = ['Beginning_of_respiratory_cycle', 'End_of_respiratory_cycle', 'Presence/absence_of_crackles', 'Presence/absence_of_wheezes']

# Respiratory cycle column values are in 'seconds'.
# Presence = 1
# Absence = 0

df_annot = pd.read_csv(path, sep="\t", header=None, names=col_names)

df_annot.head(20)


# We see that this recording contains 7 respiration cycles. We also see that crackles are present on 5 of the 7 cycles.

# In[ ]:


# Install the pysoundfile library
get_ipython().system(' pip install pysoundfile')


# In[ ]:


import soundfile as sf

# Define helper functions

# Load a .wav file. 
# These are 24 bit files. The PySoundFile library is able to read 24 bit files.
# https://pysoundfile.readthedocs.io/en/0.9.0/

def get_wav_info(wav_file):
    data, rate = sf.read(wav_file)
    return data, rate

# source: Andrew Ng Deep Learning Specialization, Course 5
def graph_spectrogram(wav_file):
    data, rate = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx


# In[ ]:


# plot the spectrogram

path = '../input/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/154_2b4_Al_mc_AKGC417L.wav'


x = graph_spectrogram(path)


# Time is on the x axis and Frequencies are on the y axis. The intensity of the different colours shows the amount of energy i.e. how loud the sound is, at different frequencies, at different times.

# ## Read an audio file as a numpy array

# In[ ]:


# choose an audio file
audio_file = '154_2b4_Al_mc_AKGC417L.wav'

path = '../input/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/' + audio_file

# read the file
data, rate = sf.read(path)

# display the numpy array
data


# ## How to slice a section from an audio file

# In[ ]:


# https://stackoverflow.com/questions/37999150/
# python-how-to-split-a-wav-file-into-multiple-wav-files

from pydub import AudioSegment

# note: Time is given in seconds. Will be converted to milliseconds later.
start_time = 0
end_time = 7

t1 = start_time * 1000 # pydub works in milliseconds
t2 = end_time * 1000
newAudio = AudioSegment.from_wav(path) # path is defined above
newAudio = newAudio[t1:t2]
newAudio.export('new_slice.wav', format="wav")


# In[ ]:


# Lets listen to the new slice

IPython.display.Audio('new_slice.wav')


# ## Helpful Resources

# I found these resources very helpful:
# 
# 1. Andrew Ng Sequence Models Course, Week 3, Trigger Word Detection Assignment<br>
# https://www.coursera.org/learn/nlp-sequence-models
# 
# 2. PySoundFile Library<br>
# https://pysoundfile.readthedocs.io/en/0.9.0/
# 
# 3. How to read a 24 bit wav file<br>
# https://stackoverflow.com/questions/16719453/how-to-read-and-write-24-bit-wav-file-using-scipy-or-common-alternative
# 
# 4. PyDub Library<br>
# https://github.com/jiaaro/pydub
# 

# <hr>
# I hope you enjoy working with this dataset.

# In[ ]:




