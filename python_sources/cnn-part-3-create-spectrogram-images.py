#!/usr/bin/env python
# coding: utf-8

# This is the part 3 of my mini series on Detecting Respiratory Disease. You can check the other parts here:
# - Part 1: [Creating slices from the audio defined by the .txt files](https://www.kaggle.com/danaelisanicolas/cnn-part-1-create-subslices-for-each-sound)
# - Part 2: [Splitting to train and test](https://www.kaggle.com/danaelisanicolas/cnn-part-2-split-to-train-and-test)
# - Part 4: [Creating a model and training spectrogram images](https://www.kaggle.com/danaelisanicolas/cnn-part-4-training-and-modelling-with-vgg16)
# 
# Now.. To be honest I can't run this fully on Kaggle due to memory constraints. But don't worry, i'll upload the resulting output on the next part of this series. Rather, you can try running this code on your local machine (of course, you have to download all needed input files for this kernel)
# 
# Without further adeu, let's get into the code

# In[ ]:


get_ipython().system("ls '../input/cnn-part-2-split-to-train-and-test/output'")


# For creating spectrogram images, I always use the librosa python library. It has a nifty function of creating spectrogram images. You can learn more about it here: https://librosa.github.io/librosa

# In[ ]:


import librosa as lb
from librosa.display import specshow

import matplotlib.pyplot as plt

import os
from os import listdir
from os.path import isfile, join

import pandas as pd


# First we have to create the directories where we'll store our output. And set the location of the wav files (split into train and val) as input -- generated from part 2.

# In[ ]:


os.makedirs('output')
os.makedirs('output/train')
os.makedirs('output/val')


# In[ ]:


files_loc = '../input/cnn-part-2-split-to-train-and-test/output/'


# In order to loop through the directories which are named based on the diagnosis, I need to get all its unique values. We'll name these as categories.

# In[ ]:


diagnosis_csv = '../input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv'
diagnosis = pd.read_csv(diagnosis_csv, names=['pId', 'diagnosis'])
diagnosis.head()


# In[ ]:


categories = diagnosis['diagnosis'].unique()
categories


# Then create the directories for each category in both train and validation directories

# In[ ]:


for cat in categories:
    os.makedirs('output/train/' + cat)
    os.makedirs('output/val/' + cat)


# In converting audio to images, we must check whether we're reading an audio file right? I defined is_wav function to check the file format.

# In[ ]:


def is_wav(filename):
    '''
        Checks if files are .wav files
        Utility tool in converting wav to png files
    '''
    return filename.split('.')[-1] == 'wav'


# The next function is the core of this kernel. This loads each file in each directory in each split ([split]/[category]/[file]) and converts it to a spectrogram image then saves it on the output directory that should have the same file structure--[split]/[category]/[file]

# In[ ]:


#create images using librosa spectogram
def convert_to_spec_image(file_loc, filename, category, is_train=False, verbose=False):
    ''' 
        Converts audio file to spec image
        Input file includes path
        Saves the file to a png image in the save_directory
    '''
    train_ = 'train/'
    val_ = 'val/'
    
    loc = file_loc + train_ + category + '/' + filename
    if is_train == False:
        loc = file_loc + val_ + category + '/' + filename

    if verbose == True:
        print('reading and converting ' + filename + '...')
        
    y, sr = lb.load(loc)

    #Plot signal in
    plt.figure(figsize=(10,3))
    src_ft = lb.stft(y)
    src_db = lb.amplitude_to_db(abs(src_ft))
    specshow(src_db, sr=sr, x_axis='time', y_axis='hz')  
    plt.ylim(0, 5000)
    
    save_directory = 'output/'
    filename_img = filename.split('.wav')[0]
    
    save_loc = save_directory + train_ + category + '/' + filename_img + '.png'
    if is_train == False:
        save_loc = save_directory + val_ + category + '/' + filename_img + '.png'
        
    plt.savefig(save_loc)
    
    if verbose == True:
        print(filename + ' converted!')
        
    plt.close()


# Now we can start converting! 
# 
# NOTE!!!! I commented out the code that converts the images directly. THe problem is Kaggle can't handle the output that's created. You can however still run this notebook into your machine. Just uncomment the *convert_to_spec_image line and you're good to go!

# In[ ]:


split = ['train', 'val']

for s in split:
    for cat in categories:
        print('-' * 100)
        print('working on ' + cat + '...')
        print('-' * 100)

        files = [f for f in listdir(files_loc + s + '/' + cat + '/') if isfile(join(files_loc + s + '/' + cat + '/', f)) and is_wav(f)]
        for f in files:
            #convert_to_spec_image(file_loc = files_loc, category=cat, filename=f, is_train=(s == 'train'), verbose=True)


# And that's it. We now have our spec images which we can now use for our CNN model. You can get download the output on Part 4 (links above)
