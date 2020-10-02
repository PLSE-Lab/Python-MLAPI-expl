#!/usr/bin/env python
# coding: utf-8

# # Part 1 - Data Exploration 
# Eu Jin Lok
# <br>Kernel post for __[Speech Accent Archive](https://www.kaggle.com/rtatman/speech-accent-archive)__ on Kaggle
# <br>2 December 2019
# 
# 
# # Understanding the data and setting an objective
# 
# 
# In this notebook we will go into the details of how to explore audio data and converge on an objective, an objective which will most likely involve some kind of **deep learning**, because its awesome. If I do write a blog post about this, I will update this kernel. But for now, just the Jupyter notebook as a kernel, and my very first one!
# 
# Before we begin, I just wanted to say that my first real heavy involvement in audio was back in March 2018 whilst doing the Audio competition on __[kaggle](https://www.kaggle.com/c/freesound-audio-tagging)__, Thanks to that competition and the awesome community support, I had learnt alot and so I wanted to contribute back to the community in the same way. So without further ado, lets begin

# In[ ]:


import pandas as pd       
import os 
import math 
import numpy as np
import matplotlib.pyplot as plt  
import IPython.display as ipd  # To play sound in the notebook
import librosa
import librosa.display
import os
os.chdir("../input")
print(os.listdir("../input"))


# ------------------------------
# After loading the libraries and setting our directory path, lets check out the meta datafile to see what we're dealing with 

# In[ ]:


#load the data 
df = pd.read_csv("speakers_all.csv", header=0)

# Check the data
print(df.shape, 'is the shape of the dataset') 
print('------------------------') 
print(df.head())


# ------------------------------
# I noticed some strange empty columns in the last 3 columns of the dataset. Lets clean it up and run some more stats and plots

# In[ ]:


df.drop(df.columns[9:12],axis = 1, inplace = True)
print(df.columns)
df.describe()


# ------------------------------
# Not sure what the differences are for age and age_onset but not important at this stage. Nothing really outstanding at the moment... lets soldier on

# In[ ]:


# Very rough plot
df['country'].value_counts().plot(kind='bar')


# In[ ]:


# Ok so that plot wasn't very good for that category. Lets try another category... 
df['native_language'].value_counts().plot(kind='bar')


# In[ ]:


# That's lots of categories too! Ok so maybe lets try a different way...
df.groupby("native_language")['age'].describe().sort_values(by=['count'],ascending=False)


# ------------------------------
# Much better. No fancy visuals unfortunately but at least the insight comes through more in this format. The thing to note here is the lower number of Hindi speakers, which according to __[wikipedia](https://en.wikipedia.org/wiki/List_of_languages_by_number_of_native_speakers)__ is the 4th most spoken language, with alot of caveats of course. Eitherways, this tables looks like a pretty good representative sample in general to me. Lets look at country of origin again

# In[ ]:


# Check country of origin again...
df.groupby("country")['age'].describe().sort_values(by=['count'],ascending=False)


# ------------------------------
# There's more native languages than there are countries which I suppose makes sense, although a hypothesis withstanding. A sankey type plot here would be interesting and very apprpriate to visualise this relationship, but lets park it for now as a seperate task. Right now, lets continue on with our main objective...

# In[ ]:


# Create DTM of counts 
df.groupby("sex")['age'].describe()


# ------------------------------
# hmmm... must be a typo. Lets notify @Rachel Tatman about this observation... continue on 

# In[ ]:


# birthplace
df.groupby("birthplace")['age'].describe().sort_values(by=['count'],ascending=False)


# ------------------------------
# Birthplace is a very sparce datapoint with 1290 unique categories with very few observations in each one. Again could be interesting to see the patterns of Birthplace and Country relationship. Either a Network analysis or a Sankey plot may shed some light on whether all the Seoul birthplace observation equates to country. Ie. Could they be South Koreans living else where such as China or USA? 
# 
# Eithercase, park for a seperate mini project. Lets look at the file_missing column

# In[ ]:


# file_missing
df.groupby("file_missing?")['age'].describe().sort_values(by=['count'],ascending=False)


# ------------------------------
# 2140 files with 32 missing. What does this actually mean? I read the overview page and there's no mention of this. So, lets go see it for ourselves... by counting the number of audio files we got

# In[ ]:


# Count the total audio files given
print (len([name for name in os.listdir('../input/recordings/recordings') if os.path.isfile(os.path.join('../input/recordings/recordings', name))]))


# ------------------------------
# huh? 2138.... We have 2 missing audio files. Well, lets keep that in mind for now. We'll eventually find out which one's we're missing, and is really inconsequential either way given its just 2 files. Let continue on to the last column, the filename column

# In[ ]:


# filename column. This time we just print out the first 10 records. 
df.groupby("filename")['age'].describe().sort_values(by=['count'],ascending=False).head(10)


# ------------------------------
# Wait, there's some files that have the same filename. On closer inspection however, I suspect these filenames also have missing audio files. In which case it is ok. I have a suspicion that these dupe filenames also have missing audio files. So, lets have a look at doing a cross-tab of the 'filename' and 'file_missing?' column...

# In[ ]:


# Cross-tab. Again, just print the first 10 record 
df.groupby("filename")['file_missing?'].describe().sort_values(by=['count'],ascending=False).head(10)
# pd.crosstab(df['filename'],df['file_missing?']) as an alternative method 


# --------------------------------------
# Ok so our suspicion is right, the filename with duplicate names have all missing audio files. Perfect! Everything checks out. We can go ahead a read in the audio files, and listen in to a few. We'll look at 'arikaans1' and 'mandarin46' since its on our periperal vision

# In[ ]:


# Play afrikaans1
fname1 = 'recordings/recordings/' + 'afrikaans1.mp3'
ipd.Audio(fname1)


# In[ ]:


# Play mandarin46
fname2 = 'recordings/recordings/' + 'mandarin46.mp3'
ipd.Audio(fname2)


# In[ ]:


# lets have a listen to a male voice. 
print(df.groupby("filename")['sex'].describe().head(10))
fname3 = 'recordings/recordings/' + 'agni1.mp3'   
ipd.Audio(fname3)


# ------------------------------
# Ok, so we've come to a point where we need to make a decision now. There's a few objectives worth pursuing on top of my head and they are: 
# - building a gender predictor from voice
# - building an accent predictor by Country from voice (or Birthplace)
# 
# All we could build all 2 applications above, starting from the easiest first being the gender predictor. The gender predictor will serve as our prototype and once we've built it, we'll expand to Country, and then maybe by Birthplace. I'm not even sure if Birthplace is viable objective but lets re-evaluate when we circle back to this. For now, **lets run with Gender first**. Also note that we don't have to limit ourselves with supervised modelling. There's many more we can do: 
# - Audio fingerprinting 
# - Emotion analysis (Text and Voice)
# - Speed, inflection etc etc
# - Others 
# 
# There's alot you can do with audio, but we'll look at these at a later stage. Meantime, the show must go on, and we will stick to our simple objective. Lets now run a few more examples of male and female audio files. This time, I want to hear the US Southern Accent. Cause I've always liked that accent and find it fascinating. =D

# In[ ]:


print(df[df['birthplace'].str.contains("kentucky",na=False)])
fname4 = 'recordings/recordings/' + 'english385.mp3'   
ipd.Audio(fname4)


# In[ ]:


fname5 = 'recordings/recordings/' + 'english462.mp3'   
ipd.Audio(fname5)


# ------------------------------
# The male version which is filename 'english462' doesn't have a strong Southern accent. And there's some distrotion of the audio at the start. Could pose a problem for our accent predictor by Birthplace, but nothing to worry about for Gender. Looking at the previous tables and plots, seems like there's some potential age correlation here. So lets hear one final one! 

# In[ ]:


fname6 = 'recordings/recordings/' + 'english381.mp3'   # An older male 
ipd.Audio(fname6)


# ------------------------------
# Ok so I'll end this kernel here now, and we'll go ahead with creating a gender predictor as our first mini-objective, with the ultimate objective being to create an accent predictor. The next logical step after this is to analyse the audio files itself and extract features from it, which we'll do in the Part 2 of this series, coming soon in the next 2 to 3 weeks hopefully!
