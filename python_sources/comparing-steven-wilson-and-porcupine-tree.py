#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

sw = pd.read_csv('../input/Steven Wilson.csv') # Steven Wilson
pt = pd.read_csv('../input/Porcupine Tree.csv', nrows=len(sw)) # Porcupine Tree

# remove useless columns 
ignore = ['analysis_url', 'id', 'track_href', 'uri', 'type', 'album', 'name', 'artist', 'lyrics']
sw.drop(ignore, axis=1, inplace=True)
pt.drop(ignore, axis=1, inplace=True)

sw.describe()


# In[ ]:


# custom color palette 
red_blue = ['#19B5FE', '#EF4836']
palette = sns.color_palette(red_blue)
sns.set_palette(palette)


# In[ ]:


# let's compare the songs of SW and PT using histograms
fig = plt.figure(figsize=(15,15))

for i, feature in enumerate(sw):
    ax = plt.subplot(4,4,i+1)
    ax.set_title(feature)
    sw[feature].hist(alpha=0.7, label='Steven Wilson')
    pt[feature].hist(alpha=0.7, label='Porcupine Tree')
    
plt.legend(loc='upper right')


# For full documentation on these features, go [here](https://developer.spotify.com/web-api/get-audio-features/).  
#   
# The reason I compared these two artists is because Steven Wilson used to be a member of Porcupine Tree, but he moved on and is now doing solo albums. However, their sound, style and genre are very similar. That's because Steven Wilson was Porcupine Tree's frontman, main writer and producer. He's really good at music production and his work is worshiped in the progressive rock scene. 
# 
# Anyways, I want to use a machine learning algorithm (specifically, a classifier) that tells me if a song is similar Wilson's style. So in order to build the classifier I have to train it with Wilson's songs. So, **The initial question was...** *should I use Porcupine Tree's songs as training data? *
# 
# Apparently, yes. 
# 
# There are a few differences though. Steven Wilson tends to be sadder (see *valence*), less energetic (see *energy*) and quieter (see *loudness*). Porcupine Tree used to be a little more noisy and energetic, I guess you could call them more "danceable".
# 
# Also, there are some tempo differences... PT is slower than SW. That's because early PT was more experimental. They had more psychodelic tracks, you know: ambient noises, echoes, electronics, people talking (see *speechiness*)... no instruments or lyrics at all. On the other hand, Steven Wilson's solo work is more traditional, he has more classic influences like *King Crimson* and *Yes*. That's why most of his tempos fall in the safe-120-area. Some even say that he just follows the "progressive rock blueprints", but let's not deviate...
# 
# These differences are not *that* significant though, it's normal for some songs to be kind of different from each other. Imagine an album where all the songs have the same valence, tempo and energy... that'd be very boring. 
# 
# There are major similarities between these two artists. Both are kind of sad and use chords in similar ways, i.e. minor and major chords (see *mode*). They like to use voiceovers in their songs (see *speechiness*), both follow the 4/4 "standard" measure. (see *time signature*). Both use the same musical notes in similar ways (see *key*). Finally, they are instrumentally related and you can *kind of* dance to both PT and SW.
