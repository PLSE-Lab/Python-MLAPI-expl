#!/usr/bin/env python
# coding: utf-8

# #Testing manipulating and displaying data

# First, we import the libraries we need:

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pylab as pl


# Then, we read in the data, prepping it so we can delete NA values in a single shot:

# In[ ]:


sentinels = {'speaking_line': ['FALSE'], 'normalized_text': ['']}
cols = ['id', 'episode_id',	'number',  'raw_text',	'timestamp_in_ms',	'speaking_line',	
        'character_id',	'location_id',	'raw_character_text',	'raw_location_text',
        'spoken_words',	'normalized_text',	'word_count']
df = pd.read_csv("../input/simpsons_script_lines.csv",
                    names = cols,
                    error_bad_lines=False,
                    warn_bad_lines=False,
                    low_memory=False,
                    na_values = sentinels)
print(len(df))
df = df.dropna()
print(len(df))


# In[ ]:


df.describe()


# In[ ]:


lines_ep_raw = df.groupby(['episode_id']).size().sort_values(ascending=True)
lines_per_episode = lines_ep_raw[1:]
lines_per_episode.describe()


# In[ ]:


lines_per_char_ep = df['word_count'].groupby([df['character_id'], df['episode_id']])
lines_per_char_ep.describe()


# In[ ]:


data = [ lines_per_episode ]

bins = np.linspace(0, 350,50)
bins2 = np.linspace(0, 350, 100)
plt.figure(1)
plt.boxplot(data)


# In[ ]:


# Calculate the statistics from the data to create the "best fit" normal distributions
mean = np.mean(data)
var = np.var(data)
sdev = np.sqrt(var)
pl.hist(data,bins,normed = 'true',color = 'blue')
pl.plot(bins2,pl.normpdf(bins2,mean,sdev), color = 'red')
pl.xlabel('lines per episode')

