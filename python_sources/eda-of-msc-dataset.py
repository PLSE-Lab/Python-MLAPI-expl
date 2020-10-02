#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# ## Setup

# In[ ]:


get_ipython().system(' git clone https://gitlab.com/smc/msc/')


# In[ ]:


import pandas as pd
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
import plotly.express as px


# In[ ]:


speech = pd.read_csv('msc/speech.tsv', sep='\t')
sentences = pd.read_csv('msc/sentences.tsv', sep='\t')
users = pd.read_csv('msc/users.tsv', sep='\t')


# ## Peeking at the data

# In[ ]:


speech.head()


# In[ ]:


sentences.head()


# In[ ]:


users.head()


# In[ ]:


print(f"{speech.shape[0]} sounds has been recorded by {users.shape[0]} users")


# ## Checking for null values

# In[ ]:


speech.isnull().sum()


# In[ ]:


sentences.isnull().sum()


# - Fortunately there is no null values in our dataset

# ## Sentences based on various categories

# In[ ]:


df = sentences.category.value_counts()
colors = ["gold", "mediumturquoise", "darkorange", "lightgreen"]
fig = px.pie(df, df.index, df.values, labels={"index": "Categories"})
fig.update_traces(
    hoverinfo="label+percent",
    textinfo="value",
    textfont_size=20,
    marker=dict(colors=colors, line=dict(color="#000000", width=2)),
)
fig.update_layout(title="No of sentences belonging to each category in MSC dataset")

fig.show()


# ## Top 15 most contributed users to MSC

# In[ ]:


users_speech = pd.merge(speech, users, left_on='user', right_on='id')
users_speech.name.value_counts()[:15]


# ## Total time duration

# In[ ]:


sum = 0
filenames = os.listdir('audio/')
for f in filenames:
    y, sr = librosa.load(os.path.join('audio/', f))
    sum = sum + librosa.get_duration(y=y, sr=sr)


# ## Categories of Speech sentences

# In[ ]:


speech_sentences = pd.merge(speech, sentences, left_on='sentence', right_on='id')
df = speech_sentences.category.value_counts()
colors = ["gold", "mediumturquoise", "darkorange", "lightgreen"]
fig = px.pie(df, df.index, df.values, labels={"index": "Categories"})
fig.update_traces(
    hoverinfo="label+percent",
    textinfo="value",
    textfont_size=20,
    marker=dict(colors=colors, line=dict(color="#000000", width=2)),
)
fig.update_layout(title="Categories of user utterances in MSC dataset")

fig.show()


# ## Good quality speech 
# 
# - criteria: min 3 upvotes

# In[ ]:


sound = speech[speech.vote != 'default']
sound['vote'] = sound['vote'].astype(str).astype(float)
good_sound = sound[sound['vote']>=3]


# In[ ]:


good_sound = good_sound.reset_index()
print("No of good sound sampeles is", good_sound.shape[0])
good_sound.head()


# ## fin.
