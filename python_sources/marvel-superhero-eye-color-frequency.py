#!/usr/bin/env python
# coding: utf-8

# This is my first ever kernel. 
# 
# My goal is to make a simple histogram that measures the frequency of eye colors in Marvel Superheroes.
# 
# prediction: brown is most frequent.

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

df_marvel = pd.read_csv('../input/marvel-wikia-data.csv', index_col=1)

# cleaning up dataframes
df_marvel['src'] = 'marvel'
df_marvel = df_marvel.drop(columns=["GSM", "urlslug", "page_id"])

df_marvel.head()


# This is the data that I am working with. More specifically, I want the eye colors.

# In[ ]:


df_marvel.EYE = df_marvel.EYE.str.split().str.get(0).astype('category')
df_eye = df_marvel[["EYE"]]
df_eye = df_eye.dropna(thresh=1)

df_eye.head()


# As can be seen, I have a dataframe of marvel superheroes and their corresponding eye colors now. But what are all of the colors?

# In[ ]:


eye_colors = []

for label, color in df_eye.iterrows():
    if color['EYE'] not in eye_colors:
        eye_colors.append(color['EYE'])

print(eye_colors)


# Very interesting colors. I wonder, who has pink eyes?

# In[ ]:


df_eye[df_eye['EYE'] == 'Pink']


# Never heard of them. Onto plotting.

# In[158]:


df_eye.apply(pd.value_counts).plot.bar(title='Marvel Superhero Eye Color Frequency', figsize=(20, 9), fontsize=16, stacked=True, color='g')
plt.xlabel("Eye Color")
plt.ylabel("Frequency")
plt.rcParams.update({'font.size': 16})
plt.legend(['Color'])

plt.show()


# Looks like blue is the most common eye color. Brown is a close runner up. 
