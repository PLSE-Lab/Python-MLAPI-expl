#!/usr/bin/env python
# coding: utf-8

# After taking the [Data Leakage](https://www.kaggle.com/dansbecker/data-leakage) course for a while, I found it happen every now and then in the real world cases. However, since most of the data in each companies are confidential. I want to take this open dataset as an example. What is Data Leakage and we should be careful of it.

# In[1]:


# Importing libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up matplotlib style 
plt.style.use('ggplot')

# Libraries for wordcloud making and image importing
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image

import tensorflow as tf
from tensorflow import keras
import random


# In[2]:


input_dir = '../input/'
file_path = os.path.join(input_dir, 'Sarcasm_Headlines_Dataset.json')

data = pd.read_json(file_path, lines=True)


# In[3]:


data.shape


# In[4]:


data.head()


# In[5]:


data['is_sarcastic'].value_counts()


# In[6]:


data['website_domain'] = data['article_link'].apply(lambda x: x.split('.com')[0].split('.')[-1])


# In[7]:


data.head()


# In[8]:


data['website_domain'].value_counts()


# In[9]:


data.groupby(['website_domain','is_sarcastic'])['headline'].aggregate('count').unstack().fillna(0)


# So we can see the website domain is a column of data leakage since it literally tell the is sarcastic or not. It seems the theonion website itself post sarcastic posts or the data owner classify in this way. We should drop it and only look at the headline to study this topic instead!
# 
# This is just a short case study. Feel free to let me know if you have any suggestion or feedback. I'll see you guys around.

# In[ ]:




