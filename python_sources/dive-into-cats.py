#!/usr/bin/env python
# coding: utf-8

# ## Diving into Categories
# 
# It looks like there have been many categories introduced in this competition. Let's take a look into some of those categories and find out what each categories logically shows. For this purpose, in this short notebook, I will try to select top three most frequent and least frequent images. Hopefully, knowing those categories and showing some of the picture would be helpful in tackling the problem.

# In[16]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from IPython.core.display import HTML  # for plotting images in a simpler format
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Import data 
# Let's load the data and gain some insights on different categories 

# In[3]:


#load data
data = json.load(open('../input/train.json'))
#Saved On Data Frame
data_url = pd.DataFrame.from_dict(data['images'])
labels = pd.DataFrame.from_dict(data['annotations'])
train_data = data_url.merge(labels, how='inner', on=['image_id'])
train_data['url'] = train_data['url'].str.get(0)
del data, data_url, labels
train_data.head(5)


# Let's gain some information on the distrbution of the labels and how frequency of each labels were distributed throughout the train, test, and validation data. Next step, I would dig more into the images to find out if train/validation/test kind of have similar distributions or not

# In[31]:


label_frequency = pd.DataFrame(train_data.label_id.value_counts())
label_frequency.reset_index(level=0, inplace=True)
label_frequency.plot('index', 'label_id', kind='bar', figsize=(20,10), title="distribution of the labels in training data")


# In[4]:


#train_data.label_id.plot(kind='bar', alpha=0.5)
train_data.describe()
print('top three labels \n', str(train_data.label_id.value_counts().head(3)))  
print('least frequent three labels \n', str(train_data.label_id.value_counts().tail(3)))  
# Top 3 and bottom 3 labels in the images are below though all the images are not distributed Uniformly


# Let's pick some of the images with the top 3 and bottom 3 labels. We might end up getting more pictures for the least frequent images in the training sets

# In[5]:


def display_image(label_id, number_of_display, seed=123):
    labeled_data = train_data[train_data.label_id == label_id]
    labeled_data = labeled_data.sample(number_of_display, random_state=seed)
    img_style = "width: 180px; margin: 0px; float: left; border: 1px solid black;"
    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for u in labeled_data.url])
    display(HTML(images_list))


# In[6]:


display_image(label_id=20, number_of_display=9)


# In[7]:


display_image(label_id=42, number_of_display=9)


# In[8]:


display_image(label_id=92, number_of_display=9)


# In[9]:


display_image(label_id=124, number_of_display=9)


# In[10]:


display_image(label_id=66, number_of_display=9)


# In[11]:


display_image(label_id=83, number_of_display=9)


# In[ ]:





# In[ ]:




