#!/usr/bin/env python
# coding: utf-8

# I have made an exploration of the dataset. I have shown how to access and display images. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#reading the training and test data
train_data = pd.read_csv('../input/google-landmark-dataset/index.csv')
test_data = pd.read_csv('../input/google-landmark-dataset/test.csv')

print("Training data size:",train_data.shape)
print("Test data size:",test_data.shape)


# In[ ]:


train_data.head()


# In[ ]:


train_data.head()
train_data['url'][33]


# In[ ]:


#Displaying number of unique URLs & ids
len(train_data['url'].unique())
len(train_data['id'].unique())


# Therefore, each image has a unique URL and ID
# 
# ### Displaying the images

# In[ ]:


#Downloading the images 
from IPython.display import Image
from IPython.core.display import HTML 
def display_image(url):
    img_style = "width: 500px; margin: 0px; float: left; border: 1px solid black;"
    #images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in urls.head(20).iteritems()])
    image=f"<img style='{img_style}' src='{url}' />"
    display(HTML(image))


# In[ ]:


#Displaying the images
display_image(train_data['url'][55])


# In[ ]:




