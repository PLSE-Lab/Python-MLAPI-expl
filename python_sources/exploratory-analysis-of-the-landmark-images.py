#!/usr/bin/env python
# coding: utf-8

# In this version of the notebook, I have made an exploration of the dataset.  I have shown how to access and display images. I have also added a feature source website for these images.
# 
# Look forward to adding more features in my next version of the notebook.

# In[1]:


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


# In[2]:


#reading the training and test data
train_data = pd.read_csv('../input/index.csv')
test_data = pd.read_csv('../input/test.csv')
submission = pd.read_csv("../input/sample_submission.csv")
print("Training data size:",train_data.shape)
print("Test data size:",test_data.shape)


# In[3]:


train_data.head()


# In[26]:


train_data.head()
train_data['url'][44]


# In[29]:


#Displaying number of unique URLs & ids
len(train_data['url'].unique())
len(train_data['id'].unique())


# Therefore, each image has a unique URL and ID

# **Displaying the images**

# In[9]:


#Downloading the images 
from IPython.display import Image
from IPython.core.display import HTML 
def display_image(url):
    img_style = "width: 500px; margin: 0px; float: left; border: 1px solid black;"
    #images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in urls.head(20).iteritems()])
    image=f"<img style='{img_style}' src='{url}' />"
    display(HTML(image))


# In[20]:


#Displaying the images
display_image(train_data['url'][68])


# **Adding the feature website**

# In[13]:


#Adding feature: website from which the image is taken
site_image_train=list()
for url in train_data['url']:
    site_image_train.append((url.split('//', 1)[1]).split('/', 1)[0])
#adding column in train data
train_data['website']=site_image_train

site_image_test=list()
for url in test_data['url']:
    site_image_test.append((url.split('//', 1)[1]).split('/', 1)[0])
#adding column in test data
test_data['website']=site_image_test


# **Displaying the unique websites and distribution of the images among the different websites**

# In[14]:


print(len(train_data['website'].unique()))
train_data['website'].unique()


# In[15]:


ax = sns.factorplot(x="website", kind="count", data=train_data,size=10, palette="muted")
ax.set_ylabels("Number of images")
plt.xticks(rotation=90)
plt.show()


# **There are 17 unique websites and the maximum number of images are from - lh3.googleusercontent.com**

# 

# 

# 

# 

# 
