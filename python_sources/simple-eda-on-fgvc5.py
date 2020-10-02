#!/usr/bin/env python
# coding: utf-8

# Hello
# In this notebook i try to explore given  data and apply some visulization if possible. Dont forget to vote for me.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:



train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
val = pd.read_json('../input/validation.json')


# In[ ]:


train.head()


# In[ ]:


val.head()


# In[ ]:


test.head()


# In[ ]:


train.columns


# In[ ]:


train['image_id'] = train.annotations.map(lambda x: x['image_id'])
train['label_id'] = train.annotations.map(lambda x: x['label_id'])
train['url'] = train.images.map(lambda x: x['url'][0])
train.drop(columns=['annotations', 'images'], inplace=True)
train.head()


# In[ ]:


train.isnull().sum()


# In[ ]:


val['image_id'] = val.annotations.map(lambda x: x['image_id'])
val['label_id'] = val.annotations.map(lambda x: x['label_id'])
val['url'] = val.images.map(lambda x: x['url'][0])
val.drop(columns=['annotations', 'images'], inplace=True)
val.head()


# In[ ]:


val.isnull().sum()


# In[ ]:


# test['image_id'] = test.annotations.map(lambda x: x['image_id'])
# test['label_id'] = test.annotations.map(lambda x: x['label_id'])
test['url'] = test.images.map(lambda x: x['url'][0])
test.drop(columns=[ 'images'], inplace=True)
test.head()


# In[ ]:


test.isnull().sum()


# In[ ]:


from IPython.display import Image
from IPython.core.display import HTML 
Image(url= train.url[50],width=200,height=200)


# In[ ]:


from IPython.display import Image
from IPython.core.display import HTML 

def display_category(urls, category_name):
    img_style = "width: 180px; margin: 0px; float: left; border: 1px solid black;"
    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in urls.head(12).iteritems()])

    display(HTML(images_list))


# In[ ]:


urls = train['url'][15:30]
display_category(urls, "")


# In[ ]:


urls = test['url'][15:30]
display_category(urls, "")


# In[ ]:


urls = val['url'][15:30]
display_category(urls, "")


# In[ ]:


train.label_id.value_counts().sort_values(ascending=False).head()


# The top most labels are shown above, let us display them

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


plt.figure(figsize=(30,5))
sns.countplot(train.label_id)
plt.show()


# In[ ]:


(train.url[11])


# In[ ]:


train.columns


# In[ ]:


a = train.label_id.unique()
a


# In[ ]:


from IPython.core.display import HTML 
from ipywidgets import interact
from IPython.display import display

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from IPython.display import Image
from IPython.core.display import HTML 

def display_category1(urls, category_name):
    img_style = "width: 180px; margin: 0px; float: left; border: 1px solid black;"
    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in urls.head(12).iteritems()])

    display(HTML(images_list))


# 
# so how can we proceed to solve this problem...any idea please comment below.
#  
#  **If you like my  notebook please vote for me.**
#  
#  more in pipeline, stay tuned.
#  
#  Thank you.

# In[ ]:




