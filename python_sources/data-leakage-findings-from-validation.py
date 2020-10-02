#!/usr/bin/env python
# coding: utf-8

# I find some data leakage from the validation set. Though I am not sure whether it is intentional for it being called 'validation' set. I am actually not sure about the validation set purpose.

# In[1]:


import time
script_start_time = time.time()

import pandas as pd
import numpy as np
import json
import gc

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.plotly as py
import cufflinks as cf
cf.set_config_file(offline=True, world_readable=True, theme='ggplot')
plt.rcParams["figure.figsize"] = 12,8
sns.set(rc={'figure.figsize':(20,12)})
plt.style.use('fivethirtyeight')

pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 50)
import warnings
warnings.filterwarnings('ignore')

# Data path
data_path = '../input/'


# ### 1. Load Data
# Reference: [](https://www.kaggle.com/badalgupta/simple-data-exploration)

# In[2]:


# 1. Load data =================================================================
print('%0.2f min: Start loading data'%((time.time() - script_start_time)/60))
train={}
test={}
validation={}
with open('%s/train.json'%(data_path)) as json_data:
    train= json.load(json_data)
with open('%s/test.json'%(data_path)) as json_data:
    test= json.load(json_data)
with open('%s/validation.json'%(data_path)) as json_data:
    validation = json.load(json_data)

print('Train No. of images: %d'%(len(train['images'])))
print('Test No. of images: %d'%(len(test['images'])))
print('Validation No. of images: %d'%(len(validation['images'])))

# JSON TO PANDAS DATAFRAME
# train data
train_img_url=train['images']
train_img_url=pd.DataFrame(train_img_url)
train_ann=train['annotations']
train_ann=pd.DataFrame(train_ann)
train=pd.merge(train_img_url, train_ann, on='imageId', how='inner')

# test data
test=pd.DataFrame(test['images'])

# Validation Data
val_img_url=validation['images']
val_img_url=pd.DataFrame(val_img_url)
val_ann=validation['annotations']
val_ann=pd.DataFrame(val_ann)
validation=pd.merge(val_img_url, val_ann, on='imageId', how='inner')

del (train_img_url, train_ann, val_img_url, val_ann)
gc.collect()

print('%0.2f min: Finish loading data'%((time.time() - script_start_time)/60))
print('='*50)


# ### 2. Check Data for Missing or Duplicated Values
# Findings:
# a. No missing values
# b. No duplicates in each data set, but val urls are from test.
# 

# In[3]:


datas = {'Train': train, 'Test': test, 'Validation': validation}

total_url = []
dataset_url = {}
for data_name, data in datas.items():
    print('%s shape: %s'%(data_name, str(data.shape)))
    print('%s Unique imageId: %s'%(data_name, len(data['imageId'].unique())))
    print('%s Unique url: %s'%(data_name, len(data['url'].unique())))
    print('%s NA: '%(data_name)) # No missing values
    print(data.isnull().sum()) # No missing values
    print('%s total unique url: %d'%(data_name, len(set(data['url'].tolist()))))
    total_url = total_url + data['url'].tolist()
    dataset_url[data_name] = data['url'].tolist()

    print('-'*50)


print('Total images: %d'%(len(total_url)))
print('Total unique images: %d'%(len(set(total_url))))
print('Duplicated url: %d'%(len(total_url) - len(set(total_url))))


# 9897? A bit familiar? Yes, same as the number of val rows. 
# Still, let's create a grid to search for the duplicated the url

# In[4]:


# Find the duplicated url
from itertools import product
combinations = list(product(*[datas.keys(), datas.keys()]))
for comb in combinations:
    print('%s inter %s: %d | %d'%(comb[0], comb[1], len(set(dataset_url[comb[0]])), len(set(dataset_url[comb[0]]).intersection(set(dataset_url[comb[1]])))))


# 'Test inter Validation: 39706 | 9897'
# What? Val is part of test?
# Let's confirm the duplicated url by merging test and val on url

# In[6]:


# Confirm the duplicated url
test[['url']].merge(validation[['url']], how = 'inner').shape
test_ = test.merge(validation[['url', 'labelId']], on = 'url',how = 'left')
print('%s NA: '%('Test')) # No missing values
print(test_.isnull().sum() / test_.isnull().count())


# Let's which part of test are leaked

# In[7]:


# Plot imageId of of leaked lable
test_leaked_lableId = test_[~pd.isna(test_['labelId'])]
test_unleaked_lableId = test_[pd.isna(test_['labelId'])]
number_of_known_lableId = validation.shape[0]
plt.plot(test_leaked_lableId['imageId'], '.')
plt.plot(test_unleaked_lableId['imageId'], '.')


# Display images to confirm again the leakage

# In[8]:


from IPython.display import Image
from IPython.core.display import HTML

def display_category(urls, category_name):
    img_style = "width: 180px; margin: 0px; float: left; border: 1px solid black;"
    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in urls.head(12).iteritems()])
    display(HTML(images_list))


# In[10]:


#test data Images
urls = test['url'][1:5]
display_category(urls, "")


# In[11]:


#validation Images
urls = validation['url'][1:5]
display_category(urls, "")


# In[13]:


#train_data Images
urls = train['url'][1:5]
display_category(urls, "")


# Now, we are 100% sure that the first 9897 of test set has their label id in val.
