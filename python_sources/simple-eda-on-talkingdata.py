#!/usr/bin/env python
# coding: utf-8

# ![](https://media.giphy.com/media/nXOds9I8K8gUg/giphy.gif)
# 
#                                                           **Please dont forget to vote for me at the end.**

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


import os
for i in os.listdir('../input'):
#     print(i)
# print(os.path.getsize('../input/'+i ))
        print(i +'   ' +  str(round(os.path.getsize('../input/' + i) / 1000000,2)) + 'MB')
    


# In[ ]:


train = pd.read_csv('../input/train.csv',nrows=1000000)
test = pd.read_csv('../input/test.csv')
train_sample = pd.read_csv('../input/train_sample.csv')


# In[ ]:


train.shape,test.shape


# In[ ]:


train.columns


# In[ ]:


train.head()


# In[ ]:


test.head()


# Lets us check are there any null values in train dataset

# In[ ]:


train.isnull().sum()


# from the above we can say that 998307 rows dont have attribute_time means they are not converted, we will cross check it later 1000000-998307

# In this data set ip is always categorical and unique.
# 
# app,device,os,channel ther are all given with encoded digits and they are categorical.
# 
# Now we have to convert the above mentioned features into categorical features in both train and test sets

# In[ ]:


variables = ['ip', 'app', 'device', 'os', 'channel']
for v in variables:
    train[v] = train[v].astype('category')
    test[v]=test[v].astype('category')


# In[ ]:


#set click_time and attributed_time as timeseries
train['click_time'] = pd.to_datetime(train['click_time'])
train['attributed_time'] = pd.to_datetime(train['attributed_time'])
test['click_time'] = pd.to_datetime(test['click_time'])

#set as_attributed in train as a categorical
train['is_attributed']=train['is_attributed'].astype('category')


# so now we converted the features accordingly and now we will check how many were converted actually as said above, out of 1000000 values 998307 values dont have attribute time and so attribute value. Lets check.

# In[ ]:


train[['attributed_time', 'is_attributed']][train['is_attributed']==1].describe()


# with the above table we can say that only 1693 were converted (attribute value==1) out of 1000000 which is so less

# In[ ]:


train['is_attributed'].value_counts()


# In[ ]:


train[train['is_attributed']==1].sum()
#this is the blunder mistakes beginners do, actually attribute value==1 it sums up to 1693 and also it sums up remaining features values also which gives wrong
#results so beware


# In[ ]:


train.describe()


# In[ ]:


test['click_id']=test['click_id'].astype('category')
test.describe()


# Now lets check are there any null values in test set.

# In[ ]:


test.isnull().sum()


# we will check how many unique ips are there actually.

# In[ ]:


temp = train['ip'].value_counts().reset_index(name='counts')
temp.columns = ['ip', 'counts']
temp[:10]


# In[ ]:


train['timegap'] = train.attributed_time - train.click_time
train['timegap'].describe()


# Here timegap is the gap between the click time and atrrtibuted time. 

# In[ ]:


train['timegap'].value_counts().sort_values(ascending=False)[:10]


# It took 57secs to get converted ie is_atrributed / downloaded the app. Like this there are 11 downloads. This table shows number of downloads for a given amount of time.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.figure(figsize=(10, 6))
cols = ['ip', 'app', 'device', 'os', 'channel']
uniques = [len(train[col].unique()) for col in cols]
# sns.set(font_scale=1.2)
ax = sns.barplot(cols, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature (from 10,000,000 samples)')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center")


# **More coming, Plase vote for me to do more.**
# 
# **Thank you.**
