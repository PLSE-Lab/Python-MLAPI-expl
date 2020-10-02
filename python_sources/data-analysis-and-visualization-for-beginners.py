#!/usr/bin/env python
# coding: utf-8

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


# The description of the data files from the data page:
# 
# * train.csv - Train data.
# * test.csv - Test data. Same schema as the train data, minus deal_probability.
# * train_active.csv - Supplemental data from ads that were displayed during the same period as train.csv. Same schema as the train data, minus deal_probability.
# * test_active.csv - Supplemental data from ads that were displayed during the same period as test.csv. Same schema as the train data, minus deal_probability.
# * periods_train.csv - Supplemental data showing the dates when the ads from train_active.csv were activated and when they where displayed.
# * periods_test.csv - Supplemental data showing the dates when the ads from test_active.csv were activated and when they where displayed. Same schema as periods_train.csv, except that the item ids map to an ad in test_active.csv.
# * train_jpg.zip - Images from the ads in train.csv.
# * test_jpg.zip - Images from the ads in test.csv.
# * sample_submission.csv - A sample submission in the correct format.
# 
# Let us start with the train file.

# In[ ]:


train_df = pd.read_csv("../input/train.csv", parse_dates=["activation_date"])
test_df = pd.read_csv("../input/test.csv", parse_dates=["activation_date"])
train_df.head()


# The train dataset description is as follows:
# 
# * item_id - Ad id.
# * user_id - User id.
# * region - Ad region.
# * city - Ad city.
# * parent_category_name - Top level ad category as classified by Avito's ad model.
# * category_name - Fine grain ad category as classified by Avito's ad model.
# * param_1 - Optional parameter from Avito's ad model.
# * param_2 - Optional parameter from Avito's ad model.
# * param_3 - Optional parameter from Avito's ad model.
# * title - Ad title.
# * description - Ad description.
# * price - Ad price.
# * item_seq_number - Ad sequential number for user.
# * activation_date- Date ad was placed.
# * user_type - User type.
# * image - Id code of image. Ties to a jpg file in train_jpg. Not every ad has an image.
# * image_top_1 - Avito's classification code for the image.
# * deal_probability - The target variable. This is the likelihood that an ad actually sold something. It's not possible to verify every transaction with certainty, so this column's value can be any float from zero to one.
# 
# So deal probability is our target variable and  is a float value between 0 and 1 as per the data page. Let us have a look at it. 

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sns.distplot(train_df.deal_probability.values)


# In[ ]:


plt.figure(figsize=(8,6))
plt.scatter( range(train_df.shape[0]),np.sort(train_df['deal_probability'].values))
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
sns.barplot(y=train_df.region, x="deal_probability", data=train_df)


# In[ ]:


train_df.city.value_counts()


# In[ ]:


plt.figure(figsize=(12,8))
sns.barplot(x=train_df.parent_category_name, y="deal_probability", data=train_df)


# In[ ]:


plt.figure(figsize=(12,8))
sns.boxplot(x="parent_category_name", y="deal_probability", data=train_df)
plt.ylabel('Deal probability', fontsize=12)
plt.xlabel('Parent Category', fontsize=12)
plt.title("Deal probability by parent category", fontsize=14)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(x="category_name",data=train_df)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(x="user_type",data=train_df)
plt.show()


# more in pipeline, if you like it please upvote for me.
# 
# Thank you :)
