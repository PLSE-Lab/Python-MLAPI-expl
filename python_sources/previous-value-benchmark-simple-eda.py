#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# Hello everyone, this kernel will serve as a starting point for many coming from the Coursera course "How to Win a Data Science Competition".
# As many of you know, this competition "Predict Future Sales" by Russian software firm 1C Company is part of the final project for the course. 
# 
# The challenge here is to predict the total monthly sales of a product in each individual shop of a store chain, specifically November 2015. We are given the historical daily sales data from January 2013 to October 2015 in the provided file 'sales_train.csv'. From the training data we are to predict the sales numbers using the data given in 'test.csv', and create a submission file in the same format as 'sample_submission.csv'. Three other csv files are provided to give more insight into the data in the training set.
# 
# This kernel serves as the process for one of the exercises given in the course - to simply create a prediction for November 2015 using the exact historical sales data of the previous month, October 2015. The steps taken throughout this kernel may not be the most efficient, but it was my personal workflow of tackling the competition as a beginner data scientist without looking at the other kernels. 
# 
# In the future I should do visualizations for my EDA process, but I made some assumptions that the data is relatively clean, haha.
# 
# Please feel free to use this kernel as you like.
# 
# We will begin by importing the necessary libraries and the datasets.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv('../input/sales_train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')
items = pd.read_csv('../input/items.csv')
item_cats = pd.read_csv('../input/item_categories.csv')
shops = pd.read_csv('../input/shops.csv')


# ### The Benchmark - October 2015 historical sales to predict November 2015
# Let's try to create a predictions benchmark by creating a submissions file with the previous month's sales. 
# 
# We will be using October 2015's sales data to predict November 2015's sales, and using the score as a benchmark for the evaluation of our future models.

# In[ ]:


train.describe()


# In[ ]:


test.describe()


# In[ ]:


submission.describe()


# Seems like the training set and testing set consist of different columns. 
# 
# The test set contains only the shop_id and item_id from the training set, and the submission file only contains the monthly item count with ID.
# 
# We need to map the training set shop_id and item_id to the ID number in the test set, add up the daily item counts for just October 2015, and create a similar submission file with ID and item_cnt_month.

# In[ ]:


train.head(50).T


# In[ ]:


train['date'].describe()


# In[ ]:


'''
train['date'] = pd.to_datetime(train.date)
train = train.sort_values(by='date')
'''


# I tried to convert the 'date' column in the training set to datetime objects so I could sort it and split the October data, but it took way too long because there are 2.9 million rows in the data set. 
# 
# Luckily I noticed there is a date_block_num column that corresponds to the consecutive month of the dataset. 

# In[ ]:


train.tail(50).T


# In[ ]:


train_oct2015 = train.loc[train['date_block_num'] == 33]


# In[ ]:


train_oct2015.head()


# I've made a dataframe of just the October 2015 sales data. 
# 
# Now I will tally up a total item_cnt_month number for each unique shop_id-item_id pair.

# In[ ]:


df_m = train_oct2015.groupby(["shop_id", "item_id"])
month_sum = df_m.aggregate({"item_cnt_day":np.sum}).fillna(0)
month_sum.reset_index(level=["shop_id", "item_id"], inplace=True)
month_sum = month_sum.rename(columns={ month_sum.columns[2]: "item_cnt_month" })
month_sum.describe()


# In[ ]:


submission.describe()


# Something doesn't seem right - the number of rows is far less than the submission dataframe.

# In[ ]:


month_sum['item_id'].value_counts()


# In[ ]:


test['item_id'].value_counts()


# I see what's going on. If the total item_cnt_month is 0 then item_id for the corresponding shop_id is not included. 
# 
# This could be fixed by simply merging month_sum with the test dataframe and filling the NaNs.
# 
# We will map the shop_id-item_id to ID number in the test set for our next step, and finally make a submission dataframe.

# In[ ]:


new_submission = pd.merge(month_sum, test, how='right', left_on=['shop_id','item_id'], right_on = ['shop_id','item_id']).fillna(0)
new_submission.drop(['shop_id', 'item_id'], axis=1)
new_submission = new_submission[['ID','item_cnt_month']]


# The current score is quite horrible, at 8.53027. Let's try to clip the values within [0,20] as per the tip from the course.

# In[ ]:


new_submission['item_cnt_month'] = new_submission['item_cnt_month'].clip(0,20)
new_submission.describe()


# In[ ]:


new_submission.to_csv('previous_value_benchmark.csv', index=False)


# The score now is 1.16777 as expected! Awesome. Now we can start trying some models in another kernel. 
