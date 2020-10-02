#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # Visualising data
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


renfe = pd.read_csv("../input/renfe.csv")


# **Data Preprocessing**

# In[ ]:


renfe.info()


# **1.1 Find out the rows with Nan Values**

# In[ ]:


renfe.isnull().sum()


# **Fill Null values with mean values**

# In[ ]:


renfe.train_class.value_counts()
renfe.fare.value_counts()


# ##Fill null values in train_class with Turista##
# ##Fill null values in fare with Promo ##

# **2. Data Preprocessing**

# **Remove the rows in 'price' with null values**

# In[ ]:


renfe.dropna(axis=0,subset=['price'],inplace=True)


# In[ ]:


renfe.train_class.fillna('Turista',inplace=True)
renfe.fare.fillna('Promo',inplace=True)


# >** Data Preprocessing **
# 
# ###Drop column "insert_date" as it looks irrelevant.As per insert_date, i see so many prices were collected at same time##
# ###Dropped first column('unnamed') as we already have index available###

# In[ ]:


renfe = renfe.loc[:, ~renfe.columns.str.contains('Unnamed')]
renfe.drop('insert_date',inplace=True,axis=1)


# **Create Additional Features**
# ##Covert the columns with date to dataframe##
# ##Create  new features like day of week,time to travel##

# In[ ]:


for i in ['start_date','end_date']:
    renfe[i] = pd.to_datetime(renfe[i])

renfe['time_to_travel'] = renfe['end_date'] - renfe['start_date']
renfe['day_of_week'] = renfe['start_date'].dt.weekday_name
renfe['date'] = renfe['start_date'].dt.date
renfe['time_in_seconds'] = renfe['time_to_travel'].dt.total_seconds()


# **Categorizing price in different Groups**

# In[ ]:


renfe_greater_150 = renfe[(renfe['price'] > 150)]
renfe_between_100_150 = renfe[(renfe['price'] > 100) & (renfe['price'] < 150)]
renfe_between_50_100 = renfe[(renfe['price'] > 50) & (renfe['price'] < 100)]
renfe_less_50 = renfe[(renfe['price'] < 50)]


# In[ ]:


sns.jointplot("time_in_seconds","price",data=renfe_greater_150)
sns.catplot(data=renfe_greater_150,x="train_type",y="price")
sns.catplot(data=renfe_greater_150,x="fare",y="price")


# **Insights :** We have very less trains for the price range greater than 150 euros. Also for all trains with price more than 150, minimum travelling time is 5700 seconds. Most of the trains are of the type 'AVE' & fare is 'Flexible' in this range. 

# In[ ]:


sns.catplot(data=renfe_greater_150,x="train_type",y="price")


# In[ ]:


sns.jointplot("time_in_seconds","price",data=renfe_between_100_150)
sns.catplot(data=renfe_between_100_150,x="train_type",y="price")
sns.catplot(data=renfe_between_100_150,x="fare",y="price")
percentage_100_150 = len(renfe_between_100_150)/ len(renfe) * 100
print(percentage_100_150)


# **Insights :** For the price range between 100 & 150, most of the fare's are 'Promo' & 'Flexible',Train_type is 'AVE' & 'AVE-TGV'. Time range is 9000-11500 seconds.

# In[ ]:


sns.jointplot("time_in_seconds","price",data=renfe_between_50_100)
sns.catplot(data=renfe_between_50_100,x="train_type",y="price")
sns.catplot(data=renfe_between_50_100,x="fare",y="price")
percentage_50_100 = len(renfe_between_50_100)/ len(renfe) * 100
print(percentage_50_100)


# **Insights :** 55% of train prices are in the range 50-100 euros. 'Promo' & 'Flexible' are the main fare's at this range. Major trian type's are AVE 

# In[ ]:


sns.jointplot("time_in_seconds","price",data=renfe_less_50)
sns.catplot(data=renfe_less_50,x="train_type",y="price")
sns.catplot(data=renfe_less_50,x="fare",y="price")
percentage_lesser_50 = len(renfe_less_50)/ len(renfe) * 100
print(percentage_lesser_50)


# **Insights :** 34% of train prices are less than 50 euros. Under 50, we have train_type as 'AVE', 'intercity','AVE-LD','MD-LD'. 'Promo' & 'Promo +' are the major train fare's under 50 euros price. 

# **Day wise train numbers**

# In[ ]:


sns.barplot(x=renfe.day_of_week.value_counts(),y=renfe.day_of_week.value_counts().index)


# **Insights :** Thursday have highest number of trains and Saturday with the least one. Even though thursdays have highest number of trains, we can see that all weekdays have almost same number of trains. 
