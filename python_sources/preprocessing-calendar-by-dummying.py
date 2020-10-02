#!/usr/bin/env python
# coding: utf-8

# Thank you for opening this notebook!

# In this notebook, I'd like to talk about one of the ways to deal with "calendar.csv".

# # What to do in this article?
# ## I preprocess "calendar.csv" so that we can use to Machine Learning
# # What not to do in this article?
# ## I don't make a prediction using other files.

# # Loading data and check missing values

# At first, I load calendar.csv on this notebook.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


calendar = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")


# I finished loading. Then, let's check the begining of the data. Here, I use 'head()'.

# In[ ]:


calendar.head()


# There are lots of 'NaN'. Let's check the num of missing values. Here, I use "isnull().sum()".

# In[ ]:


calendar.isnull().sum()


# We can see missing values in event_name_1, event_type_1, event_name_2 and event_type_2.

# It seems that the number is not few. Let's check the outline by using "describe()".

# In[ ]:


calendar.describe()


# We can see that the rate of missing value is high.

# This might lead to trouble of machine learning.

# To avoid the situation, we can select two actions.

# 1 We eliminate four columns.

# 2 We make new feature values by using them.

# Here, we choose second option.

# # See the contents of each data.

# Then, let's see the contents of each data.

# ## event_name_1

# In[ ]:


calendar["event_name_1"].unique()


# In[ ]:


len(calendar["event_name_1"].unique())


# There are 31 kinds of data in "event_name_1". It seems that they are names of special days such as 'SuperBowl' and 'ValentinesDay'.

# ## event_type_1

# Next, let's check "event_type_1".

# In[ ]:


calendar["event_type_1"].unique()


# It seems that these datas mention the kinds of special days.

# ## event_name_2, event_type_2

# By the way there is a question that why "calendar.csv" has "event_name_2" and "event_type_2".

# Let's check.

# In[ ]:


calendar["event_name_2"].unique()


# In[ ]:


calendar["event_type_2"].unique()


# If we watched carefully, we can realize that there are same contents in both name ans event.

# I use "set" to check the duplication.

# In[ ]:


event_name = set(calendar["event_name_1"].unique()) & set(calendar["event_name_2"].unique())
print(event_name)


# In[ ]:


event_type = set(calendar["event_type_1"].unique()) & set(calendar["event_type_2"].unique())
print(event_type)


# We can say that there are duplicates in event_name and event_type.

# We can say that some day corresponds to TWO SPECIAL DAYS.

# # Dummying special days and kinds.

# To make special days feature values, we need to deal with "nan".

# Then, I express if each days adapts to each special days by using 0 and 1.

# If one day did not adapt to any special days, its all columns are 0.

# In[ ]:


event_name_one = calendar["event_name_1"]
event_name_one = pd.get_dummies(event_name_one)
event_name_one.head()


# Here, I made calendar's "event_name_1" dummy variable. I do same thing to "event_name_2".

# In[ ]:


event_name_two = calendar["event_name_2"]
event_name_two = pd.get_dummies(event_name_two)
event_name_two.head()


# I completed making dummy variable. Next, I coalesce two data.

# In[ ]:


event_names = pd.merge(event_name_one, event_name_two, right_index=True, left_index=True)
event_names.head()


# In[ ]:


event_names.columns


# I used "merge". Duplicated data like Cinco De Mayo has "_x" or "_y".

# Next, I make Cinco De Mayo, OrthodoxEaster, Easter and Father's day again.

# If something_x or something_y had 1, something's value make 1. 

# In[ ]:


event_names['Easter'] = 0
event_names['Cinco De Mayo'] = 0
event_names['OrthodoxEaster'] = 0
event_names["Father's day"] = 0


# In[ ]:


for index in event_names.index:
    if event_names.loc[index,"Cinco De Mayo_x"] == 1 or event_names.loc[index,"Cinco De Mayo_y"] == 1:
        event_names.loc[index,"Cinco De Mayo"] = 1        
    if event_names.loc[index,"Easter_x"] == 1 or event_names.loc[index,"Easter_y"] == 1:
        event_names.loc[index,"Easter"] = 1    
    if event_names.loc[index,"Father's day_x"] == 1 or event_names.loc[index,"Father's day_y"] == 1:
        event_names.loc[index,"Father's day"] = 1    
    if event_names.loc[index,"OrthodoxEaster_x"] == 1 or event_names.loc[index,"OrthodoxEaster_y"] == 1:
        event_names.loc[index,"OrthodoxEaster"] = 1    
        
event_names.drop('Cinco De Mayo_x', axis=1, inplace=True)
event_names.drop('Cinco De Mayo_y', axis=1, inplace=True)
event_names.drop('Easter_x', axis=1, inplace=True)
event_names.drop('Easter_y', axis=1, inplace=True)
event_names.drop("Father's day_x", axis=1, inplace=True)
event_names.drop("Father's day_y", axis=1, inplace=True)
event_names.drop('OrthodoxEaster_x', axis=1, inplace=True)
event_names.drop('OrthodoxEaster_y', axis=1, inplace=True)


# In[ ]:


event_names.columns


# I eliminated duplicated data. There is not any more "nan".

# I deal with event_type by using same way of thinking.

# In[ ]:


event_type_one = calendar["event_type_1"]
event_type_one = pd.get_dummies(event_type_one)
event_type_one.head()


# In[ ]:


event_type_two = calendar["event_type_2"]
event_type_two = pd.get_dummies(event_type_two)
event_type_two.head()


# In[ ]:


event_types = pd.merge(event_type_one, event_type_two, right_index=True, left_index=True)
event_types['Cultural'] = 0
event_types['Religious'] = 0
event_types.head()


# In[ ]:


for index in event_types.index:
    if event_types.loc[index,"Cultural_x"] == 1 or event_types.loc[index,"Cultural_y"] == 1:
        event_types.loc[index,"Cultural"] = 1        
    if event_types.loc[index,"Religious_x"] == 1 or event_types.loc[index,"Religious_y"] == 1:
        event_types.loc[index,"Religious"] = 1    
        
        
event_types.drop('Cultural_x', axis=1, inplace=True)
event_types.drop('Cultural_y', axis=1, inplace=True)
event_types.drop('Religious_x', axis=1, inplace=True)
event_types.drop('Religious_y', axis=1, inplace=True)

event_types.head()


# # Finalize calendar

# Finally, I coalesce original data, event_name and event_types.

# In[ ]:


calendar = pd.concat([calendar, event_names, event_types], axis=1)
calendar.drop('event_name_1', axis=1, inplace=True)
calendar.drop('event_type_1', axis=1, inplace=True)
calendar.drop('event_name_2', axis=1, inplace=True)
calendar.drop('event_type_2', axis=1, inplace=True)


# In[ ]:


calendar.head()


# In[ ]:


calendar.isnull().sum()


# There is no more missing values.

# Below code is last action.

# In[ ]:


calendar.to_csv('calendar_dummied.csv', index=False)


# In this notebook, I preprocessed calendar.csv to make special days dummy values and eliminate missing values.

# Thank you for reading to the last.
