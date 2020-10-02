#!/usr/bin/env python
# coding: utf-8

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


open_file=open('../input/app-store-apple-data-set-10k-apps/AppleStore.csv')


# In[ ]:


from csv import reader
read_file=reader(open_file)


# In[ ]:


dataset=list(read_file)
dataset[0]


# # To find average user-rating

# In[ ]:


ratings=[]
for eachlist in dataset[1:]:
    rates=float(eachlist[8])
    ratings.append(rates)
#ratings


# In[ ]:


avg_user_rating=sum(ratings)/len(ratings)
print("The average User rating is :",avg_user_rating)


# # Average user rating of free apps and non-free apps
# 

# In[ ]:


#Price of each apps
for eachlist in dataset[1:5]:
    print(eachlist[5])


# ## As we can see there are many free apps and many non free apps.

# In[ ]:


free_apps_ratings = []
for row in dataset[1:]:
    rating = float(row[8])
    # Complete the code from here
    price=float(row[5])
    if price==0.0:
        free_apps_ratings.append(rating)
avg_rating_free=sum(free_apps_ratings)/len(free_apps_ratings)
print("The average User rating of free aps is :",avg_rating_free)


# In[ ]:


non_free_apps_ratings = []
for row in dataset[1:]:
    rating = float(row[8])
    # Complete the code from here
    price=float(row[5])
    if price!=0.0:
        non_free_apps_ratings.append(rating)
avg_rating_non_free=sum(non_free_apps_ratings)/len(non_free_apps_ratings)
print("The average User rating of Non-free aps is :",avg_rating_non_free)
