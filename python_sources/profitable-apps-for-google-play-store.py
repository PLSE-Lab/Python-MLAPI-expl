#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from statistics import mode
import statistics
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # First of all reading csv file using pandas library

# In[ ]:


data = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')


# We need to find the number of columns and number of rows in our dataset. 

# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


data["Last Updated"].head()


# # In the above code we get the column names for our dataset

# In[ ]:


data.info()


# The 'Rating' column is of float type whereas all other columns contain string characters in our dataset.
# There are total of 10841 rows in all columns but in 'Rating' column only 9367 rows exists so there are missing rows for column Rating in our dataset. Similarly, columns such as 'Type' is missing one row , "Content Rating' column is also missing one row, and columns 'Current Ver' and 'Android Ver' are also missing some rows. So, let's find out how many rows each column is missing in the next code shell.

# In[ ]:


data.isnull().sum()


# **Filling Missing Rows In Rating with Average Ratings Of the rows belonging to a particular Category**
# 
# To fill the missing rows in 'Rating' column, we first find the rows with missing column 'Rating' and then we look at the 'Category' column of those missing rows and find the average rating of an app belonging to a particular 'Category' column and finally replace the missing rows with the mean or average ratings of an app belonging to a particular 'Category' column.
# 

# In[ ]:


def getcategory(dataset):
    missing_ratings = data.loc[data["Rating"].isnull(),"Category"].unique()
    for category_name in missing_ratings:   
        nonullcategory = dataset.loc[(dataset["Category"] == category_name) & (dataset["Rating"].notnull()), ["Category","Rating"]]
        dataset.loc[(dataset["Category"] == category_name) & (dataset["Rating"].isnull()),"Rating"] = nonullcategory["Rating"].mean()
    return dataset


# In[ ]:


ratings_filled = getcategory(data)
ratings_filled.isnull().sum()


# We have filled the missing rows in the 'Rating' Column. Now, filling the missing row in the 'Type' column with the mode of that column.

# In[ ]:


ratings_filled.loc[ratings_filled["Type"].isnull() , "Type"] = ratings_filled["Type"].mode().values


# In[ ]:


ratings_filled.isnull().sum()


# We have successfully filled the missing row of the 'Type' column with the mode of the given column and now we are going to fill the missing row of the Content Rating with the appropriate value. When we look at the index 10472 of our dataset, we found out that the values are misplaced in the same row. For example: the value of Rating column is situated in Category column. the value of 'Reviews' column is situated in 'Rating' column and so on. 

# In[ ]:


null = ratings_filled.loc[ratings_filled['Content Rating'].isnull() ,:]
print(null)


# We are going to create a dictionary to arrange the misplaced value at 10472 row of our dataset. We will use the dictionary keys as the column names of our data frames and the values as the values of the row which were misplaced and we will iterate over the keys to arrange our row with the appropriate values.

# In[ ]:


replacable_value = {"Rating" : null['Category'].values ,"Reviews" : null['Rating'].values , "Size" : null['Reviews'].values,
                    "Installs" : null['Size'].values , "Type": null['Installs'].values, "Price" : null['Type'].values, 
                    "Content Rating" : null['Price'].values}


# In[ ]:



for item in replacable_value:
    ratings_filled.loc[ratings_filled['Content Rating'].isnull(),item] = replacable_value[item]


# **Replacing empty rows at index 10742**
# 
# 
# After replacing the misplaced value at this index, we are left with two missing values in that row in columns Category and Genres and we replace the two empty columns by searching the appropriate Genre and Category for the app name in our Chrome Browser.

# In[ ]:


ratings_filled.loc[ratings_filled['Category'] == '1.9' , ["Category" , "Genres"]] = "Lifestyle"


# In[ ]:


replaced = {"Last Updated": null['Genres'].values , "Current Ver": null["Last Updated"].values, "Android Ver": null["Current Ver"].values}


# In[ ]:


for items in replaced:
    ratings_filled.loc[ratings_filled['Rating']=='1.9',items] = replaced[items]


# In[ ]:


ratings_filled.isnull().sum()


# In[ ]:


ratings_filled.loc[ratings_filled['Android Ver'].isnull(),:]


# In[ ]:


ratings_filled.loc[ratings_filled['Android Ver'].isnull(),'Android Ver']
model = ratings_filled.loc[ratings_filled['Category'] == "PERSONALIZATION",'Android Ver']
model.mode().values


# In[ ]:


ratings_filled.loc[ratings_filled['Android Ver'].isnull(),'Android Ver'] = model.mode().values


# In[ ]:


ratings_filled.loc[ratings_filled['Android Ver'].isnull(),:]


# In[ ]:


ratings_filled.iloc[4453]


# In[ ]:


ratings_filled.loc[ratings_filled['Current Ver'].isnull(),:]


# In[ ]:


ratings_filled["Current Ver"].value_counts()


# In[ ]:


ratings_filled.loc[ratings_filled['Current Ver'].isnull(),"Current Ver"] = ratings_filled["Current Ver"].mode().values


# In[ ]:


ratings_filled.isnull().sum()


# In[ ]:




