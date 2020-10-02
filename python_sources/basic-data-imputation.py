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


# In[ ]:


df = pd.read_csv('../input/googleplaystore.csv', header=0)
df.head()


# Let's check what kind of schema/datatypes we are dealing with here...

# In[ ]:


df.dtypes


# In[ ]:


df['Price'].value_counts()


# In the dataset `Price` and `Size` are of object type. Price is either '0' or something like, '$3.99' and size is something like '2.4M' or '50.9k', so we will try to extract the price & size using regular expression groups. Remember to only extract price values when its not zero or the regex will return `NA`.
# 
# _NOTE_: To analyze `Price` and `Size` run a simple `df['Price'].value_counts()`

# In[ ]:


df['SizeS'] = df['Size'].str.extract('([0-9\.]+)[Mk]')
df['PriceS'] = df['Price']
df['PriceS'] = np.where(df['Price'] != '0', df['Price'].str.extract('\$([0-9\.]*)'), df['Price'])
df.head()


# Cool...now let's do some `dtype` casting...Also, we need to convert `Last Updatedf` from `object` to `datetime` type.

# In[ ]:


df['Ratingf'] = df['Rating'].astype("float64")
df['Pricef'] = df['PriceS'].astype("float64")
df['Reviewsf'] = df['Reviews'].astype("float64", errors="ignore")
df['Sizef'] = df['SizeS'].astype("float64")
df['Last Updatedf'] = pd.to_datetime(df['Last Updated'], format='%B %d, %Y', errors="ignore")
df.head()


# In[ ]:


df.dtypes


# In[ ]:


print(df.shape)


# Let's check where is the missing data...

# In[ ]:


df.isna().sum()


# In[ ]:


df['Category'].value_counts()


# We have missing data at `Size`, `Rating` and at `Price`. We can easily impute this data with categorical mean. So we will fill `NA` by grouping over `Category` and taking mean of each category.

# In[ ]:


df_fill = df
df_fill['Sizem'] = df_fill['Sizef'].fillna(df_fill.groupby("Category")['Sizef'].transform('mean'))
df_fill['Ratingm'] = df_fill['Ratingf'].fillna(df_fill.groupby("Category")['Ratingf'].transform('mean'))
df_fill['Pricem'] = df_fill['Pricef'].fillna(df_fill.groupby("Category")['Pricef'].transform('mean'))
df_fill.head()


# We still can see that there is one record which has missing data, This is just missaligned data in the CSV, can fix the alignment manually or we can just drop it. Here we will drop it if any of `Rating`, `Size` and `Price` is `NA`.

# In[ ]:


df_fill = df_fill.dropna(how='any', subset=['Ratingm', 'Sizem', 'Pricem'])


# Here we go.... :)

# In[ ]:


df_fill.isna().sum()


# In[ ]:


df_fill.head()

