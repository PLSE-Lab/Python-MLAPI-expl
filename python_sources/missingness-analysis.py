#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
import missingno as msno 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# First off... this is intended to by my first notebook to share. It is basic, and simple, and probably useless for this particular competition (spoiler: it kind of is). But I think the process taught me a lot, and should be useful in real word datasets!
# 
# In this notebook I plan to analyze the missing data in our dataset. In the real world we typically have very messy data, and this data can be lost when aquiring, or deleted accidentally afterwards. The workflow will be:
# 
# 1. Check for missing values (are they NaN? are they some other representation for NULL, if so, convert these to null values).
# 2. Analyze the amount of missing values, and the type.
# 3. Either delete, or impute these values.
# 4. Evaluate and compare the performance of each imputed option.
# 
# This notebook will cover the first 2 steps in this process using the missingno python library.  Intuitively, I expect this data to be completely random with regards to missing values, but that is an assumption!
# 
# We will concatenate our training and test sets for now (simply for analysis). And we will drop the target column, assuming our training set is complete.
# 

# In[ ]:


X_train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv', index_col='id')
X_test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv', index_col='id')

cat_df = X_train.append(X_test, sort=False)
cat_df.drop('target', axis=1, inplace=True)

cat_df.head()


# Now let's do some basic EDA.

# In[ ]:


cat_df.info()


# Some unique values on our features.

# In[ ]:


cols = list(cat_df.columns)
cols

for col in cols:
    temp = cat_df[col].unique()
    print('Column Name: ', col)
    print('Column Unique Values: ', temp)


# It certainly appears that all our missing values are correctly stored as NaN's. Lets get some basic statistics of our numeric data. (We will have to deal with the non-numeric data in the future)

# In[ ]:


cat_df.describe()


# Now lets see how much data is missing from each column.

# In[ ]:


for col in cols:
    print('Column Name: ', col, ' ', sum(cat_df[col].isnull()), ' is missing.')
    


# This looks like each column has a similar amount of data missing. Let's see if there are any patterns to our missing data. First using a matrix plotted through the missingno library.

# In[ ]:


msno.matrix(cat_df)


# Now let's see a heatmap. 

# In[ ]:


msno.heatmap(cat_df)


# The heatmap shows no correlations, and a dendrogram can be done as well (but... I think we can see where this is heading). Any closely correlated features will be seen "close to each other" on leaves.  Long, "deep" branches show no correlations.

# In[ ]:


msno.dendrogram(cat_df)


# We can classify this dataset as having missing values completely at Random (MCAR).  From here on out, we will have to properly encode our data before imputing the missing values.  But we can safely say, that our data has no unique patterns within the missingness, that we will be concerned about!
