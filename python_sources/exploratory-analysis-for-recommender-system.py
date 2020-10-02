#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


reviews = pd.read_csv("../input/ds5230usml-project/Reviews.csv")


# In[ ]:


reviews.head()


# > Finding the Average rating for each product and creating a Dataframe for it.

# In[ ]:


ratings = pd.DataFrame(reviews.groupby("ProductId")["Score"].mean())
ratings.head()


# Finding the Rating Count i.e number of ratings that each product has got

# In[ ]:


ratings['rating_count'] = pd.DataFrame(reviews.groupby("ProductId")["Score"].count())
ratings.head()


# Taking a look at Statistical Analysis of the rating Data Frame.

# In[ ]:


ratings.describe()


# In[ ]:


# Arranging the ratings DataFrame in Descending Order.
ratings.sort_values('rating_count',ascending = False).head()


# Building a user * item unity matrix using pivote table function.
# 
# This function will cross tabulate each user against each place, and output a matrix.
# 
# 1. Index: User ID
# 2. Column: Product ID
# 3. Vaues: Values from Score column

# In[ ]:


#products_cross_table = pd.pivot_table(data=reviews, values="Score", index="UserId", columns = "ProductId")


# In[ ]:


#Displaying the first 5 rows
#products_cross_table.head()


# In[ ]:




