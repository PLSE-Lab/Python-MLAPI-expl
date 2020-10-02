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


# This Exploratory data analysis will delve into the different airbnb listings and what kind of insights can be derived from them. In particular, there is particular interest in looking at how each of the specific regions (column neighbourhood_groups) differ, in terms of ratings, stays, and price. A correlation analysis between price and number of reviews is also explroed as well.

# ## 1. Create and preview the dataset

# In[ ]:


listing=pd.read_csv("/kaggle/input/singapore-airbnb/listings.csv")
listing.head()


# ## 2. How many listings (rows) are there in this dataset?

# In[ ]:


listing.shape[0]


# ## 3. Return the id, name, and price of all airbnb's with a price under 50.

# In[ ]:


listing.query('price<50').loc[:,["id","name","price"]]


# ## 4. Let's start delving deeper into the data. How many airbnb's are there in every neighboorhood group? Use groupby() to group the data together.

# In[ ]:


region=listing.groupby("neighbourhood_group")
region.size()


# ## 5. What was the average price of airbnb's located in the "North Region", rounded to two decimal places?

# In[ ]:


region.price.mean().loc["North Region"].round(2)


# ## 6. List the top 10 airbnb's with the most ratings in the "West Region"

# In[ ]:


listing[listing.neighbourhood_group=="West Region"].sort_values("number_of_reviews",ascending=False).iloc[:10]


# ## 7. What is the most expensive airbnb per region?

# In[ ]:


region.price.max()


# ## 8. Create a scatter plot between price and number of reviews.

# In[ ]:


listing.plot.scatter(x="price",y="number_of_reviews")

