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


# 
# 
# # Book Recommendation system
# 
# # Dataset url: http://www2.informatik.uni-freiburg.de/~cziegler/BX/
# 
# ## going to construct a book recommendation system based on user ratings of books 

# In[ ]:


#import the data
books = pd.read_csv('/kaggle/input/bookrecommendation/BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")

users = pd.read_csv('/kaggle/input/bookrecommendation/BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")

ratings = pd.read_csv('/kaggle/input/bookrecommendation/BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")




# In[ ]:


#primary look on the data 
ratings.head(),books.head(),users.head()


# In[ ]:


#looking at the shape of each dataset, along witht he names of columns
print(ratings.shape,list(ratings.columns)),print(books.shape,list(books.columns)),print(users.shape,list(users.columns))


# In[ ]:


#importing seaborn and matplot for visualization
import seaborn as sns
import matplotlib.pyplot as plt

#visual settings for seaborn 
sns.set(style='darkgrid')
sns.set_palette("coolwarm", 16)


# ## counting how many of each rating is present in our data
# 
# ## remember, ratings can only be whole numbers between, 0 and 10

# In[ ]:



plt.figure(figsize=(13, 5))
sns.countplot(x="Book-Rating", data=ratings)


# ## we observe that most entries have 0 as the rating, looking at the graph we can clearly observe that there is something imbalanced with this , 
# 
# ## looking at the graph the first intuition is that most of the null values are inclueded in our data as 0, i.e people who didnt rate a book, 
# 
# ## it would be quite odd if so many people chose to give absolute 0 ratings to books, instead of 1 or 2
# 
# ## also we can see that only a tiny fraction of our users choose to rate books 

# ## looking at the age distribution in our data

# In[ ]:


plt.figure(figsize=(13, 5))
#try to understand this line carefully 
sns.distplot(users[users["Age"]<100]["Age"],bins=17,color = "y")


# ## looking at the age distribution in our data

# In[ ]:


users["Location"].nunique()

