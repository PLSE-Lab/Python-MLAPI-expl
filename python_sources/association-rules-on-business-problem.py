#!/usr/bin/env python
# coding: utf-8

# # Business Problem
# 
# ### What is the **Association Rules**?
# 
# It is a rule-based machine learning technique used to find patterns (relationships, structures) in the data.
# 
# Association analysis applications are among the most common applications in data science. It will also coincide as Recommendation Systems.
# 
# These applications may have come up in the following ways, such as "bought this product that bought that product" or "those who viewed that ad also looked at these ads" or "we created a playlist for you" or "recommended video for the next video".
# 
# These scenarios are the most frequently encountered scenarios within the scope of e-commerce data science data mining studies.
# 
# In Turkey and the world's largest e-commerce companies spotify, amazon, it uses many platforms like netflix recommendation systems can know a little more closely.
# 
# ### So what does this association analysis summarize?
# 
# #### Apriori Algorithm
# 
# It is the most used method in this field.
# 
# Association rule analysis is carried out by examining some metrics:
# 
# * Support
#     Support(X, Y) = Freq(X,Y)/N
#         X: Product
#         Y: Product
#         N: Total Shopping
# 
# * Confidence
# 
#         Confidence (X, Y) = Freq (X, Y) / Freq (X)
# 
# * Lift (The purchase of one product increases the level of purchase of the other.)
# 
#         Lift = Support (X, Y) / (Support (X) * Support (Y))
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mlxtend.frequent_patterns import apriori, association_rules

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Data Understanding

# In[ ]:


df = pd.read_csv('/kaggle/input/online-retail-data-set-from-ml-repository/retail_dataset.csv', sep=',')
df.head()


# Now we have to convert this DF, which is made up of categorical variables to DF, which consists of 0's and 1's.

# In[ ]:


df.shape


# # Data Preprocessing

# In[ ]:


items = (df['0'].unique())
items


#  The main purpose now is to ensure that the variables in the column are on the line. One-Hot Encoding method will help us to do this.

# In[ ]:


encoded_vals = []
for index, row in df.iterrows(): 
    labels = {}
    uncommons = list(set(items) - set(row))
    commons = list(set(items).intersection(row))
    for uc in uncommons:
        labels[uc] = 0
    for com in commons:
        labels[com] = 1
    encoded_vals.append(labels)


# In[ ]:


ohe_df = pd.DataFrame(encoded_vals)


# Let's see what happenned after One-Hot Encoding method:

# In[ ]:


ohe_df


# # Association Rules
# 
# For apriori, you need to do one by giving DF with hot encoding.

# In[ ]:


freq_items = apriori(ohe_df, min_support = 0.2, use_colnames = True, verbose = 1)


# Thus, support values are calculated. Let's check it:

# In[ ]:


freq_items.head()


# Finally, we will see the function association_rules (togetherness analysis), we need to use support (frequency items) DF.

# In[ ]:


association_rules(freq_items, metric = "confidence", min_threshold = 0.6)


# ## We can easily see how often there is a connection between which products.

# 
# # Conclusion
# 
# After this notebook, my aim is to prepare 'kernel' which is 'not clear' data set.
# 
# If you have any suggestions, please could you write for me? I wil be happy for comment and critics!
# 
# Thank you for your suggestion and votes ;)
# 
# 
