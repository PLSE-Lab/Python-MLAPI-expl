#!/usr/bin/env python
# coding: utf-8

# ## View missing values with `missingno`
# This is minimalist example of the [missingno](https://github.com/ResidentMario/missingno) tool by [Aleksey Bilogur](https://github.com/ResidentMario) applied to the [Titanic](https://www.kaggle.com/c/titanic) data. From the GitHub page:
# 
# > Messy datasets? Missing values? missingno provides a small toolset of flexible and easy-to-use missing data visualizations and utilities that allows you to get a quick visual summary of the completeness (or lack thereof) of your dataset.
# 
# It could not be more simple to use!

# In[ ]:


import pandas  as pd
# and now missingno
import missingno as msno
# read in the Titanic training data
train_data = pd.read_csv('../input/titanic/train.csv')


# ## Matrix

# In[ ]:


msno.matrix(train_data);


# ## Bar chart

# In[ ]:


msno.bar(train_data);


# ## Heat map

# In[ ]:


msno.heatmap(train_data);


# ## Dendrogram

# In[ ]:


msno.dendrogram(train_data);


# ## Links:
# * [missingno on GitHub](https://github.com/ResidentMario/missingno)
# 
# ## Related reading:
# Here are two excellent notebooks on the subject of missing values, written by kaggle Notebooks Grandmaster [Rachael Tatman](https://www.kaggle.com/rtatman):
# - [Handling missing values](https://www.kaggle.com/rtatman/data-cleaning-challenge-handling-missing-values)
# - [Imputing missing values](https://www.kaggle.com/rtatman/data-cleaning-challenge-imputing-missing-values/)
