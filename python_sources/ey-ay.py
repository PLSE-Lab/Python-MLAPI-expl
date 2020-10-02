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


get_ipython().system('pip install missingno ')


# In[ ]:


import missingno as msng # Perfect visualization tool for looking at data integrity
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Lets load dataset into respective pandas dataframe
surveySchema = pd.read_csv('../input/SurveySchema.csv')
freeFormResponses = pd.read_csv('../input/freeFormResponses.csv')
multileChoiceResponses = pd.read_csv('../input/multipleChoiceResponses.csv')


# **Survey Schema**

# In[ ]:


# Sniffing the schema
surveySchema.head()


# **Free Form Responses**

# In[ ]:


# Sniffing free responses
freeFormResponses.head()


# In[ ]:


# Shape of responses dataframe
freeFormResponses.shape


# In[ ]:


# Distance between individual features is calculated and they are clustered
msng.dendrogram(freeFormResponses.sample(500))


# In[ ]:


# Patterns are detected in data
msng.matrix(freeFormResponses.sample(500))


# In[ ]:


# Nullity correlation is measured with heatmap function
msng.heatmap(freeFormResponses.sample(500))


# In[ ]:


# Nullity status of variables is visualized
msng.bar(freeFormResponses.sample(500))


# In[ ]:


# Based on nullity percentage, selecting some columns from data
# Columns of at least 20% completeness
# No more than 5 columns
freeFormResponsesFiltered = msng.nullity_filter(freeFormResponses, 
                                                filter='top', 
                                                p=.20, 
                                                n=5)
freeFormResponsesFiltered.shape


# In[ ]:


# Filtered responses
freeFormResponsesFiltered.head()


# In[ ]:


# Data is rearranged by ascending or descending completeness
freeFormResponsesFiltered = msng.nullity_sort(freeFormResponsesFiltered, 
                                              sort=None)
freeFormResponsesFiltered.head()


# In[ ]:


# Data is rearranged by ascending or descending completeness
freeFormResponsesFilteredAsc = msng.nullity_sort(freeFormResponsesFiltered, 
                                                 sort='ascending')
freeFormResponsesFilteredAsc.head()


# In[ ]:


# Data is rearranged by ascending or descending completeness
freeFormResponsesFilteredDesc = msng.nullity_sort(freeFormResponsesFiltered, 
                                                  sort='descending')
freeFormResponsesFilteredDesc.head()


# In[ ]:


# Patterns are detected in filtered data
msng.matrix(freeFormResponsesFiltered.sample(500))


# **Multile Choice Responses**

# In[ ]:


# Sniffing multiple choice selections
multileChoiceResponses.head()


# In[ ]:




