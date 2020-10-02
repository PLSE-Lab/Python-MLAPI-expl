#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Following
# http://mailchi.mp/422c4b65434f/data-challenge-day-1-read-in-and-summarize-a-csv-file-2576433


# In[ ]:


import numpy as np 
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


dataset = pd.read_csv('../input/anonymous-survey-responses.csv')


# In[ ]:


dataset.describe().transpose()


# In[ ]:


scipy.stats.chisquare(
    dataset["Have you ever taken a course in statistics?"].value_counts())


# In[ ]:


scipy.stats.chisquare(
    dataset["Do you have any previous experience with programming?"].value_counts())


# In[ ]:


contingencyTable = pd.crosstab(
    dataset["Have you ever taken a course in statistics?"],
    dataset["Do you have any previous experience with programming?"]) 


# In[ ]:


scipy.stats.chi2_contingency(contingencyTable)

