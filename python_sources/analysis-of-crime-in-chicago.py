#!/usr/bin/env python
# coding: utf-8

# I generally think discussion, articles, lectures, and data analysis about crime in Chicago lack nuance. I also believe, and have observed, that much of the analysis using this dataset will describe what is happening instead of providing any valid reasons of why crime is Higher in Chicago. 

# In[ ]:


import pandas as pd
from pandas import read_csv
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# reading in file

# In[ ]:


crime = pd.read_csv('../input/Chicago_Crimes_2012_to_2017.csv')
crime.head(3)


# In[ ]:


Types of crime


# In[ ]:


crime_type = crime['Primary Type']
crime_type.head


# In[ ]:


sns.barplot()


# By Neighborhood

# By Community Area

# Homicide in Chicago . 
# 
# I find the discussion of crime in Chicago 
