#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pandas-profiling==2.2.0')


# In[ ]:


import pandas_profiling as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Data Section 1 Files For Men and Women

# In[ ]:


mteams = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MTeams.csv')
wteams = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WTeams.csv')
mseasons = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MSeasons.csv')
wseasons = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WSeasons.csv')


# In[ ]:


pp.ProfileReport(mteams)


# In[ ]:


pp.ProfileReport(wteams)


# In[ ]:


pp.ProfileReport(mseasons)


# In[ ]:


pp.ProfileReport(wseasons)

