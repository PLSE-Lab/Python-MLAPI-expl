#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib as mpl
import matplotlib.pylab as plt
plt.style.use('ggplot')
import os
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


subject_incidents = pd.read_csv("../input/cpe-data/Dept_37-00027/37-00027_UOF-P_2014-2016_prepped.csv")
subject_incidents = subject_incidents[1:] # Remove first row


# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
subject_incidents.groupby(['SUBJECT_RACE', 'SUBJECT_GENDER', 'SUBJECT_DESCRIPTION']).count()['RIN'].unstack().plot.bar(ax=ax)


# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
subject_incidents['INCIDENT_REASON'][1:].value_counts().plot.bar(ax=ax)


# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
subject_incidents.groupby(['SUBJECT_RACE','INCIDENT_REASON']).count()['RIN'][1:].unstack().plot.bar(ax=ax)


# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
subject_incidents.groupby(['SUBJECT_RACE','REASON_FOR_FORCE']).count()['RIN'][1:].unstack().plot.bar(ax=ax)


# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
pd.to_numeric(subject_incidents['OFFICER_YEARS_ON_FORCE']).plot.hist(ax=ax, bins=20)


# A majority of these cases were handled by officers between 0-3 years experience, there are few with over 20 years exp, and it would be interesting to compare how those officers handled crimes in relation to the newer officers. ie does experience have a correlation with use of force?

# In[ ]:




