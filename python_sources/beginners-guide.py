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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('../input/corruption-in-india/Cases_registered_under_PCA_act_and_related_sections_IPC_2013.csv')
data.head()


# In[ ]:


data.columns


# *Above code gives list of all the available Column Headings*

# In[ ]:


data.info()

From the above data we can see that there are only 35 values which are in integer form.
# In[ ]:


data.isna().sum()


# ***There are no null values in the data which is shown by the above code.***

# In[ ]:


data.describe()


# In[ ]:


State_tot_cases = data.groupby(['STATE/UT'])['Total Cases For Investigation'].sum()
#State_tot_cases.sort(how =ascending)
State_tot_cases


# > ***Above you can see the State wise data for total number of cases***

# In[ ]:


state_seized_prop = data.groupby(['STATE/UT'])['Value Of Property Recovered / Seized (In Rupees)'].sum()
state_seized_prop


# > ***Above you can see the State wise details of property recovered in INR***

# In[ ]:


data['YEAR'].unique()


# ***Data is available only for 2013***

# In[ ]:


data.groupby(['STATE/UT'])['Cases Sent Up For Trial And Also Reported For Dept. Action'].sum()


# In[ ]:


data.plot(x ='STATE/UT', y = 'Total Cases For Investigation', kind = 'bar', figsize = (10,6))


# In[ ]:


data.plot(x ='STATE/UT', y = 'Value Of Property Recovered / Seized (In Rupees)', kind = 'bar', figsize = (10,5), color = 'r')


# In[ ]:




