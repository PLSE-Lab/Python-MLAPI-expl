#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Analysis in India

# ## In this analysis, let us look into how covid-19 has been in India.

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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns  
from sklearn.model_selection  import train_test_split

covid_19_india = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
covid_19_india.head()


my_tab = pd.crosstab(index=covid_19_india["State/UnionTerritory"],  # Make a crosstab
                     columns="count")                  # Name the count column
my_tab


# In[ ]:


covid_19_india.groupby(by=['State/UnionTerritory'])['Confirmed'].sum().reset_index().sort_values(['Confirmed']).tail(10).plot(x='State/UnionTerritory',
                                                                                                           y='Confirmed',
                                                                                                           kind='bar',
                                                                                                           figsize=(15,5))
plt.show()


# In[ ]:


covid_19_india.groupby(by=['State/UnionTerritory'])['Cured'].sum().reset_index().sort_values(['Cured']).tail(10).plot(x='State/UnionTerritory',
                                                                                                           y='Cured',
                                                                                                           kind='bar',
                                                                                                           figsize=(15,5))
plt.show()


# ### Currently, in the above bar chart, we see Maharashtra has high number confirmed cases followed by Kerala.

# In[ ]:


AgeGroupDetails = pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv')
AgeGroupDetails.head()
plt.figure(figsize=(10,5))  # setting the figure size
ax = sns.barplot(x='AgeGroup', y='TotalCases', data=AgeGroupDetails, palette='muted')  # barplot


# ### Age group between 20-29 has highest confirmed cases 

# In[ ]:


ind_df = pd.read_csv('../input/covid19-in-india/IndividualDetails.csv')
ind_df.head(10)


# In[ ]:


plt.figure(figsize=(10,5))  # setting the figure size
#sns.countplot(x="children",data=insur_data);

sns.countplot(x='detected_state', data=ind_df, hue='current_status');  # barplot


# In[ ]:




