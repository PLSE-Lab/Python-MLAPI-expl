#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# # Operations on combined data

# In[3]:


data_dict = pd.read_excel('../input/combined-data/Output.xls', sheet_name=None)


# In[4]:


data_dict.keys()


# In[16]:


id_cols = ["Area_Name", "Year", "Group_Name", "Sub_Group_Name", "Subgroup"]


# In[28]:


modified_dict = {}
for key, value in data_dict.items():
    modified_dict[key] = value.melt(id_vars=value.columns.intersection(id_cols), var_name="Type", value_name="Count")


# In[30]:


df_combined = pd.concat(modified_dict, axis=0, join='outer')


# In[31]:


df_combined.head()


# # Operations on individual files (q1, q2)

# In[ ]:


#data = pd.read_csv('../input/39_Specific_purpose_of_kidnapping_and_abduction.csv')


# In[ ]:


#data.head()


# In[ ]:


#id_cols = ['Area_Name', 'Year', 'Group_Name', 'Sub_Group_Name', 'K_A_Cases_Reported']


# In[ ]:


#maha = data[data['Area_Name'] == "Maharashtra"]
#maha.head()


# In[ ]:


#maha.loc[:, maha.columns.difference(id_cols)] = maha[maha.columns.difference(id_cols)].apply(lambda x: x.astype('float'))


# In[ ]:


#maha['Total_cases'] = maha[maha.columns.difference(id_cols)].sum(axis=1)


# In[ ]:


#maha.head()


# In[ ]:


#maha_summary = maha.groupby(by='Year').sum()


# In[ ]:


#maha_summary.to_csv('Maharashtra_kidnappings.csv')


# In[ ]:


#data_rape = pd.read_csv('../input/20_Victims_of_rape.csv')


# In[ ]:


#data_rape.head()


# In[ ]:


#data_rape_clean = data_rape[data_rape['Subgroup'] == "Total Rape Victims"]


# In[ ]:


data_rape_clean.groupby('Area_Name')['Victims_of_Rape_Total'].sum()


# In[ ]:




