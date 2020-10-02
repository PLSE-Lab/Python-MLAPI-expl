#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# importing libraries 

import csv
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


data_tickets = pd.read_csv('../input/Parking_Violations_Issued_-_Fiscal_Year_2014__August_2013___June_2014_.csv')
data_tickets.head()


# In[3]:


#grouping the data by Registration State and counting the number of tickets issued
ny_datatick=data_tickets.groupby(by='Registration State',as_index=False).count()
ny_datatick.head()


# In[4]:


columns_ny=ny_datatick[['Registration State','Summons Number']] # selecting the columns for further analysis 
columns_ny.head()


# In[5]:


#filtering dataset further and removing 99 as an error
columns_city=columns_ny[columns_ny['Registration State']!='99']
columns_city.head() #all other cities


# In[6]:


columns_NYC=columns_ny[columns_ny['Registration State']=='NY']
columns_NYC.head() #all other cities


# In[7]:


columns_cities=columns_city[columns_city['Registration State']!='NY']
columns_cities.head() #all other cities except 'NY'


# In[8]:


plt.figure(figsize=(13,6))
x=columns_ny['Summons Number']
y=columns_ny['Registration State']

plt.bar(y,x, color = 'b', width = 0.75)
plt.xticks( rotation=90)

plt.title("Number of Parking Tickets Given for Each State registered Car", fontsize=16)
plt.xlabel("Registration State", fontsize=18)
plt.ylabel("No. of cars", fontsize=18)
plt.show()


# ### Conclusion: From this we can conclude, that majorities tickets were issued to the locals of New York, followed by New Jersey and Pennsylvania
# 

# Analysis: In this analysis, I am analysing monthly tickets issues to the locals of new york with other states.

# In[11]:


data_tickets.head()


# In[12]:


data_tickets['date'] = pd.to_datetime(data_tickets['Issue Date']) # Convert date to datetime
data_tickets['month'] = data_tickets['date'].dt.month #extract month from Issue Date


# In[13]:


data_tickets.head()


# In[14]:


tikt_nycity=data_tickets[data_tickets['Registration State']=='NY'] # NY data for all months

month_grp_nycity=tikt_nycity.groupby(by=tikt_nycity['month'],as_index=False).count() #group NY data for each month
month_grp_nycity.head()
month_grp_nycity_sel=month_grp_nycity[['month','Summons Number']]
month_grp_nycity_sel


# In[15]:


tikt_othercity=data_tickets[data_tickets['Registration State']!='NY'] # other cities data for all months

month_grp_othercity=tikt_othercity.groupby(by=tikt_othercity['month'],as_index=False).count()#group other cities data for all the months
month_grp_othercity.head()
month_grp_othercity_sel=month_grp_othercity[['month','Summons Number']]
month_grp_othercity_sel


# In[16]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k') #size of the plot

labels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
plt.bar(month_grp_nycity_sel['month'] , month_grp_nycity_sel['Summons Number'], color = 'b', width = 0.5) # plotting the graph
plt.bar(month_grp_othercity_sel['month'], month_grp_othercity_sel['Summons Number'], color = 'g', width = 0.5)
plt.xticks(month_grp_nycity_sel['month'], labels, rotation=45) #providing xticks to the graph

blue_patch=mpatches.Patch(color='b',label='New York') 
green_patch=mpatches.Patch(color='g',label='Other Cities')
plt.legend(handles=[blue_patch,green_patch]) #providing the labels
plt.xlabel('Month',fontsize=16)
plt.ylabel('Number of cars',fontsize=16)
plt.show()


# ### Conclusion:The maximum tickets were created in October and majority of them by the New York locals followed by May. The least tickets were issued in July.

# In[ ]:




