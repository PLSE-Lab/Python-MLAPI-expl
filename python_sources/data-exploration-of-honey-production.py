#!/usr/bin/env python
# coding: utf-8

# # Data Analysis of Honey Production (1998 - 2012)

# Dataset Name : honeyproduction.csv
# 
# Variable Names 
# 
# numcol     : Number of honey producing colonies.
# yieldpercol: Honey yield per colony. Unit is pounds
# totalprod  : Total production (numcol x yieldpercol). Unit is pounds
# stocks     : Refers to stocks held by producers. Unit is pounds
# priceperlb : Refers to average price per pound based on expanded sales. Unit is dollars.
# prodvalue  : Value of production (totalprod x priceperlb). Unit is dollars.
# Year       : The observations captured by every year (1998 - 2012)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#Import Basic Libraries for Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Data Exploration

# In[ ]:


# Read Honey Production Dataset.

hp = pd.read_csv(r'/kaggle/input/honey-production/honeyproduction.csv')


# In[ ]:


#Reading First five Rows

hp.head() 


# In[ ]:


#Reading Last Five Rows

hp.tail()


# In[ ]:


#Summarizing hp dataset

hp.describe()


# In[ ]:


# Creating summary table to understand the trend using year variable

hp_year = hp[['numcol','totalprod','year','yieldpercol','stocks','prodvalue']].groupby('year').sum()

hp_year.head()


# In[ ]:


# Resetting index value

hp_year.reset_index(level=0,inplace=True)
hp_year.head()


# Questions:
# 
# 1.How has honey production yield changed from 1998 to 2012? 
# 
#   First I am going to visualize the yield per colony variable,Total Honey Production variable as well as the trend of honey production value from the honey production dataset.

# In[ ]:


# Visualizing the trend of Yield per Colony from year 1998 to 2012

plt.figure(figsize=(25,8))
plt.plot(hp_year['year'],hp_year['yieldpercol'])
plt.title('Trend of Honey Yield per Colony' ,fontsize=25)
plt.xlabel('Year',fontsize=25)
plt.ylabel('Yield per Colony',fontsize = 25)


# * The trend is showing there is a consistent declined honey yield per colony from year 2002 to 2007.

# In[ ]:


# Visualizing the total honey production from the year 1998 to 2012.

plt.figure(figsize=(25,8))
plt.plot(hp_year['year'],hp_year['totalprod'])
plt.title('Total Honey Production in USA' ,fontsize=25)
plt.xlabel('Year',fontsize=25)
plt.ylabel('Total Production of Honey (lbs.)',fontsize = 25)


# * The value of Total production is keep on changing (Decreasing and Increasing) from the year 1998 to 2004 and the year 
#   2007 to 2012.

# In[ ]:


# Visualizing the trend of Production Value from year 1998 to 2012

plt.figure(figsize=(25,8))
plt.plot(hp_year['year'],hp_year['prodvalue'])
plt.title('Trend of Production Value' ,fontsize=25)
plt.xlabel('Year',fontsize=25)
plt.ylabel('Production Value',fontsize = 25)


# * Production value is showing upward trend from the year 1998 to 2012.

# # State Analysis

# 2.Over time, which states produce the most honey? Which produce the least? Which have experienced the most change in honey yield?

# In[ ]:


# Group the dataset by states and using sum method to get the total honey production value descending order. 

US_state = hp[['state','totalprod','yieldpercol']].groupby('state').sum()
US_state.reset_index(level=0,inplace=True)
US_state.sort_values(by='totalprod',ascending=False,inplace=True)
US_state.head()


# In[ ]:


#Creating a Bar chart to visualize the total honey production by states.

plt.figure(figsize=(20,7))
sns.barplot(x=US_state['state'],y = US_state['totalprod'])
plt.title('Statewise Total Honey production in USA',fontsize =20)
plt.xlabel('States',fontsize=20)
plt.ylabel('Total Production of Honey in USA',fontsize=20)


# The Bar chart explains the states ND,CA,SD and FL are the top honey productions states.
# The states SC,OK,MD and KY are least honey production states in the US.

# In[ ]:


# Creating a table to find out maximum production value from the states

US_state_max = hp[['state','totalprod']].groupby('state').max()
US_state_max.reset_index(level=0,inplace=True)
US_state_max.columns = ['State','Max Prod']
US_state_max.head()


# In[ ]:


# Creating a table to find out minimum production value from the states

US_state_min = hp[['state','totalprod']].groupby('state').min()
US_state_min.reset_index(level=0,inplace=True)
US_state_min.columns = ['State','Min Prod']
US_state_min.head()


# In[ ]:


# Merging the Max Prod and Min Prod varible to find the range.

st_range = pd.merge(US_state_max,US_state_min,how='inner',on='State')
st_range.head()


# In[ ]:


#Create a Per_Change Column in the st_range dataset to understand honey production changes by states.


st_range['Per_Change'] = ((st_range['Max Prod']-st_range['Min Prod'])/st_range['Max Prod'])*100
st_range.sort_values(by='Per_Change',ascending=False,inplace=True)
st_range.head()


# In[ ]:


#Create a Bar chart to visualize the statewise decline trend.

plt.figure(figsize=(20,7))
sns.barplot(x='State',y='Per_Change',data= st_range)
plt.title('Statewise Production Decline Trend',fontsize=20)
plt.xlabel("State",fontsize=15)
plt.ylabel("% Decline",fontsize=15)


# The Bar chart explains the states MO,NM and ME are the highest percentage of decline in the honey production.
# The states OR,SC and MI are lowest decline in the honey production.
