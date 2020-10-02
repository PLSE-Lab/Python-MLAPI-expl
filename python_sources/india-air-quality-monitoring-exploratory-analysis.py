#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv('../input/data.csv',encoding = "ISO-8859-1")
df.head()


# In[ ]:


### Explore the data for understanding the information


# In[ ]:


# Number of records 
print( "The number of records  or rows of the data is ",df.shape[0])
print("The number of fields and columns is ", df.shape[1])


# In[ ]:


# Unique station codes involved in collecting the data
print(" Number of Unique stations across India involved in collecing the data", df.stn_code.nunique())


# In[ ]:


## Number of states 
print('Number of states ', df.state.nunique())


# In[ ]:


## Creating a new column as Year


# In[ ]:


df['date'] = df['date'].astype('str')
def get_year(s):
    if s == 'nan':
        return(0)
    return(int(s[0:4]))
convert = lambda x: get_year(x)
df['Year'] = df['date'].apply(convert)


# In[ ]:


#### what are the top 5 highly polluted states in the country
### make a subset of the data with states and the so2, no2, rspm, spm data values. 


# In[ ]:


### make a subset of the data with states and the so2, no2, rspm, spm data values. 
set1 = df[['state', 'so2','no2','rspm','spm','Year']]


# In[ ]:


### NaN Values missing data values are replaced with the average value of the other data points
set1 = set1.copy()
so2_ave = round(set1['so2'].mean(), 2)
no2_ave = round(set1['no2'].mean(),2)
rspm_ave = round(set1['rspm'].mean(),2)
spm_ave = round(set1['spm'].mean(),2)
set1['so2'].fillna(so2_ave, inplace = True)
set1['no2'].fillna(no2_ave, inplace = True)
set1['rspm'].fillna(rspm_ave, inplace = True)
set1['spm'].fillna(spm_ave, inplace = True)


# ### States & SO2 Levels

# In[ ]:


## Group by this set  state as the pivot and the mean values of the poulltion paramenters ( All the years average data is taken)
state_pollution = set1[['state', 'so2']].groupby('state').mean()
state_pollution.reset_index(inplace = True)
state_pollution.sort_values('so2', ascending= False, inplace = True)
state_pollution.plot(kind = 'bar', figsize= (20,5), x = 'state', fontsize= 15, title = 'States & SO2 Levels')
plt.show()


# ### TOP 5 states that have high so2 emission

# In[ ]:


state_pollution.head(5)


# ###  States & NO2  levels

# In[ ]:


## Group by this set  state as the pivot and the mean values of the poulltion paramenters ( All the years average data is taken)
state_pollution = set1[['state', 'no2']].groupby('state').mean()
state_pollution.reset_index(inplace = True)
state_pollution.sort_values('no2', ascending= False, inplace = True)
state_pollution.plot(kind = 'bar', figsize= (20,5), x = 'state', fontsize= 15, title = 'States & NO2 Levels')
plt.show()


# ### Top 5 States that have high NO2 Emissions.

# In[ ]:


state_pollution.head()


# ###  States & rspm, spm

# In[ ]:


state_pollution = set1[['state', 'spm', 'rspm']].groupby('state').mean()
state_pollution.reset_index(inplace = True)
state_pollution.sort_values('spm', ascending= False, inplace = True)
state_pollution.plot(kind = 'bar', figsize= (20,5), x = 'state', fontsize= 15, title = 'States & spm rspm')
plt.show()


# In[ ]:


state_pollution.head()


# ###  States  where the  ratio rspm/spm percentage is high

# In[ ]:


state_pollution['Ratio'] = round((state_pollution['rspm']/state_pollution['spm'])*100, 2)


# In[ ]:


state_pollution = state_pollution.sort_values('Ratio', ascending = False)


# In[ ]:


state_pollution.plot(kind = 'bar', figsize = (20, 5), fontsize = 15, x = 'state')
plt.show()


# ### Top 5 states where the  ratio is high. 

# In[ ]:


state_pollution.head(5)


# ### State wise emissions trend over years

# In[ ]:


states = set1['state'].unique()


# In[ ]:


for st in states:
    state_set = set1[set1['state'] == st]
    state_set = state_set.groupby('Year').mean()
    state_set.reset_index(inplace = True)
    state_set = state_set[state_set['Year'] > 0]
    x = state_set['Year'].values
    y1 = state_set['so2'].values
    y2 = state_set['no2'].values
    y3 = state_set['spm'].values
    y4 = state_set['rspm'].values
    plt.figure(figsize = (15,10))
    plt.subplot(2,2,1)
    plt.plot(x, y1, marker = "*", color = 'b', label='SO2')
    plt.plot(x, y2, marker = "<", color = 'r', label = 'NO2')
    plt.title("SO2, NO2 Emissions --- "+str(st))
    plt.legend()
    plt.subplot(2,2,2)
    plt.plot(x, y3, marker = "*", color = 'b', label = 'spm')
    plt.plot(x, y4, marker = "<", color = 'r', label = 'rspm')
    plt.title("spm, rspm Emissions --- " + str(st))
    plt.legend()
    plt.show()


# ###  All India SO2 and NO2 emissions year on year 

# In[ ]:


trace = set1[['Year', 'so2', 'no2']]
trace = trace[trace['Year'] != 0]
grouped = trace.groupby('Year').mean()
grouped.reset_index(inplace = True)
x = grouped.Year
y = grouped.so2
y1 = grouped.no2
plt.figure(figsize = (15,5))
plt.title("SO2, NO2 Emissions")
plt.plot(x, y, marker = "*", color = 'b', label = 'SO2')
plt.plot(x, y1, marker = ">", color = 'r', label = 'NO2')
plt.legend()
plt.show()


# ### All India spm and rspm levels

# In[ ]:


trace = set1[['Year', 'spm', 'rspm']]
trace = trace[trace['Year'] != 0]
grouped = trace.groupby('Year').mean()
grouped.reset_index(inplace = True)
x = grouped.Year
y = grouped.spm
y1 = grouped.rspm
plt.figure(figsize = (15,5))
plt.title("spm, rspm levels")
plt.plot(x, y, marker = "*", color = 'b', label = 'spm')
plt.plot(x, y1, marker = ">", color = 'r', label = 'rspm')
plt.legend()
plt.show()


# In[ ]:




