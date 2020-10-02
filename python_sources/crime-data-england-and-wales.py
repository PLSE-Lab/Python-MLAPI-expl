#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Load lib
import numpy as np
import pandas as pd

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')


# In[ ]:


#Load dataset
data = pd.read_csv('../input/rec-crime-pfa.csv')
data.head(3)


# ### Understand some basic details from dataset 

# In[ ]:


basic = {'unique_values' : data.nunique(),
        'na_values' : data.isna().sum(),
        'data_type' : data.dtypes}
print("Data shape : " , data.shape)
pd.DataFrame(basic)


# Convert date column type to datetime 

# In[ ]:


data['12 months ending'] = pd.to_datetime(data['12 months ending'])

#Get year, month, day
data['Year'] = data['12 months ending'].dt.year
data['Month'] = data['12 months ending'].dt.month
data['Day'] = data['12 months ending'].dt.day


# ### Unique values
# 

# In[ ]:


# Unique values
{'unique_pfa': data['PFA'].unique(),
'unique_region' : data['Region'].unique(),
'unique_offence' : data['Offence'].unique(),
'unique_year' : data['Year'].unique(),
'unique_month' : data['Month'].unique(),
'unique_day' : data['Day'].unique()}


# ### These data are taken from 2003 to 2018 in every quarter end.
# 
# Here we are going answer below mentioned questions
# 1. Which PFA handling more crime?
# 2. Region wise find top most offence? 
# 3. Crime rate based on year
# 4. Quaterly Crime rate trends for all years
# 
# ### 1. Which PFA handling more crime?

# In[ ]:


pfa = data[['PFA','Rolling year total number of offences']].groupby(['PFA']).sum().reset_index().sort_values('Rolling year total number of offences', ascending = False)

plt.figure(figsize = (10,10))
sns.barplot(y = 'PFA', x = 'Rolling year total number of offences', data = pfa)
plt.xlabel('CrimeRate', size = 15)
plt.ylabel('PFA', size = 15)
plt.title('PFA vs CrimeRate', size = 15)


# ### 2. Region wise find top most offence? 

# In[ ]:


region = data[['Region','Offence','Rolling year total number of offences']].groupby(['Region','Offence']).sum().reset_index()
#Get index
idx = region.groupby(['Region'])['Rolling year total number of offences'].transform(max) == region['Rolling year total number of offences']
region[idx]


# ### Crime rate based on year 

# In[ ]:


year = data[['Year','Rolling year total number of offences']].groupby(['Year']).sum().reset_index().sort_values('Rolling year total number of offences', ascending = False)

plt.figure(figsize = (5,5))
sns.barplot(x = 'Year', y = 'Rolling year total number of offences', data = year)
plt.xticks(rotation = 45)
plt.xlabel('CrimeRate', size = 15)
plt.ylabel('Year', size = 15)
plt.title('Year vs CrimeRate', size = 15)


# ### Quaterly Crime rate trends for all years

# In[ ]:


quater = data[['Year','Month','Rolling year total number of offences']].groupby(['Year','Month']).sum().reset_index().sort_values(['Year','Month'])
quater

