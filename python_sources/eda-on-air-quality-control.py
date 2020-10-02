#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
import numpy as np


# ### Loading Data

# In[ ]:


air = pd.read_csv('../input/india-air-quality-data/data.csv',encoding='cp1252', low_memory=False)
air.sample(n=5)


# 1. encoding='cp1252'- Encoding type.
# 2. low_memory=False - Since the datatype in column 'stn_code' has mixed data type

# ### Data Insight

# In[ ]:


air.shape


# The dataset comprises of 435742 observations and 13 chracteriestics, of which all are independent variables.
# 
# The column 'date' is a cleaned version of the 'sampling_date' and the same can be dropped.

# In[ ]:


air.info()


# The column 'st_code' & 'date' is a object since it has values in string format
# 
# Few of the columns has null/missing values
# 

# ### Summary Statistics

# In[ ]:


air.describe()


# For all the numeric columns the difference in Mean and Median(50th percentile) is less, 
# apart from rspm and spm which has more difference we can incur that the data will not be 
# symmetrical and will be skewed to the right.
# 
# The difference between the 75th percentile and max value shows that the data will have high outliers
# 

# ### Data Visualization

# In[ ]:


sns.heatmap(air.isnull(),cbar=False,yticklabels=False,cmap = 'GnBu_r')


# The column 'pm2_5' has considerably large Null/NAN values.
# 
# The 'sampling_date' can be dropped of since it's a duplicate of 'date' 

# ### Check for outliers

# In[ ]:


c = ['so2','no2','rspm','spm','pm2_5']
plt.figure(figsize=(10,5))
for i in range(0,len(c)):
    plt.subplot(5,1,i+1)
    sns.boxplot(air[c[i]],color='green',fliersize=5,orient='h')
    plt.tight_layout()


# All the feature columns have outliers

# In[ ]:


plt.figure(figsize=(6,5))
for i in range(0,len(c)):
    plt.subplot(5,1,i+1)
    sns.distplot(air[c[i]],kde=True)
    plt.tight_layout()
    


# All the features are positively skewed 

# In[ ]:


# sns.pairplot(air)


# In[ ]:


sns.heatmap(air.corr(),annot=True,cmap= 'Blues')


# The darker shades shows positive correlation while lighter shades represents negative/no correlation
# 
# pm2_5 have negative/no correlation with spm
# 
# pm2_5 has very low or no correlation with so2, hence the column pm2_5 can be dropped 
# 
# rspm and spm have a positivie correlation

# ### Cleaning the data

# In[ ]:


air['date'] = pd.to_datetime(air.date,format='%Y-%m-%d')
air.info()


# ###### Dropping the columns

# In[ ]:


air.drop(labels=['stn_code','sampling_date','pm2_5'],axis=1,inplace=True)


# In[ ]:


air.sample(n=2)


# Fill the NaT values in date column by forward fill and seperate year from the date

# In[ ]:


air['date'].isna().sum()


# In[ ]:


air['date'].fillna(method='ffill',inplace=True)


# In[ ]:


air.shape


# In[ ]:


air['date'].isnull().sum()


# In[ ]:


air['year'] = air['date'].dt.year


# In[ ]:


air.sample(2)


# Merging the type's value, by replacing the values and reducing it to 4 types

# In[ ]:


air['type'].value_counts().plot(kind='bar')


# In[ ]:


air['type'].replace("Sensitive Areas","Sensitive",inplace=True)
air['type'].replace("Sensitive Area","Sensitive",inplace=True)
air['type'].replace("Industrial Areas","Industrial",inplace=True)
air['type'].replace("Industrial Area","Industrial",inplace=True)
air['type'].replace("Residential and others","Residential",inplace=True)
air['type'].replace("RIRUO","Residential",inplace=True)


# In[ ]:


air['type'].value_counts().plot(kind='bar')


# # ___Analysis_________________

# ### Top 10 states with higest pollutent 

# In[ ]:


st_wise = air.pivot_table(values=['so2','no2','rspm','spm'],index='state').fillna(0)


# In[ ]:


maxso2 = st_wise.sort_values(by='so2',ascending=False)
maxso2.loc[:,['so2']].head(10).plot(kind='bar')


# In[ ]:


maxno2 = st_wise.sort_values(by='no2',ascending=False)
maxno2.loc[:,['no2']].head(10).plot(kind='bar')


# In[ ]:


maxrspm = st_wise.sort_values(by='rspm',ascending=False)
maxrspm.loc[:,['rspm']].head(10).plot(kind='bar')


# In[ ]:


maxspm = st_wise.sort_values(by='spm',ascending=False)
maxspm.loc[:,['spm']].head(10).plot(kind='bar')


# Drilling down to a particular state and then a particular location

# In[ ]:


kar_st = air.query('state=="Karnataka" ')


# In[ ]:


kar_st.head()


# In[ ]:


kar_st.type.value_counts().plot(kind='bar')


# The contribution of industrial's are consideriably more for the pollution for Karnataka state

# In[ ]:


kar_st['spm'].mean()


# In[ ]:


kar_st['so2'].fillna(method='ffill',inplace=True)
kar_st['no2'].fillna(method='ffill',inplace=True)
kar_st['rspm'].fillna(method='ffill',inplace=True)
kar_st['spm'].fillna(168,inplace=True)


# In[ ]:


plt.figure(figsize=(18,6))
plt.xticks(np.arange(1987,2015))
mysore = kar_st.loc[ (kar_st['location']=='Mysore')]
sns.lineplot(x='year',y='so2',data=mysore)
sns.lineplot(x='year',y='no2',data=mysore)
plt.legend(['so2','no2'])


# The above graph for a particular location ie Mysore, we can see that the level of So2 and No2 was high during the 90's untill 20's and gradually
# in a study state from 2013

# In[ ]:


plt.figure(figsize=(18,6))
plt.xticks(np.arange(1987,2015))
mangalore = kar_st.loc[ (kar_st['location']=='Mangalore')]
sns.lineplot(x='year',y='so2',data=mangalore)
sns.lineplot(x='year',y='no2',data=mangalore)
plt.legend(['so2','no2'])


# The above graph for a particular location ie Mangalore, we can see that the level of So2 and No2 have a sudden dip in 2006 and then have maintained the level.. 

# In[ ]:


kar_st.tail()


# Survey done by Agency

# In[ ]:


air.agency.value_counts()


# Maharashtra State Pollution Control Board have maximum number of survey being conducted

# In[ ]:


agent = air.loc[ (air['agency']=="Maharashtra State Pollution Control Board") ]


# In[ ]:


agent.type.value_counts().plot(kind='bar')


# "Residential, Rural and other Areas" are the majour contribution for pollution in Maharastra state or we can also incure that the agency would have conducted lot of survey in these areas 

# In[ ]:


agent.year.value_counts().plot(kind='bar')


# The agency have started the survey in the 90's and can be seen that survey count have increased starting from 2010

# In[ ]:


agent.location.value_counts().plot(kind='bar')


# Higest number of survey is been conducted in Chandrapur and Navi Mumbai location

# In[ ]:


mah_loc = agent.loc[ (agent['location']=='Chandrapur') | (agent['location']=='Navi Mumbai') ] 
mah_loc.head()


# In[ ]:


mah_loc.so2.fillna(method='ffill',inplace=True)


# In[ ]:


mah_loc.groupby('location')['type'].value_counts().plot(kind='bar')


# The higest number of survey is been conducted in "Residential, Rural and other Areas" for both the location, but from  the below graph "Industrial" are the majour contributor for 'So2' and "Residential, Rural and other Areas" are the contributor's for 'No2'

# In[ ]:


sns.barplot(x='type',y='so2',data=mah_loc)


# In[ ]:


sns.barplot(x='type',y='no2',data=mah_loc)

