#!/usr/bin/env python
# coding: utf-8

# ***Please UpVote if you like the work!!!***

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('../input/airquality/data.csv')


# In[ ]:


df.head()


# In[ ]:


df['type'].replace('Residential, Rural and other Areas','Residential',inplace = True)
df['type'].replace('Residential and others','Residential',inplace = True)
df['type'].replace('Industrial Areas','Industrial',inplace = True)
df['type'].replace('Industrial Area','Industrial',inplace = True)
df['type'].replace('Sensitive Area','Sensitive',inplace = True)
df['type'].replace('Sensitive Areas','Sensitive',inplace = True)


# We have replaced all the similar types of locations and got them down to only 4 following categories:
# 
# Residential
# 
# Industrial
# 
# Sensitive
# 
# RIRUO

# In[ ]:


df['type'].value_counts()


# In[ ]:


df['type'].value_counts().plot(kind = 'bar')


# From the above figure we can see that the data was recorded more from the residential areas as compared to other areas.

# In[ ]:


g = df.groupby(['state','type'])
d = dict(list(g))
kar_ind = d[('Karnataka','Industrial')].median()
kar_res = d[('Karnataka','Residential')].median()
kar_sen = d[('Karnataka','Sensitive')].median()
print(kar_ind,kar_res,kar_sen)
# kar_riruo = d[('Karnataka','RIRUO')].mean()


# In[ ]:


print(df['so2'].isnull().sum())
print(df['no2'].isnull().sum())


# In[ ]:


df.loc[(df['state'] == 'Karnataka') & (df['type'] == 'Industrial') & (df['so2'].isnull()),'so2'] = kar_ind['so2']
df.loc[(df['state'] == 'Karnataka') & (df['type'] == 'Residential') & (df['so2'].isnull()),'so2'] = kar_res['so2']
df.loc[(df['state'] == 'Karnataka') & (df['type'] == 'Sensitive') & (df['so2'].isnull()),'so2'] = kar_sen['so2']
# df.loc[(df['state'] == 'Karnataka') & (df['type'] == 'RIROU') & (df['so2'].isnull()),'so2'] = kar_rirou['so2']


# In the above code we have replaced the na values for so2 in the state of karnataka with the medians depending on their type of locations 

# In[ ]:


df.loc[(df['state'] == 'Karnataka') & (df['type'] == 'Industrial') & (df['no2'].isnull()),'no2'] = kar_ind['no2']
df.loc[(df['state'] == 'Karnataka') & (df['type'] == 'Residential') & (df['no2'].isnull()),'no2'] = kar_res['no2']
df.loc[(df['state'] == 'Karnataka') & (df['type'] == 'Sensitive') & (df['no2'].isnull()),'no2'] = kar_sen['no2']
# df.loc[(df['state'] == 'Karnataka') & (df['type'] == 'RIROU') & (df['so2'].isnull()),'so2'] = kar_rirou['so2']


# In the above code we have replaced the na values for no2 in the state of karnataka with the medians depending on their type of locations

# In[ ]:


df['date'] = pd.to_datetime(df['date'])


# In[ ]:


df['year'] = df['date'].dt.year


# In[ ]:


df['year'].fillna(method = 'ffill',inplace = True)


# In the above code we have extracted the year from the date replaced na values for the same with forward fill

# In[ ]:


df['year'] = df['year'].astype(int)


# In[ ]:


df['year'].isnull().sum()


# In[ ]:


d = dict(list(df[['location','year','so2','no2']].groupby('location')))


# In[ ]:


data = d['Bangalore'].groupby('year').median().reset_index()


# In[ ]:


data


# In[ ]:


plt.figure(figsize=(15,5))
plt.xticks(np.arange(1980,2016))
sns.lineplot(x='year',y='so2',data=data)
sns.lineplot(x='year',y='no2',data=data)
plt.legend(['so2','no2'])


# From the above plot we can see that in BANGALORE there is an extreme peak in so2 level after 1995 till 1998. Furthur we see a sudden dip in so2 level in 1999 till 2000 which kept furthur gradually decreasing till 2005.

# ![](http://)From the above plot we can also see that in BANGALORE there is an extreme peak in no2 level after 2001 till 2004 which gradually decreased after 2004.

# ---
# ---
# ---
# ---
# ---

# In[ ]:


df.head()


# In[ ]:


print(df.rspm.isnull().sum())
print(df.spm.isnull().sum())


# In[ ]:


df1 = dict(list(df.groupby(['location','type'])))
data = pd.DataFrame()
for key in df1:
    df2 = df1[key].sort_values('date')
    df2['rspm'].fillna(method = 'ffill',inplace = True)
    df2['spm'].fillna(method = 'ffill',inplace= True)
    data = pd.concat([data,df2])


# In the above code I have done a forward fill for spm and rspm NA data depending on its location and type of location.

# In[ ]:


df1 = dict(list(data.groupby(['location','type'])))
data1 = pd.DataFrame()
for key in df1:
    df2 = df1[key].sort_values('date')
    df2['rspm'].fillna(method = 'bfill',inplace = True)
    df2['spm'].fillna(method = 'bfill',inplace= True)
    data1 = pd.concat([data1,df2])


# Furthurmore, if there are any values which are NA at first position according to location and type, I have done a backward fill for spm and rspm data depending on its location and type of location.

# In[ ]:


data1.head()


# In[ ]:


print(data1.rspm.isnull().sum())
print(data1.spm.isnull().sum())


# In[ ]:


df1 = dict(list(data1.groupby(['state','type'])))
data2 = pd.DataFrame()
for key in df1:
    df2 = df1[key]
    df2['rspm'].fillna(df2['rspm'].median(),inplace = True)
    df2['spm'].fillna(df2['spm'].median(),inplace= True)
    data2 = pd.concat([data2,df2])


# Still we can see that there seem to be some locations with location types with all values as NA. Here the backward or forward fill won't work. So I have replaced such values with median grouped by state and location type.

# In[ ]:


print(data2.rspm.isnull().sum())
print(data2.spm.isnull().sum())


# In[ ]:


data2


# In[ ]:


df1 = dict(list(data2.groupby('type')))
data3 = pd.DataFrame()
for key in df1:
    df2 = df1[key]
    df2['rspm'].fillna(df2['rspm'].median(),inplace = True)
    df2['spm'].fillna(df2['spm'].median(),inplace= True)
    data3 = pd.concat([data3,df2])


# Furthur any remaining NA values for rspm and spm have been replaced by medians grouped by type.

# In[ ]:


data3


# In[ ]:


print(data3.rspm.isnull().sum())
print(data3.spm.isnull().sum())


# In[ ]:


data3['type'].value_counts()


# Still there is 'RIRUO' location type for which there exists NA values for spm.

# In[ ]:


data3.reset_index(inplace=True)


# In[ ]:


data3.drop(columns=['index','stn_code','sampling_date','agency','location_monitoring_station'],inplace = True)


# The above columns don't seem to be very useful for any furthur analysis so I prefer to drop these colums to concentrate more on the essential features.

# In[ ]:


data3.head()


# In[ ]:


data3.groupby('state').median()['rspm'].sort_values(ascending = False).plot(kind = 'bar', figsize = (17,5))


# In[ ]:


data3.groupby('location').median()['rspm'].sort_values(ascending = False).head(50).plot(kind = 'bar', figsize = (17,5))


# It quite evident from the above plots that the rspm levels are high for the following cities from Uttar pradesh and Punjab: 
# 
# UP : Ghaziabad, Allahabad, Bareilly, Mathura, Khurja, Lucknow, Kanpur
# 
# Punjab : Ludhiana, Khanna, Amritsar, Bhatinda
# 
# We can verify the same with the following article from indian express in september 2008 : http://archive.indianexpress.com/news/khurja-state-s-most-polluted-town-lucknow-comes-second/360739/

# In[ ]:


data3.groupby('location').median()['rspm'].sort_values(ascending = False).tail(50).plot(kind = 'bar', figsize = (17,5))


# We can also see that the lowest rspm level are in Pathanamthitta which is a city in Kerela which is quite obvious as Kerela is among those states with lowest rspm levels

# A very high rspm level could directly affect the people with many breathing/respiratory problems. 

# ---
# ---
# ---
# ---
# ---

# In[ ]:


data3.groupby('state').median()['spm'].sort_values(ascending = False).plot(kind = 'bar', figsize = (17,5))


# In[ ]:


data3.groupby('location').median()['spm'].sort_values(ascending = False).head(50).plot(kind = 'bar', figsize = (17,5))


# Similary we can see from the above plots that there is high level of spm for locations in Uttar Pradesh, Delhi and Rajasthan.

# ***Please UpVote if you like the work!!!***
