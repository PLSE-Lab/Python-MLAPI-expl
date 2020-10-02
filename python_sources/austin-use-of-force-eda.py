#!/usr/bin/env python
# coding: utf-8

# # Austin use of force - EDA
# 
# Simple exploration of the Austin incidents dataset.
# 
# No plots or anything fancy :D

# In[ ]:


import pandas as pd


# ## Incidents file

# In[ ]:


loc = '../input/cpe-data/Dept_37-00027/37-00027_UOF-P_2014-2016_prepped.csv'
df = pd.read_csv(loc, header=[1])
df['Date Occurred'] = pd.to_datetime(df['Date Occurred'])
df.head().T


# In[ ]:


df.shape


# In[ ]:


pct_nan = df.isnull().sum() / df.shape[0]
pct_nan = pct_nan[pct_nan > 0.01]
pct_nan.name = "nan"
pct_nan.to_frame().style     .format("{:.1%}")


# In[ ]:


df['RIN'].nunique()


# In[ ]:


df['Primary Key'].nunique()


# In[ ]:


df[df['Primary Key'] == 2015541517]

# same incident, different people (:


# In[ ]:


df['Date Occurred'].min()


# In[ ]:


df['Date Occurred'].max()


# In[ ]:


df['Date Occurred'].dt.year.value_counts()


# In[ ]:


df['Area Command'].value_counts()


# In[ ]:


df['Nature of Contact'].value_counts()


# In[ ]:


df['Reason Desc'].value_counts()


# In[ ]:


df['Subject Sex'].value_counts()


# In[ ]:


df['Race'].value_counts()


# In[ ]:


df['Subject Role'].value_counts().head(10)


# In[ ]:


df['Subject Conduct Desc'].value_counts()


# In[ ]:


df['Subject Resistance'].value_counts().head(10)
# needs cleaning (?)


# In[ ]:


df['Weapon Used 1'].value_counts()


# In[ ]:


df['Weapon Used 2'].value_counts()


# In[ ]:


df['Weapon Used 3'].value_counts()


# In[ ]:


df['Weapon Used 4'].value_counts()


# In[ ]:


df['Number Shots'].value_counts()


# In[ ]:


df['Subject Effects'].value_counts()


# In[ ]:


df['Effect on Officer'].value_counts()


# In[ ]:


df['Officer Organization Desc'].value_counts().head(10)


# In[ ]:


df['Officer Yrs of Service'].value_counts().head()


# In[ ]:


df['X-Coordinate'].nunique()


# In[ ]:


df['Y-Coordinate'].nunique()


# In[ ]:


df['City Council District'].value_counts()


# In[ ]:


df['Geolocation'].nunique()


# In[ ]:


df['City'].value_counts()


# In[ ]:


df['State'].value_counts()


# In[ ]:


df['Latitude'].nunique()


# In[ ]:


df['Longitude'].nunique()

