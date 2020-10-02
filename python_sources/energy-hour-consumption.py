#!/usr/bin/env python
# coding: utf-8

# In[3]:


from IPython.display import HTML
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math
import scipy.stats as ss
import warnings 
import pandas_profiling
warnings.simplefilter('ignore')


# In[4]:


df=pd.read_csv('../input/PJME_hourly.csv', parse_dates=['Datetime'])


# In[5]:


df.shape


# In[6]:


df.head()


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


pandas_profiling.ProfileReport(df)


# In[10]:


df.PJME_MW.plot(kind='hist')


# In[11]:


df.PJME_MW.plot()


# In[12]:


df['timeStamp'] = pd.to_datetime(df['Datetime'])


# In[13]:


df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Date'] = df['timeStamp'].apply(lambda t: t.day)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Year'] = df['timeStamp'].apply(lambda t: t.year)
df['dayname']=df['Datetime'].dt.day_name()


# In[14]:


print('unique years',df['Year'].unique())
print('unique months',df['Month'].unique())


# In[15]:


plt.figure(figsize=(15,5))
plt.title(' Total sell throughout 2002-2018')
sns.countplot(x='Year', data=df, color='lightblue');


# In[16]:


plt.figure(figsize=(10,5))
plt.title(' Total PJME_MW on each month throughout 2002-2018')
plt.ylabel('PJME_MW')
df.groupby('Month').PJME_MW.sum().plot(kind='bar',color='lightblue')


# In[17]:


data_2002 = df[df["Year"] == 2002]
data_2003 = df[df["Year"] == 2003]
data_2004 = df[df["Year"] == 2004]
data_2005 = df[df["Year"] == 2005]
data_2006 = df[df["Year"] == 2006]
data_2007 = df[df["Year"] == 2007]
data_2008 = df[df["Year"] == 2008]
data_2009 = df[df["Year"] == 2009]
data_2010 = df[df["Year"] == 2010]
data_2011 = df[df["Year"] == 2011]
data_2012 = df[df["Year"] == 2012]
data_2013 = df[df["Year"] == 2013]
data_2014 = df[df["Year"] == 2014]
data_2015 = df[df["Year"] == 2015]
data_2016 = df[df["Year"] == 2016]
data_2017 = df[df["Year"] == 2017]
data_2018 = df[df["Year"] == 2018]



# In[18]:


plt.figure(figsize=(10,7))
plt.title(' PJME_MW on each month throughout 2002')
plt.ylabel('PJME_MW')
data_2002.groupby('Month').PJME_MW.sum().plot(kind='bar',color='lightblue')


# In[19]:


plt.figure(figsize=(10,7))
plt.title(' PJME_MW on each month throughout 2003')
plt.ylabel('PJME_MW')
data_2003.groupby('Month').PJME_MW.sum().plot(kind='bar')


# In[20]:


plt.figure(figsize=(10,7))
plt.title(' PJME_MW on each month throughout 2004')
plt.ylabel('PJME_MW')
data_2004.groupby('Month').PJME_MW.sum().plot(kind='bar')


# In[22]:


df.tail()


# In[23]:


# df.drop(['Datetime','timeStamp'], axis=1, inplace=True)
df.index


# In[28]:


df.tail()


# In[29]:


df.Date.unique()


# In[30]:


sns.scatterplot(x='Hour',y='PJME_MW',data=df)


# In[31]:


sns.scatterplot(x='Date',y='PJME_MW',data=df)


# In[32]:


sns.scatterplot(x='Month',y='PJME_MW',data=df)


# In[33]:


sns.scatterplot(x='Year',y='PJME_MW',data=df)


# In[21]:



sns.heatmap(df.corr(),annot=True)


# In[38]:


sns.lineplot(x='Year', y='PJME_MW', data=df)


# In[39]:


sns.lineplot(x='Month', y='PJME_MW', data=df)


# In[34]:


features = ['Hour', 'Day of Week']


# In[35]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[36]:


X = df[features]
y = df['PJME_MW']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 125)
linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test)


# In[37]:


from sklearn import metrics
np.sqrt(metrics.mean_squared_error(y_test,y_pred))


# In[ ]:




