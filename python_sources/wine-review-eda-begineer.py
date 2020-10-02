#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.api as sm
from sklearn.preprocessing import scale

from scipy import stats
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df=pd.read_csv('../input/winedat/wine.csv')


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


df.head(3)


# In[ ]:


df.tail(3)


# In[ ]:


df.shape


# In[ ]:


dupli = df[df.duplicated()]
print(dupli)


# In[ ]:


df[df.duplicated('description',keep=False)].sort_values('description').head(5)


# In[ ]:


df = df.drop_duplicates('description')
df = df[pd.notnull(df.price)]
df.shape


# In[ ]:


df.dtypes


# In[ ]:


df1=df.groupby('country')['price','points'].describe()


# In[ ]:


df1.head(3)


# In[ ]:


df1.tail(5)


# In[ ]:


df1.max()


# In[ ]:


sns.scatterplot(df['price'], df['points']);


# In[ ]:


sns.distplot(df['price'],kde=False,color='red', bins=100);
#plt.ylabel('Frequency',fontsize=10);


# In[ ]:


print('correlation in betwn price and points:',pearsonr(df.price, df.points))
print(sm.OLS(df.points, df.price).fit().summary())
chart =sns.lmplot(y= 'price', x='points', data=df)


# In[ ]:


fig, ax = plt.subplots(figsize = (30,10))
chart = sns.boxplot(x='country', y= 'points', data= df)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize = (30,10))
chart = sns.boxplot(x='country', y= 'price', data= df)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


#removing county less than 150 sample
county=df.groupby('country').filter(lambda x:len(x)>150)


# In[ ]:


county.shape


# In[ ]:


fig, ax = plt.subplots(figsize = (30,10))
chart = sns.boxplot(x='country',y='price', data=county, ax = ax)
plt.xticks(rotation = 90)
plt.show()


# In[ ]:


df2 = pd.DataFrame({col:vals['points'] for col,vals in county.groupby('country')})
md=df2.median()
print(md)


# In[ ]:


fig, ax = plt.subplots(figsize = (30,10))
chart = sns.boxplot(x='country',y='points', data=county, order=md.index,ax = ax)
plt.xticks(rotation = 90)
plt.show()


# In[ ]:


ndf1 = county.groupby('variety').filter(lambda x: len(x) >200)
df4 = pd.DataFrame({col:vals['points'] for col,vals in ndf1.groupby('variety')})
med = df4.median()
med.sort_values(ascending=False, inplace=True)
fig, ax = plt.subplots(figsize = (30,10))
chart = sns.boxplot(x='variety',y='points', data=ndf1, order=med.index, ax = ax)
plt.xticks(rotation = 90)
plt.show()


# In[ ]:


sns.violinplot(x='variety', y='points',data=df[df.variety.isin(df.variety.value_counts()[:10].index)])
plt.xticks(rotation=90)
plt.show()


# In[ ]:


df.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])


# In[ ]:


df6=df.groupby(['country', 'province']).apply(lambda df: df.loc[df.price.idxmax()])


# In[ ]:


df6


# In[ ]:


value_counts = df["variety"].value_counts()
value_counts.head()
#value_counts.tail()


# In[ ]:


fig, ax = plt.subplots(figsize = (30,10))
sns.countplot(x="country", data=df)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


fig, ax= plt.subplots(figsize =(30,10))
sns.barplot(x='country', y='price',data=df);
plt.xticks(rotation=90);


# In[ ]:


fig, ax= plt.subplots(figsize =(30,10))
sns.lineplot(x='variety', y='price',data=county);
plt.xticks(rotation=90);


# In[ ]:


var=county[county['country']=='France'] #as france wine is costliest


# In[ ]:


var


# In[ ]:


cnt=var['variety'].value_counts()


# In[ ]:


cnt


# In[ ]:


sns.scatterplot(x='variety', y='price', data=var) #Its very noisey so plotting is not a good idea


# In[ ]:


#from the point, variety list we are going to takeout the best 6 variety and find the price in france
varietylist = ['Pinot Noir','Cabernet Sauvignon','Chardonnay','Port','Sangiovese Grosso']


# In[ ]:


newlst=var[var['variety'].isin(varietylist)]


# In[ ]:


newlst.count()


# In[ ]:


sns.countplot(x="variety", data=newlst,)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


plt.figure(figsize=(30,10)) #variation of price in fance from top variety
sns.violinplot(x="variety", y="price", data=newlst, inner=None);


# In[ ]:


#top review country
df.country.value_counts().head()


# In[ ]:


#review per country
fig, ax=plt.subplots(figsize=(30,10))
sns.countplot('country',data=df,edgecolor=sns.color_palette('dark',8),order=df['country'].value_counts().head(20).index)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


#top wine tester
df.taster_name.value_counts().head()


# In[ ]:


fig, ax=plt.subplots(figsize=(30,10))
sns.countplot('taster_name', data=df, order=df['taster_name'].value_counts().index)
plt.xticks(rotation=90)
plt.show()


# In[ ]:




