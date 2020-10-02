#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math


# In[ ]:


df_transfer = pd.read_csv('../input/top250-00-19.csv')


# In[ ]:


df_transfer.head()


# In[ ]:


df_transfer.info()


# In[ ]:


print(df_transfer.describe())


# In[ ]:


# Adding a column with transfer and market values round up for better readability

def to_million(value):
    if not (math.isnan(value)):
        return str(round(value/1000000,1)) + 'mn'

market_value = df_transfer['Market_value']
transfer_fee = df_transfer['Transfer_fee']
df_transfer['Market_mn'] = market_value.apply(to_million)
df_transfer['Transfer_mn'] = transfer_fee.apply(to_million)

df_transfer.head()


# In[ ]:


# Top ten transfers ever
df_transfer.sort_values('Transfer_fee', ascending= False)[['Name','Team_from','Team_to','Season','Transfer_mn']].head(10)


# In[ ]:


# Most expensive transfer of each season

df_transfer.groupby(['Season']).max()[['Transfer_fee']]


# In[ ]:


# Most transferred player by position
dff = df_transfer.groupby(['Season', 'Position'], as_index=False)['Name'].count()
dff.groupby('Season').apply(lambda x: x.Position[x.Name.idxmax()])


# In[ ]:


plt.figure(figsize=(20,12))
sns.countplot(x='Age', data = df_transfer)


# As you are reading this you are not a footballer because a footballer(professionals) wouldn't really know about Kaggle, would he? So if you want your kid to become a professional footballer and talk of the market, you know what position to make him play in. :P #OkSorryBye

# In[ ]:


#Average transfer value per season
plt.figure(figsize=(20,12))
sns.barplot(x='Season', y='Transfer_fee', data = df_transfer)


# In[ ]:


#The leagues that attract most players
plt.figure(figsize=(15,10))
sns.countplot(x='League_to', data=df_transfer,order= df_transfer['League_to'].value_counts().head(10).index, palette='RdBu')


# In[ ]:


#The teams that attract most players
plt.figure(figsize=(15,10))
sns.countplot(x='Team_to', data=df_transfer,order= df_transfer['Team_to'].value_counts().head(10).index, palette='rocket')


# In[ ]:


#Teams with the highest spending power
plt.figure(figsize=(15,10))
dff = df_transfer.groupby('Team_to')['Transfer_fee'].agg('sum').reset_index()
df_highest = dff.sort_values(by='Transfer_fee', ascending=False).head(10)
sns.barplot(x='Team_to',y='Transfer_fee',data=df_highest, palette ='Blues_d')


# In[ ]:


#Teams that cashed in on good players
plt.figure(figsize=(15,10))
dff = df_transfer.groupby('Team_from')['Transfer_fee'].agg('sum').reset_index()
df_highest = dff.sort_values(by='Transfer_fee', ascending=False).head(10)
sns.barplot(x='Team_from',y='Transfer_fee',data=df_highest, palette ='GnBu_d')


# In[ ]:


plt.figure(figsize=(20,12))
dff = df_transfer.groupby(['Season','Team_to'])['Transfer_fee'].mean().reset_index()
df_filter = dff[dff.Team_to.isin(['Paris SG','Real Madrid','FC Barcelona','Man City','Juventus'])].sort_values('Season').reset_index()
df_filter =  df_filter[df_filter['Season']!='2005-2006'] # Removed because messes the graph
sns.lineplot(x='Season',y='Transfer_fee', hue ='Team_to', data=df_filter, palette= 'husl')


# In[ ]:


dff = df_transfer.groupby('Season')['Transfer_fee'].agg('mean').reset_index()
plt.figure(figsize=(20,12))
sns.lineplot(x='Season', y='Transfer_fee', data = dff)


# In[ ]:


dff = df_transfer.groupby('Age')['Transfer_fee'].agg('mean').reset_index()
plt.figure(figsize=(20,12))
sns.barplot(x='Age', y='Transfer_fee', data = dff)


# In[ ]:




