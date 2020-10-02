#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()
import missingno
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


transfers = pd.read_csv('../input/top250-00-19.csv')


# In[ ]:


#Let's now Check the Head of the data to get some insights on what kind of data we are dealing with. 
transfers.head()


# In[ ]:


transfers.info()


# In[ ]:


transfers.describe().transpose()


# In[ ]:


transfers.columns


# In[ ]:


transfers.index


# In[ ]:


transfers.isnull().sum()


# In[ ]:


#Plotting the Graphic of missing values.
missingno.matrix(transfers, figsize = (30,10))


# In[ ]:


#Let's Drop Market Value since it has some missing values 
transfers.drop('Market_value',axis='columns', inplace=True)


# In[ ]:


transfers.columns


# In[ ]:


missingno.matrix(transfers, figsize = (30,10))


# In[ ]:


#Observing which position had the most transfers during the period
transfers['Position'].value_counts()


# In[ ]:


plt.figure(figsize=(30,7))
sns.countplot(x='Position', data=transfers)


# In[ ]:


transfers.head()


# In[ ]:


#The Age in Which Most Players Transferred 
transfers['Age'].value_counts().head()


# In[ ]:


plt.figure(figsize=(15,7))
sns.countplot(x='Age', data=transfers)


# In[ ]:


#Which Team had the most Transfers in the period Top 5
transfers['Team_to'].value_counts().head()


# In[ ]:


#Let's Take a look at the maximum Transfer fee and player
transfers.loc[transfers['Transfer_fee'].idxmax()]


# In[ ]:


#And the Minimum
transfers.loc[transfers['Transfer_fee'].idxmin()]


# In[ ]:


#Let's Look into the Most Expensive Transfers for Each Season 
met=transfers.groupby(['Season']).max()[['Transfer_fee']]
print(met)


# In[ ]:


#Let's Now Examine the Top 20 Transfers of all time 
transfers.sort_values('Transfer_fee', ascending= False)[['Name','Team_from','Team_to','Season','Transfer_fee']].head(20)


# In[ ]:


#The Average of the Transfer Values Each Season
transfers.iplot(kind='bar',x='Season',y='Transfer_fee')


# In[ ]:


#Observing the Top 10 Leagues that Most Players were Transferred to. 
plt.figure(figsize=(15,7))
sns.countplot(x='League_to', data=transfers,order=transfers['League_to'].value_counts().head(10).index, palette='rainbow')


# In[ ]:


#Which Team had the Most Transfers Over the Period
plt.figure(figsize=(15,7))
sns.countplot(x='Team_to', data=transfers,order=transfers['Team_to'].value_counts().head(10).index, palette='rainbow')


# In[ ]:


#Which Teams Spent the Most Money on Transfers
newtf = transfers.groupby('Team_to')['Transfer_fee'].agg('sum').reset_index()

highestspent = newtf.sort_values(by='Transfer_fee', ascending=False).head(15)

print(highestspent)


# In[ ]:


highestspent.iplot(kind='bar',x='Team_to',y='Transfer_fee',title='Top 15 Teams With the Highest Spending',xTitle='Teams',yTitle='Money Spent')


# In[ ]:


#Teams Which made the Most Money By Selling their Players
mm = transfers.groupby('Team_from')['Transfer_fee'].agg('sum').reset_index()
mm2 = mm.sort_values(by='Transfer_fee', ascending=False).head(15)
print(mm2)


# In[ ]:


mm2.iplot(kind='bar',x='Team_from',y='Transfer_fee',title='Top 15 Teams that Made Money From Selling Players',xTitle='Teams',yTitle='Money Made')


# In[ ]:


agetransfer = transfers.groupby('Age')['Transfer_fee'].agg('mean').reset_index()


# In[ ]:


agetransfer.plot()


# In[ ]:


agetransfer.iplot(kind='line',x='Age',y='Transfer_fee')


# In[ ]:


#How the Transfer Fee Progessed as Season Passed By
df1 = transfers.groupby('Season')['Transfer_fee'].agg('sum').reset_index()
plt.figure(figsize=(20,10))
sns.lineplot(x='Season', y='Transfer_fee', data = df1)


# In[ ]:




