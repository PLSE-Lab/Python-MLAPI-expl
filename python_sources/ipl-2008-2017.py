#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


ipl_data = pd.read_excel('../input/DIM_MATCH.xlsx')


# In[4]:


ipl_data.head()


# In[5]:


#info about the data
ipl_data.info()


# In[6]:


#To chechk whether any null values are present
ipl_data.isnull().sum()


# In[8]:


#Remove the rows with null values
ipl_data = ipl_data.dropna(axis=0,how='any')
ipl_data.isnull().sum()


# In[11]:


ipl_data.shape


# In[12]:


ipl_data.columns


# In[13]:


#Total number of matches played
ipl_data.Match_SK.count()


# In[14]:


#Team with maximium win-margin
ipl_data.loc[ipl_data['Win_Margin'].idxmax()]


# In[15]:


#To get the Team
ipl_data.loc[ipl_data['Win_Margin'].idxmax()]['match_winner']


# In[17]:


#Team with minimum win-margin
ipl_data.loc[ipl_data['Win_Margin'].idxmin()]['match_winner']


# *****Now let's do some vizualisations****

# ****

# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')


# In[22]:


#Teams won most number of matches in every ipl season
plt.figure(figsize=(20,10))
sns.countplot(x='Season_Year', data= ipl_data,hue='match_winner',saturation=1)
plt.legend(loc=1)
plt.show()


# In[24]:


#maximun number of matches won by team in ipl
sns.countplot(y='match_winner',data=ipl_data)
plt.show()


# In[25]:


plt.figure(figsize=(12,8))
sns.countplot(y='Venue_Name',data=ipl_data,palette='rainbow')
plt.tight_layout()


# In[26]:


# Highest Man of the matches in ipl
top_player = ipl_data.ManOfMach.value_counts()[:10]


# In[27]:


top_player


# In[28]:


sns.barplot(x=top_player,y=top_player.index)
plt.show()


# In[29]:


#Teams won the match with win margin
ipl_data[ipl_data['Win_Margin']>0].groupby(['match_winner'])['Win_Margin'].apply(np.median).sort_values(ascending=False)


# In[30]:


plt.figure(figsize=(12,10))
sns.boxplot(y='match_winner',x='Win_Margin',data=ipl_data[ipl_data['Win_Margin']>0],orient='h')
plt.show()


# In[ ]:




