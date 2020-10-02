#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # beauty plts
import matplotlib.pyplot as plt #plts


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/2017.csv')
df.head()


# In[ ]:


df.columns = ['Country','HRank','HScore','Whisker_High','Whisker_low','GDP_Capita','Family','LifeExp','Freedom','Generosity','GovTrust','Dystopia']
df.head()


# In[ ]:


sns.relplot(x="GDP_Capita", y="HScore", data=df);


# **Money makes Happiness?**
# 
# On this graph, we can clearly see that money leads to happiness... for a while, it interestingly seems like once you get to a certain level (GDP/capita = 0.60), money is not so correlated to hapiness anymore...
# 

# In[ ]:


df_filtered = df[(df['GDP_Capita'] > 0.60)]


# In[ ]:


sns.relplot(x="GDP_Capita", y="HScore", data=df_filtered);


# We can see that, even if money still kinda makes happiness, correlation between the two of them is not as pronounced as before...

# In[ ]:


df_filtered2 = df[(df['GDP_Capita'] > 1.0 )]
df_filtered2 = df_filtered2[(df_filtered['GDP_Capita'] < 1.2 )]


# In[ ]:


sns.relplot(x="GDP_Capita", y="HScore", data=df_filtered2);


# For GDP between 1.0 and 1.2, we clearly cannot find any correlation here, let's see which countries are in this !

# In[ ]:


df_filtered2['Country']


# In[ ]:


df_filtered2.sort_values(by=['HScore'], ascending=False)


# It apperas that Costa Rica is another level, let's track Costa Rica in various metrics and see why it is up there

# In[ ]:



sns.relplot(x="Family", y="HScore", data=df_filtered2)
plt.scatter(df_filtered2.GDP_Capita.iloc[0], df_filtered2.HScore.iloc[0], color='r')


# In[ ]:


sns.relplot(x="LifeExp", y="HScore", data=df_filtered2)
plt.scatter(df_filtered2.LifeExp.iloc[0], df_filtered2.HScore.iloc[0], color='r')


# In[ ]:


sns.relplot(x="Freedom", y="HScore", data=df_filtered2)
plt.scatter(df_filtered2.Freedom.iloc[0], df_filtered2.HScore.iloc[0], color='r')


# In[ ]:


sns.relplot(x="Generosity", y="HScore", data=df_filtered2)
plt.scatter(df_filtered2.Generosity.iloc[0], df_filtered2.HScore.iloc[0], color='r')


# In[ ]:


sns.relplot(x="GovTrust", y="HScore", data=df_filtered2)
plt.scatter(df_filtered2.GovTrust.iloc[0], df_filtered2.HScore.iloc[0], color='r')


# Finnaly, two metrics seem to explain the hapiness of Costa Rica, Life Expency and Freedom, but it doesn't explain it very well for other countries ! 

# In[ ]:


sns.relplot(x="LifeExp", y="HScore", data=df);
plt.scatter(df_filtered2.LifeExp.iloc[0], df_filtered2.HScore.iloc[0], color='r')


# In[ ]:


sns.relplot(x="Freedom", y="HScore", data=df);
plt.scatter(df_filtered2.Freedom.iloc[0], df_filtered2.HScore.iloc[0], color='r')


# **What can we say about richer countries ? **

# In[ ]:


df_filtered3 = df[(df['GDP_Capita'] > 1.4 )]


# In[ ]:


sns.relplot(x="GDP_Capita", y="HScore", data=df_filtered3);


# Here also, it seems like money doesn't make the people happy, actually the most happy countries are the poorest, related to this category

# In[ ]:


df_filtered3['Country']


# In[ ]:


df_filtered3.sort_values(by=['HScore'], ascending=False)


# **Norway** seem to be the happiest of all, once again, lets track their happiness levels compared with other metrics

# In[ ]:


sns.relplot(x="Family", y="HScore", data=df_filtered3)
plt.scatter(df_filtered3.GDP_Capita.iloc[0], df_filtered3.HScore.iloc[0], color='r')


# In[ ]:


sns.relplot(x="LifeExp", y="HScore", data=df_filtered3)
plt.scatter(df_filtered3.LifeExp.iloc[0], df_filtered3.HScore.iloc[0], color='r')


# In[ ]:


sns.relplot(x="Freedom", y="HScore", data=df_filtered3)
plt.scatter(df_filtered3.Freedom.iloc[0], df_filtered3.HScore.iloc[0], color='r')


# In[ ]:


sns.relplot(x="Generosity", y="HScore", data=df_filtered3)
plt.scatter(df_filtered3.Generosity.iloc[0], df_filtered3.HScore.iloc[0], color='r')


# In[ ]:


sns.relplot(x="GovTrust", y="HScore", data=df_filtered3)
plt.scatter(df_filtered3.GovTrust.iloc[0], df_filtered3.HScore.iloc[0], color='r')


# **It feels like** family is the only deciding factor for the richer country, while having Norway leeding the charge as "most happy place on Earth"

# In[ ]:


sns.relplot(x="Family", y="HScore", data=df);
plt.scatter(df_filtered3.Family.iloc[0], df_filtered3.HScore.iloc[0], color='r')


# **How does things matter now for the poorest countries?**

# In[ ]:


df_filtered4 = df[(df['GDP_Capita'] < 0.50)]


# In[ ]:


sns.relplot(x="GDP_Capita", y="HScore", data=df_filtered4);


# **In terms of money** once again, it's not very promising ... 

# In[ ]:


df_filtered4['Country']


# In[ ]:


df_filtered4.sort_values(by=['HScore'], ascending=False)


# **In the poorest countries, Somalia and Nepal seem out of their leagues**

# In[ ]:


sns.relplot(x="Family", y="HScore", data=df_filtered4)
plt.scatter(df_filtered4.GDP_Capita.iloc[0], df_filtered4.HScore.iloc[0], color='r')
plt.scatter(df_filtered4.GDP_Capita.iloc[1], df_filtered4.HScore.iloc[1], color='g')


# **Clearly** family is not their reciepe for success (Nepal in Green, Somalia in Red)

# In[ ]:


sns.relplot(x="LifeExp", y="HScore", data=df_filtered4)
plt.scatter(df_filtered4.LifeExp.iloc[0], df_filtered4.HScore.iloc[0], color='r')
plt.scatter(df_filtered4.LifeExp.iloc[1], df_filtered4.HScore.iloc[1], color='g')


# **Life Expectancy** is making Nepal a happier place, but not Somalia, it's not indicative for other countries either

# In[ ]:


sns.relplot(x="Freedom", y="HScore", data=df_filtered4)
plt.scatter(df_filtered4.Freedom.iloc[0], df_filtered4.HScore.iloc[0], color='r')
plt.scatter(df_filtered4.Freedom.iloc[1], df_filtered4.HScore.iloc[1], color='g')


# **Freedom** plays a much bigger role for these countries

# In[ ]:


sns.relplot(x="Generosity", y="HScore", data=df_filtered4)
plt.scatter(df_filtered4.Generosity.iloc[0], df_filtered4.HScore.iloc[0], color='r')
plt.scatter(df_filtered4.Generosity.iloc[1], df_filtered4.HScore.iloc[1], color='g')


# But **Generosity** doesn't account for their success.

# In[ ]:


sns.relplot(x="GovTrust", y="HScore", data=df_filtered4)
plt.scatter(df_filtered4.GovTrust.iloc[0], df_filtered4.HScore.iloc[0], color='r')
plt.scatter(df_filtered4.GovTrust.iloc[1], df_filtered4.HScore.iloc[1], color='g')


# Governement trust is not really Helpful either !

# In[ ]:


sns.relplot(x="Freedom", y="HScore", data=df);
plt.scatter(df_filtered2.Freedom.iloc[0], df_filtered2.HScore.iloc[0], color='r')
plt.scatter(df_filtered3.Freedom.iloc[0], df_filtered3.HScore.iloc[0], color='y')
plt.scatter(df_filtered4.Freedom.iloc[0], df_filtered4.HScore.iloc[0], color='g')


# **Freedom Scores** Yellow : Norway, Red : Costa Rica, Green : Somalia

# In[ ]:


sns.relplot(x="GDP_Capita", y="HScore", data=df);
plt.scatter(df_filtered2.GDP_Capita.iloc[0], df_filtered2.HScore.iloc[0], color='r')
plt.scatter(df_filtered3.GDP_Capita.iloc[0], df_filtered3.HScore.iloc[0], color='y')
plt.scatter(df_filtered4.GDP_Capita.iloc[0], df_filtered4.HScore.iloc[0], color='g')


# **After All** Money does buy happiness, but some countries use other ways to propel their happiness forward.

# In[ ]:


sns.relplot(x="GDP_Capita", y="Generosity", data=df);


# **Bonus** The one who give are not always the richest

# In[ ]:


sns.relplot(x="Generosity", y="HScore", data=df);


# **But also not the most happy**

# In[ ]:




