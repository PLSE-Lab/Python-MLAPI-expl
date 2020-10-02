#!/usr/bin/env python
# coding: utf-8

# ## One Hour Challenge
# ### Using Seaborn to Perform EDA of Pokemon Stats
# 
# I typically use R with ggplot to run my EDA and thought it would be nice to get some experience in python. With this
# in mind, I pulled the Pokemon stats data available here to get some practice through a one hour project. Goals are to
# perform univariate, bivariate, and multivariate visual exploration of of the data set to mine for whatever trend might 
# be apparent. I've found open-ended and short time-framed projects like this really help focus on one or two specific
# skills or tools. So  here we go.. 
# 
# My comments will be missing for the most part as I ran through these in a stream-of-conscious type style without concern for
# documentation. Just looking to put in the hour to get a better feel of using Seaborn. <(^.^)>

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 

########################
# contacts epxloration #
########################

# import
df = pd.read_csv('../input/Pokemon.csv', encoding = "ISO-8859-1")

# understand df
df.shape
df.describe()


# In[2]:


df.head()


# In[3]:


df.info()
df.columns


# In[4]:


# drop id row
df = df.drop('#', axis = 1)
df = df.drop(['Sp. Atk', 'Sp. Def'], axis = 1)
# check for missing
df.isnull().sum()
             


# In[6]:


# create feature Is two Types
def set_dual_type(row):
    if type(row['Type 2']) == str:
        return 1
    else:
        return 0
    
df['dual_type'] = df.apply(lambda row: bool(set_dual_type(row)), axis=1)       
        


# In[7]:


# Univariate Plots
fig, axes = plt.subplots(nrows=2, ncols=2)
sns.distplot(df['HP'], norm_hist=False, kde=True, bins=30, ax=axes[0,0]).set(xlabel='HP', ylabel='Count')
sns.distplot(df['Attack'], norm_hist=False, kde=True, bins=30, ax=axes[0,1]).set(xlabel='Atk', ylabel='Count')
sns.distplot(df['Defense'], norm_hist=False, kde=True, bins=30, ax=axes[1,0]).set(xlabel='Dfn', ylabel='Count')
sns.distplot(df['Speed'], norm_hist=False, kde=True, bins=30, ax=axes[1,1]).set(xlabel='Spd', ylabel='Count')


# In[8]:


fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
sns.countplot(df['Type 1'], ax=axes[0])
sns.countplot(df['Type 2'], ax = axes[1])


# In[9]:



sns.countplot(df['Generation'])


# In[10]:



#bivariate plots
sns.jointplot(x='Attack', y='Defense', data=df)


# In[11]:


sns.jointplot(x='HP', y='Defense', data=df, kind = 'hex')


# In[12]:


sns.jointplot(x='Speed', y='Attack', data=df, kind = 'kde')


# In[13]:



# facets
g = sns.FacetGrid(df, row="Generation", col="dual_type", margin_titles=True)
g.map(sns.regplot, "Attack", "Defense", color=".3", fit_reg=False, x_jitter=.1);


# In[14]:



#pair plot
df_subset = df.drop(['Name', 'Type 1', 'Type 2', 'Total', 'Generation', 'Legendary', 'dual_type'], axis = 1)
pp = sns.PairGrid(df_subset)
pp.map_diag(sns.kdeplot)
pp.map_offdiag(sns.kdeplot, n_levels = 6)


# In[15]:



# categorical bivar
sns.catplot(x = 'Generation', y = 'Total', kind = 'swarm', data = df)


# In[16]:



fig, axes = plt.subplots(nrows=2, ncols=1)
sns.catplot(x="Type 1", y = "Total", kind = 'swarm', data = df, ax = axes[0])
sns.catplot(x="Type 1", y="Total", kind="swarm",
            data=df.query("Generation == 1"), ax=axes[1]);


# In[17]:



      
# Multivariate plots
sns.catplot(x = "Generation", y = "Total", hue = "dual_type", kind = 'violin', split = True, data = df)


# In[18]:



sns.relplot(x="Attack", y="Defense", hue="Generation", data=df);


# In[19]:



sns.relplot(x="HP", y="Defense", hue="dual_type", style="Generation", data=df)


# In[20]:


sns.relplot(x="Attack", y="Speed", hue="dual_type", style="Generation", data=df)


# In[21]:



query_types = ['Grass', 'Fire', 'Water']
sns.relplot(x="Attack", y="Defense", hue="Speed", style = "Type 1", data=df[df['Type 1'].isin(query_types)])


# In[22]:



sns.relplot(x="Attack", y="Total", kind="line", ci="sd", data=df)


# In[23]:



sns.relplot(x="Attack", y="Total", kind="line", hue = "Type 1", ci="sd", data=df[df['Type 1'].isin(query_types)])

