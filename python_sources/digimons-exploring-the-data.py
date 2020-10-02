#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df_train = pd.read_csv('../input/DigiDB_digimonlist.csv')
# pd.read_csv('../input/DigiDB_movelist.csv')
# pd.read_csv('../input/DigiDB_supportlist.csv')


# In[3]:


# WE will dislpay the number of rows and colum
df_train.shape


# In[4]:


df_train.describe()


# In[5]:


df_train.head()


# In[6]:


# Displaying the columns in our dataset
df_train.columns


# In[7]:


df_train.info()
# This command gives basic information about each column in dataset


# In[8]:


f, ax = plt.subplots(figsize=(15,15))
sns.heatmap(df_train.corr(),annot=True, linewidths=.1, fmt= '.1f',ax=ax, cmap="YlGnBu")


# We see that, there are no null values in a row. If there were any missing values, than we would need to choose a strategy for dealing with them :)

# In[9]:


# let's look for the maximum HP
df_train['Lv 50 HP'].max()


# ## Visual exploratory data analysis

# Box plots: visualize basic statistics like outliers, min/max or quantiles
# * I want to compare level of digimons and their attack.
# 

# In[10]:


df_train.boxplot(column='Lv 50 HP', by="Lv50 Atk")


# In[11]:


data = df_train.loc[:,['Lv 50 HP','Lv50 Atk', 'Lv50 Def']]
data.plot()


# In[12]:


# subplots
data.plot(subplots=True)


# In[13]:


# scatter plot
data.plot(kind = 'scatter', x='Lv50 Atk', y='Lv50 Def')


# In[14]:


# hist plot
data.plot(kind='hist', y='Lv50 Def', bins = 50, range=(0,300), normed=True)


# In[15]:


df_train['Type'].unique()


# In[16]:


# Now i want to know, what type of digimons have the biggest attack ?
pd.crosstab(df_train['Type'], df_train['Lv50 Atk'])


# In[17]:


df_train[df_train['Type'] == 'Data']['Lv50 Atk'].max()


# In[18]:


df_train[df_train['Type'] == 'Free']['Lv50 Atk'].max()


# In[19]:


df_train[df_train['Type'] == 'Vaccine']['Lv50 Atk'].max()


# In[20]:


df_train[df_train['Type'] == 'Virus']['Lv50 Atk'].max()


# The next step, i will plot the type of digimons and theit amount.

# In[22]:


digimons_movelist = pd.read_csv("../input/DigiDB_digimonlist.csv")


# In[24]:


digimons_movelist['Type'].value_counts().plot(kind='bar')
plt.title('Digimonsters type')
plt.ylabel('Count')
plt.show()

