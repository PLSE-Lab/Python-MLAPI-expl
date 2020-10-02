#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[ ]:


plt.rcParams["figure.figsize"] = (10,6) # define figure size of pyplot


# In[ ]:


pd.set_option("display.max_columns", 100) # set max columns when displaying pandas DataFrame
pd.set_option("display.max_rows", 200) # set max rows when displaying pandas DataFrame


# Dataset description:
# * Pclass: ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
# * Name: passengger's name
# * Sex: gender
# * Age: age in years
# * SibSp: # of siblings / spouse aboard the Titanic
# * Parch: # of parents / children aboard the Titanic
# * Ticket: ticket number 	
# * Fare: passenger fare 	
# * Cabin: cabin number 	
# * Embarked: port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

# In[ ]:


df = pd.read_csv('/kaggle/input/uisummerschool/Online_sales.csv')


# In[ ]:


df


# In[ ]:


Question: "How many ..."


# In[ ]:


df.describe() # statistical description of DataFrame columns, numerical only


# In[ ]:


df.info() # get DataFrame general info


# In[ ]:


df.head() # get first 5 records of DataFrame


# In[ ]:


df = df.iloc[:, 2:] # get columns from Pclass to Embarked


# In[ ]:


df.head()


# In[ ]:


cols = [x.lower() for x in df.columns]
df.columns = cols


# In[ ]:


df.head()


# In[ ]:


df[df['age'].isnull()].tail() # get last 5 records of DataFrame


# In[ ]:


df[df['cabin'].isnull()].head()


# In[ ]:


null_age = df[df['age'].isnull()].copy()


# In[ ]:


null_age.groupby('sex')['name'].count().reset_index(name='total_passengers')


# In[ ]:


grouped_null_age = null_age.groupby('sex')['name'].count().reset_index(name='total_passengers')


# In[ ]:


grouped_null_age.plot(kind='bar', x='sex');


# In[ ]:


df.plot(kind='box');


# In[ ]:


df['fare'].plot(kind='box');


# In[ ]:


df.hist();


# In[ ]:


df['age'].hist();


# In[ ]:


male_filler_age = df[df['sex']=='male']['age'].median()
female_filler_age = df[df['sex']=='female']['age'].median()


# **Numpy where function:**<br>
# > np.where({condition}, {if meets condition}, {if doesn't meet condition})

# In[ ]:


# using median of each gender to fill null values
df['age'] = np.where(df['age'].isnull(), np.where(df['sex'] == 'male', male_filler_age, female_filler_age), df['age'])


# In[ ]:


df[df['age'].isnull()] # there's no null values anymore


# In[ ]:


group_embarked = df.groupby('embarked')['name'].count().reset_index(name='total_passengers')


# In[ ]:


# see the distribution of passengers by place they embarked
group_embarked.plot(kind='bar', x='embarked');


# In[ ]:


grouped_age = df.groupby('age')['name'].count().reset_index(name='total_passengers')


# In[ ]:


# see the distribution of passengers by age (similar to histogram)
grouped_age.plot(kind='line', x='age');

