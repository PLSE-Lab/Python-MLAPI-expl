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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/titanic/train.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


import seaborn as sns


# In[ ]:


sns.countplot('Sex', data = df)


# In[ ]:


def male_female_child(passenger):
    age,sex = passenger
    if age<16:
        return 'Child'
    else:
        return sex


# In[ ]:


df['Person'] = df[['Age','Sex']].apply(male_female_child,axis =1)


# In[ ]:


df.head()


# In[ ]:


sns.countplot('Person', data= df,hue='Pclass',palette = 'summer')


# In[ ]:


fig = sns.FacetGrid(df, hue = 'Sex', aspect=4)
fig.map(sns.kdeplot, 'Age', shade = 'True')
oldest = df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()


# In[ ]:


fig = sns.FacetGrid(df, hue = 'Person', aspect=4)
fig.map(sns.kdeplot, 'Age', shade = 'True')
oldest = df['Age'].max()
fig.set(xlim = (0,oldest))
fig.add_legend()


# In[ ]:


fig = sns.FacetGrid(df,hue='Pclass', aspect=4)
fig.map(sns.kdeplot, 'Age', shade = 'True')
oldest = df['Age'].max()
fig.set(xlim = (0,oldest))
fig.add_legend()


# In[ ]:


df.head()


# In[ ]:


dech = df['Cabin'].dropna()


# In[ ]:


from pandas import DataFrame


# In[ ]:


from pandas import DataFrame
levels = []
for level in dech:
    levels.append(level[0])
    
    cabin_df = DataFrame(levels)
    cabin_df.columns = ['Cabin']
    sns.countplot('Cabin', data = cabin_df, palette = 'winter')


# In[ ]:


cabin_df = cabin_df[cabin_df['Cabin'] != 'T' ]


# In[ ]:


sns.countplot('Cabin', data = cabin_df, order = ['A', 'B', 'C', 'D', 'E', 'F', 'G'])


# In[ ]:


df.head()


# In[ ]:


df['Alone'] = df.SibSp + df.Parch


# In[ ]:


df['Alone'] = df['Alone'].loc[]


# In[ ]:


df['Alone'] = pd.to_numeric(df['Alone'])


# In[ ]:


df['Alone'].loc[df['Alone']>0] = 'With Parets'
df['Alone'].loc[df['Alone']==0] = 'Alone'


# In[ ]:


df.head(10)


# In[ ]:


sns.countplot('Alone', data = df, palette = 'spring')


# In[ ]:


sns.boxplot(x='Fare', y= 'Sex', data =df, hue = 'Pclass')


# In[ ]:


sns.factorplot('Pclass', 'Survived', data = df, hue = 'Person')


# In[ ]:


sns.lmplot('Age', 'Survived', data = df, palette ='winter')


# In[ ]:


generation = [10,20,40,60,80]


# In[ ]:


sns.lmplot('Age', 'Survived', data = df,hue = 'Pclass', palette ='winter', x_bins = generation)


# In[ ]:


sns.lmplot('Age', 'Survived', data = df,hue = 'Sex', palette ='winter', x_bins = generation)

