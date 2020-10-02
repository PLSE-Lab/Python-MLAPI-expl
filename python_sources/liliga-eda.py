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


df1=pd.read_csv('../input/Laliga1.csv')


# In[ ]:


df1.head()


# Filling - values in dataset with zero
# 

# In[ ]:


dfn=df1.replace('-',0)
dfn.tail()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sns.countplot('BestPosition',data=dfn)
plt.legend()


# Graphs Shows the No.of times Each Position has been reached .

# In[ ]:


dfn
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_column',1000)


# The Team which has Debut b/w 1930 & 1980

# In[ ]:


dfn['Debuttem']=dfn['Debut'].apply(lambda x: str(x).split('-'))


# In[ ]:


def format_year(x):
    if len(x)>1:
        if int(x[0])<1999:
            return [x[0],int(x[1])+1900]
        else:
            return [x[0],int(x[1])+2000]
    else:
        return x


# In[ ]:


dfn['Debut_temp']=dfn['Debuttem'].apply(format_year)


# In[ ]:


dfn.head()


# In[ ]:


def check_debut(x):
    if len(x)==1:
        return int(x[0])>=1930 and int(x[0])<=1980
    else:
        return ((int(x[0])>=1930 and int(x[0])<=1980)or(int(x[1])>=1930 and int(x[1])<=1980))


# In[ ]:


a=dfn[dfn['Debut_temp'].apply(check_debut)]['Team']


# In[ ]:


print(a)
print('No.of teams that debut b/w 1930-1980 is',a.count())


# Before Continuing Further Lets make required columns into integer for further analysis

# In[ ]:


cols=['Pos', 'Seasons', 'Points', 'GamesPlayed', 'GamesWon',
       'GamesDrawn', 'GamesLost', 'GoalsFor', 'GoalsAgainst', 'Champion',
       'Runner-up', 'Third', 'Fourth', 'Fifth', 'Sixth', 'T','BestPosition']


# In[ ]:


dfn[cols]=dfn[cols].apply(pd.to_numeric)


# In[ ]:


dfn.info()


# Top Five Teams in points table

# In[ ]:


dfn['rank']=dfn['Points'].rank(method='dense',ascending=False)


# In[ ]:


dfn[['Team','rank']].head()


# Calculate Winning Percentage

# In[ ]:


dfn['Winning %']=(dfn['GamesWon']/dfn['GamesPlayed'])*100


# In[ ]:


dfn[['Team','Winning %']].head()


# Top Five Team Winning Percentage

# In[ ]:


dfn.info()


# In[ ]:


dfn['Winning %']=dfn['Winning %'].fillna(0)


# In[ ]:


sns.scatterplot('BestPosition','Winning %',data=dfn)


# Best Postion Team Have More winning Percentage

# In[ ]:


dfn['Goaldif']=(dfn['GoalsFor']-dfn['GoalsAgainst'])


# In[ ]:


dfn[['Team','Goaldif']].head()


# In[ ]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from statsmodels.stats.anova import anova_lm


# In[ ]:


z=smf.ols(formula='BestPosition ~ Goaldif+GamesPlayed',data=dfn).fit()


# In[ ]:


z.summary()


# From This we can conclude Goal Difference and Game Played can can explain why team is top position and also has probability of winning a cup.
