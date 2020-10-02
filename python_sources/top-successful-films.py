#!/usr/bin/env python
# coding: utf-8

# Let's rank some of the features in this database and determine an overall 'Score' that we can attribute to each film, giving us a best and worst list of superhero films.

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


df = pd.read_csv("/kaggle/input/marvel-vs-dc/db.csv", encoding = "ISO-8859-1", index_col=0)
df.head()


# The dataframe already has an index so lets include that and minus 1 from each value to reset our index to start at 0.

# In[ ]:


df.index -= 1
df.shape


# In[ ]:


df.head()


# Now to replace the spaces in the column names to underscores so to make it easier when referencing.
# Gross USA has an odd character in place of the space so we'll directly rename this one.

# In[ ]:


df.columns = [x.replace(" ", "_") for x in df.columns]
df = df.rename(columns={df.iloc[:,8].name:'Gross_USA'})
df.Budget = df.Budget.astype('int64')


# It'll be interesting to see how much a film makes in comparison to it's budget so we'll create a column to calculate this as a percent. We don't want to badly penalise a film for not making a lot if it only had a small budget.

# In[ ]:


df['Profit_in_%'] = (df['Gross_Worldwide']/df['Budget']*100).astype('int64')


# If a films comes top of a features list i.e. The Dark Knight has the highest 'Rate' of 9.0, then we award it a score of 0, and the film below it would get a score of 1 etc. This gives us a scoring for each film by adding these values for each feature together.
# 
# I've only included four features here. I could have used Opening_Weekend_USA and Gross_USA but this would give the profitability of the film too much weight as it already as Gross_Worldwide and it's profit in relation the it's budget.

# In[ ]:


top_features = ['Rate','Metascore','Gross_Worldwide','Profit_in_%']
films = {}
for i in top_features:
    for h, j in enumerate(df.sort_values(by=i, ascending=False).index):
        if j in films:
            films[j] += h
        else:
            films[j] = h

df['Score'] = df.index.map(films)


# In[ ]:


df.sort_values(by='Score').head(10)


# Avengers: Endgame is the clear winner with only a score of 7.

# In[ ]:


df.sort_values(by='Score', ascending=False).head(10)


# Catwoman, Jonah Hex, and Green Lantern coming last should make as no surprise.

# Now lets check the pandas documentation for a much easier way to rank rows, using the rank() function...

# In[ ]:


df['Score2'] = 0
for i in top_features:
    df['Score2'] += df[i].rank(ascending=False)


# In[ ]:


df['Score'] = df['Score'].rank()
df['Score2'] = df['Score2'].rank()


# In[ ]:


df.sort_values(by='Score2').head(10)


# In[ ]:


df.sort_values(by='Score2', ascending=False).head(10)


# In[ ]:




