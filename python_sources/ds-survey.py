#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
import itertools

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


schema = pd.read_csv('../input/SurveySchema.csv')

display(schema)


# In[ ]:


df1 = pd.read_csv('../input/freeFormResponses.csv')

df1.head()


# In[ ]:


df2 = pd.read_csv('../input/multipleChoiceResponses.csv', skiprows=[1])

df2.head()


# In[ ]:


df = df2.copy()


# In[ ]:


df.dtypes.value_counts()


# In[ ]:


fig = plt.figure(figsize=(8,10))

plt.subplot(211)
sns.heatmap(df.select_dtypes(['object']).isnull().T, cbar=False)
plt.title('Heatmap of missing text responses')

plt.subplot(212)
sns.heatmap(df.select_dtypes(['number']).isnull().T, cbar=False)
plt.title('Heatmap of missing numeric responses')

plt.tight_layout()
plt.show()


# # Text responses appear to have many more missing responses than numeric responses

# In[ ]:


plt.subplots(1,1, figsize=(12,4))

sns.countplot(x='Q2', data=df, palette='winter', dodge=True)
plt.title('Kaggle Survey Responses by Age Groups')
plt.xlabel('')
sns.set(style="darkgrid")

plt.legend(loc=4)
plt.show()


# # Age group distributions between 18 and 50 appear normal with outliers over 50 years old.

# In[ ]:


plt.subplots(1,1, figsize=(12,4))
sns.set_style('darkgrid')
colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
sns.countplot(x='Q1', data=df, facecolor=[0,0,0,0],
              linewidth=10,
              edgecolor=sns.xkcd_palette(colors)
             )
plt.title('Kaggle Survey Responses by Reported Gender')
plt.xlabel('')

plt.legend(loc=4)
plt.show()


# In[ ]:


cols = pd.Series(df.columns)
cols = cols[~cols.str.contains('_')]

plt.style.use('ggplot')
palette = itertools.cycle(sns.color_palette())
for col in df[cols.loc[4:]].columns:
    plt.suptitle('2018 Kaggle Survey Responses')
    w=4;l=10;
    y=col;x=None
    if len(df[col].unique()) < 6:
        w=10;l=4
        x=col;y=None
    ax, fig = plt.subplots(1,1, figsize=(w,l))
    
    sns.countplot(y=y, x=x, data=df)
    plt.title(schema.loc[0][col])

plt.show()


# In[ ]:




