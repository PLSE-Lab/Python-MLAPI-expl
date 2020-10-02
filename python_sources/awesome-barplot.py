#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt 
import seaborn as sns 

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except:
    print('Inline Failed !!!')

sns.set()
plt.style.use('ggplot') 
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False 
plt.rcParams['figure.figsize'] = (10.0, 5.0)


# # awesome barplot in python

# ## 1. read fashion titianic data

# In[ ]:


df = pd.read_csv('../input/train.csv')
df.head()


# ### create some count data for plot

# In[ ]:


# embarked counts
embarked_df = df.groupby('Embarked').size().to_frame('cnt').reset_index()
embarked_df.head()


# In[ ]:


# counts groupby embarked and sex level
embarked_sex_survived_df = df.groupby(['Embarked','Sex','Survived']).size().to_frame('cnt').reset_index()
embarked_sex_survived_df.head()


# In[ ]:


# counts groupby embarked and sex level
embarked_sex_df = df.groupby(['Embarked','Sex']).size().to_frame('cnt').reset_index()
embarked_sex_df.head()


# ## head first barplot

# ### `plt.bar` basic function for barplot

# In[ ]:


plt.bar(embarked_df['Embarked'], embarked_df['cnt'])
plt.show();


# In[ ]:


plt.barh(embarked_df['Embarked'], embarked_df['cnt']);


# ### `sns.barplot` use `seaborn` draw a bar plot

# In[ ]:


sns.barplot(x='Embarked',y='cnt',data=embarked_df);


# In[ ]:


sns.barplot(y='Embarked',x='cnt',data=embarked_df);


# ### Add color for `plt.bar` via `color` parameter

# In[ ]:


# add color by name
plt.bar(embarked_df['Embarked'], embarked_df['cnt'], color = ('red','yellow','blue'))
plt.show();


# In[ ]:


# add color use sns.color_palette()
from matplotlib.colors import ListedColormap
plt.bar(embarked_df['Embarked'], embarked_df['cnt'],color=sns.color_palette())
plt.show();


# In[ ]:


sns.barplot(x='Embarked',y='cnt',data=embarked_df, palette="Blues_d");


# ## two level dimension barplot

# ### hue plot

# In[ ]:


sns.barplot(x='Embarked',y='cnt',hue='Sex',data=embarked_sex_df);


# ### stack bar plot

# In[ ]:


embarked_sex_df.head()


# In[ ]:


tmp = embarked_sex_df.set_index(['Sex','Embarked']).unstack()
tmp.columns = tmp.columns.levels[1]
tmp.head()


# In[ ]:


tmp.plot(kind='bar',stacked=True);


# In[ ]:


tmp = embarked_sex_df.set_index(['Sex','Embarked']).groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack()
tmp.columns = tmp.columns.levels[1]
tmp.head()


# In[ ]:


tmp.plot(kind='bar',stacked=True);


# ### grid plot for three dimension

# In[ ]:


g = sns.catplot(x="Embarked", y="cnt",hue="Sex", col="Survived",
                data=embarked_sex_survived_df, kind="bar",
                height=6, aspect=.7);

