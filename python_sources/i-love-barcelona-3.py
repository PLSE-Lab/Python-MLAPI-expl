#!/usr/bin/env python
# coding: utf-8

# # I love Barcelona EDA
# I worked on one of the competitions and I didn't know how EDA is crucial and could help in feature engineering. I decided to craft my skill in making the data visual and where better to start other than a dataset about one of the most beautiful city in the world: Barcelona <3.
# 
# Data visualization libraries are full of charts, but which one to use? and how one could help more than the others? I hope this kernel will teach some techniques that helps in this area.
# 

# * Visualizing Amounts
#     * Population by district (Bar plots)
#     * Dot plots (Life expentency by age and district for males)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
sns.set(style="whitegrid")

import os
deaths_df = pd.read_csv('../input/deaths.csv')
population_df = pd.read_csv('../input/population.csv')
life_expentency_df = pd.read_csv('../input/life_expectancy.csv')
# Any results you write to the current directory are saved as output.


# ## Visualize Amounts
# 

# ### Population by district:
# Which district has **higher** population? Let's plot a bar chart showing population by district:

# In[ ]:


data = population_df.groupby(['District.Name']).sum().sort_values(by=['Number'],ascending=False)

f, ax = plt.subplots(1, 1, figsize=(16, 4))
sns.barplot(x=data.index, y=data.Number, palette="rocket", ax=ax)


# There are some issues with this chart and some enhancments can be made:
# * **Replace large numbers with percentages:** percentages are better when no exact numbers are requested. 
# * **Choose qualitative color palette:** the rocket color palette is sequential which gives oreder impression to the reader. But here each district has its own existence so let's change it to "deep". 
# * **Make the columns horizontal**: The plot above uses so much space because labels are too long. We could solve it by rotating labels, but they become hard to read. The best solution is horizontal bars. 

# In[ ]:


data = population_df.groupby(['District.Name']).sum().apply(lambda g: round(g / g.sum() * 100, 2)).sort_values(by=['Number'],ascending=False)
f, ax = plt.subplots(1, 1, figsize=(16, 6))
sns.barplot(x=data.Number, y=data.index, palette="deep", ax=ax)
ax.set(ylabel='District', xlabel='Population percentage %')


# #### Take away notes:
# * **Color as a tool to distinguish:**
#  Sometimes we use color to distingish between objects. In this case remember two things:
#     * Choose list colors that don't create order impression.
#     * Choose colors that make objects non-relative to each others. 
# this site helps [ColorBrewer](http://colorbrewer2.org/#type=sequential&scheme=BuGn&n=3) 

# ### Life expentency by age and district for males:
# 
# Bars are not the only option for visualizing amounts. One important limitation of bars is that they need to start at zero, so that the bar length is proportional to the amount shown. For some datasets, this can be impractical or may obscure key features.
# In this case, we can indicate amounts by placing dots at the appropriate locations
# along the x or y axis.
# 

# In[ ]:


data = life_expentency_df[life_expentency_df['Gender']  == 'Male'][['2009-2013', 'Neighborhood']].    sort_values(by='2009-2013', ascending=False)[:25]
f, ax = plt.subplots(1, 1, figsize=(8, 10))
ax = sns.stripplot(x='2009-2013', y="Neighborhood", data=data, jitter=True, ax=ax, color='green', size=7)


# This dataset is not suitable for being visualized with bars. The bars are too long
# and they draw attention away from the key feature of the data, the differences in life
# expectancy among the different districts.

# In[ ]:


f, ax = plt.subplots(1, 1, figsize=(8, 10))
sns.barplot(x='2009-2013', y="Neighborhood", data=data, ax=ax, color='green')

