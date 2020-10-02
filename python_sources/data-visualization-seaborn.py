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


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


x = np.linspace(0,5,11)
y = x**2


# In[ ]:


x


# In[ ]:


y


# In[ ]:


plt.plot(x,y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Graph')


# In[ ]:


plt.subplot(1,2,1)
plt.plot(x,y,'r')
plt.subplot(1,2,2)
plt.plot(y,x,'b')


# In[ ]:


fig = plt.figure()
axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.plot(x,y)
axes.set_xlabel('X Label')
axes.set_ylabel('Y Label')
axes.set_title('Title')


# In[ ]:


fig = plt.figure()
axes1 = fig.add_axes([0.1,0.1,0.9,0.9])
axes2 = fig.add_axes([0.2,0.5,0.4,0.3])


# In[ ]:


fig = plt.figure()
axes1 = fig.add_axes([0.1,0.1,0.9,0.9])
axes2 = fig.add_axes([0.2,0.5,0.4,0.3])

axes1.plot(x,y)
axes1.set_title('Bigger Plot')
axes2.plot(y,x)
axes2.set_title('Smaller Plot')


# In[ ]:


fig,axes = plt.subplots()
axes.plot(x,y)

for current_ax in axes:
    current_ax.plot(x,y)


# In[ ]:





# In[ ]:


fig,axes = plt.subplots(nrows=3,ncols=3)
plt.tight_layout()


# In[ ]:


fig,axes = plt.subplots(nrows=1,ncols=2)
axes[0].plot(x,y)
axes[1].plot(y,x)


# In[ ]:


#FIGURE SIZE AND DPI


# In[ ]:


fig = plt.figure(figsize=(8,2))
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y)


# In[ ]:


fig,axes = plt.subplots(nrows=2, ncols=1, figsize=(8,2))
axes[0].plot(x,y)
axes[1].plot(x,y)
plt.tight_layout()


# In[ ]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

ax.plot(x,x**2, label = 'X Squared')
ax.plot(x,x**3, label = 'X Cubed')
ax.legend(loc = 0)
#loc=0 best location


# In[ ]:


fig = plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.plot(x,y,color = '#FF8C00',linewidth=1,linestyle='-',
        marker='o',markersize=15, markerfacecolor = 'yellow',
       markeredgewidth=3, markeredgecolor='green')


# In[ ]:


plt.scatter(x,y)


# In[ ]:


import seaborn as sns
import numpy as np 
import pandas as pd


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


tips = pd.read_csv('../input/seaborn-tips-dataset/tips.csv')


# In[ ]:


tips.head()


# In[ ]:


sns.distplot(tips['total_bill'],kde=True,bins=50)


# In[ ]:


sns.jointplot(x='total_bill',y='tip',data=tips, kind='hex')


# In[ ]:


sns.jointplot(x='total_bill',y='tip',data=tips, kind='reg')


# In[ ]:


sns.jointplot(x='total_bill',y='tip',data=tips, kind='kde')


# In[ ]:


sns.jointplot(x='total_bill',y='tip',data=tips)


# In[ ]:


sns.pairplot(tips)


# In[ ]:


sns.pairplot(tips, hue='sex')


# In[ ]:


sns.pairplot(tips, hue='sex',palette='coolwarm')


# In[ ]:


sns.rugplot(tips['total_bill'])


# In[ ]:


sns.kdeplot(tips['total_bill'])


# In[ ]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


tips = pd.read_csv('../input/seaborn-tips-dataset/tips.csv')


# In[ ]:


tips.head()


# In[ ]:


sns.barplot(x='sex', y='total_bill' , data=tips)


# In[ ]:


import numpy as np


# In[ ]:


sns.barplot(x='sex', y='total_bill',data=tips, estimator=np.std)


# In[ ]:


sns.barplot(x='tip', y='total_bill' , data=tips)


# In[ ]:


sns.countplot(x='sex', data=tips)


# In[ ]:


sns.boxplot(x='day',y='total_bill',data=tips)


# In[ ]:


sns.boxplot(x='day',y='total_bill',data=tips, hue='smoker')


# In[ ]:


sns.violinplot(x='day', y='total_bill', data=tips)


# In[ ]:


sns.violinplot(x='day', y='total_bill', data=tips, hue='sex')


# In[ ]:


sns.stripplot(x='day',y='total_bill',data=tips)


# In[ ]:


sns.stripplot(x='day',y='total_bill',data=tips,jitter=False)


# In[ ]:


sns.stripplot(x='day',y='total_bill',data=tips, hue='sex', split=True)


# In[ ]:


sns.stripplot(x='day',y='total_bill',data=tips, hue='sex', split=True, jitter=False)


# In[ ]:


sns.swarmplot(x='day',y='total_bill',data=tips)


# In[ ]:


sns.violinplot(x='day',y='total_bill',data=tips)


# In[ ]:


sns.violinplot(x='day',y='total_bill',data=tips)
sns.swarmplot(x='day',y='total_bill',data=tips, color='black')


# In[ ]:


sns.factorplot(x='day',y='total_bill',data=tips, kind='bar')


# In[ ]:


flights = pd.read_csv('../input/wwwkagglecomtanyildizderyaflightscsv/flights.csv')


# In[ ]:


flights.head()


# In[ ]:


tips.corr()


# In[ ]:


tc = tips.corr()


# In[ ]:


sns.heatmap(tc, annot=True)


# In[ ]:


flights.pivot_table(index='month',columns='year',values='passengers')


# In[ ]:


fp = flights.pivot_table(index='month',columns='year',values='passengers')


# In[ ]:


sns.heatmap(fp, cmap='Blues')


# In[ ]:


sns.heatmap(fp, cmap='Blues_r', linecolor='white', linewidths=1)


# In[ ]:


sns.clustermap(fp)


# In[ ]:


flights.head()


# In[ ]:


flights['year'].unique()


# In[ ]:


sns.pairplot(flights)


# In[ ]:


tips.head()


# In[ ]:


sns.pairplot(tips)


# In[ ]:


g = sns.PairGrid(tips)
g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)


# In[ ]:


g = sns.FacetGrid(data=tips,col='time',row='smoker')
g.map(sns.distplot,'total_bill')


# In[ ]:


g = sns.FacetGrid(data=tips,col='time',row='smoker')
g.map(plt.scatter,'total_bill','tip')


# In[ ]:


sns.lmplot(x='total_bill',y='tip', data=tips,hue='sex',
           markers=['o','v'],scatter_kws={'s':100})


# In[ ]:


sns.lmplot(x='total_bill',y='tip', data=tips)


# In[ ]:


sns.lmplot(x='total_bill',y='tip', data=tips,col='sex',row='time')


# In[ ]:


sns.lmplot(x='total_bill',y='tip', data=tips,col='day', hue='sex')


# In[ ]:


sns.lmplot(x='total_bill',y='tip', data=tips,col='day', hue='sex',
          aspect=0.6,size=8)


# In[ ]:


sns.set_style('darkgrid')
sns.countplot(x='sex',data=tips)
sns.despine(left=True,bottom=True)


# In[ ]:


tips.head()


# In[ ]:


plt.figure(figsize=(12,3))
sns.countplot(x='day',data=tips)


# In[ ]:


sns.set_context('poster')
plt.figure(figsize=(12,3))
sns.countplot(x='day',data=tips)


# In[ ]:


sns.set_context('paper')
sns.set_style('whitegrid')
sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',palette='seismic')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


("")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




