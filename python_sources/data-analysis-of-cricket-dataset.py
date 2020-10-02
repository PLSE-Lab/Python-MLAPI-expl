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
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns




# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv('../input/Teams.csv')


# In[ ]:


data


# In[ ]:


data.info


# In[ ]:


data.sort_values(ascending='Rank',axis=0,inplace=True,by='Rank')


# In[ ]:


data


# In[ ]:


data.sort_values(ascending='Year',axis=0,inplace=True,by='Year')
data


# In[ ]:


data.plot.bar(x='Teams',y='Rank')


# In[ ]:


data.plot.barh(x='Teams',y='Rank')
plt.xlabel('Rank')
plt.title('teams vs rank')


# In[ ]:


data.plot.barh(x='Teams',y='Points')
plt.xlabel('Points')
plt.title('Teams vs points')


# In[ ]:


data_2014=data[data['Year']==2014]
data_2014
data_2014.plot.barh(x='Teams',y='Rank')
plt.xlabel('Rank')
plt.title('2014')


# In[ ]:





# In[ ]:


data_2015=data[data['Year']==2015]
data_2015
data_2015.plot.barh(x='Teams',y='Rank')
plt.xlabel('Rank')
plt.title('2015')


# In[ ]:





# In[ ]:


data_2016=data[data['Year']==2016]
data_2016
data_2016.plot.barh(x='Teams',y='Rank')
plt.xlabel('Rank')
plt.title('2016')


# In[ ]:


data_2017=data[data['Year']==2017]
data_2017
data_2017.plot.barh(x='Teams',y='Rank')
plt.xlabel('Rank')
plt.title('2017')


# In[ ]:


data_2014=data[data['Year']==2014]
data_2014
data_2014.plot.bar(x='Teams',y='Points')
plt.ylabel('Points')
plt.title('2014')


# In[ ]:


data_riders=data[data['Teams']=='Riders']
data_riders
data_riders.plot.bar(x='Year',y='Rank')
plt.xlabel('Year')
plt.ylabel('Rank')
plt.title('riders')


# In[ ]:


data_royals=data[data['Teams']=='Royals']
data_royals
data_royals.plot.bar(x='Year',y='Rank')
plt.xlabel('Year')
plt.ylabel('Rank')
plt.title('Royals')


# In[ ]:


data_kings=data[data['Teams']=='Kings']
data_kings
data_kings.plot.bar(x='Year',y='Rank')
plt.xlabel('Year')
plt.ylabel('Rank')
plt.title('Kings')


# In[ ]:


data_devils=data[data['Teams']=='Devils']
data_devils
data_devils.plot.bar(x='Year',y='Rank')
plt.xlabel('Year')
plt.ylabel('Rank')

plt.title('devils')


# In[ ]:


data


# In[ ]:


sns.pairplot(data,hue='Teams')



# In[ ]:


sns.distplot(data['Points'])


# In[ ]:


sns.pairplot(data_2014,hue='Teams')


# In[ ]:


sns.pairplot(data_riders,hue='Teams')


# In[ ]:


sns.pairplot(data_kings,hue='Teams')


# In[ ]:


sns.pairplot(data_devils,hue='Teams')


# In[ ]:



sns.pairplot(data_royals,hue='Teams')


# In[ ]:


sns.pairplot(data_2014,hue='Teams')


# In[ ]:



sns.pairplot(data_2015,hue='Teams')



# In[ ]:



sns.pairplot(data_2016,hue='Teams')



# In[ ]:



sns.pairplot(data_2017,hue='Teams')


# In[ ]:


sns.heatmap(data.corr(),cmap='coolwarm',annot=True)


# In[ ]:


sns.heatmap(data_2014.corr(),cmap='coolwarm',annot=True)


# In[ ]:


sns.heatmap(data_riders.corr(),cmap='coolwarm',annot=True)


# In[ ]:


sns.lmplot(x='Teams',y='Rank',data=data_2014)


# In[ ]:


sns.lmplot(x='Teams',y='Rank',data=data_2015)


# In[ ]:


sns.lmplot(x='Teams',y='Rank',data=data_2016)


# In[ ]:


sns.lmplot(x='Teams',y='Rank',data=data_2017)


# In[ ]:


sns.lmplot(x='Points',y='Rank',data=data_riders)


# In[ ]:


sns.lmplot(x='Points',y='Rank',data=data_kings)


# In[ ]:


sns.lmplot(x='Points',y='Rank',data=data_royals)


# In[ ]:


sns.lmplot(x='Points',y='Rank',data=data_devils)


# In[ ]:




