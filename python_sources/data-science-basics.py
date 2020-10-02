#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding='ISO-8859-1')
data.rename(columns={'iyear': 'Year', 'imonth':'Month', 'iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname': 'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
data = data[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]
data['casualities']=data['Killed']+data['Wounded']
data.head(10)


# In[ ]:


data.info()


# In[ ]:


data.corr()


# In[ ]:


f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax)
plt.show()


# In[ ]:


data.columns


# In[ ]:


data.isnull().sum()


# In[ ]:


print("The most dangerous city: ",data['Country'].value_counts().index[0])
print("The most dangerous region: ",data['Region'].value_counts().index[0])
print("The most people died in an attack: ",data['Killed'].max(),'that took place in', data.loc[data['Killed'].idxmax()].Country)


# In[ ]:


plt.subplots(figsize=(15,6))
sns.countplot('Year', data=data, palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Terrorist Attack Count')
plt.show()


# In[ ]:


#Scatter Plot
data.Year.plot(kind = 'hist', bins=50, figsize = (12,12))
plt.show()


# In[ ]:


plt.clf()


# In[ ]:


x = data['casualities'] > 5000
data[x]


# In[ ]:


data[np.logical_and(data['Killed'] > 500, data['Year']==2001)]

