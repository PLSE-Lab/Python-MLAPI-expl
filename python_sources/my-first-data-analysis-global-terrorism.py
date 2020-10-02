#!/usr/bin/env python
# coding: utf-8

# # Simply Data Analysis for  Global Terrorism

# In[ ]:



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns #visualization tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


#   ## Getting Data  
#   
#  Some situation we should use encoding method.Otherwise we can not call data

# In[ ]:


data=pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')


# In[ ]:


data.info()


# **First 5 data relationship correlation**

# In[ ]:


data.corr().head()


# In[ ]:


data.columns


# **Data arrangement for make process**

# In[ ]:


data.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
data.head()


# **Find total 'NaN' value each features**

# In[ ]:


data.isnull().sum()


# **Data arrange for see data clearly**

# In[ ]:


data1=data[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]
data1.head()


# **Correlation Map**

# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(25, 18))
sns.heatmap(data1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# ### The 10 most effective terrorist groups . (Except  unknown groups)

# In[ ]:


data['Group'].value_counts().head(11)


# ## Some Analysis

# ### First five country with Highests Terrorist Attacks

# In[ ]:


data1['Country'].value_counts().head()


# ### First five region with Highest Terrorist Attacks

# In[ ]:


data1['Region'].value_counts().head()


# In[ ]:


print('Maximum people killed in an attack are: ',data1['Killed'].max(),'\nCountry:',data1.loc[data1['Killed'].idxmax()].Country)
print('Minimum people killed in an attack are: ',data1['Killed'].min(),'\nCountry:',data1.loc[data1['Killed'].idxmin()].Country)


# In[ ]:


#Line Plot
data1.Killed.plot(kind='line',color='g',label='Killed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data1.Wounded.plot(color = 'r',label = 'Wounded',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()


# In[ ]:


# Scatter Plot 
data1.plot(kind='scatter', x='Killed', y='Wounded',alpha = 0.5,color = 'red')
plt.xlabel('Killed')              # label = name of label
plt.ylabel('Wounded')
plt.title('Killed Wounded Scatter Plot')            # title = title of plot
plt.show()


# In[ ]:


data1.Killed.plot(kind = 'hist',bins = 13,range= (0,13))
#plt.clf()
plt.show()


# ## Terrorist activity by region

# In[ ]:


#Number Of Terrorist Activities By Region
plt.subplots(figsize=(17,7))
sns.countplot('Region',data=data,palette='PRGn',edgecolor=sns.color_palette('dark',7),order=data['Region'].value_counts().index)
plt.xticks(rotation=30)
plt.title('Number Of Terrorist Activities By Region')
plt.show()


# In[ ]:


terror_region=pd.crosstab(data.Year,data.Region)
terror_region.plot(color=sns.color_palette('Set2',12))
fig=plt.gcf()
fig.set_size_inches(17,7)
plt.show()


# In[ ]:


#Number Of Terrorist Activities Each Year
plt.subplots(figsize=(17,7))
sns.countplot('Year',data=data, palette='gist_heat_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()


# ## Terrorist activity by country

# In[ ]:


plt.subplots(figsize=(18,6))
sns.barplot(data['Country'].value_counts()[:15].index,data['Country'].value_counts()[:15].values,palette='YlOrBr_r')
plt.title('Top Affected Countries')
plt.show()


# ## Most effective terrorist groups

# In[ ]:


sns.barplot(data['Group'].value_counts()[1:15].values,data['Group'].value_counts()[1:15].index,palette=('BrBG_r'))
plt.xticks(rotation=90)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.title('Terrorist Groups with Highest Terror Attacks')
plt.show()


# In[ ]:




