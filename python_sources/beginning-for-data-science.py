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

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_terror = pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')


# In[ ]:


df_terror.head()


# In[ ]:


df_terror.info()


# There are too many colums. I am selecting the columns with useful info and then I will rename them for easy understanding

# In[ ]:


terror=df_terror[['iyear','imonth','iday','country_txt','region_txt','city','latitude','longitude',
                  'attacktype1_txt','nkill','nwound','target1','summary','gname','targtype1_txt', 
                  'targsubtype1_txt', 'weaptype1_txt','motive','specificity']]
print(terror.columns)


# In[ ]:


terror.columns = ['year', 'month', 'day', 'country', 'region', 'city',
       'latitude', 'longitude', 'attack_type', 'killed', 'wounded',
       'target', 'summary', 'group', 'target_type', 'target_sub_type','weapon_type',
       'motive', 'specifity']


# In[ ]:


terror.info()


# In[ ]:


f,ax = plt.subplots(figsize = (15, 15))
corr_data = terror.corr()
sns.heatmap(corr_data, annot=True, linewidths=0.5, fmt='.1f', ax=ax)
plt.show()


# There is a correlation between number of killed and wounded people. Except this, we can say there is no correlation.
# 
# Let's get a summary of our data.

# In[ ]:


terror.head()


# In[ ]:


terror.plot(x='year', y='killed', kind='line',color = 'r',label = 'Killed',linewidth=1,
            alpha = 0.5,grid = True,linestyle = ':')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.title('Line Plot')            # title = title of plot
plt.show()


# In[ ]:


terror.Year.plot(kind = 'hist',bins = 10,figsize = (12,12))
plt.show()


# In[ ]:


series = terror['Country']        # data['Defense'] = series
print(type(series))
data_frame = terror[['Region']]  # data[['Defense']] = data frame
print(type(data_frame))


# In[ ]:


terror.region.unique()


# In[ ]:


terror[terror['Killed']>100].head()


# In[ ]:


terror[np.logical_and(terror['region']=='Middle East & North Africa', terror['killed']>100)].head()


# In[ ]:


terror.shape


# In[ ]:


terror.describe()


# In[ ]:


terror.month.value_counts()


# In[ ]:


terror.boxplot(column="Killed")
plt.show()


# In[ ]:


terror.dtypes


# In[ ]:


terror.attack_type.value_counts(dropna=False)


# In[ ]:


terror.loc[0,"Killed"]


# In[ ]:


terror.killed[0]


# In[ ]:


terror[['killed', 'wounded']]


# In[ ]:


terror.loc[0:10, "killed":]


# In[ ]:


terror.killed[terror.region=='Middle East & North Africa'].mean()


# In[ ]:


df_ndf_terror.nwound.head()


# In[ ]:


terror.index.name = "index_name" 


# In[ ]:


terror.head()


# In[ ]:


terror.tail()


# In[ ]:


terror.index = range(1,181692,1)


# In[ ]:


terror.head()


# In[ ]:




