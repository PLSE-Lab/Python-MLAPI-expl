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


data = pd.read_csv('../input/Pokemon.csv')


# In[ ]:


data.info()


# In[ ]:


data.corr()


# In[ ]:


data.head()


# In[ ]:


f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


data.columns


# In[ ]:


data.Speed.plot(kind='line' ,color='g',label='Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Defense.plot(kind='line',color='r',label='Defense',linewidth=1,alpha=0.5,grid=True,linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')              
plt.ylabel('y axis')
plt.title('defense & speed Line Plot')            
plt.show()


# In[ ]:


data.plot(kind='scatter',x='Attack',y='Defense',alpha=0.5,color='r')
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.title('Attack & Defense Scatter plot')
plt.show()


# In[ ]:


data.Speed.plot(kind='hist',bins=50,figsize=(12,12))
plt.title('Speed hist')
plt.show()


# In[ ]:


series = data['Defense']        # data['Defense'] = series
print(type(series))
data_frame = data[['Defense']]  # data[['Defense']] = data frame
print(type(data_frame))


# In[ ]:


x=data['Defense']>200
data[x]


# In[ ]:


data[(data['Defense']>200) & (data['Attack']>100)]


# In[ ]:


threshold=sum(data.Speed)/len(data.Speed)
print(threshold)
data['Speed_level']=['hight' if i>threshold else 'low' for i in data.Speed]
data.loc[:10,['Speed_level','Speed']]


# In[ ]:


print(data['Type 1'].value_counts(dropna=False))


# In[ ]:


data.describe()


# In[ ]:


data.boxplot(column='Attack',by='Legendary')
plt.show()


# In[ ]:


data_new=data.head()
data_new


# In[ ]:


melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Attack','Defense'])
melted


# In[ ]:


data2=data.tail()
data1=data.head()
conc_data_row=pd.concat([data1,data2],axis=0,ignore_index=True)
conc_data_row


# In[ ]:


data1=data['Attack'].head()
data2=data['Defense'].head()
conc_data_col=pd.concat([data1,data2],axis=1)
conc_data_col


# In[ ]:


data["Type 2"].value_counts(dropna=False)


# In[ ]:


data1=data 
data1["Type 2"].dropna(inplace = True)


# In[ ]:


assert 1==1 


# In[ ]:


assert  data['Type 2'].notnull().all() 


# In[ ]:


data["Type 2"].fillna('empty',inplace = True)


# In[ ]:


assert  data['Type 2'].notnull().all()


# In[ ]:


data1=data.loc[:,["Attack","Defense","Speed"]]
data1.plot()


# In[ ]:


data1.plot(subplots=True)
plt.show()


# In[ ]:


data1.plot(kind="scatter",x="Attack",y="Defense")
plt.show()


# In[ ]:


data1.plot(kind="hist",y="Defense",bins=50 , range=(0,250),normed=True)
plt.show()


# In[ ]:


fig,axes=plt.subplots(nrows=2,ncols=1)
data1.plot(kind="hist",y="Defense",bins=50,range=(0,250),normed=True,ax=axes[0])
data1.plot(kind="hist",y="Attack",bins=50,range=(0,250),normed=True,ax=axes[1])
plt.show()


# In[ ]:


data_melted=data.head()
melted=pd.melt(frame=data_melted,id_vars='Name',value_vars=['Attack','Defense'])
melted

