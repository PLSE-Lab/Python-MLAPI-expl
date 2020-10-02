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



from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting

import os # accessing directory structure

# Any results you write to the current directory are saved as output.


# In[ ]:


import os
print(os.listdir('../input'))


# In[ ]:


import pandas as pd
nRowsRead = 2500
df1 = pd.read_csv('../input/FieldSampleSoilResults_2.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = '/FieldSampleSoilResults_2.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


df1.info()


# In[ ]:


df1.columns


# In[ ]:


df1.describe()


# In[ ]:


df1.head(10)


# In[ ]:


import matplotlib.pyplot as plt 
plt.hist(df1['Analysis Id'], color = 'blue', edgecolor = 'black',
         bins = int(180/5))


# In[ ]:


plt.hist(df1['Sample Id'], color = 'red', edgecolor = 'brown',
         bins = int(180/5))


# In[ ]:


plt.hist(df1['Bearing'], color = 'green', edgecolor = 'black',
         bins = int(180/5))


# In[ ]:


for i, binwidth in enumerate([10, 50, 75, 100]):
    
   
    ax = plt.subplot(2, 2, i + 1)
    
    ax.hist(df1['Distance(miles)'], bins = int(200/binwidth),
             color = 'purple', edgecolor = 'black')
    
    # Title and labels
    ax.set_title('Histogram with Binwidth = %d' % binwidth, size = 10)
    ax.set_xlabel('Distance(miles)', size = 10)
    ax.set_ylabel('Soil', size=10)

plt.tight_layout()
plt.show()


# In[ ]:


plt.matshow(df1.corr())
plt.show()


# In[ ]:


import seaborn as sns
corr = df1.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=50,
    horizontalalignment='right'
);


# In[ ]:


f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


df1.Latitude.plot(kind = 'line', color = 'g',label = 'Latitude',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
df1.Longitude.plot(color = 'r',label = 'Result',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')    
plt.xlabel('x axis')              
plt.ylabel('y axis')
plt.title('Line Plot')           
plt.show()


# In[ ]:


df1.plot(kind='scatter', x='Result', y='Longitude',alpha = 0.5,color = 'blue')
plt.xlabel('Result')              # label = name of label
plt.ylabel('Longitude')
plt.title('Result Longitude Scatter Plot')


# In[ ]:


for index,value in df1[['Distance(miles)']][0:100].iterrows():
    print(index," : ",value)


# In[ ]:


df1.boxplot(column='Result',by = 'Longitude',grid=True, rot=1000, fontsize=10,figsize=(25,15))


# In[ ]:


data1 = df1.loc[:,["Latitude","Longitude","Distance(miles)","Result"]]
data1.plot()


# In[ ]:


data1.plot(subplots = True,figsize=(25,15))


# In[ ]:


data1.plot(kind = "scatter",x="Longitude",y = "Latitude",figsize=(25,15))


# In[ ]:


data1.plot(kind = "hist",y = "Distance(miles)",bins = 50,range= (0,250),normed = True)


# In[ ]:



fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "Result",bins = 50,range= (0,250),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "Result",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt

