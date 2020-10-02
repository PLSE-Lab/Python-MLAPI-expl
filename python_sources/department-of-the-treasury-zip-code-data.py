#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Not a complete notebook. Mostly using it as a rough space to test out different ideas on the Dataset. 
## Any ideas how to take it private? 


# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

taxes = pd.read_csv('../input/14zpallagi.csv')


# In[ ]:


taxes.shape


# In[ ]:


taxes.head()


# In[ ]:


# one_col = taxes[taxes['STATE']=='AL']


# In[ ]:


# one_col.head()


# In[ ]:


# two_col = taxes[taxes['STATE']=='IL']
# two_col.head()


# In[ ]:


# one_col[one_col[['STATE','zipcode','agi_stub','mars1','MARS2','MARS4','A00100']]
# one_col[['STATE','zipcode','agi_stub','mars1','MARS2','MARS4','A00100']]


# In[ ]:


# arr = one_col[['STATE','zipcode','agi_stub','mars1','MARS2','MARS4','A00100']]
# arr


# In[ ]:


# indx = np.arange(arr.shape[0])    
# indx


# In[ ]:


# arr.set_index(indx)
# arr


# In[ ]:


# df_red = taxes[(taxes['zipcode'] < 99999) & (taxes['zipcode'] > 0)][['STATE','zipcode','agi_stub','N1','mars1','MARS2','MARS4','A00100']]
# df_red


# In[ ]:


# df_red.reset_index(inplace=True)
# df_red


# In[ ]:


# df_red.drop(['index'],axis=1,inplace=True)


# In[ ]:


# df_red


# In[ ]:


# plt.plot(df_red['zipcode'],df_red['A00100'])


# In[ ]:


# sns.distplot(df_red['A00100'])


# In[ ]:


# df_red.head()


# In[ ]:


# sns.jointplot(x='agi_stub',y='N1',data=df_red,size=8)


# In[ ]:


# sns.pairplot(df_red)


# In[ ]:


# df_red.head()


# In[ ]:


taxes.describe()


# In[ ]:


data = taxes[(taxes['zipcode'] < 99999) & (taxes['zipcode'] > 0)]        [['STATE','zipcode','agi_stub','N1','mars1','MARS2','MARS4',          'NUMDEP','A00100','A00300','A00700','N01000','A00101','A18500']]


# In[ ]:


data.head(36)


# In[ ]:


# remove outliers
   
from scipy import stats
data = data[(np.abs(stats.zscore(data.drop(['STATE'],axis=1))) < 3).all(axis=1)]
data


# columns that need further investigation:
# * N01000
# 

# In[ ]:


# sns.pairplot(data)


# In[ ]:


# data[data['A00101']>100000]


# In[ ]:


data_2 = taxes[(taxes['zipcode'] < 99999) & (taxes['zipcode'] > 0)]        [['STATE','zipcode','agi_stub','N1','mars1','MARS2','MARS4',          'NUMDEP','A00100','A00101','A18500','A19700']]


# In[ ]:


data_2.head(36)


# In[ ]:


sns.pairplot(data[:1000])


# In[ ]:


g = sns.PairGrid(data[:1000])
g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map(sns.kdeplot)


# In[ ]:


data.drop(['A00101','A18500','NUMDEP','A00300','A00700'],axis=1,inplace=True)


# In[ ]:


data.head()


# In[ ]:


subset = data[:1000]
subset


# In[ ]:


sns.barplot(x='agi_stub',y='A00100',data=subset)


# In[ ]:


sns.barplot(x='agi_stub',y='A00100',data=data)


# In[ ]:


sns.barplot(x='agi_stub',y='A00100',data=taxes)


# In[ ]:


sns.barplot(x='zipcode',y='A00100',data=data[:10000],estimator=np.mean) # expensive


# In[ ]:


sns.boxplot(x='agi_stub',y='A00100',data=data[:]) # after outliers


# In[ ]:


sns.boxplot(x='agi_stub',y='A00100',data=subset)


# In[ ]:


sns.boxplot(x='agi_stub',y='A00100',data=subset[:10],hue='zipcode')


# In[ ]:


fig, ax = plt.subplots(figsize=(12,8))
sns.violinplot(x='agi_stub',y='A00100',data=subset)


# In[ ]:


sns.violinplot(x='agi_stub',y='A00100',data=data)


# In[ ]:


sns.stripplot(x='agi_stub',y='A00100',data=subset,jitter=True)


# In[ ]:


sns.swarmplot(x='agi_stub',y='A00100',data=subset)


# In[ ]:


sns.violinplot(x='agi_stub',y='A00100',data=subset)
sns.swarmplot(x='agi_stub',y='A00100',data=subset)


# In[ ]:


sns.factorplot(x='agi_stub',y='A00100',data=subset,kind='swarm',size=8) # also --violin, box, strip etc..


# In[ ]:


data.head(12)


# In[ ]:


data_corr = data.corr()


# In[ ]:


fig, ax = plt.subplots(figsize=(14,12))
sns.heatmap(data_corr,annot=True,cmap='coolwarm',linecolor='white',linewidths=.5)


# In[ ]:


data.head(12)


# In[ ]:


data_pvtable = data.pivot_table(index='STATE',columns='agi_stub',values='N1')
data_pvtable


# In[ ]:


fig, ax = plt.subplots(figsize=(14,12))
sns.heatmap(data_pvtable,cmap='coolwarm')


# In[ ]:


sns.clustermap(data_pvtable,cmap='coolwarm')


# In[ ]:


sns.clustermap(data_pvtable,cmap='coolwarm',standard_scale=1)


# In[ ]:


data.head()


# In[ ]:


grd = sns.FacetGrid(data=data,col='agi_stub',row='STATE')
grd.map(sns.distplot,'mars1')


# In[ ]:


subset.info()


# In[ ]:


# regression plots
sns.lmplot(x='N01000',y='A00100',data=data[:100],hue='agi_stub',size=8)


# In[ ]:


# regression plots
sns.lmplot(x='N01000',y='A00100',data=data[:100],hue='agi_stub',size=8,palette='coolwarm')


# In[ ]:


sns.lmplot(x='N01000',y='A00100',data=data[:100],hue='agi_stub',size=8,markers='v',
          scatter_kws={'s':100})


# In[ ]:


sns.lmplot(x='N01000',y='A00100',data=data[:10000],col='agi_stub',row='STATE')


# In[ ]:


sns.lmplot(x='N01000',y='A00100',data=data[:5000],col='agi_stub',row='STATE',
          aspect=.6,size=10)


# In[ ]:


fig, ax = plt.subplots(figsize=(12,8))

# setting styles
# sns.set_style('white')
sns.set_style('ticks')

sns.boxplot(x='agi_stub',y='A00100',data=subset)


# In[ ]:


plt.figure(figsize=(12,8))

# setting styles
# sns.set_style('white')
sns.set_style('ticks')

sns.boxplot(x='agi_stub',y='A00100',data=subset)


# In[ ]:


data.head(12)


# In[ ]:


data['A00100'].hist(bins=20)


# In[ ]:


data['A00100'].plot(kind='hist')


# In[ ]:


data['A00100'].plot.hist()


# In[ ]:


data['A00100'].plot.area(alpha=0.4)


# In[ ]:


data['A00100'][:20].plot.bar()


# In[ ]:


# Index Levels
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)
df = pd.DataFrame(np.random.randn(6,2),index=hier_index,columns=['A','B'])
df


# In[ ]:


df['A'].plot.bar()


# In[ ]:


df['A'].plot.hist()


# In[ ]:


data['A00100'][:20].plot.line()


# In[ ]:


data.plot.scatter(x='N1',y='N01000',c='agi_stub',figsize=(18,10),cmap='coolwarm')


# In[ ]:


data.plot.scatter(x='A00100',y='N01000',c='agi_stub',figsize=(18,10),cmap='coolwarm')


# In[ ]:


data.plot.scatter(x='A00100',y='N01000',s=data['agi_stub']*1.1,figsize=(10,8))


# In[ ]:


subset.plot.box(figsize=(12,8))


# In[ ]:


data.plot.box(figsize=(12,8))


# In[ ]:


data.plot.hexbin(x='MARS2',y='agi_stub',gridsize=25,figsize=(12,8))


# In[ ]:


df.plot.hexbin(x='A',y='B',gridsize=25,figsize=(12,8))


# In[ ]:


data.plot.density()


# In[ ]:


data['N1'].plot.density()


# In[ ]:


data.head(20)


# In[ ]:


from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
print(__version__) # requires version >= 1.9.0
import plotly.graph_objs as go
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()


# In[ ]:


df3 = pd.DataFrame({'x':[1,2,3,4,5],'y':[10,20,30,20,10],'z':[5,4,3,2,1]})
df3.iplot(kind='surface',colorscale='rdylbu')


# In[ ]:


subset.iplot(kind='scatter',x='MARS2',y='MARS4',mode='markers',size=10)


# In[ ]:


subset.iplot(kind='surface',colorscale='rdylbu')


# In[ ]:


subset.iplot(kind='box')


# In[ ]:


subset.iplot(kind='bar',x='agi_stub',y='N1')


# In[ ]:


subset[['mars1','MARS2']].iplot(kind='spread')


# In[ ]:


subset['A00100'].iplot(kind='hist',bins=25)


# In[ ]:


subset.iplot(kind='bubble',x='mars1',y='MARS2',size='zipcode')


# In[ ]:


subset.iplot(kind='bubble',x='zipcode',y='MARS2',size='agi_stub')


# In[ ]:


subset[["N1",'mars1','MARS2','MARS4','A00100']].scatter_matrix()


# In[ ]:


subset.head()


# In[ ]:


# subset[["N1",'mars1','MARS2','MARS4','A00100']]


# In[ ]:


dat = dict(type = 'chloropleth',
          locations = data['STATE'].unique(),
          locationmode = 'USA-states',
          colorscale = 'Portland',
          text = data['STATE'].unique(),
          z = data['STATE'].unique(),
          colorbar = {'title' : 'Colorbar Title Goes Here'})


# In[ ]:


dat


# In[ ]:


dat


# In[ ]:


layout = dict(geo={'scope':'usa'})


# In[ ]:


# choromap = go.Figure(data=[dat],layout=layout)
# iplot(choromap)


# In[ ]:





# In[ ]:




