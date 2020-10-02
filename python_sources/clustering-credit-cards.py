#!/usr/bin/env python
# coding: utf-8

# # Importing the necessary Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,scale
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
init_notebook_mode(connected=True)  
import cufflinks as cf  
cf.go_offline() 
import os
df = pd.read_csv('../input/CC GENERAL.csv')
df.head()


# # Performing Data Cleaning/Preprocessing Sequences
# This involve steps like:-
# 1. Removal of Unnamed Column(s)
# 2. Column(s) Renaming
# 2. Checking the correct data type(s) of the respective column(s)
# 3. Finding empty instance(s) and replacing them with suitable statistical figure such as mean/median

# In[ ]:


df.isna().sum()


# In[ ]:


df = df.fillna(df.mean())


# In[ ]:


df.isna().sum()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# # EDA

# In[ ]:


cols = df.columns.tolist()
cols.pop(0)


# In[ ]:


for i in ['BALANCE',
 'PURCHASES',
 'ONEOFF_PURCHASES',
 'INSTALLMENTS_PURCHASES',
 'CASH_ADVANCE',
 'CASH_ADVANCE_FREQUENCY',
 'CASH_ADVANCE_TRX',
 'PURCHASES_TRX',
 'CREDIT_LIMIT',
 'PAYMENTS',
 'MINIMUM_PAYMENTS']:
    print(i)
    df[i].iplot()


# In[ ]:


df.iplot(kind='box')


# # Hierarhical Clustering

# In[ ]:


import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import scale as s
from scipy.cluster.hierarchy import dendrogram, linkage


# In[ ]:


le = LabelEncoder()
df['CUST_ID'] = le.fit_transform(df['CUST_ID'])


# In[ ]:


Z = sch.linkage(df,method='ward')
den = sch.dendrogram(Z)
plt.tick_params(
axis='x',          
which='both',      
bottom=False,     
top=False,         
labelbottom=False) 


# **Graph Description:** This Hierarhical Clustering is done with ward linkage. In Ward's linkage, minimum variance criterion minimizes the total within-cluster variance

# Given below is special function made to serve the purpose of drawing the line which cuts the generated dendrogram to determine the number of clusters and the dendrogram node(s) which are below the cutting line

# In[ ]:


def fd(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


# In[ ]:


Z = linkage(df,method='ward')
fd(Z,leaf_rotation=90.,show_contracted=True,annotate_above=260000,max_d=320000)
plt.tick_params(
    axis='x',          
    which='both',      
    bottom=False,     
    top=False,         
    labelbottom=False) 


# **Graph Description:** Graph Description: Following the main critera of the cutting the dendrogram appropriatly we discover that there are basically 2 clusters, also observed from the above Graph. Observing the height of each dendrogram division we decided to go with 320000 where the line would be drawn and 260000 to determine the dendrogram nodes
