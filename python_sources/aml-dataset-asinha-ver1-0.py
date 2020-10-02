#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


base_dir = "/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_bitcoin_dataset/"
feature_file = "elliptic_txs_features.csv"
edgelist_file = "elliptic_txs_edgelist.csv"
classes_file = "elliptic_txs_classes.csv"


# In[ ]:


feature_data = pd.read_csv(base_dir + feature_file,header=None)
edge_data = pd.read_csv(base_dir + edgelist_file)
class_data = pd.read_csv(base_dir + classes_file)


# In[ ]:


feature_data.shape, edge_data.shape, class_data.shape


# In[ ]:


feature_data.head(2)


# In[ ]:


feature_data.columns = [str(x) for x in np.arange(0,167)]
feature_data.columns


# In[ ]:


edge_data.head()


# In[ ]:


class_data.head()


# In[ ]:


g = pd.DataFrame(class_data.groupby(["class"]).count()["txId"] / class_data.shape[0] * 100).reset_index()
plt.bar(g["class"],g["txId"])
plt.show()


# In[ ]:


feature_data.iloc[:,1:].describe()


# In[ ]:


a = pd.merge(left=edge_data,right=class_data,left_on="txId1",right_on="txId",how="left").rename(columns={"class" : "txId1_class"}).drop(columns=["txId"])
edge_data = pd.merge(left=a,right=class_data,left_on="txId2",right_on="txId",how="left").rename(columns={"class" : "txId2_class"}).drop(columns=["txId"])


# In[ ]:


edge_data.head()


# In[ ]:


a = pd.merge(left=edge_data,right=feature_data[["0","1"]],left_on="txId1",right_on="0",how="left").rename(columns={"1" : "txId1_timestep"}).drop(columns=["0"])
edge_data = pd.merge(left=a,right=feature_data[["0","1"]],left_on="txId2",right_on="0",how="left").rename(columns={"1" : "txId2_timestep"}).drop(columns=["0"])


# In[ ]:


edge_data.head()


# Checking it each source/destination transaction belong to same timestep?

# In[ ]:


np.where(edge_data.txId1_timestep != edge_data.txId2_timestep)


# In[ ]:


edge_data = edge_data.assign(class_comb = edge_data.txId1_class + "_" + edge_data.txId2_class)
edge_data.head()


# In[ ]:


edge_data.groupby(["class_comb"]).agg({"txId1" : "count",
                                     "txId2" : lambda x : len(np.unique(x))/edge_data.shape[0] * 100}).reset_index()


# Building unique sets for txId1 and txId2

# In[ ]:


source_set = set(edge_data.txId1.values)
dest_set = set(edge_data.txId2.values)


# checking which all were only present in source, which were present in both and which are only considered as destination.

# In[ ]:


len(source_set - dest_set),len(source_set.intersection(dest_set)), len(dest_set - source_set)


# In[ ]:


len(source_set - dest_set) + len(source_set.intersection(dest_set))+ len(dest_set - source_set)


# In[ ]:


only_source = source_set - dest_set
only_dest = dest_set - source_set
common_nodes = source_set.intersection(dest_set)


# In[ ]:


'''
loop_list = []
for s in only_source:
    df = edge_data[edge_data.txId1 == s]
    if df.txId2.isin(common_nodes):
'''


# In[ ]:




