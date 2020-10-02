#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Import Dataset**

# In[ ]:


data = pd.read_csv('/kaggle/input/market-basket-optimization/Market_Basket_Optimisation.csv', header=None)


# **Check Head of Dataset**

# In[ ]:


data.head()


# In[ ]:


data.shape # Dataset Shape and Size


# **Visualization**

# In[ ]:


# Most Frequent Items Bar plot

plt.rcParams['figure.figsize'] = (10,6)
color = plt.cm.inferno(np.linspace(0,1,20))
data[0].value_counts().head(20).plot.bar(color = color)
plt.title('Top 20 Most Frequent Items')
plt.ylabel('Counts')
plt.xlabel('Items')
plt.show()


# In[ ]:


# Tree Map of Most Frequent Items
import squarify
plt.rcParams['figure.figsize']=(10,10)
Items = data[0].value_counts().head(20).to_frame()
size = Items[0].values
lab = Items.index
color = plt.cm.copper(np.linspace(0,1,20))
squarify.plot(sizes=size, label=lab, alpha = 0.7, color=color)
plt.title('Tree map of Most Frequent Items')
plt.axis('off')
plt.show()


# In[ ]:


data['Items'] = 'items'
df = data.truncate(before=-1,after=15)

import networkx as nx

Items = nx.from_pandas_edgelist(df, source = 'Items', target = 0, edge_attr = True)

plt.rcParams['figure.figsize'] = (20,20)
nx.draw_networkx_nodes(G=Items,pos=nx.spring_layout(Items), node_size=15000,node_color='green')
nx.draw_networkx_edges(G=Items,pos=nx.spring_layout(Items), alpha=0.6, width=3 ,edge_color='black')
nx.draw_networkx_labels(G=Items,pos=nx.spring_layout(Items),font_size=20, font_family = 'sans-serif')
plt.axis('off')
plt.grid()
plt.title('Top 15 First Choices', fontsize = 20)
plt.show()


# In[ ]:


data.drop(columns='Items',axis=1, inplace=True)
data.head()


# **Association Analysis** **And**
# **Apriori Algorithm** 

# In[ ]:


# list of list is needed as an input for transaction encoder
transactions = []
for i in range(0,7501):
    transactions.append([str(data.values[i,j]) for j in range(0,20)])


# In[ ]:


# TransactionEncoder learns the unique labels in the dataset, and via the transform method, 
# it transforms the input dataset (a Python list of lists) into a one-hot encoded NumPy boolean array:

from mlxtend.preprocessing import TransactionEncoder
transac = TransactionEncoder()
dataset = transac.fit_transform(transactions)
dataset


# In[ ]:


df = pd.DataFrame(dataset, columns= transac.columns_)
df.head()


# In[ ]:


#Apriori Algorithm to find out most frequent itemset with min support of 0.003

from mlxtend.frequent_patterns import apriori, association_rules
frequent_itemsets = apriori(df, min_support=0.003, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x : len(x))


# In[ ]:


frequent_itemsets.head(20)


# In[ ]:


#Filter Frequent itemset of minimum length 2

frequent_itemsets[frequent_itemsets['length'] >= 2].head(20)


# In[ ]:


# Association Rules Mining to generate the rules with their coresponding support
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
rules.head(50)


# In[ ]:


rules[(rules['lift'] >= 5) & (rules['confidence'] >= 0.4)]


# **Conclusion**

# In[ ]:


#association analysis is easy to run and relatively easy to interpret. 
#the most frequent association itemset are mineral water and whole wheat pasta with 
#olive oil and people always buy this three items together!!!
#more significant rules can be find with lower lift and confidence and suport!!!

