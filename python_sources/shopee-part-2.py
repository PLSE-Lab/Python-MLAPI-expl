#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm_notebook

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from IPython.display import display, HTML
def displayer(df): display(HTML(df.head(4).to_html()))
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


# In[ ]:


import pandas as pd

class StringConverter(dict):
    def __contains__(self, item):
        return True

    def __getitem__(self, item):
        return str

    def get(self, default=None):
        return str


# In[ ]:


df1pd = pd.read_csv("/kaggle/input/ungrd-rd2-auo/credit_cards.csv", converters=StringConverter())
df2pd = pd.read_csv("/kaggle/input/ungrd-rd2-auo/devices.csv", converters=StringConverter())
df3pd = pd.read_csv("/kaggle/input/ungrd-rd2-auo/bank_accounts.csv", converters=StringConverter())
df4 = pd.read_csv("/kaggle/input/ungrd-rd2-auo/orders.csv", converters=StringConverter())

displayer(df1pd)
displayer(df2pd)
displayer(df3pd)

df_all = pd.concat([df1pd.copy(), df2pd.copy(), df3pd.copy()], sort=False)


# In[ ]:


df1 = df1pd.set_index('userid').to_dict()['credit_card']
df2 = df2pd.set_index('userid').to_dict()['device']
df3 = df3pd.set_index('userid').to_dict()['bank_account']


# In[ ]:


from collections import defaultdict

dx = defaultdict(list)

for k,v in df1.items():
    dx[v].append(k)
    dx[k].append(v)
for k,v in df2.items():
    dx[v].append(k)
    dx[k].append(v)
for k,v in df3.items():
    dx[v].append(k)
    dx[k].append(v)


# In[ ]:


from collections import defaultdict
import networkx as nx

G = nx.Graph()

for k,v in df1.items():
    G.add_edge(k,v)
for k,v in df2.items():
    G.add_edge(k,v)
for k,v in df3.items():
    G.add_edge(k,v)


# In[ ]:


nx.bidirectional_dijkstra(G,"207471076","233033146")


# In[ ]:


fraud_arr = []
for row in tqdm_notebook(df4.values):
    x,a,b = row
    try: 
        _,_ = nx.bidirectional_dijkstra(G,a,b)
        fraud_arr.append(1)
    except:
        fraud_arr.append(0)


# In[ ]:


arr = fraud_arr
print(len(fraud_arr))
print(sum(fraud_arr))

dfs = df4.copy()
dfs["is_fraud"] = arr
dfs = dfs.drop(["buyer_userid", "seller_userid"], axis = 1)
dfs.to_csv("submission.csv", index=False)


# In[ ]:


pd.read_csv("submission.csv")


# In[ ]:


sum(list(dfs["is_fraud"]))


# In[ ]:




