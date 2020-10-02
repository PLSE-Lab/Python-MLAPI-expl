#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df = pd.read_csv('/kaggle/input/close-contact-status-of-corona-in-japan/COVID-19_Japan_Mar_07th_2020.csv')
df.head()


# In[ ]:


set(df['Surrounding patients *'])


# In[ ]:


import numpy as np
df['Surrounding patients new'] = df['Surrounding patients *'].str.replace('None','')
df['Surrounding patients new'] = df['Surrounding patients new'].str.replace('Same person as No.8',list(df[df['No.'] == 8]['Surrounding patients new'].values)[0])
df['Surrounding patients new'] = df['Surrounding patients new'].str.replace('Onset after cruise ship disembarkation','cruise ship')
df['Surrounding patients new'] = df['Surrounding patients new'].str.replace('Onset after cruise ship disembarks','cruise ship')
df['Surrounding patients new'] = df['Surrounding patients new'].str.replace('investigating','')
df['Surrounding patients new'] = df['Surrounding patients new'].str.replace('unknown','')
df['Surrounding patients new'] = df['Surrounding patients new'].str.replace('No.','')
df['Surrounding patients list'] = [i.split(', ') for i in df['Surrounding patients new']]
df['edges'] = [[(str(df.at[i,'No.']),j) for j in df.at[i,'Surrounding patients list'] if j != ''] for i in df.index]

places = ['cruise ship','Table tennis school','exhibition','live house','sports gym']

import networkx as nx
G = nx.Graph()
G.add_edges_from(df['edges'].sum())
G.add_nodes_from(places)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20, 10))
pos=nx.spring_layout(G, k=0.15, seed=1)

nx.draw_networkx(G,pos,node_size=300, node_color='aqua')
nx.draw_networkx(G,pos,nodelist=places,node_size=300, node_color='pink')


# In[ ]:




