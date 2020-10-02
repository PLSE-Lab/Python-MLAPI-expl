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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.book import FreqDist

biology=pd.read_csv("../input/biology.csv") 
tags=[]
for tag in biology["tags"]:
    for t in tag.split():
        tags.append(t)
fdist1 = FreqDist(tags)
#fdist1.most_common(50)
fdist1.plot(50, cumulative=True)


# In[ ]:


cooking=pd.read_csv("../input/cooking.csv") 
tags=[]
for tag in cooking["tags"]:
    for t in tag.split():
        tags.append(t)
fdist1 = FreqDist(tags)
fdist1.plot(50, cumulative=True)


# In[ ]:


crypto=pd.read_csv("../input/crypto.csv") 
tags=[]
for tag in crypto["tags"]:
    for t in tag.split():
        tags.append(t)
fdist1 = FreqDist(tags)
fdist1.plot(50, cumulative=True)


# In[ ]:


fdist1.most_common(10)


# In[ ]:


diy=pd.read_csv("../input/diy.csv") 
tags=[]
for tag in diy["tags"]:
    for t in tag.split():
        tags.append(t)
fdist1 = FreqDist(tags)
fdist1.plot(50, cumulative=True)


# In[ ]:


travel=pd.read_csv("../input/travel.csv") 
tags=[]
for tag in travel["tags"]:
    for t in tag.split():
        tags.append(t)
fdist1 = FreqDist(tags)
fdist1.plot(50, cumulative=True)


# In[ ]:


import networkx as nx
used = []
V=[used.append(x) for x in tags if x not in used]# list of vertices
g=nx.Graph()
g.add_nodes_from(V)
E=[]
#E=[e for e in travel["tags"]]
for tag in travel["tags"]:
    s=tag.split()
    if len(s)==2: 
        E.append(s)
    elif len(s)>2:
        for i in (0,len(s)):
            for j in range(i+1,len(s)):
                E.append([s[i],s[j]])

g.add_edges_from(E)

pos=nx.fruchterman_reingold_layout(g) 


# In[ ]:


import plotly.plotly as py
from plotly.graph_objs import *

Xv=[pos[k][0] for k in V]
Yv=[pos[k][1] for k in V]
Xed=[]
Yed=[]
for edge in E:
    Xed+=[pos[edge[0]][0],pos[edge[1]][0], None]
    Yed+=[pos[edge[0]][1],pos[edge[1]][1], None] 
    
trace3=Scatter(x=Xed,
               y=Yed,
               mode='lines',
               line=Line(color='rgb(210,210,210)', width=1),
               hoverinfo='none'
               )
trace4=Scatter(x=Xv,
               y=Yv,
               mode='markers',
               name='net',
               marker=Marker(symbol='dot',
                             size=5, 
                             color='#6959CD',
                             line=Line(color='rgb(50,50,50)', width=0.5)
                             ),
               text=V,
               hoverinfo='text'
               )

annot="This networkx.Graph has the Fruchterman-Reingold layout<br>Code:"+"<a href='http://nbviewer.ipython.org/gist/empet/07ea33b2e4e0b84193bd'> [2]</a>"

data1=Data([trace3, trace4])
fig1=Figure(data=data1, layout=layout)
fig1['layout']['annotations'][0]['text']=annot
py.iplot(fig1, filename='Coautorship-network-nx')

