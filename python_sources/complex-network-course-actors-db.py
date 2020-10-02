#!/usr/bin/env python
# coding: utf-8

# #  What's a Bacon number?

# https://oracleofbacon.org/

# In[ ]:


import pandas as pd
import matplotlib
from itertools import combinations
import networkx as nx # the main libary we will use
from networkx.algorithms import bipartite
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
import pandas as pd
import matplotlib


# #  We will import our dataset from a Wikipedia scraper that is also used to calculate the Bacon numbers
# link: https://oracleofbacon.org/how.php
# size: 15 Mb

# In[ ]:


df = pd.read_json('../input/data.txt',lines=True)
df.sample(5,random_state=12228)


# # The majority of our movies are millenials 

# In[ ]:


df.year.hist(bins=200)


# # Preprocessing makes the tables ready for a network representation -
# * Nodes - actors
# * Edges - played together in a movie

# In[ ]:


df = df[df['year'].between(2014,2018)]
df['cast_coup']=df['cast'].apply(lambda L: [comb for comb in combinations(L, 2)])
df['title_tup']=df['title'].apply(lambda x: (x,))
df['year_tup']=df['year'].apply(lambda x: (str(x),))

def generate_touples(tup,row):
    return tup+row['title_tup']+row['year_tup']

df['cast_coup_step2']=df.apply(lambda row: [generate_touples(tup,row) for tup in row['cast_coup']],axis=1)
final_tup_list = df['cast_coup_step2'].apply(pd.Series).stack().reset_index(True)[0].tolist()
df = pd.DataFrame(final_tup_list, columns=['Player1', 'Player2', 'Movie_name','Year'])
df.sample(5,random_state =6)


# #  Building our first netowork with Movie name and year as edge attributes 

# In[ ]:


G = nx.from_pandas_edgelist(df,'Player1','Player2',edge_attr = ['Year','Movie_name'], create_using=nx.Graph())
print (nx.info(G))


# In[ ]:


def Bacon_number(G,Player1,Player2):
        length =nx.shortest_path_length(G,Player1,Player2)
        path = nx.shortest_path(G,Player1,Player2) # get the length of shortest path between 2 nodes (d)
        attr_year = nx.get_edge_attributes(G,'Year')
        attr_name = nx.get_edge_attributes(G,'Movie_name')
        print('%s Bacon number to actor %s is %d' %(Player1,Player2,length))
        for edge in range(length):
            try:
                print('At the year of %s, %s and %s Played in a movie called %s' %(str(attr_year[(path[edge],path[edge+1])])[:-2],path[edge],path[edge+1], attr_name[(path[edge],path[edge+1])]))
            except:
                print('At the year of %s, %s and %s Played in a movie called %s' %(str(attr_year[(path[edge+1],path[edge])])[:-2],path[edge],path[edge+1], attr_name[(path[edge+1],path[edge])]))
            #attr_year[(path[edge],path[edge+1])],attr_name[(path[edge],path[edge+1])]

Bacon_number(G,'Brad Pitt', 'Gal Gadot')


# In[ ]:


Bacon_number(G,'Gal Gadot', 'William Shatner')


# #  Goal - find out who are the "new Bacons", or - who are the most linked actors in the industry 

# In[ ]:




