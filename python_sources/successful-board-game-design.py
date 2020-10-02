#!/usr/bin/env python
# coding: utf-8

# This kernel investigates a great dataset from [BoardGameGeek](http://boardgamegeek.com) (which I personally thought needed a little more love) with the goal of identifying trends in factors weighing into the design of more highly rated games. The data consists of over 90k records of games which users of the source site have rated in different categories along with some metrics related to these ratings.

# First let's import the modules we intend to use and implement a script which I have selfishly borrowed from the parent kernel for classifying our columns.

# In[ ]:


import plotly.offline
import plotly.graph_objs as go
import networkx as nx
import numpy as np
import pandas as pd
import sqlite3
import re
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

conn = sqlite3.connect('../input/database.sqlite')

df_games = pd.read_sql_query('SELECT * FROM BoardGames ', conn)

conn.close()

text_columns = []
number_columns = []
other_columns = []

c = list(df_games.columns)

for i in range(df_games.shape[1]):
    if df_games.iloc[:,i].dtype == 'object':
        text_columns.append(c[i])
    elif (df_games.iloc[:,i].dtype == 'float64') or (df_games.iloc[:,i].dtype == 'int64'):
        number_columns.append(c[i])
    else:
        other_columns.append(c[i])
        


# Now let's take a look at the attributes we are dealing with here.

# In[ ]:


print("TEXT COLUMNS:",len(text_columns))
for tcol in text_columns:
    print(tcol)
print('\n')
print("NUMBER COLUMNS:",len(number_columns))
for ncol in number_columns:
    print(ncol)
print('\n')
print("OTHER COLUMNS:",len(other_columns))


# Some other kernels have already investigated correlations among our "number_columns" listed here and have found nothing surprising going on. What I would like to investigate instead is correlation among our "text_columns" as well as any correlation these columns may have with the "stats.bayesaverage" column from the second set. I have chosen this last column because according to the source site it is the most representative of their overall ranking for a game. I will use this as a somewhat naive measure of success for a game.
# 
# The first thing I notice about our "text_columns" is the prevalence of numbers in the "polls.suggested.numplayers..." columns. I would rather not spend time implementing a bunch of NLP algorithms on this stuff if I don't have to, so let's see if there is a way to quickly quantify some the data without losing its integrity. 

# In[ ]:


text_df = df_games[text_columns]
nmplyrs_df = text_df.iloc[:, 18:29]
print(list(nmplyrs_df))
print('\n')
print(nmplyrs_df.head(10))
print(list(nmplyrs_df['polls.suggested_numplayers.1'].unique()))


# It turns out that each of these columns has only four possible values which I will reduce to two: 1 or 0. I am interested in when a given number of players has been  reccomended for a game versus when it hasn't been. It is a safe assumption that the best number of players for a game is also a reccomended number of players. Since I am only interested in affrimative information, I can regard 'NotRecommended' as equivalent to having no data i.e. a value of None.

# In[ ]:


mp_dict = {'Best': 1, np.nan: 0, 'Recommended': 1, 'NotRecommended': 0}
remap_df = nmplyrs_df.replace(mp_dict)


# Now that we have remapped all of the strings and missing values to numbers, we are able to look at correlations within this set of columns as well as with stats.bayesaverage. To view these correlations we create a new dataframe of numerical columns in the next input field from the results of our mapping concatenated with the stats.bayesaverage column. 

# In[ ]:


num_df = df_games[number_columns]
framesCrrctd = (num_df["stats.bayesaverage"], remap_df)
num_dfCrrctd = pd.concat(framesCrrctd, axis = 1)
print(num_dfCrrctd.corr())


# ...And as long as we can, let's go ahead and view these correlations as a heatmap.

# In[ ]:


f,ax = plt.subplots(figsize=(13, 12))
sns.heatmap(num_dfCrrctd.corr(), cmap='YlGnBu', annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# Right away we can see that a game is a bit more likely to be successful if it is designed to play well with 2, 3, or 4 players. Some careful scrutiny of this map reveals that games reccomended for X or Y number of players tend to cluster together for certain values of X and Y. What we really want here in order to see the clustering clearly is a cool little graph which focuses on it specifically.

# First, let's setup some reasonable values for our node labels.

# In[ ]:


def nKey(node):
    se = re.search('(?<=\.)\w+$', node)
    return(se.group(0))
#the regex above reads as:
# "a substring of one or more alphanumeric characters preceded by '.' and followed by the end of the string."
#note that '_' is alphanumeric, but '.' is not.

ndDict = {}
for node in list(num_dfCrrctd):
    ndDict[node] = nKey(node)


# Now we get cracking on our little graph. We will only graph edges for statistically significant correlations, typically .5 or greater but let's make it variable.

# In[ ]:


G = nx.Graph()
G.add_nodes_from(list(num_dfCrrctd))

df_corr = num_dfCrrctd.corr()        # The correlation matrix for our dataframe

nodelist = list(G.nodes)

uniqueEdges = list(combinations(nodelist, 2))
def weight(edge):
    return(df_corr.loc[edge])

threshold = .5                       # Controls significant correlation threshold for when edges will appear

edgelist = []
for edge in uniqueEdges:
    if weight(edge) >= threshold:
         edgelist.append(edge)

G.add_edges_from(edgelist)

edgelist = list(G.edges)             # Ensures that the order of output is static

weightD={}                           # edges and edge weights as a dict
weightE=[]                           # edge weights as a list
for edge in edgelist:
    weightD[edge] = round(weight(edge),3)
    weightE.append(round(weight(edge),3))

poslist = [(1,8),(2,0),(2,8),(1,6),(2,6),(1,4),(2,4),(1,2),(2,2),(1,0),(1,-2),(3,3)]
#these points are contrived for a better image

pos = dict(zip(sorted(nodelist), poslist))
print(pos)

plt.figure(1,figsize=(13,12))
nx.draw_networkx(G, pos, node_color='violet', node_size=350, labels=ndDict)
nx.draw_networkx_edges(G, pos, edgelist=edgelist, cmap=plt.get_cmap('Accent'), edge_color=weightE)
nx.draw_networkx_edge_labels(G, pos, edge_labels=weightD)

plt.axis('off')
plt.show()


# What this graph is showing us is how often a game that is reccomended for X number of players is also reccomended for Y number of players. Would you rather design a game for 5 players or for 4? Would you want to design a game for 5 - 7 players? A lot of sales and marketing insights are gleaned from this graph!   

# *Still more to come!*
