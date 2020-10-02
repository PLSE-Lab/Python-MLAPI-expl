#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import networkx as nx
from PIL import Image
import matplotlib as plt
import pandas as pd
import numpy as np


# In[ ]:


## We first import our adjacency matrix as a DataFrame. 


# In[ ]:


df = pd.read_csv('../input/adjacency_matrix_the_wire.csv', index_col='Character')


# In[ ]:


# We also load the files characters.csv and characters2.csv, which simply contain the information regarding where the characters belong to in the series (The Law, The Streets, Politicians, etc.). I took this characters' division from Wikipedia and thought it could be useful to compare the centrality of characters with those that belong to the same world. This information is in separate files because it would not fit in the adjacency matrix itself.
characters = pd.read_csv('../input/characters.csv')
characters2 = pd.read_csv('../input/characters2.csv')


# In[ ]:


## Now we import the adjacency matrix to NetworkX for doing our analysis
Graphtype = nx.Graph()
G = nx.from_pandas_adjacency(df)


# In[ ]:


print(nx.info(G))


# In[ ]:


## So, we have a total of 65 characters and 298 connections between them. Quite a lot for a fictional series, very impressive storytelling indeed.


# # **Degree of Characters**
# #### Below we will calculate the Degree of each character. That is simply the number of connections each of them have.

# In[ ]:


degrees = np.array(nx.degree(G))


# In[ ]:


Characters_Degree = pd.DataFrame(degrees)


# In[ ]:


Characters_Degree.columns = ['Character', 'Degree']


# In[ ]:


Characters_Degree['Degree'] = Characters_Degree['Degree'].astype(int)


# In[ ]:


Characters_Degree['From'] = characters['From']


# In[ ]:


Characters_Degree.set_index('Character', inplace=True)


# In[ ]:


Characters_Degree.sort_values(by='Degree', ascending=False).head()


# # Calculation of Centrality of Characters
# #### Below we will calculate 4 different kinds of centrality measures for the characters:
# #### 1) Degree Centrality
# #### 2) Betweenness Centrality
# #### 3) Closeness Centrality
# #### 4) PageRank Centrality
# 
# #### Later on, when analyzing the results, we will talk about each of these 4 centrality measures, in which ways they differ, and explain why different characters appear as more 'central' or 'powerful' when using different of these measures.

# ## Betweenness Centrality of Characters

# In[ ]:


betweenness = nx.betweenness_centrality(G)


# In[ ]:


Characters_Betweenness = pd.DataFrame.from_records([betweenness], index=[0])


# In[ ]:


Characters_Betweenness = Characters_Betweenness.transpose()


# In[ ]:


Characters_Betweenness.columns = ['Betweenness_Centrality']


# In[ ]:


Characters_Betweenness.index.rename('Character', inplace=True)


# ## Degree Centrality of Characters

# In[ ]:


centrality = nx.degree_centrality(G)


# In[ ]:


Characters_Centrality = pd.DataFrame.from_records([centrality], index=[0])


# In[ ]:


Characters_Centrality = Characters_Centrality.transpose()


# In[ ]:


Characters_Centrality.columns = ['Degree_Centrality']


# In[ ]:


Characters_Centrality.index.rename('Character', inplace=True)


# ## Closeness Centrality of Characters

# In[ ]:


closeness = nx.closeness_centrality(G)


# In[ ]:


Closeness_Centrality = pd.DataFrame.from_records([closeness], index=[0])


# In[ ]:


Closeness_Centrality = Closeness_Centrality.transpose()


# In[ ]:


Closeness_Centrality.columns = ['Closeness_Centrality']


# In[ ]:


Closeness_Centrality.index.rename('Character', inplace=True)


# ## PageRank Centrality of Characters

# In[ ]:


pagerank = nx.pagerank(G)


# In[ ]:


PageRank_Centrality = pd.DataFrame.from_records([pagerank], index=[0])


# In[ ]:


PageRank_Centrality = PageRank_Centrality.transpose()


# In[ ]:


PageRank_Centrality.columns = ['PageRank_Centrality']


# In[ ]:


PageRank_Centrality.index.rename('Character', inplace=True)


# ## Merging all centrality measures in one DataFrame

# In[ ]:


TheWire = pd.concat([Characters_Degree, Characters_Betweenness, Characters_Centrality, Closeness_Centrality, PageRank_Centrality], axis=1)


# In[ ]:


cols = ['From', 'Degree', 'Degree_Centrality', 'Betweenness_Centrality', 'Closeness_Centrality', 'PageRank_Centrality']


# In[ ]:


TheWire = TheWire[cols]


# In[ ]:


TheWire.head()


# # The Law Characters - Centrality Measures

# In[ ]:


The_Law = TheWire[TheWire['From']=='The Law']


# ## Characters from 'The Law' with the highest Degree Centrality:
# 

# In[ ]:


The_Law.sort_values(by='Degree_Centrality', ascending=False, inplace=True)
The_Law[['Degree_Centrality', 'Degree', 'Betweenness_Centrality', 'Closeness_Centrality', 'PageRank_Centrality']]


# #### Degree Centrality is the simplest of the centrality measurements. It simply accounts for the number of connections an actor has. We see that McNulty is the most central Law-related character according to this measurement.

# ## Characters from 'The Law' with the highest Betweenness Centrality:

# In[ ]:


The_Law.sort_values(by='Betweenness_Centrality', ascending=False, inplace=True)
The_Law[['Betweenness_Centrality', 'Degree', 'Degree_Centrality', 'Closeness_Centrality', 'PageRank_Centrality']]


# #### Betweenness Centrality measures the number of times an actor lies on the shortest path between other actors. It shows which actors act as "bridges" within the network, and it is a great metric to analyze who is in a position to influence everyone in the network.  We see that McNulty is again first in the list, but some interesting conclusions appear when looking at the rest of the top 5. We see that Maurice Levy, despite having a pretty low degree (not a lot of connections) is second on this list. An explanation for this may be that he can act as a 'bridge' connecting drug-dealers from different sides, and also connecting drug-dealers with Law Enforcement characters.
# #### We also see that Beadie Russell is third on the list, despite not having many connections and being a secondary character on the series. But this also has a reasonable explanation: Beadie is the only one who an act as a "bridge" between all the Police characters and the Port characters (Sobotkas, etc.). Moreover, Beadie is very close to McNulty, so she can also lie in the shortest path between any Port character and any of the characters that McNulty knows (which are a lot).

# ## Characters from 'The Law' with the highest Closeness Centrality:

# In[ ]:


The_Law.sort_values(by='Closeness_Centrality', ascending=False, inplace=True)
The_Law[['Closeness_Centrality', 'Degree', 'Degree_Centrality', 'Betweenness_Centrality', 'PageRank_Centrality']]


# #### Closeness Centrality scores each node according to how "close" they are to all other actors in the network. McNulty is first as usual, so it is becoming obvious that he may be the most central characters in the series. But again, we see some interesting things in the top 5 below him. Now Herc appears as the second most central character, who would have thought? An explanation for this is that along the series he forms close connections with Mayor Clarence Royce, and later on with Maurice Levy. Added to the connections he has with fellow Policemen, then he is basically all over the place and has bonds with all the different "worlds" within the series. Therefore, he is close to everyone else. We see that Levy again scores very high in this kind of centrality measure as well.

# ## Characters from 'The Law' with the highest PageRank Centrality:

# In[ ]:


The_Law.sort_values(by='PageRank_Centrality', ascending=False, inplace=True)
The_Law[['PageRank_Centrality', 'Degree', 'Degree_Centrality', 'Betweenness_Centrality', 'Closeness_Centrality']]


# #### PageRank Centrality refers to the famous algorithm behind Google. It is similar to Degree Centrality, in the way that it scores each actor according to the number of connections they have. But, unlike Degree Centrality, it also takes into account how well connected those connections are. So, not all connections hold the same value. The most interesting thing in the results is the appearance of Lester Freamon (who was nowhere close the most central before) in the third position.

# ## 'The Law' Characters Centrality - Conclusions
# #### It is concluded that McNulty is by far the most central law-enforcement character in the series. This is not surprising: after all, he is kind of the main character in the series, and he gets around everything and knows a lot of people. He doesn't care at all about the chain of command thing. We also see that depending on the centrality measure we choose, we get different characters as the most central behind him. According to the number or quality of connections, Daniels seems to be the second most central character. But when looking at how much influence characters have according to how they can act as bridges among other characters, Maurice Levy is the second most important one.

# # The Street Characters - Centrality Measures

# In[ ]:


The_Street = TheWire[TheWire['From']=='The Street']


# ## Characters from 'The Streets' with the highest Degree Centrality:

# In[ ]:


The_Street.sort_values(by='Degree_Centrality', ascending=False, inplace=True)
The_Street[['Degree_Centrality', 'Degree', 'Betweenness_Centrality', 'Closeness_Centrality', 'PageRank_Centrality']]


# #### We see that Avon is the street-level character who is most connected, followed by Stringer Bell and Prop Joe.

# ## Characters from 'The Street' with the highest Betweenness Centrality:

# In[ ]:


The_Street.sort_values(by='Betweenness_Centrality', ascending=False, inplace=True)
The_Street[['Betweenness_Centrality', 'Degree', 'Degree_Centrality', 'Closeness_Centrality', 'PageRank_Centrality']]


# #### Interestingly, we find Boadie as the the character who has the most influence when looking at the betweenness measure. This may be explained by the fact that he works both with the Barksdale crew and the Marlo crew (even though in reality this happens at two different points in time!). Also, he forms a light bond with some policemen as well, like McNulty, which makes him a bridge between the streets characters and the law-enforcement ones.

# ## Characters from 'The Street' with the highest Closeness Centrality:

# In[ ]:


The_Street.sort_values(by='Closeness_Centrality', ascending=False, inplace=True)
The_Street[['Closeness_Centrality', 'Degree', 'Degree_Centrality', 'Betweenness_Centrality', 'PageRank_Centrality']]


# #### Boadie is again the street-level character who is closest to everyone else in the series when looking at these measures. Surprisingly, we also see Omar here, even though he only has 7 connections in total. This may be because he is connected to both McNulty and Bunk, two very central characters.

# ## Characters from 'The Street' with the highest PageRank Centrality:

# In[ ]:


The_Street.sort_values(by='PageRank_Centrality', ascending=False, inplace=True)
The_Street[['PageRank_Centrality', 'Degree', 'Degree_Centrality', 'Betweenness_Centrality', 'Closeness_Centrality']]


# #### Nothing very revealing here, as the ranking is pretty much the same as the Degree Centrality ranking. The only interesting thing is that Boadie drops considerably. Even though he has 13 connections (more than Marlo Stanfield) he is behind Marlo in this measure. This is because some of Boadie's strongest connections are pretty useless (like Poot or Wallace) and don't add him much.

# ## 'The Streets' Characters Centrality - Conclusions
# > #### It is concluded that Avon, Stringer, and Prop Joe are the street-level characters who are most connected. But Boadie is the one who is best connected when looking at how he can act as bridge between characters of different places in the network.

# # Politicians Characters - Centrality Measures

# In[ ]:


Politicians = TheWire[TheWire['From']=='Politicians']


# ## Characters from 'Politicians' with the highest Degree Centrality:

# In[ ]:


Politicians.sort_values(by='Degree_Centrality', ascending=False, inplace=True)
Politicians[['Degree_Centrality', 'Degree', 'Betweenness_Centrality', 'Closeness_Centrality', 'PageRank_Centrality']]


# #### Tommy Carcetti is by far the politician with the highest number of connections in the series. We see that a very secondary character like Odell Watkins is surprisingly second.

# ## Characters from 'Politicians' with the highest Betweenness Centrality:

# In[ ]:


Politicians.sort_values(by='Betweenness_Centrality', ascending=False, inplace=True)
Politicians[['Betweenness_Centrality', 'Degree', 'Degree_Centrality', 'Closeness_Centrality', 'PageRank_Centrality']]


# #### As usual, the betweenness measure gives us very meaningful information. It seems that despite having only 8 connections (a bit over half of the connections Carcetti has), Clay Davis is by very far the character with the most influence when looking at this measure ("Shiiiiiiiiit"). He can act as a bridge between very different parts of the network, as he forms bonds with people like Stringer Bell or Ervin Burrel alike.

# ## Characters from 'Politicians' with the highest Closeness Centrality:

# In[ ]:


Politicians.sort_values(by='Closeness_Centrality', ascending=False, inplace=True)
Politicians[['Closeness_Centrality', 'Degree', 'Degree_Centrality', 'Betweenness_Centrality', 'PageRank_Centrality']]


# #### Clay Davis is also the politician who is closest to every other actor in the network. He manages to achieve this with only 8 connections, which is remarkable.

# ## Characters from 'Politicians' with the highest PageRank Centrality:

# In[ ]:


Politicians.sort_values(by='PageRank_Centrality', ascending=False, inplace=True)
Politicians[['PageRank_Centrality', 'Degree', 'Degree_Centrality', 'Betweenness_Centrality', 'Closeness_Centrality']]


# ## Politicians Characters Centrality - Conclusions
# #### It is concluded that Carcetti is the character with the highest number of connections among the Politicians, followed by Odell Watkins. However, Clay Davis steals the show here, as with only 8 connections he is definitely the most powerful one.

# In[ ]:




