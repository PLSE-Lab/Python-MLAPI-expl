#!/usr/bin/env python
# coding: utf-8

# # Summary
# We have used the python module  **networkx**  to analyze the relation between kaggle users in team competitions. We have found out who are the most connected ones and we have build a user recommender based on the rule: "the friends of my friends are also my friends".
# 

# ## Data loading

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 


# In[ ]:


users_df = pd.read_csv("../input/Users.csv")


# In[ ]:


teams_df = pd.read_csv("../input/Teams.csv", low_memory=False)
teams_df['Score'] = teams_df['Score'].astype(float)


# In[ ]:


user_teams_df = pd.read_csv("../input/TeamMemberships.csv") 


# In[ ]:


user_teams_df.head()


# ## Analysis

# Let's see the size of the teams. Most of them have size 1 but there are a small amout which are very big:

# In[ ]:


teams_size = user_teams_df.groupby("TeamId")['UserId'].count()
teams_size = teams_size.sort_values(ascending=False)


# In[ ]:


teams_size.head()


# In[ ]:


teams_df[teams_df['Id']==9021]


# In[ ]:


teams_size.describe()


# More than 91% of the teams have only 1 participant. 

# In[ ]:


from scipy import stats
stats.percentileofscore(teams_size, 1, kind='weak')


# In[ ]:


plt.figure(figsize=(14, 6))
sns.distplot(teams_size.values)
plt.show()


# In[ ]:


import networkx as nx


# In[ ]:


from itertools import combinations
teams_graph = nx.MultiGraph()
teams_graph.add_nodes_from(user_teams_df['UserId'].unique())


# In[ ]:


len(teams_graph.nodes())


# In[ ]:


grouped = user_teams_df.groupby('TeamId')
for name, group in grouped:
    for n1, n2 in combinations(group['UserId'].values, 2):
        teams_graph.add_edge(n1, n2, TeamId=name)


# In[ ]:


teams_graph.edges(data=True)[:10]


# In[ ]:


len(teams_graph.edges())


# In[ ]:


user_teams_df.loc[user_teams_df['TeamId']==1000, 'UserId'].values


# In[ ]:


teams_graph[811]


# In[ ]:


user_teams_df.loc[user_teams_df['TeamId']==9021, 'UserId'].values


# In[ ]:


user_teams_df.loc[user_teams_df['UserId']==16399]


# We calculate the degrees of centrality to see which are the most important nodes:

# In[ ]:


degrees = {n: len(teams_graph.neighbors(n)) for n in teams_graph.nodes()}


# In[ ]:


max(degrees, key=degrees.get)


# In[ ]:


print("Neighbors: {} Edges: {}".format(degrees[24478], len(teams_graph.edges(24478))))


# The kaggle user most connected in teams with other users is the number *24478*  ([Mariahbarrio](https://www.kaggle.com/Mariahbarrio)) who is connected to *49* users

# In[ ]:


users_df[users_df.Id==24478]


# In[ ]:


deg_cent = nx.degree_centrality(teams_graph)


# In[ ]:


max(deg_cent, key=deg_cent.get)


# The user with higher degree of centrality is *97201* ([jimmykuo](https://www.kaggle.com/jimmykuo)). That's because although he has not so many neighbors as *24478* , he is near and has participated more times with them

# In[ ]:


users_df[users_df.Id==97201]


# In[ ]:


print("Neighbors: {} Edges: {}".format(degrees[97201], len(teams_graph.edges(97201))))


# Let's see the cliques of the graph:

# In[ ]:


cliques = list(nx.find_cliques(teams_graph))


# In[ ]:


len(cliques)


# In[ ]:


max(list(map(len, cliques)))


# We are going to process our previous graph to obtain a non directed and non multigraph graph: each node will be connected with other no more than 1 time (1 edge) but we will add an edge attribute with the times the two nodes (kaggle users) has been in a team together. This way the resulting graph will have less edges and less nodes (only non individual participants are included)

# In[ ]:


from collections import defaultdict
tmp = defaultdict(int)
for n1, n2 in teams_graph.edges():
    tmp[(n1, n2)] += 1


# In[ ]:


G = nx.Graph()
for u, v, d in teams_graph.edges_iter(data=True):
    if G.has_edge(u,v):
        G[u][v]['weight'] += 1
    else:
        G.add_edge(u, v, weight=1)


# In[ ]:


G.edges(data=True)[:10]


# In[ ]:


len(G.nodes())


# In[ ]:


len(G.edges())


# In[ ]:


cliques_2 = list(nx.find_cliques(G))


# In[ ]:


len(cliques)


# In[ ]:


len(G.nodes())


# In this case, the max degree of centrailty is for the user with more neighbors, altough he is not who has been in more teams:

# In[ ]:


deg_cent_2 = nx.degree_centrality(G)
max(deg_cent_2, key=deg_cent_2.get)


# In[ ]:


user_teams_df.loc[user_teams_df['UserId']==24478]


# In[ ]:


user_teams_df.loc[user_teams_df['UserId']==97201]


# In[ ]:


user_teams_df.head()


# These are the users who have participated in more teams (One of the users have participated in 106 teams, although most of the time with the same people):

# In[ ]:


user_teams_df.groupby('UserId')['TeamId'].agg(['count'])     .sort_values(by=['count'], ascending=False).head(20)


# In[ ]:


users_df[users_df.Id==16491]


# In[ ]:


len(G.neighbors(16491))


# ## User recommendation
# We are going to create an user recommendator, similar to user suggested friends. The base of the recommendator will be: if a user "A" has been in a team with "B" and "B" has been in a team with "C", but "A" and "C" have never  been in a team, we will recommend "C" to the user "A". This technique es called "finding open triangles".

# In[ ]:


recommended = defaultdict(int)
for n, d in G.nodes(data=True):
    for n1, n2 in combinations(G.neighbors(n), 2):
        if not G.has_edge(n1, n2):
            recommended[(n1, n2)] += 1


# In[ ]:


# Identify the top pairs of users
all_counts = sorted(recommended.values())
top_pairs = [pair for pair, count in recommended.items() if count > 5]
print(top_pairs)


# In[ ]:


[recommended[e] for e in top_pairs]


# In[ ]:


max_pairs = [pair for pair, count in recommended.items() if count ==21]


# Someone should present these people because they have a lot of contacts in common:

# In[ ]:


max_pairs


# In[ ]:


from IPython.display import HTML
users_df.head()


# In[ ]:


recommend = pd.DataFrame(max_pairs, columns=['user1', 'user2'])

for i in range(len(recommend)):
    tmp_id1 = recommend.loc[i, "user1"]
    tmp_username1 = users_df.loc[users_df.Id==tmp_id1, "DisplayName"].values[0]
    
    tmp_id2 = recommend.loc[i, "user2"]
    tmp_username2 = users_df.loc[users_df.Id==tmp_id2, "DisplayName"].values[0]    

    recommend.loc[i, "user1"] = "<a target='_blank' href='https://www.kaggle.com/u/"         + str(tmp_id1) + "'>" + tmp_username1 + "<" + "/a>"
    recommend.loc[i, "user2"] = "<a target='_blank' href='https://www.kaggle.com/u/"         + str(tmp_id2) + "'>" + tmp_username2 + "<" + "/a>"        


# In[ ]:


pd.set_option("display.max_colwidth", -1)
HTML(recommend.to_html(escape=False))


# Let's try subgraphs. We are going to build a subgraph of the neighbors of the user 24478 (the one with most neighbors) and himself

# In[ ]:


G_maxconnected = G.subgraph(G.neighbors(24478) + [24478])


# In[ ]:


import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(12,8))
nx.draw(G_maxconnected)
plt.show()


# It looks like that the user 24478 connects two different cliques: 

# In[ ]:


cliques_2 = list(nx.find_cliques(G_maxconnected))


# In[ ]:


len(cliques_2)

