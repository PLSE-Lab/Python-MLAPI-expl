#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import datetime as dt
import os
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
import seaborn as sns
sns.set_palette(sns.color_palette('tab20', 20))
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from igraph import Graph
import urllib.request
from tqdm import tqdm
from collections import Counter
from itertools import combinations 


# In[ ]:


pd.set_option('display.max_columns', 99)
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14


# In[ ]:


class MetaData():
    def __init__(self, path='../input'):
        self.path = path

    def Competitions(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'Competitions.csv'), nrows=nrows)
        return df.rename(columns={'Id': 'CompetitionId'})

    def TeamMemberships(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'TeamMemberships.csv'), nrows=nrows)

    def Teams(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'Teams.csv'), nrows=nrows)
        return df.rename(columns={'Id': 'TeamId'})

    def Users(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'Users.csv'), nrows=nrows)
        return df.rename(columns={'Id': 'UserId'})

    def PerformanceTiers(self):
        df = pd.DataFrame([
            [0, 'Novice', '#5ac995'],
            [1, 'Contributor', '#00BBFF'],
            [2, 'Expert', '#95628f'],
            [3, 'Master', '#f96517'],
            [4, 'GrandMaster', '#dca917'],
            [5, 'KaggleTeam', '#008abb'],
        ], columns=['PerformanceTier', 'PerformanceTierName', 'PerformanceTierColor'])
        return df


# ## Users
# Kaggle has almost 3M registered users.

# In[ ]:


start = dt.datetime.now()
md = MetaData()

users = md.Users()
tiers = md.PerformanceTiers()
users = users.merge(tiers, on='PerformanceTier')
users.shape
users.head()


# ## Competitions
# 
# There were almost 300 competitions (Inclass excluded) hosted on kaggle.

# In[ ]:


competitions = md.Competitions()
competitions = competitions[competitions.HostSegmentTitle != 'InClass']
competitions['DeadlineDate'] = pd.to_datetime(competitions.DeadlineDate)
competitions = competitions.sort_values(by='DeadlineDate')
competitions.shape
competitions.tail(2)


# ## Teams

# In[ ]:


teams = md.Teams()
teams.shape
teams.head(2)
members = md.TeamMemberships()
members.shape
members.head(2)


# In[ ]:


real_teams = members.groupby('TeamId')[['Id']].count().reset_index()
real_teams = real_teams[real_teams.Id > 1]
real_teams.columns = ['TeamId', 'TeamSize']
real_teams.shape
real_teams.head()
real_teams.TeamSize.sum(), real_teams.TeamSize.mean()

team_members = members.merge(real_teams, on='TeamId')
team_members.shape
team_members.head(3)


# ## Actual team members
# Most of the registered users did not participate in competitions. Even if they did they are likely to had only a few submission for a single competition. Only a small portion of users actually teamed up.

# In[ ]:


YOUR_USER_ID = 18102


# In[ ]:


user_ids = team_members.UserId.unique()
user_ids = sorted(user_ids)
len(user_ids)


# In[ ]:


user_edges = {u: Counter() for u in user_ids}
for team_id, df in tqdm(team_members.groupby('TeamId')):
    for a, b in combinations(df.UserId.values, 2):
        user_edges[a][b] += 1
        user_edges[b][a] += 1


# In[ ]:


degrees = [[u, len(cnt)] for u, cnt in user_edges.items()]
degree_df = pd.DataFrame(degrees, columns=['UserId', 'TeamMateDegree'])
degree_df = degree_df.merge(users, on='UserId', how='left')
degree_df = degree_df.sort_values(by='TeamMateDegree', ascending=False)
degree_df['VertexId'] = np.arange(len(degree_df))
degree_df.shape
degree_df.head(20)
degree_df = degree_df.fillna({'PerformanceTierColor': 'grey'})
degree_df = degree_df.fillna('')


# ## Construct the graph

# In[ ]:


vertex_map = {u: v for u, v in degree_df[['UserId', 'VertexId']].values}


# In[ ]:


edges = []
edge_weights = []
for u1, cnt in tqdm(user_edges.items()):
    for u2, weight in cnt.items(): 
        if u1 < u2:
            edges.append((vertex_map[u1], vertex_map[u2]))
            edge_weights.append(weight)


# In[ ]:


g = Graph()
g.add_vertices(len(user_ids))
g.add_edges(edges)
g.vs['UserId'] = degree_df['UserId'].values
g.vs['UserName'] = degree_df['UserName'].values
g.es['Weight'] = edge_weights
g.summary()


# In[ ]:


clusters = g.components(mode='WEAK')
for sg in clusters.subgraphs():
    if sg.vcount() > 50:
        sg.summary()
for sg in clusters.subgraphs():
    if sg.vcount() > 1000:
        large_cluster = sg


# In[ ]:


graph_layout = large_cluster.layout('fr')
coords = np.array(graph_layout.coords)
large_user_coords = pd.DataFrame(coords, columns=['x', 'y'])
large_user_coords['UserId'] = large_cluster.vs['UserId']

large_user_coords = large_user_coords.merge(degree_df, on='UserId')


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 9))
plt.scatter(large_user_coords.x, large_user_coords.y, s=np.sqrt(large_user_coords.TeamMateDegree) * 5,
            c=large_user_coords.PerformanceTierColor.values, alpha=0.5)
for e in tqdm(large_cluster.es[:]):
    u1, u2 = e.source, e.target
    plt.plot([large_user_coords.loc[u1, 'x'], large_user_coords.loc[u2, 'x']],
             [large_user_coords.loc[u1, 'y'], large_user_coords.loc[u2, 'y']], 'grey', alpha=0.02)
plt.title('Kaggle users - Giant Component')
plt.axis('off')
fig.savefig('users.png', dpi=600, figsize=(16, 10))
plt.show();


# In[ ]:


p = 20
center_coords = large_user_coords.copy()
center_coords = center_coords[center_coords.x > np.percentile(center_coords.x, 50 - p)]
center_coords = center_coords[center_coords.x < np.percentile(center_coords.x, 50 + p)]
center_coords = center_coords[center_coords.y > np.percentile(center_coords.y, 50 - p)]
center_coords = center_coords[center_coords.y < np.percentile(center_coords.y, 50 + p)]
center_coords.shape


# In[ ]:


data = []
for e in tqdm(large_cluster.es[:]):
    u1, u2 = e.source, e.target
    if u1 in center_coords.index and u2 in center_coords.index:
        trace = go.Scatter(
            x = [center_coords.loc[u1, 'x'], center_coords.loc[u2, 'x']],
            y = [center_coords.loc[u1, 'y'], center_coords.loc[u2, 'y']],
            mode = 'lines',
            line=dict(color='grey', width=1),
            opacity=0.1
        )
        data.append(trace)
data.append(
    go.Scatter(
        y = center_coords['y'],
        x = center_coords['x'],
        mode='markers',
        marker=dict(sizemode='diameter',sizeref=1,
                    opacity=0.5,
                    size=np.log(center_coords.TeamMateDegree + 1)*2,
                    color=center_coords.PerformanceTierColor.values),
        text=center_coords.DisplayName,
        hoverinfo = 'text',
    )
)
layout = go.Layout(
    autosize=True,
    title='Center of the top cluster',
    hovermode='closest',
    yaxis = dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),
    xaxis = dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='TopCluster')


# In[ ]:


end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))

