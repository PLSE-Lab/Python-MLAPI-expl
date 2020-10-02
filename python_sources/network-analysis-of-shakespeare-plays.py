#!/usr/bin/env python
# coding: utf-8

# data analyst data scientist**strong text**

# ***Data Analysis*

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import networkx as nx
from subprocess import check_output
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system('pip install seaborn')
get_ipython().system('pip install networkx')


# In[ ]:


print(check_output(["ls", "../"]).decode("utf8"))


# In[ ]:





# In[ ]:


data =pd.read_csv('../input/Shakespeare_data.csv')


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.columns


# In[ ]:


data.describe()


# In[ ]:


data['Player'].replace(np.nan, 'Other',inplace = True)


# In[ ]:


data.head()


# In[ ]:


print("Number of plays are: " + str(data['Play'].unique()))


# In[ ]:


print("Number of plays are: " + str(data['Play'].nunique()))


# In[ ]:


pd.DataFrame(data['Play'].unique().tolist(), columns=['Play Name'])


# In[ ]:


data['Play'].unique().tolist()


# In[ ]:


pd.DataFrame(data['Play'].unique().tolist(),columns=['Play List'])


# In[ ]:


data.groupby(['Play'])['Player'].nunique().sort_values(ascending= False).to_frame().head(2)


# In[ ]:


data.groupby(['Play']).get_group('Richard III').head(2)


# In[ ]:


numberPlayers = data.groupby(['Play'])['Player'].nunique().sort_values(ascending= False).to_frame()
numberPlayers.index


# In[ ]:


numberPlayers.index.tolist()


# In[ ]:


numberPlayers = data.groupby(['Play'])['Player'].nunique().sort_values(ascending= False).to_frame()
numberPlayers['Play'] = numberPlayers.index.tolist()
numberPlayers.columns = ['Num Players','Play']
numberPlayers.index= np.arange(0,len(numberPlayers))
#numberPlayers

plt.figure(figsize=(10,10))
ax = sns.barplot(x='Num Players',y='Play',data=numberPlayers)
ax.set(xlabel='Number of Players', ylabel='Play Name')
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
ax = sns.barplot(x='Num Players',y='Play',data=numberPlayers)
ax.set(xlabel='Number of Players', ylabel='Play Name')
plt.show()


# In[ ]:


data.groupby('Play').count().sort_values(by='Player-Line',ascending=False)['Player-Line']


# In[ ]:


data.groupby('Play').count()


# In[ ]:


data.groupby('Play').count().sort_values(by='Player-Line',ascending=True)


# In[ ]:


data.groupby('Play').count().sort_values(by='Player-Line',ascending=True)


# In[ ]:


data.groupby('Play').count().sort_values(by='Player-Line',ascending=True)['Player-Line']


# In[ ]:


data.groupby('Play').count().sort_values(by='Player-Line',ascending=True)['Player-Line']


# In[ ]:


play_data = data.groupby('Play').count().sort_values(by='Player-Line',ascending=False)['Player-Line']
play_data = play_data.to_frame()
print(play_data)


# In[ ]:


play_data['Play'] = play_data.index.tolist()
print(play_data['Play'])


# In[ ]:


play_data.index = np.arange(0,len(play_data)) #changing the index from plays to numbers
play_data.columns =['Lines','Play']
play_data


# In[ ]:


numberPlayers = data.groupby(['Play'])['Player'].nunique().sort_values(ascending= False).to_frame()
numberPlayers['Play'] = numberPlayers.index.tolist()
numberPlayers.columns = ['Num Players','Play']
numberPlayers.index= np.arange(0,len(numberPlayers))
numberPlayers

plt.figure(figsize=(10,10))
ax = sns.barplot(x='Num Players',y='Play',data=numberPlayers)
ax.set(xlabel='Number of Players', ylabel='Play Name')
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
ax= sns.barplot(x='Lines',y='Play',data=play_data, order = play_data['Play'])
ax.set(xlabel='Number of Lines', ylabel='Play Name')
plt.show()


# In[ ]:


data.groupby(['Play','Player']).count()['Player-Line']


# In[ ]:


lines_per_player= data.groupby(['Play','Player']).count()['Player-Line']
lines_per_player= lines_per_player.to_frame()
lines_per_player


# In[ ]:


play_name = data['Play'].unique().tolist()
for play in play_name:
    p_line = data[data['Play']==play].groupby('Player').count().sort_values(by='Player-Line',ascending=False)['Player-Line']
    p_line = p_line.to_frame()
    p_line['Player'] = p_line.index.tolist()
    p_line.index = np.arange(0,len(p_line))
    p_line.columns=['Lines','Player']
    plt.figure(figsize=(10,10))
    ax= sns.barplot(x='Lines',y='Player',data=p_line)
    ax.set(xlabel='Number of Lines', ylabel='Player')
    plt.title(play,fontsize=30)
    plt.show()


# In[ ]:


g= nx.Graph()
g


# In[ ]:


g = nx.from_pandas_dataframe(data,source='Play',target='Player')


# In[ ]:


g


# In[ ]:


print (nx.info(g))


# In[ ]:


plt.figure(figsize=(40,40)) 
nx.draw_networkx(g,with_labels=True,node_size=100)
plt.show()


# In[ ]:


centralMeasures = pd.DataFrame(nx.degree_centrality(g),index=[0]).T
centralMeasures.columns=['Degree Centrality']
centralMeasures['Page Rank']= pd.DataFrame(nx.pagerank(g),index=[0]).T
centralMeasures['Name']= centralMeasures.index.tolist()
centralMeasures.index = np.arange(0,len(centralMeasures))
centralMeasures


# In[ ]:


centralMeasures[centralMeasures['Name'].isin(data['Player'].unique().tolist())].sort_values(by='Degree Centrality',ascending=False)


# In[ ]:


#Centrality measures only for players (or actors)
centralMeasures[centralMeasures['Name'].isin(data['Player'].unique().tolist())].sort_values(by='Page Rank',ascending=False)


# In[ ]:


#number of nodes that the "Messenger" is connected to across all plays
len(g.neighbors('Messenger'))


# In[ ]:


#getting the number of lines a messanger spoke across all the plays
data[data['Player']=='Messenger']['Player-Line'].count()


# In[ ]:


centralMeasures[centralMeasures['Name'].isin(data['Play'].unique().tolist())].sort_values(by='Degree Centrality',ascending=False)


# In[ ]:


centralMeasures[centralMeasures['Name'].isin(data['Play'].unique().tolist())].sort_values(by='Page Rank',ascending=False)


# In[ ]:


#number of nodes that "Richard III" is connected to
len(g.neighbors('Richard III'))


# In[ ]:


print("hadoop")


# In[ ]:





# In[ ]:





# In[ ]:




