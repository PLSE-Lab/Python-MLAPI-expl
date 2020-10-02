#!/usr/bin/env python
# coding: utf-8

# # Network Graph with plotly and networkx

# ## I recently had to create some network graphs for one of my projects and I realised how nice and interesting they are. So i decided to create a kernel with the AT&T dataset. Check out the dataset [here](https://www.kaggle.com/latentheat/att-network-data)

# ## We go through 6 steps to create a network graph.
# 1. [Preparing the data](#prep)
# 2. [Creating the graph](#graph)
# 3. [Get node positions](#pos)
# 4. [Add nodes and edges to plotly API](#plotly)
# 5. [Color the nodes](#adj)
# 6. [Plot the figure](#final)

# <h2 id="prep">Preparing the data</h2>

# In[ ]:


import pandas as pd
import numpy as np
import networkx as nx
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import plotly.graph_objs as go


# In[ ]:


init_notebook_mode(connected=True)


# In[ ]:


network_df = pd.read_csv("../input/network_data.csv") 


# In[ ]:


network_df.head()


# #### source_ip and destination_ip are of our interest here. so we isolate them. We then get the unique ip addresses for getting the total number of nodes. We do this by taking unique values in both columns and joining them together.

# In[ ]:


A = list(network_df["source_ip"].unique())


# In[ ]:


B = list(network_df["destination_ip"].unique())


# In[ ]:


node_list = set(A+B)


# <h2 id="graph">Creating the graph</h2>

# In[ ]:


G = nx.Graph()


# ### Graph api to create an empty graph. And the below cells we will create nodes and edges and add them to our graph

# In[ ]:


for i in node_list:
    G.add_node(i)


# ##### Uncomment the below code to see the nodes list. 

# In[ ]:


#G.nodes()


# In[ ]:


for i,j in network_df.iterrows():
    G.add_edges_from([(j["source_ip"],j["destination_ip"])])


# <h2 id="pos">Getting positions for each node.</h2>

# In[ ]:


pos = nx.spring_layout(G, k=0.5, iterations=50)


# ##### Uncomment below code to see the positions of all the nodes

# In[ ]:


#pos


# ### Adding positions of the nodes to the graph

# In[ ]:


for n, p in pos.items():
    G.node[n]['pos'] = p


# <h2 id="plotly">Adding nodes and edges to the plotly api</h2>

# In[ ]:


edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=0.5,color='#888'),
    hoverinfo='none',
    mode='lines')

for edge in G.edges():
    x0, y0 = G.node[edge[0]]['pos']
    x1, y1 = G.node[edge[1]]['pos']
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])


# In[ ]:


node_trace = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='RdBu',
        reversescale=True,
        color=[],
        size=15,
        colorbar=dict(
            thickness=10,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line=dict(width=0)))

for node in G.nodes():
    x, y = G.node[node]['pos']
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])


# <h2 id="adj">Coloring nodes</h2>

# #### Coloring based on the number of connections of each node. 

# In[ ]:


for node, adjacencies in enumerate(G.adjacency()):
    node_trace['marker']['color']+=tuple([len(adjacencies[1])])
    node_info = adjacencies[0] +' # of connections: '+str(len(adjacencies[1]))
    node_trace['text']+=tuple([node_info])


# <h2 id="final">Plotting the figure</h2>

# In[ ]:


fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>AT&T network connections',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="No. of connections",
                    showarrow=False,
                    xref="paper", yref="paper") ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

iplot(fig)
plotly.plot(fig)


# In[ ]:


from plotly.plotly import plot


# In[ ]:


from plotly import plotly


# In[ ]:


import plotly


# In[ ]:


plotly.tools.set_credentials_file(username='anand0427', api_key='5Xd8TlYYqnpPY5pkdGll')


# In[ ]:


iplot(fig,"anand0427",filename="Network Graph.html")


# a# References 
# [Plotly Network Graph Example](https://plot.ly/python/network-graphs/)
# 
# [Networkx Docs](https://networkx.github.io/documentation/networkx-1.9.1/)

# ### Thats a nice plot! So we reached the end of this kernel! Let me know if you have any doubts or what you think :)
