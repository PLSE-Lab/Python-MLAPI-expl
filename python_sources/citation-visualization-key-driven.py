#!/usr/bin/env python
# coding: utf-8

# I started out on this project right when this task was released with aspirations to build an visualization tool that could be fed specific papers that might answer some of the relevant questions and see how these papers were connected to other relevant papers. I thought this might be a decent tool as many academic papers typically reference other works that they extended. Seeing how these papers are connected might highlight themes that are more widely accepted in this community. 

# In[ ]:


# Import Stuff
import plotly.graph_objects as go
import networkx as nx
import os
from tqdm import tqdm
import json
import numpy as np
import plotly.offline as py
py.init_notebook_mode(connected=True)


# In[ ]:


keywords = ['surveillance'] ### CHANGE THESE WORDS TO PRODUCE DIFFERENT GRAPH RESULTSd


# In[ ]:


# Where are the papers to consider? You can add more to the list, but it slows it down. 
paper_dirs = ['../input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json', '../input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json']


# In[ ]:


G = nx.DiGraph() # Our graph that we'll visualize in a bit. 


total = 0
    
    # Add all nodes
for dir in tqdm(paper_dirs):
    files = os.listdir(dir)
    total += len(files)
    for file in files:
        with open(os.path.join(dir, file)) as json_data:
            data = json.load(json_data)
            if data['abstract']:
                if any(word in data['abstract'][0]['text'] for word in keywords):
                    G.add_node(data['metadata']['title'])
    
for dir in tqdm(paper_dirs):
    files = os.listdir(dir)
    for file in files:
        with open(os.path.join(dir, file)) as json_data:
            data = json.load(json_data)
            main_node = data['metadata']['title']

            bibliography = data['bib_entries']

            for bib in bibliography:
                possible_node = bibliography[bib]['title']
                if possible_node in G.nodes.keys():
                    G.add_edge(possible_node, main_node)


# In[ ]:


pos = nx.spring_layout(G)
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node_names = np.asarray(G.nodes)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
    showscale=True,

    colorscale='YlGnBu',
    reversescale=True,
    color=[],
    size=10,
    colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
            ),
        line_width=2))

node_adjacencies = []
node_text = []
for node, adjacencies in enumerate(G.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    node_text.append(f'{node_names[node]} : ' + str(len(adjacencies[1])))

node_trace.marker.color = node_adjacencies
node_trace.text = node_text

fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Network graph made with Python',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )


# In theory the following plot should let you surf to different papers quickly to see which ones are connected. The papers in the middle have the most references, so it may be worth checking them out first. When hovering, you can see the paper title followed by its citation count. 

# In[ ]:


py.iplot(fig)

