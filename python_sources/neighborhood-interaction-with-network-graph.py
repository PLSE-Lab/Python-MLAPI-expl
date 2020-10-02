#!/usr/bin/env python
# coding: utf-8

# This is a notebook visualizing the neighborhood interactions. Here the concept **'interaction'** represents the **numbers of trips between two neighborhoods**. I extracted the neighborhood and boro information using QGIS ([https://www.kaggle.com/c/nyc-taxi-trip-duration/discussion/38220](http://)). For network graph, I used the package plotly and referred to this tutorial [plot.ly/python/network-graphs/](http://). To sum up, in order to plot the network graph, I first calculated the numbers of trips between each neighborhood pairs, then identified those neighborhood pairs with the numbers of trip that are over a certain value as "having high interactions" (which means, will be shown the edge between them on the network graph).

# # Load the packages

# In[ ]:


import plotly.offline as pyo
import plotly.plotly as py
from plotly.graph_objs import *
import pandas as pd
import plotly
plotly.offline.init_notebook_mode()
from scipy import signal
pyo.offline.init_notebook_mode()
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.plotly as py
from plotly.graph_objs import *


# # Read the neighborhood files

# This file contains information of the 195 neighborhoods in NYC. The longitudes and latitudes are the approximate centroids I calculated for each neighborhood using part of the data from train set (It doesn't need to be precise since we just want to show the approximate location of each neighborhood.

# In[ ]:


nb=pd.read_csv("../input/tianyi-datasets/neighborhood graph.csv")
nb.head()


# <b> What's this file for ? </b> <br> This is like a dictionary for all the neighborhoods so that we can query the location (to place the point on the scatter plot) for a certain neighborhood. You can also build the dictionary directly in Python. 

# Let's first try to visulize all the neighborhoods.

# In[ ]:


boros=list(nb['boro'].unique())
boros


# In[ ]:


trace=[]
for boro in boros:
    trace.append({'type':'scatter',
                  'mode':'markers',
                  'y':nb.loc[nb['boro']==boro,'latitude'],
                  'x':nb.loc[nb['boro']==boro,'longitude'],
                  'name':boro,
                  'marker':{'size':10,'opacity':0.7,
                            'line':{'width':1.25,'color':'black'}}})
layout={'title':'NYC in an interesting view',
       'xaxis':{'title':'latitude'},
       'yaxis':{'title':'longitutde'}}
fig=Figure(data=trace,layout=layout)
pyo.iplot(fig)


# It's looks reasonable. 

# # Read the files for the information of edges and nodes for the network graph

# ### 1. Ignore the direction

# Let's first not distinguish the direction of the trip(e.g. from A to B or from B to A). In this case, we basically consider the "interaction" between two neighborhoods.

# In[ ]:


data=pd.read_csv("../input/tianyi-datasets/nodes ignore direction.csv")
data.head()


# ### some functions

# We will define some functions here for our network graph.

# <b>to get the edge</b>: Well I'm not sure if I get the concept right but here I will get all the pairs of neighborhood which have very high interaction level, which means, the numbers of trips between those neighborhood should surpass a certain level (which is threshold in the function)

# In[ ]:


def get_edge(nb,data,threshold):
    edge=[]
    for i in range(len(data)):
        if data['count'][i]>=threshold:
            edge.append((data['neighborhood1'][i],data['neighborhood2'][i]))      
    return edge


# <b>number of adjacencies</b>: Which neighborhoods have a lot of interaction with neighborhood A?

# In[ ]:


def get_numbers_of_adjcs(edge,nb):
    n=len(nb)
    num_of_adjacencies=[]
    for i in range(n):
        num_of_adjacencies.append(0)
    for d in edge:
        num_of_adjacencies[d[0]-1]+=1
        num_of_adjacencies[d[1]-1]+=1
    return num_of_adjacencies


# <b>preparation for the graph</b>

# In[ ]:


def prep(edge,num_of_adjacencies,text,nb):
    edge_trace = Scatter(
    x=[],
    y=[],
    line=Line(width=0.5,color='#888'),
    hoverinfo='none',
    mode='lines')
    
    for i in range(len(edge)):
        e1=edge[i][0]-1
        e2=edge[i][1]-1
        x0, y0 = nb['longitude'][e1],nb['latitude'][e1]
        x1, y1 = nb['longitude'][e2],nb['latitude'][e2]
        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]

    node_trace = Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=Marker(
            showscale=True,
            colorscale='YIGnBu',
            reversescale=True,
            color=[],
            size=10,
             colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))
    
    for i in range(len(nb)):
        x, y = nb['longitude'][i],nb['latitude'][i]
        node_trace['x'].append(x)
        node_trace['y'].append(y)

    for i in range(len(nb)):
        node_info = text[i]
        node_trace['text'].append(node_info)
        node_trace['marker']['color'].append(num_of_adjacencies[i])

        
    fig = Figure(data=Data([edge_trace, node_trace]),
                 layout=Layout(
                    title='<br>NYC texi trip neighborhood interactions',
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))
    
    return fig


# The threshold is very important here. It defines what we consider as "having high interaction".

# ## Set the threshold as 500

# In[ ]:


edge=get_edge(nb,data,500)
num_of_adjacencies=get_numbers_of_adjcs(edge,nb)
#prepare the hover text
text=[]
for i in range(len(nb)):
    t='neighborhood:'+'<b>'+str(nb['neighborhood_name'][i])+'</b>'+'<br>'+'boro:'+ '<b>'+str(nb['boro'][i])+'</b>'+'<br>'+'# of connections:'+"<b>"+str(num_of_adjacencies[i])
    text.append(t)
fig=prep(edge,num_of_adjacencies,text,nb)
pyo.iplot(fig)


# ## change the threshold to 2000

# In[ ]:


edge=get_edge(nb,data,2000)
num_of_adjacencies=get_numbers_of_adjcs(edge,nb)
text=[]
for i in range(len(nb)):
    t='neighborhood:'+'<b>'+str(nb['neighborhood_name'][i])+'</b>'+'<br>'+'boro:'+ '<b>'+str(nb['boro'][i])+'</b>'+'<br>'+'# of connections:'+"<b>"+str(num_of_adjacencies[i])
    text.append(t)
fig=prep(edge,num_of_adjacencies,text,nb)
pyo.iplot(fig)


# We can see from the graphs that 2000 could be a great threshold. And we can get some fun facts from it. Firstly, we all know that Manhattan must be way more busy than the other 4 boros, but that's partly true. Only the area below the northern border of central park is really busy. There are several other busy neighborhoods in other boros such as Airport in Queens, Melrose South in Bronx and Sunset Park West in Brooklyn. This is an interesting application to know the dynamics between neighborhoods in New York (where do people live...to where do New Yorkers usually travel by taxi...)
