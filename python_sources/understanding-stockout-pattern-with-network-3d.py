#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import gc
import pandas as pd

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist,squareform
import networkx as nx
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly as py
import plotly.graph_objs as go


# This is my first shared notebook.Feel free to comment or suggestion.
# 

# In[ ]:


sales_df = pd.read_csv('../input/m5-forecasting-accuracy//sales_train_validation.csv')


# In[ ]:


item_id_sales = sales_df.groupby('item_id').agg('sum')


# In[ ]:


item_id_sales==0


# Similar to:https://www.kaggle.com/jpmiller/grouping-items-by-stockout-pattern
# but I have tried Jaccard distance and change method to complete.
# Q: What is Jaccard?
# A: In short, Jaccard is an intersected area over the union area. In this case, when the product is out of stock the value will be True so Jaccard distance is the intersected day between 2 products that out of stock over the union day between 2 products that out of stock.

# In[ ]:


dist_martix= hierarchy.linkage(item_id_sales==0, 'complete',metric='jaccard')


# In[ ]:


group_df = pd.DataFrame(index=item_id_sales.index)

group_df['group'] = fcluster(dist_martix,0.1)


# In[ ]:


group_df[group_df['group']==1325].index


# In[ ]:


group_df.to_csv('group_jaccard.csv')


# In[ ]:


get_ipython().run_cell_magic('time', '', "dm = pdist(item_id_sales==0, 'jaccard')")


# In[ ]:


dm_df =pd.DataFrame(squareform(dm),index=item_id_sales.index,columns=item_id_sales.index,dtype='float16')


# In[ ]:


del dm


# In[ ]:


dm_df.info()


# In[ ]:


dm_df.head()


# In[ ]:


dm_df.reset_index(inplace=True)


# In[ ]:


dm_df = dm_df.melt(id_vars=['item_id'], var_name='pair', value_name='jaccard_dist')


# In[ ]:


dm_df.head()


# In[ ]:


dm_df.info()


# In[ ]:


dm_df[(dm_df['jaccard_dist']<0.02)]


# In[ ]:


dm_df = dm_df[dm_df['item_id']!=dm_df['pair']]


# In[ ]:


selected_df = dm_df[(dm_df['jaccard_dist']<0.02)]


# In[ ]:


dupes = selected_df.iloc[:,:2].T.apply(sorted).T.duplicated()


# In[ ]:


## remove duplicated
selected_df[~dupes]


# In[ ]:



G=nx.from_pandas_edgelist(selected_df[~dupes],'item_id','pair', 'jaccard_dist')


# In[ ]:


layt = nx.spring_layout(G,k=1/4, dim=3)


# In[ ]:


Edges = list(G.edges)


# In[ ]:


Xn=[layt[k][0] for k in layt]# x-coordinates of nodes
Yn=[layt[k][1] for k in layt]# y-coordinates
Zn=[layt[k][2] for k in layt]# z-coordinates
Xe=[]
Ye=[]
Ze=[]
for e in Edges:
    Xe+=[layt[e[0]][0],layt[e[1]][0], None]# x-coordinates of edge ends
    Ye+=[layt[e[0]][1],layt[e[1]][1], None]
    Ze+=[layt[e[0]][2],layt[e[1]][2], None]


# In[ ]:


group_dict = group_df['group'].to_dict()


# In[ ]:


import plotly as py
import plotly.graph_objs as go
all_tarce1=[]
for i in range(len(list(nx.get_edge_attributes(G,'jaccard_dist').values()))):
    trace1=go.Scatter3d(x=Xe[i*3:(i+1)*3],
                   y=Ye[i*3:(i+1)*3],
                   z=Ze[i*3:(i+1)*3],
                   mode='lines',
                   line=dict(color='rgb(125,125,125)', width=1),
                        text=str((list(nx.get_edge_attributes(G,'jaccard_dist').keys())[i]))+' jaccard: '+str((list(nx.get_edge_attributes(G,'jaccard_dist').values())[i])),
                   hoverinfo='text'
                   )
    all_tarce1.append(trace1)
trace2=go.Scatter3d(x=Xn,
               y=Yn,
               z=Zn,
               mode='markers',
               name='actors',
               marker=dict(symbol='circle',
                             size=4,
                             color=[group_dict[i] for i in list(G.nodes)],
                             colorscale='Viridis',
                             line=dict(color='rgb(50,50,50)', width=0.5)
                             ),
               text= [i+'_'+str(group_dict[i]) for i in list(G.nodes)],
               hoverinfo='text'
               )

axis=dict(showbackground=False,
          showline=False,
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title=''
          )

layout = go.Layout(
         title="3D Stockout Plot",
         width=1000,
         height=1000,
         showlegend=False,
         scene=dict(
             xaxis=dict(axis),
             yaxis=dict(axis),
             zaxis=dict(axis),
        ),
     margin=dict(
        t=100
    ),
    hovermode='closest',
    annotations=[
           dict(
           showarrow=False,
            text="3D Stockout Plot",
            xref='paper',
            yref='paper',
            x=0,
            y=0.1,
            xanchor='left',
            yanchor='bottom',
            font=dict(
            size=14
            )
            )
        ],    )


# In[ ]:


all_tarce1.append(trace2)
data=all_tarce1
fig=go.Figure(data=data, layout=layout)
fig.write_html('test_stockout_3d.html')


# In[ ]:


## Node named after item_id and group_id
## Link named after pair of item_id and jaccard distance value
py.offline.iplot(fig)


# We can see that there is a certain amount of product that out of stock on same day (jaccard distance = 0). Some products have a strong correlation between each department. Group index and most correlated pairs of product features might be a useful feature.
