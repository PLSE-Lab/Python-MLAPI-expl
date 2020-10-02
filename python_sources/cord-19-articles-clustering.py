#!/usr/bin/env python
# coding: utf-8

# # CORD-19 Clustering Articles and Papers related to Coronavirus

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
df[:3]


# In[ ]:


df[:3]


# So far we are interested just in title,abstract and maybe authors
# 
# 
# 
# *   First we get rid of rows with missing values
# *   Then we get rid of the duplicated values
# 
# 

# In[ ]:


df = df[['title','abstract','authors']]


# In[ ]:


df.isnull().sum()


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


df[df.duplicated()].__len__()


# In[ ]:


df.drop_duplicates(inplace=True)
df[df.duplicated()].__len__()


# In[ ]:


df[:3]


# ## Tensorflow Upgrade
# 

# Here we prepare some simple clustering functions to be used later on

# ## Text features extraction

# In[ ]:


import tensorflow as tf
import tensorflow_hub as hub
class UniversalSenteneceEncoder:

    def __init__(self, encoder='universal-sentence-encoder', version='4'):
        self.version = version
        self.encoder = encoder
        self.embd = hub.load(f"https://tfhub.dev/google/{encoder}/{version}")

    def embed(self, sentences):
        return self.embd(sentences)

    def squized(self, sentences):
        return np.array(self.embd(tf.squeeze(tf.cast(sentences, tf.string))))


# In[ ]:


encoder = UniversalSenteneceEncoder(encoder='universal-sentence-encoder',version='4')


# Since this process is quite consuming we will slice the dataset

# ## Title Feature correlation

# In[ ]:


get_ipython().run_cell_magic('time', '', "df = df[:5000]\ndf['title_sent_vects'] = encoder.squized(df.title.values).tolist()\n# df['abstract_sent_vects'] = encoder.embed(df.abstract.values)")


# In[ ]:


import plotly.graph_objects as go

sents = 50
labels = df[:sents].title.values
features = df[:sents].title_sent_vects.values.tolist()

fig = go.Figure(data=go.Heatmap(
                    z=np.inner(features, features),
                    x=labels,
                    y=labels,
                    colorscale='Viridis',
                    ))

fig.update_layout(
    margin=dict(l=40, r=40, t=40, b=40),
    height=1000,
    xaxis=dict(
        autorange=True,
        showgrid=False,
        ticks='',
        showticklabels=False
    ),
    yaxis=dict(
        autorange=True,
        showgrid=False,
        ticks='',
        showticklabels=False
    )
)

fig.show()


# ## Abstract Features Correlation

# In[ ]:


get_ipython().run_cell_magic('time', '', "df = df[:5000]\ndf['abstract_sent_vects'] = encoder.squized(df.abstract.values).tolist()\n# df['abstract_sent_vects'] = encoder.embed(df.abstract.values)")


# In[ ]:


import plotly.graph_objects as go

sents = 30
labels = df[:sents].abstract.values
features = df[:sents].abstract_sent_vects.values.tolist()

fig = go.Figure(data=go.Heatmap(
                    z=np.inner(features, features),
                    x=labels,
                    y=labels,
                    colorscale='Viridis',
                    ))

fig.update_layout(
    margin=dict(l=140, r=140, t=140, b=140),
    height=1000,
    xaxis=dict(
        autorange=True,
        showgrid=False,
        ticks='',
        showticklabels=False
    ),
    yaxis=dict(
        autorange=True,
        showgrid=False,
        ticks='',
        showticklabels=False
    ),
    hoverlabel = dict(namelength = -1)
)

fig.show()


# ## PCA & Clustering

# In[ ]:


from sklearn.cluster import KMeans
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from datetime import datetime
import plotly.express as px


# ### Title Level Clustering

# In[ ]:


get_ipython().run_cell_magic('time', '', "n_clusters = 10\nvectors = df.title_sent_vects.values.tolist()\nkmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', random_state = 0)\nkmean_indices = kmeans.fit_predict(vectors)\n\npca = PCA(n_components=512)\nscatter_plot_points = pca.fit_transform(vectors)\n\ntmp = pd.DataFrame({\n    'Feature space for the 1st feature': scatter_plot_points[:,0],\n    'Feature space for the 2nd feature': scatter_plot_points[:,1],\n    'labels': kmean_indices,\n    'title': df.title.values.tolist()[:vectors.__len__()]\n})\n\nfig = px.scatter(tmp, x='Feature space for the 1st feature', y='Feature space for the 2nd feature', color='labels',\n                 size='labels', hover_data=['title'])\nfig.update_layout(\n    margin=dict(l=20, r=20, t=20, b=20),\n    height=1000\n)\n\nfig.show()")


# ### Abstract Level Clustering

# In[ ]:


get_ipython().run_cell_magic('time', '', "n_clusters = 10\nvectors = df.abstract_sent_vects.values.tolist()\nkmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', random_state = 0)\nkmean_indices = kmeans.fit_predict(vectors)\n\npca = PCA(n_components=512)\nscatter_plot_points = pca.fit_transform(vectors)\n\ntmp = pd.DataFrame({\n    'Feature space for the 1st feature': scatter_plot_points[:,0],\n    'Feature space for the 2nd feature': scatter_plot_points[:,1],\n    'labels': kmean_indices,\n    'title': df.abstract.values.tolist()[:vectors.__len__()]\n})\n\nfig = px.scatter(tmp, x='Feature space for the 1st feature', y='Feature space for the 2nd feature', color='labels',\n                 size='labels', hover_data=['title'])\nfig.update_layout(\n    margin=dict(l=20, r=20, t=20, b=20),\n    height=1000\n)\n\nfig.show()")


# ## Graphs

# #### Preparing the data for graphs

# In[ ]:


import networkx as nx


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity


# Make a copy of our dataframe and create the similarity matrix for the extracted title vectors

# In[ ]:


get_ipython().run_cell_magic('time', '', 'sdf = df.copy()\nsimilarity_matrix = cosine_similarity(sdf.title_sent_vects.values.tolist())')


# Add them into a dataframe, similiar to what a heatmap looks likes

# In[ ]:


simdf = pd.DataFrame(
    similarity_matrix,
    columns = sdf.title.values.tolist(),
    index = sdf.title.values.tolist()
)


# Let's unstack them to add them easier into our graph

# In[ ]:


long_form = simdf.unstack()
# rename columns and turn into a dataframe
long_form.index.rename(['t1', 't2'], inplace=True)
long_form = long_form.to_frame('sim').reset_index()


# In[ ]:


long_form = long_form[long_form.t1 != long_form.t2]
long_form[:3]


# #### Plotly Graph

# In[ ]:


get_ipython().system('pip install python-igraph')


# In[ ]:


import igraph as ig


# In[ ]:


get_ipython().run_cell_magic('time', '', "lng = long_form[long_form.sim > 0.75] \ntuples = [tuple(x) for x in lng.values]\nGm = ig.Graph.TupleList(tuples, edge_attrs = ['sim'])\nlayt=Gm.layout('kk', dim=3)")


# In[ ]:


Xn=[layt[k][0] for k in range(layt.__len__())]# x-coordinates of nodes
Yn=[layt[k][1] for k in range(layt.__len__())]# y-coordinates
Zn=[layt[k][2] for k in range(layt.__len__())]# z-coordinates
Xe=[]
Ye=[]
Ze=[]
for e in Gm.get_edgelist():
    Xe+=[layt[e[0]][0],layt[e[1]][0], None]# x-coordinates of edge ends
    Ye+=[layt[e[0]][1],layt[e[1]][1], None]
    Ze+=[layt[e[0]][2],layt[e[1]][2], None]


# In[ ]:


import plotly.graph_objs as go


# In[ ]:


trace1 = go.Scatter3d(x = Xe,
  y = Ye,
  z = Ze,
  mode = 'lines',
  line = dict(color = 'rgb(0,0,0)', width = 1),
  hoverinfo = 'none'
)

trace2 = go.Scatter3d(x = Xn,
  y = Yn,
  z = Zn,
  mode = 'markers',
  name = 'articles',
  marker = dict(symbol = 'circle',
    size = 6, 
    # color = group,
    colorscale = 'Viridis',
    line = dict(color = 'rgb(50,50,50)', width = 0.5)
  ),
  text = lng.t1.values.tolist(),
  hoverinfo = 'text'
)

axis = dict(showbackground = False,
  showline = False,
  zeroline = False,
  showgrid = False,
  showticklabels = False,
  title = ''
)

layout = go.Layout(
  title = "Network of similarity between CORD-19 Articles(3D visualization)",
  width = 1500,
  height = 1500,
  showlegend = False,
  scene = dict(
    xaxis = dict(axis),
    yaxis = dict(axis),
    zaxis = dict(axis),
  ),
  margin = dict(
    t = 100,
    l = 20,
    r = 20
  ),

)


# In[ ]:


fig=go.Figure(data=[trace1,trace2], layout=layout)

fig.show()


# #### Finding Communities

# Next step, we filter them nodes with higher similarity

# In[ ]:


sim_weight = 0.75
gdf = long_form[long_form.sim > sim_weight]


# We create our graph from our dataframe

# In[ ]:


plt.figure(figsize=(50,50))
pd_graph = nx.Graph()
pd_graph = nx.from_pandas_edgelist(gdf, 't1', 't2')
pos = nx.spring_layout(pd_graph)
nx.draw_networkx(pd_graph,pos,with_labels=True,font_size=10, node_size = 30)


# Now let's try to find communities in our graph

# In[ ]:


betCent = nx.betweenness_centrality(pd_graph, normalized=True, endpoints=True)
node_color = [20000.0 * pd_graph.degree(v) for v in pd_graph]
node_size =  [v * 10000 for v in betCent.values()]
plt.figure(figsize=(35,35))
nx.draw_networkx(pd_graph, pos=pos, with_labels=True,
                 font_size=5,
                 node_color=node_color,
                 node_size=node_size )


# ### Now let's get our groups

# In[ ]:


l=list(nx.connected_components(pd_graph))

L=[dict.fromkeys(y,x) for x, y in enumerate(l)]

d=[{'articles':k , 'groups':v }for d in L for k, v in d.items()]


# We've got our 'clustered' dataframe of articles, however since we filtered the data to take just the most similar articles, we've ended up havin left just 660 from the 5k data

# #### Creating word clouds for grooups

# In[ ]:


gcd = pd.DataFrame.from_dict(d)


# In[ ]:


import nltk
nltk.download('stopwords')
nltk.download('punkt') 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.tokenize import RegexpTokenizer

tok = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english')) 
def clean(string):
    return " ".join([w for w in word_tokenize(" ".join(tok.tokenize(string))) if not w in stop_words])

gcd.articles = gcd.articles.apply(lambda x: clean(x))


# In[ ]:


gcd.__len__(),gcd.__len__() / df.__len__()


# In[ ]:


from wordcloud import WordCloud


# In[ ]:


get_ipython().run_cell_magic('time', '', "clouds = dict()\n\nbig_groups = pd.DataFrame({\n    'counts':gcd.groups.value_counts()\n    }).sort_values(by='counts',ascending=False)[:9].index.values.tolist()\n\nfor group in big_groups:\n    text = gcd[gcd.groups == group].articles.values\n    wordcloud = WordCloud(width=1000, height=1000).generate(str(text))\n    clouds[group] = wordcloud")


# In[ ]:


def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows,figsize=(20,20))
    for ind,title in zip(range(len(figures)), figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.jet())
        axeslist.ravel()[ind].set_title(f'Most Freqent words for the group {title+1}')
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional


# In[ ]:


plot_figures(clouds, 3, 3)
plt.show()


# In[ ]:




