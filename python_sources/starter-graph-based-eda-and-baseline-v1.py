#!/usr/bin/env python
# coding: utf-8

# Hi guys!
# <br>Please find below detailed explanation of a given tweet dataset, 
# <br>as well as ideas for baseline model construction
# 
# The notebook *is organized as follows*:
# 
# # Table of Contents
# - [Preparation](#Preparation)
#     - [Additional packages installation](#Additional-Installations)
#     - [Imports](#Imports)
#     - [Data loading & merging](#Data-Loading)
# - [Exploratory Data Analysis](#Exploratory-Data-Analysis)
#     - [Data Profiling](#Data-Profiling)
#     - [Target (`target` field)](#Target)
#     - [Keywords normalization (`keyword` field)](#Explore-keywords)
#     - [Tweet's entities (`text` field)](#Analyse-Tweet-Entities)
#         - [Hashtags](#Hashtags)
#         - [URLs](#URLs)
#         - [Users](#Users)
#     - [Location exploration (`location` field)](#Locations)
#     - [NLP-based stuff (WIP)](#NLP-based-stuff)
# - Baseline (WIP)

# # Preparation
# ---

# ## Additional Installations
# ---

# In[ ]:


# install additional dependencies

# !pip install -U stellargraph[demos]  # graph-based embeddings, etc., for v2
# !pip install geopy  # for direct/reverse geocoding, kaggle kernel already has it
get_ipython().system('pip install -U cufflinks  # interactive visualizations atop of Pandas and Plot.ly')
get_ipython().system('pip install lemminflect  # spaCy add-on for lemmatization')
get_ipython().system('pip install twitter-text-python  # for easier tweet entities extraction')
get_ipython().system('pip install folium  # for neat geo-visualizations')


# ## Imports
# ---

# In[ ]:


# Imports
import re
from os.path import join as pjoin

import cufflinks as cf
import networkx as nx
import numpy as np
import pandas as pd
import pandas_profiling as pp
from lemminflect import getInflection, getLemma
from matplotlib import pyplot as plt
from plotly.offline import init_notebook_mode

init_notebook_mode(connected=False)
cf.go_offline()

pd.options.display.max_rows = 200
pd.options.display.max_columns = 200
pd.options.display.max_colwidth = 200
plt.style.use('ggplot')


# ## Data Loading
# ---
# Let's load the data and create inline HTML report with `PandasProfiler` library 
# <br>to see what's our data look like

# In[ ]:


DATA_DIR = '/kaggle/input/nlp-getting-started/'
train = pd.read_csv(pjoin(DATA_DIR, 'train.csv'))
test = pd.read_csv(pjoin(DATA_DIR, 'test.csv'))

# glue datasets together, for convenience
train['is_train'] = True
test['is_train'] = False
df = pd.concat(
    [train, test], 
    sort=False, ignore_index=True
).set_index('id').sort_index()

print(train.shape, test.shape, df.shape)
df.head()


# # Exploratory Data Analysis
# ---

# ## Data Profiling
# ---

# In[ ]:


# get data description report (train)
pp.ProfileReport(train)


# In[ ]:


# do the same for test dataset
pp.ProfileReport(test)


# ## Target
# ---
# Let's see how balanced our dataset is and whether there is a leak in data ordering
# <br> and/or train/test balancing within groups, ordered by `id` column
# <br>(to check if ordering matters)
# <br>P.s. However, such information (as explicit `row_id` data ordering in DWH) **shouldn't be used in a real ML pipeline**

# In[ ]:


# well, almost balanced
df.loc[df.is_train, 'target'].value_counts(normalize=True)


# In[ ]:


# let's see how is train/test mixed within a step
print(f'train/test size ratio: {train.shape[0] / test.shape[0] :.2f}')
print(f'train share: {train.shape[0] / df.shape[0] :.2f}')

# if balanced, this kpi should oscillate around 0.7
grid_step = 1000
tr_share = df.sort_index().groupby(df.index // grid_step).agg({'is_train': 'mean'})
tr_share.iplot(
    dimensions=(640, 320), 
    title=f'Mean train share within groups, ordered by `id` column ({tr_share["is_train"].mean():.2f})',
    kind='bar'
)


# In[ ]:


# let's check the same  kpi against mean target
# well, not so uniform, even for bigger (1k) grid step
print(f'mean target: {train["target"].mean() :.2f}')
target_share = df.sort_index().groupby(df.index // grid_step).agg({'target': 'mean'})['target']
target_share.iplot(
    dimensions=(640, 320), 
    title=f'Mean target within groups, ordered by `id` column ({target_share.mean():.2f})',
    kind='bar'
)


# ## Explore keywords
# ---
# One may make fair assumption that there are different form of the same keyword, occuring in the dataset
# <br>Let's explore it and automatically map to the same lemma, based on installed [lemminflect](https://lemminflect.readthedocs.io/en/latest/) package

# In[ ]:


# prepare lemmatization schema (based on https://lemminflect.readthedocs.io/en/latest/)
def lemmatize(word, upos='VERB'):
    try:
        lemma = getLemma(word, upos=upos, lemmatize_oov=True)
    except AssertionError:
        lemma = getLemma(word, upos='VERB', lemmatize_oov=True)
    if not lemma:  # empty tuple
        lemma = getLemma(word, upos='VERB', lemmatize_oov=True)
    return lemma[0].lower()


# explore initial keywords, change space codes to underlines
df.keyword = df.keyword.fillna('NaN').astype(str).str.replace('(%20)+', '_')
print(df.keyword.nunique())

# check top-N keywords stats
top_n = 10
df.groupby('keyword').agg(
    tweet_cnt=pd.NamedAgg(column='target', aggfunc='count'),
    mean_target=pd.NamedAgg(column='target', aggfunc='mean'),
).sort_values(by=['tweet_cnt', 'mean_target'], ascending=[False, False]).head(top_n)


# In[ ]:


# prepare lemmatized mapping
lemmatized = dict(
    zip(
        df.keyword.drop_duplicates(),
        df.keyword.drop_duplicates().str.split('[_ ]+')
        .apply(
            lambda words: '_'.join(
                lemmatize(w, upos='VERB') for w in sorted(words))  
            # to get rid of the duplicates, like 'building_burning', 'building_burning'
        )
    )
)

# check reduction effect - 20% unique tag cnt drop
print(
    len(set(lemmatized.keys())), 
    len(set(lemmatized.values())), 
    1 - len(set(lemmatized.values())) / len(set(lemmatized.keys()))
)


# Let's check merge quality by visual inspection
# <br>Get only those lemmas with **2+ merged** candidates
# <br>Seems that merging was **valid**

# In[ ]:


merged = pd.DataFrame(
    data=lemmatized.items(), 
    columns=['init', 'new']
).groupby('new')['init'].apply(list)

merged[merged.apply(len) > 1]


# In[ ]:


# let's substitute initial keywords with their merged lemmas 
# and see updated target mean
df['keyword_normalized'] = df.keyword.map(lemmatized)
print(df.keyword.nunique())

keywords_grouped = df.groupby('keyword_normalized').agg(
    tweet_cnt=pd.NamedAgg(column='target', aggfunc='count'),
    mean_target=pd.NamedAgg(column='target', aggfunc='mean'),
).sort_values(by=['tweet_cnt', 'mean_target'], ascending=[False, False])

keywords_grouped.head(top_n*2)


# #### WordCloud visualization
# ---

# In[ ]:


from wordcloud import WordCloud

# generate 'disastrous' keywords

wc_params = dict(
    max_font_size=42,
    background_color=None,
    mode='RGBA',
    max_words=200,
    width=300,
    height=300,
    collocations=False,
    relative_scaling=0.6,
)

wordcloud_disaster = WordCloud(**wc_params).generate(
    ' '.join(
        df.loc[
            df['keyword_normalized'].replace({'nan': np.nan}).isin(
                keywords_grouped[keywords_grouped['mean_target'] >= 0.5].index.tolist()
            ), 
            'keyword_normalized'
        ].values.tolist()
    )
)

wordcloud_neutral = WordCloud(**wc_params).generate(
    ' '.join(
        df.loc[
            df['keyword_normalized'].replace({'nan': np.nan}).isin(
                keywords_grouped[keywords_grouped['mean_target'] <= 0.33].index.tolist()
            ), 
            'keyword_normalized'
        ].values.tolist()
    )
)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt


plt.figure(figsize=(7, 7))
plt.title('`Disastrous` keywords')
plt.imshow(wordcloud_disaster, interpolation="bilinear")
plt.axis("off")
plt.show()


plt.figure(figsize=(7, 7))
plt.title('`Neutral` keywords')
plt.imshow(wordcloud_neutral, interpolation="bilinear")
plt.axis("off")
plt.show()


# ## Analyse Tweet Entities
# ---
# 
# This block is dedicated to extraction and analysis of tweet's entities:
# - #hashtags
# - @users
# - http://URLs
# 
# For that purpose we'll be using [this package by Edmond Burnett](https://github.com/edmondburnett/twitter-text-python)
# <br>It allows us to extract users/mentions, hashtags, follow the links, etc.

# In[ ]:


from ttp import ttp

tparser = ttp.Parser()


def extract_tweet_entities(tweet_text):
    """Extract entities from tweet given tweet's text"""
    return tparser.parse(tweet_text)


tweet_entities = {
    tweet_id: extract_tweet_entities(text)
    for (tweet_id, text) in df.reset_index()[['id', 'text']].values.tolist()
}


# ### Hashtags
# ---

# In[ ]:


df['hashtags'] = df.index.map(tweet_entities)
df['hashtags'] = df['hashtags'].apply(lambda x: sorted(ht.lower() for ht in x.tags))
df['hashtag_cnt'] = df.hashtags.map(len)

# check whether property 'has_hashtags' correlates with target
print(df[['target', 'hashtag_cnt']].clip(0, 1).corr())

pd.crosstab(
    df.is_train.map({True: 'train', False: 'test'}),
    df.hashtag_cnt.clip(0, 4),
    normalize='index'
).astype(str).rename(columns={4: '4+'}).iplot(
    kind='bar',
    title='Unique hashtags cnt (share) in train vs. test datasets',
    dimensions=(640, 320)
)


# In[ ]:


# let's check remaining hashtag candidates
# it looks like garbage to me, ~2% of all tweets containing `#` sign
print(
    df[(df.hashtag_cnt == 0) & df.text.str.contains('#')].shape[0] 
    / df[df.text.str.contains('#')].shape[0]
)

df[(df.hashtag_cnt == 0) & df.text.str.contains('#')].sample(10, random_state=911)


# Let's see most popular hashtags as well as their **average target ratio**

# In[ ]:


col = 'hashtags'
hashtags_stats = df.loc[:, [col, 'target']].explode(col)    .reset_index(drop=True).groupby(col).agg(
    tweet_cnt=pd.NamedAgg(column=col, aggfunc='count'),
    mean_target=pd.NamedAgg(column='target', aggfunc='mean'),
).sort_values(by=['tweet_cnt', 'mean_target'], ascending=[False, False])

hashtags_stats.head()


# In[ ]:


# top "disastrous" hashtags
hashtags_stats[
    (hashtags_stats.tweet_cnt > 6)
    & (hashtags_stats.mean_target > 0.75)
].sort_values(by=['mean_target', 'tweet_cnt'], ascending=[False, False])


# In[ ]:


# top "non-disastrous" hashtags
hashtags_stats[
    (hashtags_stats.tweet_cnt > 6)
    & (hashtags_stats.mean_target <= 0.25)
].sort_values(by=['mean_target', 'tweet_cnt'], ascending=[True, False])


# As far as there are tweets with **2+ hashtags within** we can try to adapt [graph-based](https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)) approaches
# <img src="http://www.buharainsaat.net/pics/b/56/565174_drawing-graph-python.png" height="200" width="300" align="left"/>
# 
# to seek for hidden dependencies: let hashtags be **nodes** and we add an edge (connection) between 2 hashtags <-> they simultaneously occur within single tweet
# <br>Then we correspondingly update edge weigths for stronger connections to have bigger weights

# In[ ]:


# create hashtag graph
from collections import defaultdict
from itertools import combinations

G_ht = nx.Graph()

# add nodes
G_ht.add_nodes_from(hashtags_stats.index.tolist())

# add edges and set their weight
d = defaultdict(int)
for taglist in df.hashtags:
    for c in combinations(taglist, 2):
        d[c] += 1
        G_ht.add_edge((*c))

for n1, n2, attrs in G_ht.edges(data=True):
    attrs['weight'] = np.log1p(d[(n1, n2)])

# drop "weak" (ocassional) links
print(
    f'Init stats:\nNodes: {G_ht.number_of_nodes()}\nEdges: {G_ht.number_of_edges()}')
weak_edges = []
for n1, n2, attrs in G_ht.edges(data=True):
    if attrs['weight'] <= np.log1p(1):
        weak_edges.append((n1, n2))
G_ht.remove_edges_from(weak_edges)

# graph stats
print(
    f'After weak edge removal:\nNodes: {G_ht.number_of_nodes()}\nEdges: {G_ht.number_of_edges()}')

# get isolated tags and drop them from visualization
isolates = list(nx.isolates(G_ht))

G_ht.remove_nodes_from(isolates)
print(
    f'After isolates removal:\nNodes: {G_ht.number_of_nodes()}\nEdges: {G_ht.number_of_edges()}')


node_degrees = pd.DataFrame(
    [x for x in G_ht.degree()],
    columns=['node', 'degree']
).sort_values(by='degree', ascending=False).reset_index()

# drop node with highest degree - tag #news
G_ht.remove_node(node_degrees.iloc[0]['node'])
print(
    f'After highest node degree removal:\nNodes: {G_ht.number_of_nodes()}\nEdges: {G_ht.number_of_edges()}')

cc_ht = sorted(list(nx.connected_components(G_ht)),
               key=lambda x: len(x), reverse=True)
# remove very small components
min_size = 5
cc_ht = [c for c in cc_ht if len(c) > min_size]

print(f'Top connected Components: {len(cc_ht)}')


# Let's visualize top connected components as well as their mean targets + individual targets to see interconnection quality
# <br>Well, **almost all of them seems homogenous** among mean target, as well as **partially interpretable**
# 
# Unfold 2 cells below to see visualization code

# In[ ]:


import plotly.graph_objects as go


def draw_plotly_graph(G, pos, title, node_target_dict, node_degrees):
    """
    Draw network data according to graph `G` and positions `pos`
    https://plot.ly/python/network-graphs/
    """
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
        x=edge_x,
        y=edge_y,
        line=dict(width=0.75, color='#888', dash='dot'),
        hoverinfo='none',
        mode='lines',
    )

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, 
        y=node_y,
        mode='markers+text',
#         hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            reversescale=False,
            size=pd.Series(G.nodes).map(
                np.power(node_degrees.set_index('node')['degree'], 1.2)
            ).values.tolist(),
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        ),
        textposition='top center',
        textfont=dict(family='arial', size=9.5, color='black', )
    )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'<b>{adjacencies[0]}</b><br>({node_target_dict[adjacencies[0]]:.1f})')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title={
                'text': title,
                'xanchor': 'center',
                'yanchor': 'top',
                'y': 0.95,
                'x': 0.5,
            },
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=600,
            height=400,
        )
    )
    fig.show()


# In[ ]:


# plot main connected components and their mean target
hashtags_flattened = df.loc[:, [col, 'target']].explode(col).reset_index(drop=True)

hashtags_agg_target = hashtags_flattened.groupby(col).agg({'target': 'mean'})['target']

# uncomment all data instead of top-n
for (i, c) in enumerate(cc_ht[:4]):
    mean_target = hashtags_flattened[hashtags_flattened[col].isin(c)]['target'].mean()
#     ax[i].title.set_text(f'Mean target: {mean_target:.2f}')
    sG = G_ht.subgraph(c)
    pos = nx.spring_layout(sG, seed=42)
    
    draw_plotly_graph(
        G=sG, 
        pos=pos, 
        title=f'Mean target: {mean_target:.2f}', 
        node_target_dict=hashtags_agg_target, 
        node_degrees=node_degrees
    )


# Let's add obtained user components as cluster features

# In[ ]:


from tqdm.notebook import tqdm

df['hashtag_cluster'] = [[] for i in range(len(df))]
for (ri, r) in tqdm(list(df.iterrows()), total=len(df)):
    for (i, c) in enumerate(cc_ht):
        if set(r['hashtags']).intersection(c):
            df.loc[ri, 'hashtag_cluster'].append(i)
    if not df.loc[ri, 'hashtag_cluster']:
        df.loc[ri, 'hashtag_cluster'].append(-1)

# add connected component mapping as a features
t = df.explode('hashtag_cluster')['hashtag_cluster'].reset_index().rename(columns={'index': 'id'})

df_ht_clusters = pd.crosstab(t.id, t.hashtag_cluster)
df_ht_clusters.columns = [f'ht_cluster__{c}' for c in df_ht_clusters.columns]

print(df.shape)
df = df.drop(df.filter(regex='^ht_cluster').columns, axis=1)
df = pd.concat([df, df_ht_clusters], axis=1)
print(df.shape)

df_ht_clusters.head()


# ### URLs
# ---
# Many tweets have URLs within its body -
# <br>let's see whether we can extract something meaningful from this data
# <br>It seems that URL presence **increases** chances of a tweet being `disastrous`
# (At least in this particular dataset)

# In[ ]:


df['urls'] = df.index.map(tweet_entities)
df['urls'] = df['urls'].apply(lambda x: sorted(ht.lower() for ht in x.urls))
df['url_cnt'] = df.urls.map(len)

# check whether property 'has_url' correlates with target
print(df[['target', 'url_cnt']].clip(0, 1).corr())

pd.crosstab(
    df.is_train.map({True: 'train', False: 'test'}),
    df.url_cnt.clip(0, 3),
    normalize='index'
).astype(str).rename(columns={3: '3+'}).iplot(
    kind='bar',
    title='Unique URLs cnt (share) in train vs. test datasets',
    dimensions=(640, 320)
)


# In[ ]:


# get most popular links
col = 'urls'
url_stats = df.loc[:, [col, 'target']].explode(col)    .reset_index(drop=True).groupby(col).agg(
    tweet_cnt=pd.NamedAgg(column=col, aggfunc='count'),
    mean_target=pd.NamedAgg(column='target', aggfunc='mean'),
).sort_values(by=['tweet_cnt', 'mean_target'], ascending=[False, False])

print(f'distinct urls mentions: {len(url_stats)}')
print(f'mentioned 2+ times: \t{(url_stats.tweet_cnt > 2).sum()}')

url_stats[url_stats.tweet_cnt > 4].sort_values(by='mean_target', ascending=False)

# one may try to follow the links with `utils` module
# >>>import ttp
# >>>tparser = ttp.Parser()
# >>>result = tparser.parse(tweet)
# >>>ttp.utils.follow_shortlinks(result.urls)


# ### Users
# ---
# Many tweets have user mentions within its body -
# <br>let's see whether we can extract something meaningful from this data and build user graph
# <br>It seems that user mention presence **decreases** chances of a tweet being `disastrous`
# (At least in this particular dataset)

# In[ ]:


df['users'] = df.index.map(tweet_entities)
df['users'] = df['users'].apply(lambda x: sorted(ht.lower() for ht in x.users))
df['user_cnt'] = df.users.map(len)

# check whether property 'has_user_ref' correlates with target
print(df[['target', 'user_cnt']].clip(0, 1).corr())

pd.crosstab(
    df.is_train.map({True: 'train', False: 'test'}),
    df.user_cnt.clip(0, 3),
    normalize='index'
).astype(str).rename(columns={3: '3+'}).iplot(
    kind='bar',
    title='Unique users cnt (share) in train vs. test datasets',
    dimensions=(640, 320)
)


# In[ ]:


col = 'users'
user_stats = df.loc[:, [col, 'target']].explode(col)    .reset_index(drop=True).groupby(col).agg(
    tweet_cnt=pd.NamedAgg(column=col, aggfunc='count'),
    mean_target=pd.NamedAgg(column='target', aggfunc='mean'),
).sort_values(by=['tweet_cnt', 'mean_target'], ascending=[False, False])

print(f'distinct user mentions: {len(user_stats)}')
print(f'mentioned 2+ times: \t{(user_stats.tweet_cnt > 2).sum()}')

user_stats[user_stats.tweet_cnt > 5].sort_values(by='mean_target', ascending=False)


# Unfold cells below to see visualization code

# In[ ]:


# create user graph
G_u = nx.Graph()

# add nodes
G_u.add_nodes_from(user_stats.index.tolist())

# add edges and set their weight
d = defaultdict(int)
for ulist in df.users:
    for c in combinations(ulist, 2):
        d[c] += 1
        G_u.add_edge((*c))

for n1,n2,attrs in G_u.edges(data=True):
    attrs['weight'] = np.log1p(d[(n1, n2)])

# drop "weak" (ocassional) links
print(f'Init stats:\nNodes: {G_u.number_of_nodes()}\nEdges: {G_u.number_of_edges()}')
weak_edges = []
for n1,n2,attrs in G_u.edges(data=True):
    if attrs['weight'] <= np.log1p(0):
        weak_edges.append((n1, n2))
G_u.remove_edges_from(weak_edges)
    
# graph stats
print(f'After weak edge removal:\nNodes: {G_u.number_of_nodes()}\nEdges: {G_u.number_of_edges()}')

# get isolated tags and drop them from visualization
isolates = list(nx.isolates(G_u))

G_u.remove_nodes_from(isolates)
print(f'After isolates removal:\nNodes: {G_u.number_of_nodes()}\nEdges: {G_u.number_of_edges()}')


node_degrees = pd.DataFrame(
    [x for x in G_u.degree()], 
    columns=['node', 'degree']
).sort_values(by='degree', ascending=False).reset_index(drop=True)

# drop node with highest degree - tag #news
print(node_degrees.iloc[0]['node'])
G_u.remove_node(node_degrees.iloc[0]['node'])
print(f'After highest node degree removal:\nNodes: {G_u.number_of_nodes()}\nEdges: {G_u.number_of_edges()}')

cc_u = sorted(list(nx.connected_components(G_u)), key=lambda x: len(x), reverse=True)
# remove very small components
min_size = 5
cc_u = [c for c in cc_u if len(c) > min_size]

print(f'Top connected Components: {len(cc_u)}')


# Let's see what popular user's network consists of
# <br>We can clearly see groups with 
# - perfect 0.00, like 4th group with porn-based stuff :) 
# - or 1.00 in (observed in train) nodes ->
# 
# One may inference those group target mean to (observed in test) nodes -> dive into tweets level.
# <br>Be careful, some of the connected components exists only in **test dataset** (those with perfect `NaNs`)

# In[ ]:


# plot main connected components and their mean target
users_flattened = df.loc[:, [col, 'target']].explode(col).reset_index(drop=True)

users_agg_target = users_flattened.groupby(col).agg({'target': 'mean'})['target']

# uncomment all data instead of top-n
for (i, c) in enumerate(cc_u[:4]):
    mean_target = users_flattened[users_flattened[col].isin(c)]['target'].mean()
    sG = G_u.subgraph(c)
    pos = nx.spring_layout(sG, seed=42)
    
    draw_plotly_graph(
        G=sG, 
        pos=pos, 
        title=f'Mean target: {mean_target:.2f}', 
        node_target_dict=users_agg_target, 
        node_degrees=node_degrees
    )


# ## Locations
# ---
# Let's see what secrets are hidden in the `location` field.
# <br>We'll perform basic data cleaning, as well as direct/inverse geocoding with the help of [geopy package](https://geopy.readthedocs.io/en/stable/)
# <br>Then we plot the heatmap using [folium package](https://python-visualization.github.io/folium/)
# 
# **UPD**: To prevent ArcGIS API from our kaggle-DDoSing, I've created **[separate dataset atop of geocoded location data](https://www.kaggle.com/frednavruzov/disaster-tweets-geodata)**
# <br>You can find collection details & scripts in [this starter notebook](https://www.kaggle.com/frednavruzov/starter-how-the-data-was-collected/edit)
# <br>Further contribution is always welcomed!

# In[ ]:


# read parsed geodata
geodata = pd.read_csv('../input/disaster-tweets-geodata/geodata.csv').set_index('id')
geodata.head()


# In[ ]:


# append geodata to the initial dataframe
print(df.shape)
df = df.merge(geodata, left_index=True, right_index=True, how='left')
# drop possible duplicates
df = df[~df.index.duplicated(keep='first')]
print(df.shape)
df[['location', 'target'] + geodata.columns.tolist()].sample(5, random_state=911)


# In[ ]:


# print location target (by country)
df.groupby(df.country.replace('', np.nan)).agg({'target': 'mean', 'location': 'count'}).sort_values(by=['location', 'target'], ascending=[False, False]).head(20)


# Unfold cells below to see visualization code

# In[ ]:


from folium import plugins
import folium


# inspired by https://alysivji.github.io/getting-started-with-folium.html
def map_points(df, lat_col='latitude', lon_col='longitude', zoom_start=11,                 plot_points=False, pt_radius=15,                 draw_heatmap=False, heat_map_weights_col=None,                 heat_map_weights_normalize=True, heat_map_radius=15):
    """Creates a map given a dataframe of points. Can also produce a heatmap overlay

    Arg:
        df: dataframe containing points to maps
        lat_col: Column containing latitude (string)
        lon_col: Column containing longitude (string)
        zoom_start: Integer representing the initial zoom of the map
        plot_points: Add points to map (boolean)
        pt_radius: Size of each point
        draw_heatmap: Add heatmap to map (boolean)
        heat_map_weights_col: Column containing heatmap weights
        heat_map_weights_normalize: Normalize heatmap weights (boolean)
        heat_map_radius: Size of heatmap point

    Returns:
        folium map object
    """

    ## center map in the middle of points center in
    middle_lat = df[lat_col].median()
    middle_lon = df[lon_col].median()

    curr_map = folium.Map(location=[middle_lat, middle_lon],
                          zoom_start=zoom_start)

    # add points to map
    if plot_points:
        for _, row in df.iterrows():
            folium.CircleMarker([row[lat_col], row[lon_col]],
                                radius=pt_radius,
                                popup=row['location'],
                                fill_color="#3db7e4", # divvy color
                               ).add_to(curr_map)

    # add heatmap
    if draw_heatmap:
        # convert to (n, 2) or (n, 3) matrix format
        if heat_map_weights_col is None:
            cols_to_pull = [lat_col, lon_col]
        else:
            # if we have to normalize
            if heat_map_weights_normalize:
                df[heat_map_weights_col] =                     df[heat_map_weights_col] / df[heat_map_weights_col].sum()

            cols_to_pull = [lat_col, lon_col, heat_map_weights_col]

        stations = df[cols_to_pull].values
        curr_map.add_child(plugins.HeatMap(stations, radius=heat_map_radius))

    return curr_map


# In[ ]:


map_points(
    df=df[
        df.is_train 
        & ~df.lat.isnull()
    ].sample(500, random_state=911) , # for memory constrains
    lat_col='lat',
    lon_col='lon', 
    zoom_start=2, # change to `1` to set global world view
    plot_points=False, 
    pt_radius=0.75,
    draw_heatmap=True, 
    heat_map_weights_col='target',
    heat_map_weights_normalize=True,
    heat_map_radius=15,
)


# ## NLP-based stuff
# ---
# Let's see how natural language processing techniques can help us in solving this particular binary classification problem

# In[ ]:


# borrowed some hot stuff from https://www.kaggle.com/jdparsons/tweet-cleaner

# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
EMOJI_REGEX = re.compile(
    "["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE
)

NON_WORD_PATTERN = r"[^A-Za-z0-9\.\'!\?,\$\s]"

# https://stackoverflow.com/a/35041925
# replace multiple punctuation with single. Ex: !?!?!? would become ?
PUNCT_DEDUPLICATION_REGEX = re.compile(r'[\?\.\!]+(?=[\?\.\!])', re.I)


# In[ ]:


from gensim.parsing.preprocessing import STOPWORDS
from ttp.ttp import HASHTAG_REGEX, URL_REGEX, USERNAME_REGEX

# substitute links, urls, mentions, hashtags etc.
df['text_processed'] = df.text.str.replace(EMOJI_REGEX, ' EMOJI ').str.replace(PUNCT_DEDUPLICATION_REGEX, '').str.replace(URL_REGEX, ' URL ').str.replace(NON_WORD_PATTERN, '').str.replace('\s+', ' ')

df['text_processed'].sample(10, random_state=911).values.tolist()


# ### Build word wectors from `text` field

# In[ ]:


from gensim.models import FastText, Word2Vec


def compute_doc2vec(words, w2v, weight_dict, emb_size):
    """
    given word embeddings, compute weighted (by term frequency) 
    doc embeddings
    """
    doc_vector = np.zeros(emb_size, dtype=np.float32)
    weights = 0
    for w in words:
        try:
            doc_vector += w2v.wv[w]
            weights += weight_dict.get(w, 0)

        except KeyError:
            pass

    doc_vector /= max(0.01, weights)
    return doc_vector / np.linalg.norm(doc_vector, ord=2)

text_field = 'text_processed'
prepared_texts = df[text_field].str.lower().str.replace(
#     '(\\b' + '\\b|\\b'.join(STOPWORDS) + ')', ''
    '', ''
).str.extractall(
    '(?P<words>[A-Za-z]+)').groupby(level=0)['words'].apply(list).reindex(df.index).fillna('EMPTY')

w2v_size = 16

w2v = Word2Vec(
    sentences=prepared_texts,
    size=w2v_size,
    window=32,
    min_count=7,
    iter=5,
    negative=5,
    sg=1,
    batch_words=2**11,
    #     min_n=3,
    #     max_n=4,
)


word_dict = np.log1p(prepared_texts.explode().value_counts()).to_dict()


w2v_df = pd.DataFrame(
    data=np.concatenate(
        prepared_texts.apply(
            #             lambda x: np.array([w2v.wv[w] if w in w2v.wv else np.zeros(w2v_size) for w in x]).mean(axis=0)
            lambda x: compute_doc2vec(
                x, w2v, weight_dict=word_dict, emb_size=w2v_size)
        ).values
    ).reshape((len(df), w2v_size)),
    columns=[f'w2v__{i}' for i in range(w2v_size)]
)


# join with the main dataset
df = df.drop(df.filter(regex='^w2v__').columns, axis=1)
print(df.shape)
df = pd.concat([df, w2v_df], axis=1)
print(df.shape)


# ## Encode tags

# In[ ]:


hashtags_sentences = df.explode('hashtags')['hashtags'].groupby(level=[0]).apply(
#     lambda x: ' '.join(re.sub(r'[^\x00-\x7f]', r'', s) for s in list(x.astype(str)))
    lambda x: ' '.join([re.sub(r'[^\x00-\x7f]', r'', s) for s in list(x.astype(str))])
)

hashtags_sentences.sample(10, random_state=911)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


tfidf = TfidfVectorizer(
    max_df=1.0, # drop all words, whose document/sentence share > share (if float)
    min_df=5,  # drop all words, whose count < x (if int)
    dtype=np.float32,
    max_features=200,
)

tfidf.fit(hashtags_sentences)
tag_df = pd.DataFrame(
    data=tfidf.transform(hashtags_sentences).todense(),
    columns=[f'tag__{k}' for k, v in sorted(tfidf.vocabulary_.items(), key=lambda item: item[1])]
)

print(tag_df.shape)
tag_df.head()

# join with the main dataset
df = df.drop(df.filter(regex='^tag__').columns, axis=1)
print(df.shape)
df = pd.concat([df, tag_df], axis=1)
# df['tag__encoding'] = np.array(tfidf.transform(hashtags_sentences).todense().argmax(axis=1)).ravel()
print(df.shape)


# ## Prepare Universal sentence encoder embeddings

# In[ ]:


import tensorflow_hub as hub

embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

# we have to wait a little :)
df_univ_embeddings = pd.DataFrame(
    data=embedder(df[text_field].values).numpy(), 
    columns=[f'tf_emb__{i}' for i in range(512)]
)

df_univ_embeddings.head()


# In[ ]:


# join with the main dataset
df = df.drop(df.filter(regex='^tf_emb__').columns, axis=1)
print(df.shape)
df = pd.concat([df, df_univ_embeddings], axis=1)
print(df.shape)

print(f'Memory usage, Mb: {df.memory_usage(deep=True).sum() // 2**20}')


# In[ ]:


# some basic NLP feature engineering
df['text_len'] = df.text.apply(len).astype(np.int16)
df['text_word_cnt'] = df.text.str.split('\s+').map(len).astype(np.int16)
df['has_exclamations'] = df.text.str.contains(r'[?!]').astype(np.int8)
df['has_uppercased'] = df.text.str.contains(r'[A-Z]{2,}').astype(np.int8)

nlp_features = [
    'has_exclamations', 
    'has_uppercased',
    'text_len',
    'text_word_cnt',
]

df[nlp_features + ['target']].corr()


# ## Baseline
# ---

# In[ ]:


features_num = (
    [
        'hashtag_cnt',
        'url_cnt',
        'user_cnt',
        'lat',
        'lon'
    ]
#     + nlp_features
    + tag_df.columns.tolist()
    + w2v_df.columns.tolist()
    + df_ht_clusters.columns.tolist()
    + df_univ_embeddings.columns.tolist()
)

features_cat = [
    'country',
    'city',
    'keyword_normalized',
#     'tag__encoding'
]

cat_cols = []
for col in features_cat:
    df[f'{col}_code'] = df[col].astype('category').cat.codes
    cat_cols.append(f'{col}_code')
    
features = features_num + cat_cols
    
print(df[features].dtypes)

df[features].head()


# ## Cross-validation

# We'll use LightGBM model (boosting) due to its **natural strengths**
# <br>(speed, scale-invariance, NaN handling, etc.)

# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


def f1_score_lgb(preds, dtrain):
    labels = dtrain.get_label()
    f_score = f1_score(
        np.round(preds), 
        labels, 
        average='macro'
    )
    return 'f1_score', f_score, True


skf = StratifiedKFold(n_splits=5, random_state=911, shuffle=True)

lgb_params = {
#     'class_weight': 'balanced',
    'num_leaves': 63,
    'learning_rate': 0.01,
    'max_depth': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.25,
    'cat_l2': 10.,
    'cat_smooth': 10.,
    'min_data_per_group': 20,
    'min_data_in_leaf': 20,
    'reg_lambda': 0.5,
    'boost_from_average': True,
    
    'max_cat_threshold': 64,
    'max_bin': 255,
    'min_data_in_bin': 5,
#     'scale_pos_weight': 1 / df.loc[df.is_train, 'target'].mean()
    #     ''
}

sample_weight = (
    1 / df.loc[
        df.is_train, 
        'target'
    ].map(df.loc[df.is_train, 'target'].value_counts(normalize=True))
)


cv_res = lgb.cv(
    params=lgb_params,
    train_set=lgb.Dataset(
        data=df.loc[df.is_train, features],
        label=df.loc[df.is_train, 'target'],
        categorical_feature=cat_cols,
        weight=sample_weight
    ),
    folds=skf,
    metrics=['binary_logloss'],
    feval=f1_score_lgb,
    verbose_eval=50,
    early_stopping_rounds=200,
    #     eval_train_metric=True,
    num_boost_round=1000,
)


# In[ ]:


# fit simple model, based on cv rounds

model = lgb.LGBMClassifier(
    **lgb_params, 
    n_esimators=int( len(cv_res['binary_logloss-mean']) * (skf.n_splits + 1)/skf.n_splits )
)

model.fit(
    X=df.loc[df.is_train, features], 
    y=df.loc[df.is_train, 'target'], 
    sample_weight=sample_weight,
    categorical_feature=cat_cols,
)


# In[ ]:


# check feature importance
lgb.plot_importance(
    model, 
    importance_type='gain', 
    figsize=(10, 10), 
    max_num_features=50
)


# In[ ]:


# get predictions
y_pred = model.predict(df.loc[~df.is_train, features])

# prepare submission
pd.DataFrame({'target': y_pred, 'id': df[~df.is_train].index}).astype(np.int32).to_csv(
    'submission.csv', 
    index=False, 
    encoding='utf-8'
)


# ---
# That's all for now
# <br>Stay tuned, this notebook is going to be updated soon
# <br>Hope, you guys, like it and learn something new!
# <br>**As always, upvotes, comments, ideas are always welcome!**
# 
# ---
# P.s. Check my [F1-Score metric analysis notebook](https://www.kaggle.com/frednavruzov/be-careful-with-f1-score-shuffle-estimate)
