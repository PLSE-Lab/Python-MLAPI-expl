#!/usr/bin/env python
# coding: utf-8

# # Abstract
# * Small EDA
# * KMeans Clustering
# * Analysis of the clusters (Work in Progress)

# In[ ]:


import numpy as np 
import pandas as pd 

import os

files_wanted = ['movies_metadata.csv']
file_paths = list()

for dirname, _, filenames in os.walk('/kaggle/input'):
     for filename in filenames:
            if filename in files_wanted:
                file_paths.append(str(dirname + "/" + filename))
meta_df = pd.read_csv(file_paths[0], low_memory=False)


# Let's look at our data

# In[ ]:


meta_df.head()


# In[ ]:


meta_df.shape


# # Data Cleaning
# * I will remove the one's I absolutely will never need
# * There are some jsons that I want to parse (production_companies, production_countries, & genres)
# * Drop null values

# In[ ]:


meta_df.drop(['belongs_to_collection', 'homepage', 'tagline', 'poster_path', 'overview', 'imdb_id', 'spoken_languages'], inplace=True, axis=1)

column_changes = ['production_companies', 'production_countries', 'genres']

json_shrinker_dict = dict({'production_companies': list(), 'production_countries': list(), 'genres': list()})

meta_df.dropna(inplace=True)


# ## Parsing JSONs
# * It got complicated because some jsons had multiple name / genre values.
#     * While looking at some, I noticed that it wasn't alphabetical. So I assumed if a movie had genres, Action, Drama, Thriller, Action was the "main" genre. I could be wrong, but no explanations were given
#     * To solve this issue, I just took the first one
# Let's import what we need

# In[ ]:


import ast


# I noticed that this JSON payload was pretty nasty. The quotes were single quotes instead of double quotes. I used the AST library to be able to interpet the json properly.
# The naive approach was to replace all single quotes with double quotes, but some of the strings have a single quote that stands for an accent. So "L'Hotel" for example would cause an error.

# In[ ]:


for col in column_changes:
    if col == 'production_companies':
        for i in meta_df[col]:
            i = ast.literal_eval(i)
            if len(i) < 1:
                json_shrinker_dict['production_companies'].append(None)

            for element in i:
                json_shrinker_dict['production_companies'].append(element['name'])
                break
    elif col == 'production_countries':
        for i in meta_df[col]:
            i = ast.literal_eval(i)
            if len(i) < 1:
                json_shrinker_dict['production_countries'].append(None)
            for element in i:
                json_shrinker_dict['production_countries'].append(element['iso_3166_1'])
                break
    else:
        for i in meta_df[col]:
            i = ast.literal_eval(i)
            if len(i) < 1:
                json_shrinker_dict['genres'].append(None)

            for element in i:
                json_shrinker_dict['genres'].append(element['name'])
                break

for i in column_changes:
    meta_df[i] = json_shrinker_dict[i]

meta_df.dropna(inplace=True)

meta_df['budget'] = meta_df['budget'].astype(int)


# In[ ]:


meta_df.head()


# In[ ]:


meta_df.shape


# Notice I dropped null values again. This is because some JSONs were empty, but the were not NaNs. Instead, they looked like this in a row  "[]".  
# Overall we lost about ~12,000 rows.

# # EDA

# Let's import what is needed

# In[ ]:


import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)


# ## Budget and Revenue

# In[ ]:


fig = px.scatter(meta_df, x='budget', y='revenue', hover_data=['title'], color='genres', width=800, height=800)
fig.update_layout(
    title='The Relationship between Budget and Revenue',
    xaxis_title='Budget',
    yaxis_title='Revenue',
    font=dict(
        size=16
    )
)
iplot(fig)


# Pretty interesting to see that theres almost somewhat of a linear trend with Budget and Revenue.
# This trend runs to a certain extend, though. I does make sense that budget and revenue would have a trend though. Better directors, better actors, better effects team, etc.

# ## Genres and Budgets

# In[ ]:


genre_budget_df = meta_df.groupby(['genres'])['budget'].sum()

fig = go.Figure([
    go.Bar(
        x=genre_budget_df.index,
        y=genre_budget_df.values,
        text=genre_budget_df.values,
        textposition='auto',
        marker_color=['#94447f',
                      '#5796ef',
                      '#8a59c0',
                      '#288abf',
                      '#0ab78d',
                      '#4ed993',
                      '#7d3970',
                      '#b3dc67',
                      '#dc560a',
                      '#0079fe',
                      '#98d3a8',
                      '#d5105a',
                      '#d04dcf',
                      '#58c7a2',
                      '#7bf1f8',
                      '#244155',
                      '#587b77',
                      '#c64ac2',
                      '#5e805d',
                      '#ebab95']
    )])

fig.update_layout(
    title='Sum of all Movie Budgets in each Genre',
    xaxis_title='Genre',
    yaxis_title='Total Budget',
    width=800,
    height=1000,
    font=dict(
        size=16
    )
)

fig.layout.template = 'seaborn'

iplot(fig)


# I was surprised see that Drama was second. I not much of a drama movie kind of guy I guess.

# ## Budget and Runtime

# In[ ]:


fig = px.scatter(meta_df, x='budget', y='runtime', hover_data=['title'], color='genres', width=800, height=800)
fig.update_layout(
    title='The Relationship between Budget and Movie Runtime',
    xaxis_title='Budget',
    yaxis_title='Runtime',
    font=dict(
        size=16
    )
)

iplot(fig)


# I was surprised by the result. I expected the runtimes of movies to increase with budget significantly (it does a little). The way I saw it was that a larger budget would be needed to pay for all the services that take time. So when we look at an editing process of a movie: The more time needed to spend editing something, the more money. It does increase a little, but not by a lot. It seems that all movies like that sweetspot of ~100-200 minutes for a movie.

# In[ ]:


fig = px.scatter(meta_df, y='runtime', x='revenue', hover_data=['title'], color='genres', width=800, height=800)
fig.update_layout(
    title='The Relationship between Runtime and Movie Revenue',
    yaxis_title='Runtime',
    xaxis_title='Revenue',
    font=dict(
        size=16
    )
)

iplot(fig)


# I kind of expected this result because who wants to watch a 900 minute movie? My favorite movie of all time right now is Intersellar, and I felt like that movie was really long already!

# # Clustering

# While looking around at all my data that I didn't think was significant enough to graph, I ran into this:

# In[ ]:


fig = go.Figure(go.Box(
    y=meta_df['vote_count']
    
))

fig.update_layout(
    title='Vote Count Distribution',
    yaxis_title='Vote Count',
    width=800,
    height=800
)

iplot(fig)


# In my mind, for a rating to be credible, it needs a significant amount of votes. It is like saying that ONE 5 star rating means that a product is good. It's possible, but you can't take just one persons word for granted like that. On the other hand, a product with FIFTY 5 star reviews would make me trust the product a bit more (not including bots that favorite things).
# I basically got rid of everything before the 3rd quartile (~58 votes per movie). 

# In[ ]:


meta_df = meta_df[meta_df['vote_count'] >= meta_df['vote_count'].quantile(.75)]
fig = go.Figure(go.Box(
    y=meta_df['vote_count']
))

fig.update_layout(
    title='Vote Count Distribution',
    yaxis_title='Vote Count',
    width=800,
    height=800
)

iplot(fig)


# That's better. In all, this will help our clustering algorithm in the long run.

# ## KMeans Clustering

# Considering what we just saw with vote count, it is safe to say our data is not evenly distributed.  
# For that reason, I think it is best to use the MinMaxScaler from sklearn to normalize our data
# Let's import what we need

# In[ ]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scalar = MinMaxScaler()

scaled_df = meta_df[['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']]

smaller_df = scaled_df.copy()

scaled = scalar.fit_transform(meta_df[['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']])

scaled_df = pd.DataFrame(scaled, index=scaled_df.index, columns=scaled_df.columns)

scaled_df.head()


# My plan is to:
#   
# Cluster -> Get cluster labels -> Join cluster labels with old df

# In[ ]:


def apply_kmeans(df, clusters):
    kmeans = KMeans(n_clusters=clusters, random_state=0)
    cluster_labels = kmeans.fit(df).labels_
    string_labels = ["c{}".format(i) for i in cluster_labels]
    df['cluster_label'] = cluster_labels
    df['cluster_string'] = string_labels

    return df


# In[ ]:


def param_tune(df):
    scores = {'clusters': list(), 'score': list()}
    for cluster_num in range(1,31):
        scores['clusters'].append(cluster_num)
        scores['score'].append(KMeans(n_clusters=cluster_num, random_state=0).fit(df).score(df))

    scores_df = pd.DataFrame(scores)

    fig = go.Figure(go.Scatter(
        x=scores_df['clusters'],
        y=scores_df['score']
    ))

    fig.update_layout(
        xaxis_title='Cluster',
        yaxis_title='Score',
        title='Elbow Method Results',
        height=800,
        width=800
    )

    fig.show()

    return 9


# I have two methods which I'll hide above this cell, but essentially this is what they do:
# * param_tune: Works as a grid search to find optimal K betw (uses elbow)
# * apply_kmeans: Takes optimal k found from param_tune, generates cluster labels, and appends it to the dataframe

# In[ ]:


clusters = param_tune(scaled_df)

scaled_df = apply_kmeans(scaled_df, clusters)


# In[ ]:


smaller_df = smaller_df.join(scaled_df[['cluster_label', 'cluster_string']])
smaller_df = smaller_df.join(meta_df[['title', 'genres']])

smaller_df.head()


# Now we have our data clustered and can explore it some more.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style


# ## Cluster Cardinality

# In[ ]:


style.use('seaborn-poster')
fig, ax = plt.subplots(1,1)
cluster_comb = smaller_df.groupby(['cluster_label'])['title'].count()
sns.barplot(y=cluster_comb.index, x=cluster_comb.values, orient='h', palette="Spectral",
            edgecolor='black', linewidth=1)
plt.ylabel("Cluster", fontsize=18)
plt.xlabel("Records", fontsize=18)
plt.title("Records in Each Cluster", fontsize=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()


# So the distribution of records in the cluster does not seem to be even. There are a lot of records in clusters 0 and 2 for some reason.

# ## Scatter Matrix

# In[ ]:


fig = px.scatter_matrix(smaller_df, dimensions=['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count'],
                        color='cluster_string', hover_data=['title', 'genres'])
fig.update_layout(
    title='Cluster Scatter Matrix',
    height=1000,
    width=800
)

iplot(fig)


# What I find notable is notice cluster 3. It is usually on the high end of values for all the features. Cluster 3 is also the cluster with the least records in it. This definitely means there is some sort of trend going on in cluster 3 (really cool to me)

# ## Trying to Find Some Meaning of the Clusters

# In[ ]:


clusters = list(smaller_df['cluster_label'].unique())
cluster_dict = dict()

cluster_count = 0
for col in range(3):
    for row in range(3):
        cluster_df = smaller_df[smaller_df.cluster_label == clusters[cluster_count]]
        cluster_dict["{},{}".format(str(col), str(row))] = cluster_df['genres'].value_counts()
        cluster_count += 1

cluster_count = 0

fig, axs = plt.subplots(3, 3, figsize=(15,15))

for col in range(3):
    for row in range(3):
        coord = "{},{}".format(str(col), str(row))

        sns.barplot(y=cluster_dict[coord].index, x=cluster_dict[coord].values, orient='h',
                    palette={'Drama': '#94447f',
                             'Action': '#5796ef',
                             'Adventure': '#8a59c0',
                             'Comedy': '#288abf',
                             'Crime': '#0ab78d',
                             'Thriller': '#4ed993',
                             'Fantasy': '#7d3970',
                             'Horror': '#b3dc67',
                             'Science Fiction': '#dc560a',
                             'Animation': '#0079fe',
                             'Romance': '#98d3a8',
                             'Mystery': '#d5105a',
                             'Family': '#d04dcf',
                             'War': '#58c7a2',
                             'History': '#7bf1f8',
                             'Western': '#244155',
                             'TV Movie': '#587b77',
                             'Music': '#c64ac2',
                             'Documentary': '#5e805d'}, edgecolor='black', linewidth=0.9, ax=axs[col][row])

        title = "Cluster {}'s Genre Distribution".format(cluster_count)
        axs[col][row].set_title(title, fontsize=15, fontweight='bold')
        cluster_count += 1
plt.tight_layout()
plt.show()


# I find it interesting that Drama and Action movies seem to be the top in several clusters. This could be that there is just a larger quantity of these movies in the data set, but then why is Drama's count so low in Cluster 7 and 8?
# From these results, I'd like to next compare specific movies within the Drama and Action genres within several clusters and find the similarities / differences.

# ## Looking at Drama in it's significant clusters

# In[ ]:


drama_df = smaller_df[(smaller_df.genres == 'Drama') & (smaller_df.cluster_label.isin([0, 1, 2, 5]))]
drama_df = drama_df.sort_values('cluster_label')

fig = px.violin(drama_df, y='revenue', x='cluster_string', color='cluster_string', points='all', hover_data=drama_df)

fig.update_layout(
    title='Revenue Distribution in Drama Movies',
    yaxis_title='Revenue',
    xaxis_title='Cluster',
    height=1000,
    width=800
)

iplot(fig)


# It doesn't seem like there's any special indications of why clusters are the way they are on this graph alone...but  
# since plotly gives me the entire row data when I hover over a point, I was able to find out that every cluster's vote average range is usually within 1 point.  
#   
# Let's explore that idea next.

# In[ ]:


fig = px.violin(drama_df, y='vote_average', x='cluster_string',
                    color='cluster_string', points='all', hover_data=drama_df)

fig.update_layout(
    title='Vote Average Distribution in Drama Movies',
    yaxis_title='Vote Average',
    xaxis_title='Cluster',
    height=1000,
    width=800
)

iplot(fig)


# Really interesting. It seems that my hypothesis was correct. Although this is for Drama movies within these select clusters. Let's add in all clusters and see if this trend says prevalent

# ## Vote Average Distribution for Drama Movies in all Clusters 

# In[ ]:


drama_df = smaller_df[(smaller_df.genres == 'Drama')]
drama_df = drama_df.sort_values('cluster_label')
fig = px.violin(drama_df, y='vote_average', x='cluster_string',
                color='cluster_string', points='all', hover_data=drama_df)

fig.update_layout(
    title='Vote Average Distribution in Drama Movies',
    yaxis_title='Vote Average',
    xaxis_title='Cluster',
    height=1000,
    width=800
)

iplot(fig)


# Within the drama category, vote average definitely seems to have an influence on which cluster the movie will go to. Although some clusters like 3 & 4 are hard to differentiate why one movie would be in one cluster over the other.  
#   
# Let's see how all genres fall under this hypothesis

# ## Vote Average Distribution in all Movie Genres

# In[ ]:


drama_df = smaller_df
drama_df = drama_df.sort_values('cluster_label')
fig = px.violin(drama_df, y='vote_average', x='cluster_string',
                color='cluster_string', points='all', hover_data=drama_df)

fig.update_layout(
    title='Vote Average Distribution in all Movies',
    yaxis_title='Vote Average',
    xaxis_title='Cluster',
    height=1000,
    width=800
)

iplot(fig)


# So while the clusters still have some distinction regardless of genre, the mins and maxes of the vote average increased for every cluster. This means there must be some more variables determining our clusters.

# # Conclusion
# Thank you for taking the time to check out my notebook. If you have any criticisms of any kind, feel free to mention them. 
