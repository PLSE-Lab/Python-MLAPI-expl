#!/usr/bin/env python
# coding: utf-8

# # Meretz Party
# Meretz is a left-wing, social-democratic and green political party in Israel. It is also one of the 3 parties in Israel where the candidates are elected in open primaries, and the only party to publish the raw data.
# 
# While Meretz is not expected to gain a large number of seats in the upcoming elections, the story of the primaries is still interesting.
# 
# A few camps were competing against each other, with the most prominent being Ilan Gilon's socialist camp, and a camp led by several other Meretz members which, while being left-leaning economically in general, puts more emphasis on pregressive matters such as human rights, LGBTQ, the peace process with the Palestinians, etc. While representatives of both camps were elected, it was generally accepted that Gilon's camp won.
# 
# In this script I would explore the results and tell the story of Gilon's victory.

# In[ ]:


import numpy as np 
import pandas as pd 
import folium

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import SpectralClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

import networkx as nx
from community import best_partition

import plotly.offline as py
import plotly.graph_objs as go
import networkx as nx
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

py.init_notebook_mode(connected=True)


# # Load The Data

# In[ ]:


df = pd.read_csv('../input/meretz_party.csv',  encoding='latin-1')
df.drop('Hebrew Location', axis=1, inplace=True)
df = df.head(130)


# First, let's look at the final results:

# In[ ]:


df.drop(['Location', 'Lat', 'Lng', 'voters', 'votes'], axis=1).sum().sort_values(ascending=False)


# In[ ]:


df.drop(['Lat', 'Lng'], axis=1).groupby('Location').sum()['voters'].sort_values(ascending=False)


# # Preprocessing
# 
# We will save the general columns for later and for now only keep those with the candidates data. Also, normalize each booth to account for different booth sizes (in terms of voters.

# In[ ]:


locations = df['Location']
lat = df.Lat
lng = df.Lng
sizes = df.sum(axis=1)
normalized_df = df.drop(['Location', 'Lat', 'Lng', 'voters', 'votes'], axis=1)
normalized_df = normalized_df[normalized_df.sum().sort_values(ascending=False).index]
normalized_df = normalized_df.truediv(normalized_df.sum(axis=1), axis=0)


# # Cluster The Voting Booths 
# 
# First; Let's project the data onto a 2-dimensional space for visualizaions sake:
# 

# In[ ]:


plt.figure(figsize=(7, 7))
pca_df = pd.DataFrame(PCA(n_components=2).fit_transform(normalized_df))
pca_df['locations'] = locations
pca_df['sizes'] = sizes

plt.scatter(pca_df[0], pca_df[1], s=50, alpha=0.6)
plt.xlabel('PC 1')
plt.ylabel('PC 2')


# * It looks like 2-4 clusters would do. Let's see if there's a number of clusters that is significantly better than the others:

# In[ ]:


for i in np.arange(1,12):
    kmeans = KMeans(n_clusters=4).fit(normalized_df)
    print(i, silhouette_score(normalized_df, kmeans.predict(normalized_df)))


#  Silhouette score and common sense both agree that 4 clusters is a reasonable number. This is especially useful since I've already decided to use 4 clusters as it fits my internal model regarding the demography of the voters, and I needed some justification. 
#  
#  Now let's use the goold old K-Means clustering and see the different cluster-centers

# In[ ]:


kmeans = KMeans(n_clusters=4, random_state=42).fit(normalized_df)

pca_df = pd.DataFrame(PCA(n_components=2).fit_transform(normalized_df))
pca_df['label'] = kmeans.predict(normalized_df)
pca_df['locations'] = locations
pca_df['sizes'] = sizes


# In[ ]:


trace1 = go.Bar(
        x=normalized_df.columns,
        y=kmeans.cluster_centers_[0],
        name='Cluster 1'
        )

trace2 = go.Bar(
        x=normalized_df.columns,
        y=kmeans.cluster_centers_[1],
        name='Cluster 2'
        )

trace3 = go.Bar(
        x=normalized_df.columns,
        y=kmeans.cluster_centers_[2],
        name='Cluster 3'
        )

trace4 = go.Bar(
        x=normalized_df.columns,
        y=kmeans.cluster_centers_[3],
        name='Cluster 4'
        )

layout = go.Layout(
     title='<b>Cluster Centers</b>',
     titlefont=dict(size = 30, color='#7f7f7f'),
     hovermode='closest'
)

fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
py.iplot(fig)


# # Let's explore the clusters
# 1. Cluster 1: Isay Farij is by far the most dominant figure in this cluster, as well as Michal Rozin, Halil Elakovi and Nir Avishai-Cohen, This cluster has mostly Arab villages
# 2. Cluster 2: This seems like a mixture of cluster 3 and 4, where Ilan gilon is the most dominant figure, tofether with Maharth Baruch, Yaniv Sagi and Ali Salhalha. but this cluster has some representation of the competing camp, especially Michal Rozin. This is the urban cluster which is the 2nd largets cluster.
# 3. Cluster 4 has the largerst number of votes, and pretty much has all the leading candidates. This cluster is the Urban cluster, where the human\civil rights camp, led by candidates such as Mossi Raz and Gaby Lasky had a strong grip.
# 4. Cluster number 4 Led by Ilan Gilon, Ali Salhalha, Mahart Baruch and Yaniv Sagi. as we would see, most of the settlements that are assigned to this cluster are Druze villages. This makes sense since Ali Salhalha is Druze. In the primaries, every voter chooses 4 candidates, so this voting pattern definitely relflects a decision made by a group of people to vote for the same candidates
# 
# Ilan Gilon's success was based on his dominance in the rural, druze and arab cluster together with his medium popularity in the urban cluster which was enough in order to keep the first place. It also seems that other candidates that were associated with Gilon were succesfull in voting booths where he was, even though their popularity in the urban cluster was much lower.

# In[ ]:


clusters_dict ={
    pca_df[pca_df.locations == 'Haifa'].label.values[0]: 'Urban',
    pca_df[pca_df.locations == 'Gan Shmuel'].label.values[0]: 'Rural',
    pca_df[pca_df.locations == 'Beit Jan'].label.values[0]: 'Druze',
    pca_df[pca_df.locations == 'Furadis'].label.values[0]: 'Arab',
}


# Let's plot each settlement in a 2-D plain where every settlement is represented by the 2 first principals of the number of votes each candidate got:

# In[ ]:


traces = []

for label in sorted(pca_df.label.unique()):
    traces.append(go.Scatter(
            x=pca_df[pca_df.label == label][0],
            y=pca_df[pca_df.label == label][1],
            text=pca_df[pca_df.label == label]['locations'],
            mode='markers',
            hoverinfo='text',
            name = clusters_dict[label],
            marker=dict(
                size=[np.sqrt(a) for a in (pca_df[pca_df.label == label]['sizes'])],
                opacity=0.3,
          )
           )
                     )
    
layout = go.Layout(
        title= 'Meretz voting clusters',
        hovermode='closest',
        xaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False
        ),
        yaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False
        ))
fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)


# Since this data is geograhpical, let's plot it on the map:

# In[ ]:


pca_df['lat'] = lat
pca_df['lng'] = lng


# In[ ]:


df.head()


# In[ ]:


pca_df.head()


# In[ ]:


ans = []
for i in range(len(df)):    
    ans.append(list(df.loc[i, df.columns[3:-2]].transpose().sort_values().index[-5:]))
    
pca_df['winners'] = ans
pca_df['winners'] = pca_df.winners.apply(lambda x: x[::-1])

pca_df['first'] = pca_df.winners.apply(lambda x: x[0])
pca_df['second'] = pca_df.winners.apply(lambda x: x[1])
pca_df['third'] = pca_df.winners.apply(lambda x: x[2])
pca_df['forth'] = pca_df.winners.apply(lambda x: x[3])


# In[ ]:


m = folium.Map(location=[32.13,34.8],zoom_start=9, tiles="CartoDB dark_matter" )

colors = ['blue', 'crimson', 'green', 'orange']
for row in pca_df.iterrows():
    folium.Circle(
              location= (row[1].lat, row[1].lng),    
              radius=0.7*row[1].sizes,
              popup= '<b>' + row[1].locations + '</b><br>' + row[1]['first'] + '<br>' + row[1]['second'],
              color=colors[row[1].label],
              fill=True,
              fill_color=colors[row[1].label]
        ).add_to(m)


# In[ ]:


m


# In[ ]:


m.save('plot_data.html')


# Let's find the most typical settlements with respect to each cluster (The one that is the closest to cluster center)

# In[ ]:


pca_df['cluster_name'] = pca_df.label.apply(lambda x: clusters_dict[x])
distances = pd.DataFrame(euclidean_distances(normalized_df, kmeans.cluster_centers_))
pca_df.loc[distances.idxmin()][['label', 'cluster_name', 'locations']]


# In[ ]:


df['label'] = pca_df['label']
df['cluster_name'] = pca_df['cluster_name']


# In[ ]:


df.sort_values(by='voters', ascending=False)[['Location', 'voters', 'cluster_name']].head(20)


# In[ ]:


df.groupby('cluster_name').sum()[['voters']].sort_values(by='voters', ascending=False)


# # Not let's reverse the prroblem and cluster the candidates together

# In[ ]:


candidates = df.drop(['Location', 'Lat', 'Lng', 'votes', 'voters', 'label', 'cluster_name'], axis=1)
sizes = candidates.sum(axis=0).values


# In[ ]:


sorted_df = candidates[candidates.corr()['Ilan Gilon'].sort_values().index.tolist()]


# # Correlation Matrix between candidates

# In[ ]:


trace = go.Heatmap(z=sorted_df.corr(),
                  x=sorted_df.columns,
                  y=sorted_df.columns)
py.iplot([trace])


# In[ ]:


candidates = candidates.transpose() 
candidates = candidates.truediv(candidates.sum(axis=1), axis=0)


# In[ ]:


kmeans = KMeans(n_clusters=3).fit(candidates)

pca_df = pd.DataFrame(PCA(n_components=2).fit_transform(candidates))
pca_df['label'] = kmeans.predict(candidates)
pca_df['name'] = candidates.index
pca_df['size'] = sizes


# In[ ]:


traces = []

for label in sorted(pca_df.label.unique()):
    traces.append(go.Scatter(
            x=pca_df[pca_df.label == label][0],
            y=pca_df[pca_df.label == label][1],
            text=pca_df[pca_df.label == label]['name'],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                size=[np.sqrt(a) for a in (pca_df[pca_df.label == label]['size'])],
                opacity=0.3,
          )
           )
                     )
    
layout = go.Layout(
        title= 'Meretz candidates clusters',
        hovermode='closest',
        xaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False
        ),
        yaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False
        ))
fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)


# # Candidates Graph
# 
# The next section is pretty much copy-paste from Itamar Mushkin's great [kernel](http://www.kaggle.com/itamarmushkin/partitioning-the-parties])
# 
# We will use the great networkx library, as well as graph clustering based on the[ Louvain Modularity](http:/en.wikipedia.org/wiki/Louvain_Modularity/) which is a great tool for network analysis.
# 
# Every node in the graph is a candidate, and there is an edge between two candidates if the correlation between them is bigger than a certain threshold (I chose 0.8). We then partition the nodes into communities based on the Louvain Modularity and plot the results

# In[ ]:


C=np.corrcoef(candidates.transpose(),rowvar=0)
A=1*(C>0.8)
G=nx.Graph(A)
G=nx.relabel_nodes(G,dict(zip(G.nodes(),candidates.transpose().columns.values)))
communities=best_partition(G)


# In[ ]:


community_colors={0:0,1:0.1,2:0.2,3:0.3,4:0.4,5:0.5, 6:0.6, 7:0.7, 8:0.8, 9:0.9}
node_coloring=[community_colors[communities[node]] for node in G.nodes()]

nx.pos=nx.fruchterman_reingold_layout(G, dim=2, k=None, pos=None, fixed=None, iterations=5000, weight='weight', scale=1, center=None)

nx.draw_networkx(G, cmap=plt.get_cmap('jet'), with_labels=True, node_color=node_coloring,font_size=10)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)


# # Let's zoom in
# 

# In[ ]:


nx.draw_networkx(G, cmap=plt.get_cmap('jet'), with_labels=True, node_color=node_coloring,font_size=10)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.xlim([-0.4, 0.4])
plt.ylim([-0.4, 0.4])


# In[ ]:


list(G.edges)


# In[ ]:


sorted_df.corr()['Halil Elakovi']['Isay Farij']


# In[ ]:


for edge in list(G.edges):
    print('{ target: "' +edge[0] + '" ,source: "' + edge[1] + '", strength:  ' + str(sorted_df.corr()[edge[0]][edge[1]]) + '},')


# In[ ]:


communities


# In[ ]:


color=nx.get_node_attributes(G, 'labels')


# In[ ]:


color


# In[ ]:


dict(zip(G.nodes(),candidates.transpose().columns.values))


# "Maybe we can prettify the graph a little bit and use interactive plotting with Plotly to make thinks clearer

# In[ ]:


C=np.corrcoef(candidates.transpose(),rowvar=0)
A=1*(C>0.8)
G=nx.Graph(A)
pos=nx.spring_layout(G)#, dim=2, k=None, pos=None, fixed=None, iterations=5000, weight='weight', scale=1, center=None)
labels = candidates.transpose().columns.values
N = len(G.nodes)
E = G.edges
Xv=[pos[k][0] for k in range(N)]
Yv=[pos[k][1] for k in range(N)]
Xed=[]
Yed=[]
for edge in E:
    Xed+=[pos[edge[0]][0],pos[edge[1]][0], None]
    Yed+=[pos[edge[0]][1],pos[edge[1]][1], None] 
    
trace3=go.Scatter(x=Xed,
               y=Yed,
               mode='lines',
               line=dict(color='rgb(210,210,210)', width=1),
               hoverinfo='none'
               )
trace4=go.Scatter(x=Xv,
               y=Yv,
               mode='markers',
               name='net',
               marker=dict(symbol='circle-dot',
                             size=[0.8*np.sqrt(x) for x in sizes], 
                             color= node_coloring,
                            colorscale='Viridis',
                             line=dict(color='rgb(50,50,50)', width=0.5)
                             ),
               text=labels,
               hoverinfo='text'
               )


layout = go.Layout(
    title= 'Zoom in for a better view!',
        hovermode='closest',
        xaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False
        ),
        yaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False
    )
)
data1=[trace3, trace4]
fig1=go.Figure(data=data1, layout=layout)
# fig1['layout']['annotations'][0]['text']=annot
py.iplot(fig1,)


# When we zoom in we can see the 3 groups:
# 1. Ilan Gilon, Ali Salhalha and Maharta Baruch. These are the candidates that dominated the rural and druze areas. This is the socialist wing of the party.
# 2. Avi Buskila, Mossi Raz and Gabi Lasky, as well as several other candidates,, the candidates who dominated the rural areas with their human rights messaging 
# 1.5. Avi Dabush is the connecting link between these 2 clusters. This makes sense since Dabush can be seen as some kind of a mix between the two groups, being vocal both on human rights issues as well as working a lot with more rural communities.
# 3. The next cluster belong to the arab candidates, led by Isay Farij. It's interesting to see Nir Avishai-Cohen in this cluster as well
# 2.5 Michal rozin, who came up 2nd in the primaries, is the connecting link between the urban camp and Farij's cluster. Both Farij and Rozen are generally popular and therefore less correlated with specific candiates and are not in the centers of the clusters

# In[ ]:




