#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Spreading
# ## Using Graph Theory to Visualize the Spreading of the COVID-19 Virus
# 
# The purpose of this notebook is to represent the COVID-19 spreading around the World using graph / networks theory. Then, we will study the properties of such graph.
# 
# ![Virus Spreading](https://static.seattletimes.com/wp-content/uploads/2020/03/coronavirus-spread-W-780x226.jpg)
# 
# ** How would it be possible to transform a set of evolution lines / time series in a network graph? **
# > It`s not impossible at all! We will try to link the countries following a criteria: (1) each country will represent a node and (2) countries where the explosion of cases started nearly at the same time will tend to have a connection between them.
# 
# Of course there are different ways to define what would be "cases that started at the same time". We will discuss it and, then, we will create a clustering algorithm to classify the countries in "transmission groups" (i.e: groups of countries that were infected by the virus at **nearly** the same time).
# 
# Here is an example of what we are going to do! :)

# In[ ]:


img_dir = '/kaggle/input/clusterfigurescovid19/GRAPH_COVID1.png'
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread(img_dir)

fig = plt.figure(figsize = (15, 15))
ax = fig.add_subplot(111)
ax.axis('off')
ax.imshow(img, interpolation='none')

plt.show()


# # 1. Exploring the Dataset (EDA)
# 
# After loading our dataset, we will just play a little with what have:

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns

import plotly
import plotly.express as px

import networkx as nx

import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

df_in = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
df_in.iloc[:, 1:10].head()


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# We also have informations about the number of confirmed cases and recovered people but these numbers will not be used here. There is a **good** reason for that:
# > We cannot trust the number of confirmed cases for every country. Many places are having huge troubles to test all the population. Also, we can't determine exactly the number of infected people since the COVID-19 is characterized by a great number of assymptomatic people. When someone dies, the tests tend to be executed with a higher probability than for those who are infected with no symptoms.
# 
# Ok. So, imagine we have an exponential function:
# 
# $$ f(x) = K.e^x $$
# 
# We can easily calculate the average value of the function:
# 
# $$ \overline{f}(x) = \int_{x = 0}^{x = T_{final}} f(x) $$
# 
# This will lead us to a value with the same units of "f(x) = y". Alright. Let's follow a similar approach to calculate the average value of "x" instead of "y":
# 
# $$ \overline{x} = \frac{\int_{x = 0}^{x = T_{final}} x.f(x)}{\int_{x = 0}^{x = T_{final}} f(x)} $$
# 
# > You can imagine that the average value of a function is a weighted average of the function along the time axis with all different times having the same weight. So, we can generalize that to take the average "x" but, this time, we will use the function values as weights. The final value is indeed something with the same units of "x" - in our case...the time! We are calculating the average date of the deaths.

# In[ ]:


dates_vec = list(df_in.columns)[3:]
average_time_vec = [None] * df_in.shape[0]

for i, row_index in enumerate(df_in.index):

    weighted_sum, total_deaths = 0, 0
    
    for j, date in enumerate(dates_vec):
        current_term = df_in.at[row_index, date]
        weighted_sum += j * current_term
        total_deaths += current_term
    
    average_time_vec[i] = weighted_sum / total_deaths
    
df_in['avg_time'] = average_time_vec

n_lines = int((df_in.shape[0] * (df_in.shape[0] - 1)) / 2)
list_country1, list_country2, list_w, list_d =    [None] * n_lines, [None] * n_lines, [None] * n_lines, [None] * n_lines

line_index = 0
epsilon = 0.001
for i in range(0, df_in.shape[0] - 1):
    for j in range(i + 1, df_in.shape[0]):
        index_i, index_j = df_in.index[i], df_in.index[j]
        list_country1[line_index] = df_in.at[index_i, 'Country/Region']
        list_country2[line_index] = df_in.at[index_j, 'Country/Region']
        diff_time = df_in.at[index_i, 'avg_time'] - df_in.at[index_j,'avg_time']
        list_w[line_index] = (1 / (abs(diff_time) + epsilon))
        list_d[line_index] = abs(diff_time)
        line_index += 1
        
df_graph = pd.DataFrame(dict(
    Country1 = list_country1,
    Country2 = list_country2,
    Weight = list_w,
    Distance = list_d
))

df_graph.head()


# Finaly, with this datatable, we can represent our graph and explore its structure. Notice the definition of the graph weight that we used:
# 
# $$ W_{ij} = (\lVert \overline{x_i} - \overline{x_j} \rVert + \epsilon) ^{-1} $$
# 
# We take the inverse of the difference between the mean times. The epsilon factor corrects the fact that the difference may be zero. In that case, the nodes will be strongly linked since the epsilon factor is a small number (0.001).

# In[ ]:


df_graph.to_csv('df_graph.csv', index=False)
df_graph.head()


# # 2. Plotting / Exploring the Graph
# 
# The graph was generated using the Gephi editor. It's true that we could create the graph using the NetworkX python library but it's easier to use the Gephi software and, also, it's more complete and with it's customization I was able to get good insights about the structure of the COVID-19 propagation network. You can check more informations about Gephi in the [Official Website](https://gephi.org/), it's open source and very interesting to deal with networks in general.
# 
# ![https://gephi.files.wordpress.com/2008/09/logo_about.png?w=584](https://gephi.files.wordpress.com/2008/09/logo_about.png?w=584)
# 
# > **2A.** I used the size of the nodes, their colors and the colors of the edges directly related to the weighted average of each node, where the weight is bigger if the mean time of occurrences are similar. So, we have the following figure:
# 
# ![https://i.ibb.co/bg7PQ3c/GRAPH-COVID1.png](https://i.ibb.co/bg7PQ3c/GRAPH-COVID1.png)

# Our first impressions are:
# 
# * China takes a central role in the COVID-19 propagation, which makes sense **BUT**
# * It seems that something similar happens in France, Canada, United Kingdom and Denmark
# * Then, from Europe, a huge number of African / Asian countries were affected
# 
# The weighted degree of each node can be calculated by:
# 
# $$ D_i = \sum_{j \in Neighbours} W_{ij} $$
# 
# So it's the average weight of the neighbours that are linked to the node in analysis. We can continue to get other visualizations. Now:
# 
# > **2B.** Let's use a Gephi filter see just countries with a weighted degree bigger than a certain number. We will increase the threshold progressively:
# 
# ![https://i.ibb.co/YQs9gDV/GRAPH-COVID2.png](https://i.ibb.co/YQs9gDV/GRAPH-COVID2.png)
# 
# > **2C.** Filtering more to get just the main points:
# 
# ![https://i.ibb.co/1RR7b5g/GRAPH-COVID3.png](https://i.ibb.co/1RR7b5g/GRAPH-COVID3.png)

# We can take some nice conclusions looking at the last network, where we look at the points with a weighted degree bigger or equal than the weighed degree of Denmark:
# 
# * There is a strong link between Canada and France. They are connected, which makes sense when we think that there is a great cultural proximity between Quebec and France, which may interfer in the flights and in the personal contacts among different kinds of people.
# * The graph also shows us that it seems that we have a propagation following the path "China - Europe - Asia / Africa - America".
# 
# Finally, we can study the graph metrics and try to take more conclusions.

# # 3. Graph Metrics
# 
# This step will be really simple: we will just study the distribution of the weighted degree of the graph. Other metrics are possible, like the Page Rank or the Concentration. Some of them are used mainly in Social Network applications.
# 
# So, let's start by plotting the distribution of the edge weights:

# In[ ]:


fig = px.histogram(x=df_graph['Weight'])
fig.update_layout(yaxis_type='log', title='Edges Weight Histogram', xaxis_title='Weight', yaxis_title='Count (Log)')


# The edge nodes degree can be computed by taking the sum of the weights of all edges that link a given node. We can calculate it programatically, but we can also use the NetworkX library tools:

# In[ ]:


covid_graph = nx.from_pandas_edgelist(df_graph, 'Country1', 'Country2', 'Weight')
w_degrees = covid_graph.degree(weight='Weight')
df_w_degrees = pd.DataFrame(w_degrees, columns=['Country', 'Degree'])
df_w_degrees.head()


# In[ ]:


fig = px.histogram(x=df_w_degrees['Degree'])
fig.update_layout(yaxis_type='log', title='Weighted Degrees', xaxis_title='Weighted Deg.', yaxis_title='Count (Log)')


# Let's check the histogram taking the values with a weighted degree smaller than "20.000":

# In[ ]:


fig = px.histogram(x=df_w_degrees.query('Degree < 20000')['Degree'])
fig.update_layout(yaxis_type='log', title='Weighted Degrees', xaxis_title='Weighted Deg.', yaxis_title='Count (Log)')


# And if we list the countries with a degree bigger than 20.000 we will get the "big nodes" of our Gephi figure and, also, the medium sized nodes located at the center of the figure:

# In[ ]:


df_w_degrees.query('Degree > 20000').sort_values('Degree', ascending=False)


# So, what can we do with all these informations? We can consider that the inverse of the edge weight is a kind of distance between the nodes and use it to find a good clusterization algorithm in order to classify the countries in "transmission groups"! Remember that the weight of the graph is the inverse of the absolute value of the difference between the mean times:
# 
# $$ W_{ij}^{-1} = ((\lVert \overline{x_i} - \overline{x_j} \rVert + \epsilon) ^{-1})^{-1} = \lVert \overline{x_i} - \overline{x_j} \rVert + \epsilon$$

# # 3. Finding Transmission Groups with Clustering
# 
# We can try to use ahierarchical clustering model to find transmission groups.

# In[ ]:


graph_distance = nx.from_pandas_edgelist(df_graph, 'Country1', 'Country2', 'Distance')
adj_matrix = nx.adjacency_matrix(graph_distance, weight='Distance')
adj_matrix


# The matrix indexes are ordered with the same order as the dict keys:

# In[ ]:


graph_distance._node.keys()


# Checking the histogram of the distribution of the distances:

# In[ ]:


fig = px.histogram(x=df_graph['Distance'])
fig.update_layout(yaxis_type='log', title='Distances Distribution', xaxis_title='Dist.', yaxis_title='Count (Log)')


# Noting that we have a bigger concentration of points at smaller values of X:

# In[ ]:


fig = px.histogram(x=df_graph.query('Distance < 20')['Distance'])
fig.update_layout(yaxis_type='log', title='Distances Distribution', xaxis_title='Dist.', yaxis_title='Count (Log)')


# Let's check the dendrogram, since we will work with a hierarchical agglomerative clustering:

# In[ ]:


plt.rcParams['figure.figsize'] = [16, 6]
plt.style.use('ggplot')
D = dendrogram(linkage(adj_matrix.todense()), no_labels=True, truncate_mode='level')
plt.ylim(0, 600)
plt.yticks(fontsize=20)
plt.title('Dendrogram - Hierarchical Clustering', fontsize=20)


# As we can see, if we take a threshold equal to 300, we will have 6 clusters, which is a reasonable number of clusters and, in this case, we will have 2 outliers (the figure helps us to find a good trade-off between the number of clusters and the number of outliers). Let's fit our model:

# In[ ]:


ac_model = AgglomerativeClustering(n_clusters=6, affinity='precomputed', linkage='complete')
ac_model


# In[ ]:


ac_model.fit(adj_matrix.toarray())
ac_model.labels_


# In[ ]:


df_classification = pd.DataFrame(dict(Country=list(graph_distance._node.keys()), Label=ac_model.labels_.tolist()))
df_classification.to_csv('df_classification.csv', index=False)
df_classification.head()


# The obtained dataframe can be used to color the different nodes of the graphs and to draw a new Gephi Networks, and they will be presented in the next sections as our final results.
# 
# # 4. Final Visualizations / Results
# 
# We actually have 3 clusters (0, 1 and 2). The clusters 3, 4 and 5 are composed by single outliers as we can see in the table below:

# In[ ]:


df_classification.query('Label > 2')


# We can draw the clusters that we found and try to understand each one of them:

# ## 4.A. Cluster 0: The Majority
# 
# The first cluster takes a huge number of countries when compared to others. They are formed by a great group of african countries but it seems that Denmark has a strong influence in this cluster:
# 
# ![https://i.ibb.co/18LYmQy/Cluster0.png](https://i.ibb.co/18LYmQy/Cluster0.png)

# ## 4.B. Cluster 1: The Origins
# 
# China takes the main role in this cluster. We can also notice a strong presence of european countries like France and UK:
# 
# ![https://i.ibb.co/wwLhXQr/Cluster1.png](https://i.ibb.co/wwLhXQr/Cluster1.png)

# ## 4.C. Cluster 2: Australia + Other Countries
# 
# ![https://i.ibb.co/VJM4wBK/Cluster2.png](https://i.ibb.co/VJM4wBK/Cluster2.png)

# # 5. Conclusions
# 
# We can use graph / network theory to study the propagation of the virus. It may be useful to understand how the countries should act in case of a new pandemy.
# 
# A given country should aways be aware about new diseases in its "graph main neighbouts". This is not a perfect analysis since some countries have not a good testing rate and are passing by subnotification problems but it's nice to see how we can transform transmission curves in instants of time.
# 
# The sky is the limit **but** it's crucial to have a massive testing policy everywhere in order to obtain better results. And,remember:
# 
# ![https://ca-indosuez.com/var/indosuez/storage/images/_aliases/full_news/6/2/3/3/33326-15-eng-GB/Visuel%20EN.jpg](https://ca-indosuez.com/var/indosuez/storage/images/_aliases/full_news/6/2/3/3/33326-15-eng-GB/Visuel%20EN.jpg)
