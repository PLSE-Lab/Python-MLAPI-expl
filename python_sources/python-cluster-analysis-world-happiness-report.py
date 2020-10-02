#!/usr/bin/env python
# coding: utf-8

# ## For data manipulations and plotting

# In[ ]:



import numpy as np
import pandas as pd    
import matplotlib.pyplot as plt
import seaborn as sns


# ## For Clustering

# In[ ]:


from sklearn import cluster,mixture
import time


# ## For World Map

# In[ ]:


import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_context('poster')
sns.set_color_codes()


# ## For formatting the Scatter Plot

# In[ ]:


plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}


# ## Clustering parameters for various algorithms

# In[ ]:


n_clusters=6


# ## Read input file as dataframe

# In[ ]:


wh=pd.DataFrame.from_csv('../input/2017.csv', sep=',')


# ## Select all columns but Rank and Happiness Score

# In[ ]:


data=wh.iloc[:,2:11]
wh.index


# ## Standardize features

# In[ ]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
data=ss.fit_transform(data)


# ## Check the final dimenstions of the data set

# In[ ]:


data.shape


# ## Create Function to plot clusters as scatter plot and world map

# In[ ]:



def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    if algorithm == mixture.GaussianMixture:
        labels = algorithm(*args, **kwds).fit(data)
        labels = labels.predict(data)
    else:
        labels = algorithm(*args, **kwds).fit_predict(data)
        
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[1], data.T[5], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(-0.10, -2.40, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
    data = dict(type = 'choropleth', 
           locations = wh.index,
           locationmode = 'country names',
           z = labels, 
           text = wh.index,
           colorbar = {'title':'Cluster Group'})
    layout = dict(title = format(str(algorithm.__name__)), 
             geo = dict(showframe = False, 
                       projection = {'type': 'Mercator'}))
    choromap3 = go.Figure(data = [data], layout=layout)
    iplot(choromap3)


# ## Kmeans Clustering

# In[ ]:


plot_clusters(data, cluster.KMeans, (), {'n_clusters':n_clusters})


# ## MeanShift Clustering

# In[ ]:


plot_clusters(data, cluster.MeanShift, (0.1,), {'cluster_all':False})


# ## Spectral Clustering

# In[ ]:


plot_clusters(data, cluster.SpectralClustering, (), {'n_clusters':n_clusters})


# ## DBScan Clustering

# In[ ]:


plot_clusters(data, cluster.DBSCAN, (), {'eps':0.03})


# ## Birch Clustering

# In[ ]:


plot_clusters(data, cluster.Birch, (), {'n_clusters':n_clusters})


# ## AffinityPropagation Clustering

# In[ ]:


plot_clusters(data, cluster.AffinityPropagation, (), {'preference':-5.0, 'damping':0.95})


# ## GaussianMixture Clustering

# In[ ]:


plot_clusters(data, mixture.GaussianMixture, (), {'n_components':n_clusters, 'covariance_type':'full'})


# ## AgglomerativeClustering

# In[ ]:



plot_clusters(data, cluster.AgglomerativeClustering, (), {'n_clusters':n_clusters, 'linkage':'ward'})


# ## Observations:
#  Even though the data set is less, to create proper clusters.
#  We can notice that MeanShift took the most time for clustering and AgglomerativeClustering the least
