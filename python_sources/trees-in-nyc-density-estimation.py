#!/usr/bin/env python
# coding: utf-8

# # Trees in NYC &mdash; Density Estimation
# 
# This is a relatively short notebook to try out the $k$-NN density estimation
# that I investigated in my notebook [$k$-NN Distance Statistics](https://www.kaggle.com/mrganger/k-nn-distance-statistics).
# The main idea is to use the distance to the $k$-th nearest neighbor to estimate the
# local density around that point.
# 
# First, let's load the data from a tree census in NYC.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import pandas as pd
import numpy as np

R = 6371 # Radius of earth in km


# In[ ]:


def load_trees(fname):
    df = pd.read_csv(fname)
    return df[['latitude', 'longitude', 'spc_common']].copy()
tree_df = load_trees('../input/new_york_tree_census_2015.csv')
plt.figure(figsize=[10,10])
sns.scatterplot(x='longitude', y='latitude', data=tree_df.sample(10000));


# I only plotted the first 10,000 GPS points above&mdash;there's a lot data on the trees in NYC&mdash;but
# it seems to have a reasonable amount of detail. Let's try making a density plot using the $k$-NN distance
# method.

# In[ ]:


def to_cartesian(latlon):
    lat,lon = np.radians(latlon.T)
    clat = np.cos(lat)
    return R*np.column_stack([clat*np.cos(lon), clat*np.sin(lon), np.sin(lat)])

def density_estimate(coords, k, extent, grid_res=300, logscale=True, q=0.99):
    from matplotlib.colors import LogNorm
    from scipy.spatial import cKDTree as KDTree
    tree = KDTree(to_cartesian(coords))
    x0,x1,y0,y1 = extent
    dx = (x1-x0)/grid_res
    dy = (y1-y0)/grid_res
    g = np.moveaxis(np.mgrid[x0:x1:dx,y0:y1:dy], 0, -1)
    r = tree.query(to_cartesian(g.reshape(-1,2)), k, n_jobs=-1)[0][:,-1]
    d = (k/np.pi)*r**-2
    d = d.reshape(*g.shape[:-1])
    
    if logscale:
        norm = LogNorm()
    else:
        norm = None
    plt.figure(figsize=[15,12])
    plt.imshow(d, origin='lower', extent=(y0,y1,x0,x1), vmax=np.quantile(d, q), aspect='auto', norm=norm)
    plt.colorbar()
    plt.title("Density estimate of trees recorded in NYC tree census, in trees / km$^2$")
    plt.grid(False)


# In[ ]:


density_estimate(tree_df[['latitude', 'longitude']].dropna().values, 25, [40.48,40.92,-74.3,-73.65], grid_res=800, logscale=False)


# Very interesting! You can start to make out street-level details on this plot. Let's zoom in
# to see a more local picture.

# In[ ]:


density_estimate(tree_df[['latitude', 'longitude']].dropna().values, 25, [40.6,40.7,-74.0,-73.9], grid_res=800, logscale=False)


# In[ ]:


density_estimate(tree_df[['latitude', 'longitude']].dropna().values, 25, [40.62,40.65,-73.98,-73.96], grid_res=800, logscale=False, q=0.99)


# Zoomed in, we can make out the different blocks in the city! It also appears that
# there are a lot more trees along some streets than others (or at least more were reported along
# those streets). To me, this demonstrates that $k$-NN density estimation is pretty effective
# at picking out features without having to play around with too many parameters.
# 
# Just for curiosity's sake, let's try estimating the dimensionality at each location.

# In[ ]:


def dimensional_likelihood(maxdim, dists):
    dims = np.arange(1,maxdim+1)
    l = (np.expand_dims(np.log(dists), axis=-1)*(dims-1) + np.log(dims)).sum(axis=-2)
    return np.exp(l-np.expand_dims(l.max(axis=-1), axis=-1))

def plot_dimension_estimate(coords, k, extent, grid_res=300):
    from matplotlib.colors import LogNorm
    from scipy.spatial import cKDTree as KDTree
    tree = KDTree(to_cartesian(coords))
    x0,x1,y0,y1 = extent
    dx = (x1-x0)/grid_res
    dy = (y1-y0)/grid_res
    g = np.moveaxis(np.mgrid[x0:x1:dx,y0:y1:dy], 0, -1)
    r = tree.query(to_cartesian(g.reshape(-1,2)), k, n_jobs=-1)[0]
    rel = r[:,:-1]/r[:,[-1]]
    d = dimensional_likelihood(3, rel).argmax(axis=1).reshape(*g.shape[:-1])
    
    plt.figure(figsize=[12,12])
    plt.imshow(d, origin='lower', extent=(y0,y1,x0,x1), vmax=2, aspect='auto', cmap='gray')
    plt.title('Dimensionality estimate of trees from NYC tree census (black = 1, grey = 2, white = 3).')
    plt.grid(False)


# In[ ]:


plot_dimension_estimate(tree_df[['latitude', 'longitude']].dropna().values, 10, [40.48,40.92,-74.3,-73.65], grid_res=800)


# In[ ]:


plot_dimension_estimate(tree_df[['latitude', 'longitude']].dropna().values, 10, [40.6,40.7,-74.0,-73.9], grid_res=800)


# In[ ]:


plot_dimension_estimate(tree_df[['latitude', 'longitude']].dropna().values, 10, [40.62,40.65,-73.98,-73.96], grid_res=800)


# The above three images cover the same areas as the density plots. The black areas have a dimension esimate of $d=1$,
# the grey areas $d=2$, and the white areas $d \ge 3$. It seems to do a pretty decent job of picking out the streets as being
# 1-dimensional; there aren't too many open areas (i.e. Central Park) so that dataset is mostly 1-dimensional.
