#!/usr/bin/env python
# coding: utf-8

# # $k$-NN Distance Statistics
# 
# One of my current interests is in non-parametric statistical methods. The reason for this is two-fold: (1) non-parametric statistics require much weaker assumptions about data and (2) it is an interesting way to approach statistics. I encourage you to read up on some non-parametric statistics when if you get the chance.
# 
# This notebook investigates the statistics of multi-dimensional manifolds ("surfaces" of arbitrary dimension). Specifically, it considers the statistics of $k$-nearest neighbor point sets in different dimensions. This notebook only uses simulated data; for an application of some of the ideas, see my kernel ["Trees in NYC - Density Estimation"](https://www.kaggle.com/mrganger/trees-in-nyc-density-estimation).
# 
# ## Generation Realistic Distributions
# 
# First, we need to generate good examples. The first question to address is how to do this generically so that it would apply to a "real" dataset (i.e. an arbitrary distribution of points). Recall the foundation of calculus: if you zoom in enough to any continuous function, it will look linear and flat. This is good because if we sample a (continuous) probability distribution with a large number of points and look at a group of nearby points, they won't be very far apart. Locally, the probability density will be approximately constant. This is illustrated below:

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

points = pd.DataFrame(np.random.randn(100000,2), columns=['x','y'])
points['dist_std'] = np.max(np.abs(points-1), axis=1)
fig, ax = plt.subplots(2, 2, figsize=[20,20])
sns.scatterplot(x='x', y='y', ax=ax[0,0], data=points.query('dist_std < 8')[:300]);
sns.scatterplot(x='x', y='y', ax=ax[0,1], data=points.query('dist_std < 2')[:300]);
sns.scatterplot(x='x', y='y', ax=ax[1,0], data=points.query('dist_std < 0.5')[:300]);
sns.scatterplot(x='x', y='y', ax=ax[1,1], data=points.query('dist_std < 0.125')[:300]);


# Each plot shows 300 samples, each drawn a standard 2-dimensional normal distribution. However, each plot draws a successively smaller box around $(x=1,y=1)$. Visually, this supports the point made above: if you zoom in enough to *any* locally-continuous probability distribution, it will look uniform. Importantly, this is *not* true for probability distributions with sharp discontinuities. You can zoom in all you want on the edge of a discontinuity and it will continue to look the same.
# 
# ## Distance Statistics and Dimensionality
# 
# Armed with this knowledge, we are free to simulate $k$-NN scenarios on uniform distributions as long as we assume that we will have "enough" samples when given a real dataset. The first thing to investigate is the distribution of distances to the neighbors of a point, relative to the $k$th nearest neighbor.

# In[ ]:


def vol_ball(n,r=1):
    from scipy.special import gammaln
    return np.exp(np.log(np.pi)*n/2 - gammaln(n/2+1) + np.log(r)*n)

def vol_cube(n,r=1):
    return (2*r)**n

def distance_stats(dim=1, k=2, buffer=5, samples=1000):
    p = vol_ball(dim)/vol_cube(dim)
    n = k/p
    m = int(np.ceil(n+5*np.sqrt(n)))
    pop = np.random.uniform(-1, 1, size=(samples,m,dim))
    dist = np.sort(np.linalg.norm(pop, axis=2), axis=1)
    return dist[:,:k-1]/dist[:,[k-1]]


# The function `distance_stats` generates the statistics of the relative distances of
# the $k$-NN of a point in a uniform distribution. It works by assuming a point exists
# at the origin and samples points randomly in a $d$-dimensional cube. Since the distance
# to the $k$-th point corresponds to the radius of a ball that contains the closer points,
# we have to ensure that we have enough samples that the ball has a radius less than the
# cube it is contained in.
# 
# Now, let's plot the statistics for a couple different dimensions.

# In[ ]:


def plot_dim(dim):
    d = distance_stats(dim=dim, k=5, samples=2000).ravel()
    plt.plot(np.sort(d), np.linspace(0,1,len(d)+1)[1:], label=str(dim))
    plt.legend(title='Dimension')
    plt.ylabel('CDF')
    plt.xlabel('Relative Distance')

plt.figure(figsize=[10,10])
plt.title('CDF of relative distances of $k$-NN in a uniform distribution of different dimensions.')
plot_dim(1)
plot_dim(2)
plot_dim(3)
plot_dim(4)


# Ok, nothing crazy here. The number of points within a radius $r$ in a $d$-dimensional space is proportional to $r^d$ (which is propotional to the volume of a $d$-dimensional ball). The interesting part is that this allows us to estimate the local dimension of the manifold around a point using its $k$ nearest neighbors. Specifically, let $F_n(x) = x^d$ be the CDF of the relative distances to the nearest neighbors of a point. The PDF is given by $f_n(x) = nx^{n-1}$. The joint PDF of $k$ observations is given by:
# $$f_n(x_1,\ldots,x_k) = n^k x_1^{n-1} \cdots x_k^{n-1}.$$
# This is simple enough to implement. To account for runaway values, we compute it in the log domain and normalize it.

# In[ ]:


def dimensional_likelihood(maxdim, dists):
    dims = np.arange(1,maxdim+1)
    l = (np.expand_dims(np.log(dists), axis=-1)*(dims-1) + np.log(dims)).sum(axis=-2)
    return np.exp(l-np.expand_dims(l.max(axis=-1), axis=-1))


# Let's try this on some simulated neighbor sets and see how precise it is.

# In[ ]:


ks = [2,5,10,20,30]
max_dim = 20

def plot_dim(*dims):
    fig, axes = plt.subplots(len(dims), len(ks), figsize=[20,3*len(dims)], sharex=True, sharey=True)
    axes[-1,0].set_xlabel('Dimension')
    axes[-1,0].set_ylabel('Relative Likelihood')
    
    for k, axrow in zip(ks, axes):
        for dim, ax in zip(dims, axrow):
            ax.bar(np.arange(1,max_dim+1), dimensional_likelihood(max_dim, distance_stats(dim=dim, k=k, samples=1)[0]));
            ax.set_title('$d$ = {}, $k$ = {}'.format(dim,k))


# In[ ]:


plot_dim(1,2,5,10,15)


# Ok, so not surprisingly, the precision of the estimate of $d$ increases with increasing $k$ and decreases with increasing $d$. It also seems that if $d$ is large, $k$ needs to be a lot bigger to have the same amount of precision. I'm sure there's a lot more theory that can be teased out of this&mdash;I'm not really sure where to start looking for existing research on this&mdash;but it seems to be a pretty simple way to estimate the dimensionality of points in a dataset. Let's try using this to label the dimensionality of points in a mixed dataset.

# In[ ]:


def test_mixed(k, samples=1000):
    square = np.random.uniform(-2, 0, size=(samples, 2))
    line = np.random.uniform(0, 1, size=(samples,1))*[1,1]
    x = np.vstack([square,line])
    
    from scipy.spatial import cKDTree as KDTree
    tree = KDTree(x)
    dists = tree.query(x, k=k+1)[0][:,1:]
    dists = dists/dists[:,[-1]]
    
    import pandas as pd
    
    df = pd.DataFrame(x, columns=['x','y'])
    dim = dimensional_likelihood(2, dists).argmax(axis=1)
    plt.figure(figsize=[15,15]);
    for i in range(np.max(dim)+1):
        sns.scatterplot(x=x[dim==i,0], y=x[dim==i,1])
    plt.legend(np.arange(1,2+np.max(dim)).astype(str), title='Estimated Dimension')


# In[ ]:


test_mixed(30, samples=100);


# Nice! With a $k$ value of 30, we seem to be able to estimate the dimensionality at each point reasonably well.
# 
# ## Non-parametric Density Estimation
# 
# A common goal in statistics is to estimate density functions. Generally, this is done via fitting a model (parametric) or using kernel density estimation ("non-parametric"). The problem with kernel density estimation (KDE) is that it actually *does* require parameters and model-based assumptions: usually some distribution shape (the kernel) and the bandwidth (the size of the kernel). A gaussian kernel is usually a good choice unless you have some good intuition about the model. But what width should you choose? This usually requires some domain-specific knowledge such as:
# 
# 1. The typical distance between points from the distribution.
# 2. The desired fidelity of the model (how blurry it is).
# 
# Personally, I dislike #1 (#2 is reasonable). Shouldn't it be possible to infer the typical distance directly from the data at hand? Actually, it is...and it's quite easy. Easier than understanding KDE, in my opinion. The process goes like this:
# 
# 1. Load the data into a $k$-d tree.
# 2. For each location you are interested in, find the distance $r$ to the the $k$-th nearest neighbor.
# 3. Estimate the density at those locations as $\rho = \frac{k}{V_d(r)}$, where $V_d(r)$ is the volume of a $d$-ball of radius $r$, given by: $$V_d(r) = \frac{\pi^\frac{d}{2}}{\Gamma\left(\frac{d}{2}+1\right)}r^d.$$
# 
# Simple enough. The hardest thing here is to compute $V_d(\cdot)$, but you can skip this step if you're only trying to get relative densities. Let's wrap this up in a simple function:

# In[ ]:


def density_estimate(loc, x, k):
    d = x.shape[1]
    v = vol_ball(d)
    
    from scipy.spatial import cKDTree as KDTree
    tree = KDTree(x)
    r = tree.query(loc.reshape(-1,d),k, n_jobs=-1)[0][:,-1].reshape(*loc.shape[:-1])
    
    return (k/v)*r**-d


# Ok, let's test it out on a uniform distribution.

# In[ ]:


def test_for_size(size):
    k = int(np.sqrt(size))
    x = np.random.uniform(-0.5,0.5, size=(size,2))
    g = np.moveaxis(np.mgrid[-1:1:0.005,-1:1:0.005], 0,-1)
    d = density_estimate(g, x, k)/len(x)

    fig, ax = plt.subplots(1, 2, figsize=[20,10])
    sns.scatterplot(ax=ax[0], x=x[:,0], y=x[:,1])
    ax[0].set_xlim([-1,1])
    ax[0].set_ylim([-1,1])
    ax[1].imshow(d, origin='lower', vmin=0, vmax=4.0, extent=(-1,1,-1,1));
    ax[1].set_title('k={}'.format(k))
    ax[1].grid(False)


# In[ ]:


test_for_size(100)


# In[ ]:


test_for_size(1000)


# In[ ]:


test_for_size(10000)


# In[ ]:


test_for_size(1000000)


# Neat! Note that I set $k$ to be the square root of the number of samples, mostly just for display purposes. In essence, $k$ controls the acceptable variations in the output. "What's this?" you say, "We now have a *different* parameter to set instead of the bandwidth!" This is a valid point; my answer is that the choice of $k$ requires less *a priori* knowledge: just the underlying dimension of the dataset and an acceptable amount of variation in the output.
# 
# I won't prove this rigorously, I'll do it empirically instead:

# In[ ]:


def relative_error(ks, size):
    k = int(np.sqrt(size))
    x = np.random.uniform(-0.5,0.5, size=(size,2))
    g = np.moveaxis(np.mgrid[-0.25:0.25:0.005,-0.25:0.25:0.005], 0,-1)
    devs = [(density_estimate(g, x, k)/len(x)).std() for k in ks]
    
    plt.figure(figsize=[15,10])
    sns.lineplot(ks, devs)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Relative Standard Deviation')
    plt.xlabel('k')

relative_error(2**np.arange(1,11), 100000)


# Ok, looks like on a log-log plot the relative error is approximately linear. Based on this plot, I estimate that (in 2 dimensions) around $k=10^4$ the relative standard deviation of the density estimate is about 1% of the density. What does this mean? Using $k$-NN distances to estimate the density is *very* data-hungry. Obviously, methods with more parameters (and more assumptions) are much more efficient when applied to data with a small number of parameters. In return for being inefficient, however, comes a significant amount of flexibility. The major advantages that I can see are:
# 
# 1. It doesn't require *a priori* knowledge about spatial relationships.
# 2. The dimensionality can be automatically inferred (and used to make density estimates).
# 3. It is applicable in cases where points are not explicitly represented by coordinates, but it is possible to estimate pairwise relationships (i.e. any graph).
# 4. The dataset has features that occur on **wildly different scales**.
# 
# To me, #1-3 are interesting, but #4 is the most common. Classic KDE cannot handle this; the bandwidth needs to be adaptive. On the other hand, $k$-NN handles this natively. As visual proof:

# In[ ]:


def test_multi_scale(size):
    k = int(np.sqrt(size))
    x = np.random.uniform(-0.5,0.5, size=(size,2))
    n1 = size//3
    n2 = size*2//3
    x = np.vstack([x[:n1], 0.1*x[n1:n2], 0.01*x[n2:]])
    g = np.moveaxis(np.mgrid[-1:1:0.005,-1:1:0.005], 0,-1)
    d = density_estimate(g, x, k)/len(x)
    
    fig, ax = plt.subplots(1, 2, figsize=[20,10])
    sns.scatterplot(ax=ax[0], x=x[:n1,0], y=x[:n1,1])
    sns.scatterplot(ax=ax[0], x=x[n1:n2,0], y=x[n1:n2,1])
    sns.scatterplot(ax=ax[0], x=x[n2:,0], y=x[n2:,1])
    ax[0].set_xlim([-1,1])
    ax[0].set_ylim([-1,1])
    ax[1].imshow(np.log(d), origin='lower', extent=(-1,1,-1,1));
    ax[1].set_title('k={}'.format(k))
    ax[1].grid(False)


# In[ ]:


test_multi_scale(100000)


# Nice! To make the plot above I combined 3 uniformly distributed squares each with 1/3 of the total number of points. The side lengths of the squares are 1, 0.1, and 0.01. I plotted the color on a log scale because it is hard to see all three on the same plot. The key observation is that $k$-NN **automatically adapts to features with different scales**. The clarity of these features is mostly related to the number of samples, while the amount of noise in the plot is related to $k$.
# 
# To see this in action, check out my notebook [Trees in NYC - Density Estimation](https://www.kaggle.com/mrganger/trees-in-nyc-density-estimation).
