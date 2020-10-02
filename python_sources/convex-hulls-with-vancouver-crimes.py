#!/usr/bin/env python
# coding: utf-8

# # Convex Hulls with Vancouver Crimes
# 
# ## Introduction
# 
# Imagine that we have a set of points scattered around on a piece of paper. We are tasked with united these points into some kind of simple contiguous shape. How can we do that? In instances such as this, a convex hull is a simple way of achieving a "nice" polygonal shape: a way of backing a geometry out of a sequence of points.
# 
# A convex hull fitted to a set of points is defined as the smallest shape which contains every point in the set, whilst simultaneously also containing every point on any lines drawn between any two points in that set. Another way to think about it as wrapping a rubber band around our points. What is the smallest-diameter rubber band we will need to contain every point? 
# 
# The illustration below conveys this meaning: here are a bunch of points; here is a band to fit around the points; the convex hull, the blue shape, is the result of "shrinking" a rubber band to "fit" around these points.
# 
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/ConvexHull.svg/258px-ConvexHull.svg.png)
# 
# Convex hulls are a simple concept that's useful in a variety of mathematical and algorithmic contexts.
# 
# ## Formulation
# 
# (this technical formulation is based on chapter 2 of _Convex Optimization_ by Boyd & Vandenerghe)
# 
# Suppose that we have a set of points $x_1, x_2, \ldots, x_k$ such that $\forall x_i \in \mathbb{R}$.
# 
# A **convex combination** of these points is defined by $\theta_1 x_1 + \ldots + \theta_k x_k$ under the condition that (1) $\theta_1 + \ldots + \theta_k = 1$ and $\forall{\theta \in [0, 1]}. \: \theta_i > 0$.
# 
# A **convex hull** $\text{cov}(C)$ is the set of all convex combinations of points in $C$.

# ## Implementation
# 
# There are [many different algorithms](https://en.wikipedia.org/wiki/Convex_hull_algorithms#Algorithms) available for computing the convex hull of a set of points. The simplest of these is known as the [gift wrapping algorithm](https://en.wikipedia.org/wiki/Gift_wrapping_algorithm). It has time complexity $O(nh)$, where $n$ is the number of points in the data and $h$ is the number of points on the convex hull.
# 
# It's so simple it fits in a picture:
# 
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Jarvis_march_convex_hull_algorithm_diagram.svg/625px-Jarvis_march_convex_hull_algorithm_diagram.svg.png)

# ## Application
# 
# The application of convex hulls is most immediate in geospatial analytics. Suppose for example that we have a dataset of individual events geolocated to precise locations. We also have some other categorical variable corresponding with a neighborhood of some kind. However, suppose we don't know the actual shapes of those neighborhoods. We can recover some kind of *idea* of what they look like using convex hulls!
# 
# We'll grab a chunk of the Vancouver Crimes dataset that matches this description.

# In[ ]:


import pandas as pd
crimes = pd.read_csv("../input/crime.csv")


# In[ ]:


crimes[['Latitude', 'Longitude', 'NEIGHBOURHOOD']].head()


# In[ ]:


coords = crimes.query('NEIGHBOURHOOD == "Strathcona"')[['Latitude', 'Longitude']].values


# Here is what the neighborhood of Strathcona looks like in abstract:

# In[ ]:


pd.DataFrame(coords).dropna().plot.scatter(x=0, y=1)
import matplotlib.pyplot as plt
plt.gca().set_aspect('equal')


# Here's an implementation using the convex hull algorithm you should be using, the one packaged into `shapely`.

# In[ ]:


from shapely.geometry import Polygon

Polygon(
    list(
        pd.DataFrame(coords).apply(lambda srs: (srs[0], srs[1]), axis='columns').values
    )
).convex_hull

