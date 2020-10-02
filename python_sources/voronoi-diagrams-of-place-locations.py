#!/usr/bin/env python
# coding: utf-8

# # Voronoi diagrams of places
# 
# For a given `place_id`, we can find the median of all the x,y values for a check-in event. We will use this to approximate the location of the place being checked-in to.
# 
# 
# Then we will look at the [Voronoi diagrams](https://en.wikipedia.org/wiki/Voronoi_diagram) of those places, we will start with the upper left corner. Since the check-ins are pretty close to uniformly distributed over the given square, it suffices to take the square in the corner defined by `[(0,L), (0,L)]`, we can vary `L` to explore the Voronoi diagrams of a larger and larger swathe of the space, to get a feel for the landscape.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
df_train = pd.read_csv("../input/train.csv")
centroids = df_train[["x","y","place_id"]].groupby("place_id").median()


# Since there are about 100,000 points uniformly distributed in a 10x10 square, we can choose any corner and have it be a reasonably representative sample. After experimenting a bit, I went with the square with corners `(0,0)` and `(0.125,0.125)`, which contains 12 points. This is a good small example to get some familiarity with a little bit of the data, and with using scipy's Voronoi class.

# In[ ]:


cent_small = centroids[(centroids["x"] < 0.125) & (centroids["y"] < 0.125)]
cent_small


# In[ ]:


vor = Voronoi(cent_small, qhull_options="Qc")
voronoi_plot_2d(vor)
plt.show()


# In[ ]:


# Now let's look at a larger area:
vor = Voronoi(centroids[(centroids["x"] < 0.3) & (centroids["y"] < 0.3)], qhull_options="Qc")
voronoi_plot_2d(vor)
plt.show()


# In[ ]:




