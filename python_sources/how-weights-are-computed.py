#!/usr/bin/env python
# coding: utf-8

# This notebook shows how weights are computed.  First, let's import som packages.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Then let's read one event data, and put it in a single data frame.

# In[ ]:


event_id = 3
hits = pd.read_csv('../input/train_1/event00000100%d-hits.csv' % event_id)
particles = pd.read_csv('../input/train_1/event00000100%d-particles.csv' % event_id)
truth = pd.read_csv('../input/train_1/event00000100%d-truth.csv' % event_id)

hits = hits.merge(truth, how='left', on='hit_id')
hits = hits.merge(particles, how='left', on='particle_id')


# Then let's sort the data by particle id then by distance to the track origin.

# In[ ]:


hits['dv'] = np.sqrt((hits.vx - hits.tx) ** 2 +                      (hits.vy - hits.ty) ** 2 +                      (hits.vz - hits.tz) ** 2)
hits = hits.sort_values(by=['particle_id', 'dv']).reset_index(drop=True)


# We're almost there.  We compute a rank in percentagfe for each hit along a given particle track.

# In[ ]:


hits['rank'] = hits.groupby('particle_id').cumcount()
hits['len'] = hits.groupby('particle_id').particle_id.transform('count')
hits['rank'] = (hits['rank']) / (hits['len'] - 1)


#    And we normalize weights so that the largest weight on each track is 1.

# In[ ]:


hits['weight'] /= hits.groupby('particle_id').weight.transform('max')


# We can now plot the normalized weight along each track.  

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(15, 15))

ax.scatter(hits['rank'], hits['weight'], alpha=0.1, marker='+')


# The pattern is quite clear, isn't it? Note that there are surious points away from the main curve.  Aftert looking at them, they belong to particles with strange tracks, probably errors in the simulation.
