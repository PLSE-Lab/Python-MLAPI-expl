#!/usr/bin/env python
# coding: utf-8

# ## Particle trajectories in spherical coordinates
# ref.https://www.kaggle.com/wesamelshamy/trackml-problem-explanation-and-data-exploration  
# ref.https://www.kaggle.com/afaist/hdbscan-and-scaling-of-the-coordinates

# In[74]:


import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

import operator


# ### Read one event.

# In[2]:


event_prefix = 'event000001000'
hits, cells, particles, truth = load_event(os.path.join('../input/train_1', event_prefix))


# ### Method to convert to spherical coordinate.

# In[75]:


def cart2spherical(cart):
    r = np.linalg.norm(cart, axis=0)
    theta = np.degrees(np.arccos(cart[2] / r))
    phi = np.degrees(np.arctan2(cart[1], cart[0]))
    return np.vstack((r, theta, phi))


# ### Compare trajectories between Cartesian and spherical coordinates.

# In[96]:


# Get particle id with highest weights
NUM_PARTICLES = 100
truth_dedup = truth.drop_duplicates('particle_id')
truth_sort = truth_dedup.sort_values('weight', ascending=False)
truth_head = truth_sort.head(NUM_PARTICLES)

# Get points where the same particle intersected subsequent layers of the observation material
p_traj_list = []
for _, tr in truth_head.iterrows():
    p_traj = truth[truth.particle_id == tr.particle_id][['tx', 'ty', 'tz']]
    # Add initial position.
    #p_traj = (p_traj
    #          .append({'tx': particle.vx, 'ty': particle.vy, 'tz': particle.vz}, ignore_index=True)
    #          .sort_values(by='tz'))
    p_traj_list.append(p_traj)
    
# Convert to spherical coordinate.
rtp_list = []
for p_traj in p_traj_list:
    xyz = p_traj.loc[:, ['tx', 'ty', 'tz']].values.transpose()
    rtp = cart2spherical(xyz).transpose()
    rtp_df = pd.DataFrame(rtp, columns=('r', 'theta', 'phi'))
    rtp_list.append(rtp_df)

# Plot with Cartesian coordinates.
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
for p_traj in p_traj_list:
    ax.plot(
        xs=p_traj.tx,
        ys=p_traj.ty,
        zs=p_traj.tz,
        marker='o')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm) -- Detection layers')
plt.title('Trajectories of top weights particles in Cartesian coordinates.')

# Plot with spherical coordinates.
fig2 = plt.figure(figsize=(10, 10))
ax = fig2.add_subplot(111, projection='3d')
for rtp_df in rtp_list:
    ax.plot(
        xs=rtp_df.theta,
        ys=rtp_df.phi,
        zs=rtp_df.r,
        marker='o')
ax.set_xlabel('Theta (deg)')
ax.set_ylabel('Phi (deg)')
ax.set_zlabel('R  (mm) -- Detection layers')
plt.title('Trajectories of top weights particles in spherical coordinates.')
plt.show()

