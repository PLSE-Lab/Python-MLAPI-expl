#!/usr/bin/env python
# coding: utf-8

# # Quick Problem Description
# Link every *track* to one *hit*.
# 
# *Disclaimer: over simplifying physics to understand this problem*.
# 
# Every particle leaves a track behind it.  Like a car leaving tyre marks in the sand.  We did not catch the particle in action.  Now we want to link every track (tyre marks) to one hit that the particle created.
# 
# In every **event**, a large number of **particles** are released.  They move along a path leaving behind their **tracks**.  They eventually **hit** a particle detector surface on the other end.
# 
# In the training data we have the following information on each **event**:
# - **Hits**: $x, y, z$ coordinates of each hit on the particle detector.
# - **Particles**: Each particle's initial position ($v_x, v_y, v_z$), momentum ($p_x, p_y, p_z$), charge ($q$) and number of hits.
# - **Truth**: Mapping between hits and generating particles.  The particle's trajectory, momentum and the hit weight.
# - **Cells**: Precise location of where the particle hit the detector and how much energy it deposited.
# 

# # Data Exploration:
# 
# #### Import `trackml-library`
# The easiest and best way to load the data is with the [trackml-library] that was built for this purpose.
# 
# Under your kernel's *Settings* tab -> *Add a custom package* -> *GitHub user/repo* (LAL/trackml-library)
# 
# Restart your  kernel'
# s session and you will be good to go.
# 
# [trackml-library]: https://github.com/LAL/trackml-library

# In[1]:


import numpy as np
import pandas as pd

from trackml.dataset import load_event
from trackml.randomize import shuffle_hits
from trackml.score import score_event

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


hits, cells, particles, truth = load_event('../input/train_1/event000001000')


# ## Hits Data
# 
# ### Where Did it Hit?
# Here we have the $x, y, z$ global coordinates (in millimeters) of where the particles hit the detector surface.

# In[6]:


hits.head()


# Here is the distribution of $x, y, z$ location of hits in event 1000.  This is only for one out of 8,850 events.
# 
# ### Vertical Intersection ($x, y$) in Detection Layers
# As shown in the figure below, the hits are semi evenly distributed on the detector surface $x, y$.  We can see a blind circle where probably there are no detectors or there is a surface deformation.  Also, there are a few scattered spots where no hits where detected, probably due to defective detectors.
# 
# The colors represent different detector volumes.  Thanks to [Joshua Bonatt's notebook][josh].
# 
# [josh]: https://www.kaggle.com/jbonatt/trackml-eda-etc

# In[7]:


g = sns.jointplot(hits.x, hits.y,  s=1, size=12)
g.ax_joint.cla()
plt.sca(g.ax_joint)

volumes = hits.volume_id.unique()
for volume in volumes:
    v = hits[hits.volume_id == volume]
    plt.scatter(v.x, v.y, s=3, label='volume {}'.format(volume))

plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.legend()
plt.show()


# ### Horizonal Intersection ($y, z$) in Detection Layers
# You can think of the chart below as a horizontal intersection in the detection surface, where every dot is a hit.  Notice the relationship between the different activity levels in this chart and the one above for $x, y$.
# 
# Again, the colors represent different volumes in the detector surface.

# In[8]:


g = sns.jointplot(hits.z, hits.y, s=1, size=12)
g.ax_joint.cla()
plt.sca(g.ax_joint)

volumes = hits.volume_id.unique()
for volume in volumes:
    v = hits[hits.volume_id == volume]
    plt.scatter(v.z, v.y, s=3, label='volume {}'.format(volume))

plt.xlabel('Z (mm)')
plt.ylabel('Y (mm)')
plt.legend()
plt.show()


# And here is how a 10,000 random (uniform) sample points look like in 3D.  Again, a smaple from one event.  This combiles the previous two charts in 3D.
# 
# Notice how the particles penetrate the detector surface along $z$ coordinate.

# In[9]:


hits_sample = hits.sample(n=10000)
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    xs=hits_sample.x.values,
    ys=hits_sample.y.values,
    zs=hits_sample.z.values
)
ax.set_title('Sample of 1000 Hit Locations')
ax.set_xlabel('X (millimeters)')
ax.set_ylabel('Y (millimeters)')
ax.set_zlabel('Z (millimeters)')
plt.show()


# ### Affected Surface Object
# The **volume**, **layer** and **module** are nested parts on the detector surface.  The volume, is made of layers, which in turn have modules.  Analyzing their response could help us understand if some of them are dead/defective and therefore we may need to account for the bias they cause.
# 
# The figure below shows a plot of every combination of `x`, `y`, `volume`, `layer` and `module`.  The colors identify different *volumes*.  Along the main diagonal we have the variables' historgrams.
# 
# The (`hit_id`, `x`) and (`hit_id`, `y`) pairs show us how different volumes are layered.
# 
# This figure idea is taken from [Joshua Bonatt's notebook][josh].
# 
# [josh]: https://www.kaggle.com/jbonatt/trackml-eda-etc

# In[10]:


hits_sample = hits.sample(8000)
sns.pairplot(hits_sample, hue='volume_id', size=8)
plt.show()


# ## Particles' Data
# The particles data help us understand the particles intitial position, momentum and charge which we can join with the event truth data set to get the particles final position and momentum.  This is needed to identify the tracks that the particle generated.
# 
# Here is how the data look like:

# In[11]:


particles.head()


# ### Hit Rate and Charge Distribution
# Let's see the distribution of the number of hits per particle, shown below.  A significant number of particles had no attributed hits, and most of them have positive charge in this event.

# In[12]:


plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.distplot(particles.nhits.values, axlabel='Hits/Particle', bins=50)
plt.title('Distribution of number of hits per particle for event 1000.')
plt.subplot(1, 2, 2)
plt.pie(particles.groupby('q')['vx'].count(),
        labels=['negative', 'positive'],
        autopct='%.0f%%',
        shadow=True,
        radius=0.8)
plt.title('Distribution of particle charges.')
plt.show()


# ### Initial Position and Momentum
# Let's now take a look at the initial position of the particles around the global coordinates origin $(x, y)=(0,0)$, as shown in the figure below.
# 
# The initial position distribution is more concentrated around the origin (less variance) than it's hit position (shown above under the Hits Data section).  As the particles hit the detection surface they tend to scatter as shown in the particle trajectory plot at the end of this notebook.
# 
# The colors here show the number of hits for each particle.

# In[13]:


g = sns.jointplot(particles.vx, particles.vy,  s=3, size=12)
g.ax_joint.cla()
plt.sca(g.ax_joint)

n_hits = particles.nhits.unique()
for n_hit in n_hits:
    p = particles[particles.nhits == n_hit]
    plt.scatter(p.vx, p.vy, s=3, label='Hits {}'.format(n_hit))

plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.legend()
plt.show()


# And here is the initial position of the particles in a $z$, $y$ view.  Colors show number of hits.

# In[16]:


g = sns.jointplot(particles.vz, particles.vy,  s=3, size=12)
g.ax_joint.cla()
plt.sca(g.ax_joint)

n_hits = particles.nhits.unique()
for n_hit in n_hits:
    p = particles[particles.nhits == n_hit]
    plt.scatter(p.vz, p.vy, s=3, label='Hits {}'.format(n_hit))

plt.xlabel('Z (mm)')
plt.ylabel('Y (mm)')
plt.legend()
plt.show()


# ## Pair plot
# Let's now take a look at the relationship between different pair combinations of the particles' variables.  Again, the colors represent the number of hits.
# 
# There is no large skew in the distribution of the number of hits over other variables.  It looks like the particles are targetted towards the global origin $(x,y)=(0,0)$ and are evenly distributed around it.

# In[ ]:


p_sample = particles.sample(8000)
sns.pairplot(p_sample, vars=['particle_id', 'vx', 'vy', 'vz', 'px', 'py', 'pz', 'nhits'], hue='nhits', size=8)
plt.show()


# ### Particle Trajectory
# We can reconstruct the trajectories for a few particles given their intersection points with the detection layers.  As explained in the [competition evaluation page][ceval], hits from straight tracks have larger wieght, and random or hits with very short tracks have weight of zero.  The figure below show two such examples.
# 
# Thanks to [maka's notebook][traj] for the idea.
# 
# [traj]: https://www.kaggle.com/makahana/quick-trajectory-plot
# [ceval]: https://www.kaggle.com/c/trackml-particle-identification#evaluation

# In[17]:


# Get particle id with max number of hits in this event
particle = particles.loc[particles.nhits == particles.nhits.max()].iloc[0]
particle2 = particles.loc[particles.nhits == particles.nhits.max()].iloc[1]

# Get points where the same particle intersected subsequent layers of the observation material
p_traj_surface = truth[truth.particle_id == particle.particle_id][['tx', 'ty', 'tz']]
p_traj_surface2 = truth[truth.particle_id == particle2.particle_id][['tx', 'ty', 'tz']]

p_traj = (p_traj_surface
          .append({'tx': particle.vx, 'ty': particle.vy, 'tz': particle.vz}, ignore_index=True)
          .sort_values(by='tz'))
p_traj2 = (p_traj_surface2
          .append({'tx': particle2.vx, 'ty': particle2.vy, 'tz': particle2.vz}, ignore_index=True)
          .sort_values(by='tz'))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.plot(
    xs=p_traj.tx,
    ys=p_traj.ty,
    zs=p_traj.tz, marker='o')
ax.plot(
    xs=p_traj2.tx,
    ys=p_traj2.ty,
    zs=p_traj2.tz, marker='o')

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z  (mm) -- Detection layers')
plt.title('Trajectories of two particles as they cross the detection surface ($Z$ axis).')
plt.show()

