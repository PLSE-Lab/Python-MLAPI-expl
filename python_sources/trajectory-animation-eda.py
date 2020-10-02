#!/usr/bin/env python
# coding: utf-8

# ## Simple  Events EDA  - TrackML Particle Tracking 
# 
# 
# In this notebook, I have explored the dataset provied in the [TrackML Particle Tracking Challenge](https://www.kaggle.com/c/trackml-particle-identification). 
# 
# The dataset comprises of multiple independent events, where each event contains simulated measurements (essentially 3D points) of particles generated in a collision between proton bunches at the Large Hadron Collider at CERN. The goal of this challenge is to group the recorded measurements or hits for each event into tracks, sets of hits that belong to the same initial particle. A solution must uniquely associate each hit to one track. The training dataset contains the recorded hits, their ground truth counterpart and their association to particles, and the initial parameters of those particles. 
# 
# **Contents **  
# 
# 1. Read Event Files 
# 2. Read - Cells, Particles, Truth, Hits Files  
# 3. Exploring Cells Dataset   
#   3.1 Distribution of cells.ch0  
#   3.2 Distribution of cells.ch1  
#   3.3 Distribution of cells.value   
#   3.4 mean of "cells.value" by ch0 and ch1  
# 4. Exploring Hits Data  
#   4.1 X, Y, Z global coordinates of particles   
#   4.2 Layer Id for every hit   
#   4.3 Module Id for every hit  
#   4.4 Volume Id for every hit   
#   4.5 Distance of particles from Origin  
#   4.6 Initial Position of particles in 3d space  
# 5. Exploring Particles Data   
#   5.1 Animated trajectory of a positively charged particle   
#   5.2 Animated trajectory of a negatively charged particle  
#   5.3 Plotting the initial x, y, and z coordinates of particles  
#   5.4 Distribution of positively and negatively charged particles   
#   5.5 Distribution of number of hits for different particles  
# 6. Exploring Truths Data  
#   6.1 Truth Data weight per id  

# In[1]:


from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os 

from IPython.display import HTML
from matplotlib import animation
import io, base64
# plt.style.use('fivethirtyeight')

from plotly.offline import init_notebook_mode, iplot
from matplotlib.pyplot import pie, show
import plotly.graph_objs as go
import numpy as np

# from trackml.dataset import load_event, load_dataset
# from trackml.score import score_event

sns.set_style("whitegrid", {'axes.grid' : False})
init_notebook_mode(connected=True)


# ## 1. Read the Event Files
# 
# The full training dataset with 8850 events split into 5 files, In this notebook I am only using train_1 file.

# In[ ]:


train_path = '../input/train_1/'
train_files = os.listdir(train_path)
train_files[:10]


# ### Total events in train_1 file

# In[ ]:


len(train_files)


# ## 2. Read - Cells, Particles, Truth, Hits Files
# 
# In train_1, there are four different files for every event: 
# 
# 1. Event Hits - Details about where particles hit and the position in the 3 dimentional space
# 2. Event Hit Cells  - The cells file contains the constituent active detector cells that comprise each hit. The cells can be used to refine the hit to track association.   
# 3. Event Particles  - The truth file contains the mapping between hits and generating particles and the true particle state at each measured hit. Each entry maps one hit to one particle.  
# 4. Event Truth  - All entries in thie file contain the generated information or ground truth.  

# In[ ]:


cells_df, truth_df, particles_df, hits_df = [], [], [], []
for filename in sorted(train_files)[:20]:
    if "cells" in filename:
        cell_df = pd.read_csv(train_path+filename)
        cells_df.append(cell_df)
    elif "hits" in filename:
        hit_df = pd.read_csv(train_path+filename)
        hits_df.append(hit_df)
    elif "particles" in filename:
        particle_df = pd.read_csv(train_path+filename)
        particles_df.append(particle_df)
    elif "truth" in filename:
        trut_df = pd.read_csv(train_path+filename)
        truth_df.append(trut_df)


# ## 3. Exploring Cells Dataset
# 
# The cells file contains the constituent active detector cells that comprise each hit. The cells can be used to refine the hit to track association. A cell is the smallest granularity inside each detector module, much like a pixel on a screen, except that depending on the volume_id a cell can be a square or a long rectangle. It is identified by two channel identifiers that are unique within each detector module and encode the position, much like column/row numbers of a matrix. A cell can provide signal information that the detector module has recorded in addition to the position. Depending on the detector type only one of the channel identifiers is valid, e.g. for the strip detectors, and the value might have different resolution. 
# 
# hit_id: numerical identifier of the hit as defined in the hits file.  
# ch0, ch1: channel identifier/coordinates unique within one module.  
# value: signal value information, e.g. how much charge a particle has deposited.  

# In[ ]:


# cells_df[0].shape
cells_df[0].head(10)


# There are a total of 664996 entries in this file.  
# 
# ### 3.1 cells.value for single event
# 
# Lets observe the cells.value column for a single event. 

# In[ ]:


def dist(df, col, bins, color, title, kde=False):
    plt.figure(figsize=(15,3))
    sns.distplot(df[col].values, bins=bins, color=color, kde=kde)
    plt.title(title, fontsize=14);
    plt.show();

def mdist(df, col, bins, color, kde=False):
    f, axes = plt.subplots(1, 2, figsize=(15,3))
    sns.distplot(df[1][col].values, bins=bins, color=color, rug=False, ax=axes[0], kde=kde)
    sns.distplot(df[2][col].values, bins=bins, color=color, rug=False, ax=axes[1], kde=kde)

    f, axes = plt.subplots(1, 2, figsize=(15,3))
    sns.distplot(df[3][col].values, bins=bins, color=color, rug=False, ax=axes[0], kde=kde)
    sns.distplot(df[4][col].values, bins=bins, color=color, rug=False, ax=axes[1], kde=kde)

dist(cells_df[0], 'value', 10, 'red', 'cells.value')


# The cell.value takes values between 0 and 1, where most of the cells hit have value on the lower end. 
# 
# Is it true for some over events ? Lets check by plotting the same variable for other events. 
# 
# ### Distribution of cells.value for some other events

# In[ ]:


mdist(cells_df, 'value', 10, 'green', kde=True)


# ### 3.2 cells.ch0 for single event

# In[ ]:


dist(cells_df[0], 'ch0', 100, 'red', 'cells.ch0')
# cells_df[0].ch0.describe()
# cells_df[0].ch0.value_counts()


# The variable cells.ch0 takes values between 0 to 1195 for this event
# 
# ### cells.ch0 for some other events

# In[ ]:


mdist(cells_df, 'ch0', 10, 'green', kde=True)


# ### 3.3 cells.ch1 for single event

# In[ ]:


dist(cells_df[0], 'ch1', 100, 'red', 'cells.ch1')
# cells_df[0].ch1.describe()


# This variable takes values from 0 to 1275, and by looking at above graph it seems that most of the entries have ch1 = 0
# 
# ### cells.ch1 for some other events

# In[ ]:


mdist(cells_df, 'ch1', 10, 'green', kde=True)


# ### 3.4 mean of "value" for ch0 and ch1

# In[ ]:


ch0df = cells_df[0].groupby('ch0').agg({'value' : 'mean'}).reset_index()
ch1df = cells_df[0].groupby('ch1').agg({'value' : 'mean'}).reset_index()

f, axes = plt.subplots(2, 1, figsize=(15,10))
sns.regplot(x='ch0', y='value', data=ch0df, fit_reg=False, color='#ff4c64', ax=axes[0])
sns.regplot(x='ch1', y='value', data=ch1df, fit_reg=False, color='#89ea7c', ax=axes[1])
axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)


# Mean of value variable is close to zero for many values of ch1 (greater than 180), while the mean value is higher, while for ch0 it is close to zero for smaller values of ch0 and close to 1 for higher values of ch0.
# 
# ### 4. hits data
# 
# Hit data contains following variables: 
# 
# hit_id: numerical identifier of the hit inside the event.  
# x, y, z: measured x, y, z position (in millimeter) of the hit in global coordinates.  
# volume_id: numerical identifier of the detector group.  
# layer_id: numerical identifier of the detector layer inside the group.  
# module_id: numerical identifier of the detector module inside the layer.  

# In[ ]:


hits_df[0].head(10)


# 
# ### 4.1 values of x,y,z for hit in global coordinates

# In[ ]:


# f, axes = plt.subplots(1, 2, figsize=(15,5));
# sns.distplot(hits_df[0].x.values, color='red', rug=False, ax=axes[0])
# sns.distplot(hits_df[0].y.values, color='red', rug=False, ax=axes[1])
# axes[0].set_title("distribution of x coordinate of particles");
# axes[1].set_title("distribution of y coordinate of particles");

# f, axes = plt.subplots(1, 2, figsize=(15,5));
# sns.distplot(hits_df[0].z.values, color='red', rug=False, ax=axes[0])
# sns.regplot(x=hits_df[0][:2000].x.values, y=hits_df[0][:2000].y.values, fit_reg=False, color='#ff4c64', ax=axes[1])
# axes[0].set_title("distribution of z coordinate of particles");
# axes[1].set_title("plotting of x and y of particles");
# axes[1].set(xlabel='x', ylabel='y');

hits_small = hits_df[0][['x','y','z']]
sns.pairplot(hits_small, palette='husl', size=6)
plt.show()


# ### 4.2 hits.volume_id for single event

# In[ ]:


dist(hits_df[0], 'volume_id', 20, 'red', 'hits.volume_id distribution', kde=True)
# hits_df[0].volume_id.value_counts()


# volume_id takes these values - 7,8,9,12,13,14,16,17,18
# 
# ### hits.volume_id for some other events

# In[ ]:


mdist(hits_df, 'volume_id', 20, 'green')


# ### 4.3 hits.layer_id for single event

# In[ ]:


dist(hits_df[0], 'layer_id', 20, 'red', 'hits.layer_id distribution', kde=False)


# layer_id distinct values : 2,4,6,8,10,12,14
# 
# ### 4.4 hits.module_id for single event

# In[ ]:


dist(hits_df[0], 'module_id', 100, 'green', 'hits.module_id distribution')


# ### 4.5 Distance of particles from origin
# 
# lets compute the distance of particles from the origin

# In[ ]:


hits_df[0]['origin_dis'] = np.sqrt(np.square(hits_df[0].x) + np.square(hits_df[0].y) + np.square(hits_df[0].z))
dist(hits_df[0], 'origin_dis', 100, 'red', 'hits.origin_dis distribution')


# ### 4.5 Initial Position of particles 

# In[ ]:


from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

fig = pyplot.figure(figsize=(10,8))
# plt.figure();

ax = Axes3D(fig)

hits_sample = hits_df[0].sample(3000)
ax.scatter(hits_sample.x, hits_sample.y, hits_sample.z)
pyplot.show()


# ### 5. particles data
# 
# Particles data consists of following variables:  
# 
# 
# particle_id: numerical identifier of the particle inside the event.  
# vx, vy, vz: initial position or vertex (in millimeters) in global coordinates.  
# px, py, pz: initial momentum (in GeV/c) along each global axis.  
# q: particle charge (as multiple of the absolute electron charge).  
# nhits: number of hits generated by this particle.  
# 

# In[ ]:


# truth_df[0][truth_df[0]['particle_id'] == 414345390649769984]
# particle_df.sort_values(['nhits'])[particle_df.nhits == 12]
# 4513357793067008, 112595763120308224

vals = list(truth_df[0][truth_df[0]['particle_id'] == 112595763120308224].hit_id.values)
tempdf = hits_df[0][hits_df[0]['hit_id'].isin(vals)]
tempdf
fig = plt.figure(figsize = (10,10))
ax = Axes3D(fig)
plt.style.use('fivethirtyeight')
def animate(hit_id):
    ax.set_xlim([0,900])
    ax.set_ylim([-600,300])
    ax.set_zlim([-300,-10])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
#     x = tempdf.x[tempdf.hit_id == hit_id]
#     y = tempdf.y[tempdf.hit_id == hit_id]
#     z = tempdf.z[tempdf.hit_id == hit_id]
    
    
    x = tempdf.x[tempdf.hit_id == hit_id]
    y = tempdf.y[tempdf.hit_id == hit_id]
    z = tempdf.z[tempdf.hit_id == hit_id]
    s = tempdf.module_id[tempdf.hit_id == hit_id] 
#     c = tempdf.layer_id[tempdf.hit_id == hit_id]
    ax.scatter(x, y, z, s=300)# 'o', color = 'r', markersize = 10, alpha = 0.5) 
    
ani = animation.FuncAnimation(fig, animate, tempdf.hit_id.unique().tolist())
ani.save('animation1.gif', writer='imagemagick', fps=2)


# truth_df[0][truth_df[0]['particle_id'] == 414345390649769984]
# particle_df.sort_values(['nhits'])[particle_df.nhits == 12]
# 4513357793067008, 112595763120308224

vals = list(truth_df[0][truth_df[0]['particle_id'] == 968275088115761152].hit_id.values)
tempdf = hits_df[0][hits_df[0]['hit_id'].isin(vals)]
# tempdf
fig = plt.figure(figsize = (10,10))
ax = Axes3D(fig)
plt.style.use('fivethirtyeight')
def animate(hit_id):
#     ax.set_xlim([-60,20])
#     ax.set_ylim([0,800])
#     ax.set_zlim([-100,-3000])

    ax.set_xlim([20,1100])
    ax.set_ylim([-110,0])
    ax.set_zlim([0,1300])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
#     x = tempdf.x[tempdf.hit_id == hit_id]
#     y = tempdf.y[tempdf.hit_id == hit_id]
#     z = tempdf.z[tempdf.hit_id == hit_id]
    
    
    x = tempdf.x[tempdf.hit_id == hit_id]
    y = tempdf.y[tempdf.hit_id == hit_id]
    z = tempdf.z[tempdf.hit_id == hit_id]
    ax.scatter(x, y, z, s=300)# 'o', color = 'r', markersize = 10, alpha = 0.5)
    
ani = animation.FuncAnimation(fig, animate, tempdf.hit_id.unique().tolist())
ani.save('animation2.gif', writer='imagemagick', fps=2)


# In[ ]:


particles_df[0].head(10)


# ### 5.1 Lets see the animated trajectory of a Positively Charged Particle 

# In[ ]:


filename = 'animation1.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))


# ### 5.2 Animated Trajectory of a negatively charged particle 

# In[ ]:


filename = 'animation2.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))


# ### 5.3 Plotting the initial x,y,z coordinates of particles and initial momentum values for the particles

# In[ ]:


f, axes = plt.subplots(1, 2, figsize=(15,3));
sns.distplot(particles_df[0].vx.values, color='green', rug=False, ax=axes[0])
sns.distplot(particles_df[0].px.values, color='green', rug=False, ax=axes[1])
axes[0].set_title("x coordinate of particles");
axes[1].set_title("momentum of particle in x direction ");

f, axes = plt.subplots(1, 2, figsize=(15,3));
sns.distplot(particles_df[0].vy.values, color='green', rug=False, ax=axes[0])
sns.distplot(particles_df[0].py.values, color='green', rug=False, ax=axes[1])
axes[0].set_title("y coordinate of particles");
axes[1].set_title("momentum of particle in y direction ");

f, axes = plt.subplots(1, 2, figsize=(15,3));
sns.distplot(particles_df[0].vz.values, color='green', rug=False, ax=axes[0])
sns.distplot(particles_df[0].pz.values, color='green', rug=False, ax=axes[1])
axes[0].set_title("z coordinate of particles");
axes[1].set_title("momentum of particle in z direction ");

f, axes = plt.subplots(1, 2, figsize=(15,5));
sns.regplot(x=particles_df[0][:12000].vx.values, y=particles_df[0][:12000].vy.values, fit_reg=False, color='#ff4c64', ax=axes[0])
sns.regplot(x=particles_df[0][:12000].vx.values, y=particles_df[0][:12000].vz.values, fit_reg=False, color='#ff4c64', ax=axes[1])
axes[0].set_title("x and y position of particles");
axes[0].set(xlabel='x', ylabel='y');
axes[1].set_title("x and z position of particles");
axes[1].set(xlabel='x', ylabel='z');


# ### 5.4 particles charge

# In[ ]:


plt.figure(figsize=(5, 4))
cnts = particles_df[0]['q'].value_counts()
pie(cnts.values, labels=cnts.index, colors=['#8ded82', '#f45342']);
show()


# More than half particles have positive charge in this event as compared to negatively charged particles 
# 
# ### 5.5 particles.nhits

# In[ ]:


dist(particles_df[0], 'nhits', 10, 'red', 'particles.nhits distribution', kde=True)


# ### 5.6 Pair plotting of initial positions of the particles

# In[ ]:


psmall = particles_df[0][['vx','vy','vz']]
sns.pairplot(psmall, palette='husl', size=6)
plt.show()


# ### 6 Truths Data
# 
# truths data consists of following variables:  
# 
# hit_id: numerical identifier of the hit as defined in the hits file.  
# particle_id: numerical identifier of the generating particle as defined in the particles file.   
# tx, ty, tz true intersection point in global coordinates (in millimeters) between the particle trajectory and the sensitive surface.  
# tpx, tpy, tpz true particle momentum (in GeV/c) in the global coordinate system at the intersection point.  
# weight per-hit weight used for the scoring metric; total sum of weights within one event equals to one.  

# In[ ]:


truth_df[0].head(10)


# ### 6.1 Truth data Weight per hit

# In[ ]:


dist(truth_df[0], 'weight', 10, 'red', 'truth.weight distribution', kde=True)


# Thanks for viewing the notebook. An upvote wi
