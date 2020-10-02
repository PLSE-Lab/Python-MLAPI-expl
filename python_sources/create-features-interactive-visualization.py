#!/usr/bin/env python
# coding: utf-8

# ![abc](http://www.pbs.org/wgbh/nova/next/wp-content/uploads/2015/03/cms-inner-tracker-barrel1.jpg)
# Retrieved from http://www.pbs.org/wgbh/nova/next/physics/lhc-accidental-rainbow-universe/

# # Objective
# The host hands us a large dataset containing records of a bunch of particle detectors and wants us to perdict what kind of particle is hitting the detector. According to Pauli Exclusion Principle, no two fermion could occupy a same quantum state meaning every particle is unique just like us. Since the information shows the particle here is fermion, the objective is to label every jiterring paricle released in each collision event. If we see the problem in a semi-classical way, each particle would has it's unique tarjectory after bouncing off from the collision. Therefore if we could figure out a model to perdict the trajectory of the particle, we are indirectly labeling the particle.

# # Dependency
# Remember to add the trackml package to the notebook
# 
# How: 
# 
# click the ">" beside the "Commit&Run" botton 
#     
# go to the setting, Add a custom package
# 
# type "LAL/trackml-library" to the GitHub user/repo and click the arrow
# 
# wait until done and restart the kernal by clicking the refrashing button at the bottom

# # Note
# *  particle_id 0 in the "truth" file represents noise particles which has a relatively high momentum.
# *  column value in the "cells" file represents the signal value from the detector but the host do not left too many information to us (need to figure it out).
# * The apparatus is most like a pipe placed horizontally. 
# * Detectors are in a multi-layers cylindrical arragement and are coaxial with the apparatus.
# * Each detector is a square tile. (Silicon Semiconductor?)
# * x, y axis is in the cross section (transverse) plane of the apparatus while z axis represent the "width" of the pipe
# * a strong magnetic field in z-axis bend the particle into a helix trajectory
# * mostly the particle is circulating in the x-y plane 

# # Helpful Resources:
# * Announcement from the host https://www.kaggle.com/c/trackml-particle-identification/discussion/55708
# * Guideline from the host https://kaggle2.blob.core.windows.net/forum-message-attachments/321278/9331/trackml-participant-document-particle-v1.0.pdf
# * List of Elementary Particle https://en.wikipedia.org/wiki/Elementary_particle
# * Useful Background Information by Heng CherKeng https://www.kaggle.com/c/trackml-particle-identification/discussion/55726#335835
# * DL Approachs
#     * Incorporating Deep Learning https://www.kaggle.com/c/trackml-particle-identification/discussion/57503#335377
#     * cone slicing, straightening helix and fitting https://www.kaggle.com/c/trackml-particle-identification/discussion/55726#335835
# * Clustering Approchs
#     * Flattening or unrolling tracks in polar (r,s,c) coordinates https://www.kaggle.com/c/trackml-particle-identification/discussion/58078#337285
#     * metric learning + clustering https://www.kaggle.com/c/trackml-particle-identification/discussion/57931#336566
# * Visualization & Dimension Reduction
#     * Transformation Visualization (fixed) osa111 https://www.kaggle.com/osa111/transformation-visualization-fixed
#     * analyzing results of LB 0.4922 https://www.kaggle.com/c/trackml-particle-identification/discussion/57947#336192
#     * Tilted Tracks https://www.kaggle.com/c/trackml-particle-identification/discussion/57931#336566

# In the training data we have the following information on each **event**:
# - **Hits**: $x, y, z$ coordinates of each hit on the particle detector
# - **Particles**: Each particle's initial position ($v_x, v_y, v_z$), momentum ($p_x, p_y, p_z$), charge ($q$) and number of hits
# - **Truth**: Mapping between hits and generating particles; the particle's trajectory, momentum and the hit weight
# - **Cells**: Precise location of where each particle hit the detector and how much energy it deposited

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

# In[2]:


import pdb
import os
import copy

import numpy as np
import pandas as pd

from trackml.dataset import load_event
from trackml.randomize import shuffle_hits
from trackml.score import score_event

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[3]:


device = pd.read_csv('../input/detectors.csv')


# In[ ]:


event_prefix = 'event000001000'
hits, cells, particles, truth = load_event(os.path.join('../input/train_1', event_prefix))

mem_bytes = (hits.memory_usage(index=True).sum() 
             + cells.memory_usage(index=True).sum() 
             + particles.memory_usage(index=True).sum() 
             + truth.memory_usage(index=True).sum())
print('{} memory usage {:.2f} MB'.format(event_prefix, mem_bytes / 2**20))


# ## Tracjectory Data
# 
# ### the truth data
# Here we have: 
# * $tx, ty, tz$ global coordinates (in millimeters) of where the particles hit the detector surface
# * $tp_x, tp_y, tp_z$ the momentum component  (in $GeV/c$) of the particles in each direction [c means light speed, eV is electron volt]
# * $weight$ the weight for evaluation the predicted particle_id. 
# * $particle id$ our objective

# In[4]:


truth.sort_values(by="hit_id",inplace=True)
truth.head()


# ### Discovey1: the host don't care about the noise particles
# We have not penalty from the noise.

# In[5]:


truth['hasweight'] = ~np.equal( truth.weight.values, 0 )
truth[['particle_id','weight','hasweight']].head()


# In[6]:


# store the magnitude of the momentum
truth['tp'] = np.sqrt(truth['tpx']**2+truth['tpy']**2+truth['tpz']**2)
truth.head()


# ## the particles data
# Here we have: 
# * $vx, vy, vz$ global coordinates of the initial position of the particles
# * $vp_x, vp_y, vp_z$ the initial momentum component of the particles in each direction.
# * $q$ the relative electric charge w.r.t e
# * nhits the number of hit to the detectors
# * $particle id$ our objective

# In[7]:


particles.head()


# In[8]:


# add hit_id to help the merge in future
particles['hit_id'] = -1
particles['p'] = np.sqrt(particles['px']**2+particles['py']**2+particles['pz']**2)
particles.head()


# ### Discovery 2: there are several particles missing in the truth data
# * Some of the particle in "particles" could not be found in "truth". 
# * the only one particle in "truth" cannot be found in "particles" is particle_id 0, noise particles.

# In[9]:


init_truth = particles.rename({
    'vx' : 'tx',
    'vy' : 'ty',
    'vz' : 'tz',
    'px' : 'tpx',
    'py' : 'tpy',
    'pz' : 'tpz',
    'p'  : 'tp'
}, axis=1)
init_truth.drop('nhits', axis=1, inplace=True)


# In[10]:


# the number of particles in the particles and truth
inituni_par = init_truth.particle_id.unique()
uni_par = truth.particle_id.unique()
len(inituni_par), len(uni_par)


# In[11]:


# how many particles do they share
inter = np.intersect1d(uni_par, inituni_par)
len(inter)


# In[12]:


# what is the one can't find in the particles
np.setdiff1d(uni_par, inter)


# ### Let's merge truth and particles

# In[13]:


weight_map = truth.groupby('particle_id').first()['weight']
charge_map = init_truth.groupby('particle_id').first()['q']


# In[14]:


truth['q'] = truth.particle_id.map(charge_map)
truth.fillna(0, inplace=True)


# In[15]:


init_truth['weight'] = init_truth.particle_id.map(weight_map)
init_truth.fillna(0, inplace=True)
init_truth['hasweight'] = ~np.equal(init_truth.weight, 0)


# In[16]:


truth.set_index('hit_id',inplace=True)
init_truth.set_index('hit_id',inplace=True)
fulltruth = init_truth.append(truth, sort=True)
fulltruth.head()


# ### Create a feature $R_r$ or relative rotation radius from the Equation:
# \begin{equation}
#     m\frac{v^2}{r}=qvB
# \end{equation}
# \begin{equation}
# r = \frac{mv}{qB} = \frac{p}{qB}
# \end{equation}
# 
# \begin{equation}
# R_r = reB = \frac{p}{q_r}
# \end{equation}

# In[17]:


fulltruth['R'] =  np.sqrt( fulltruth['tpx']**2 + fulltruth['tpy']**2 ) / fulltruth['q']


# ### Discovery 3: the particle the host do not care about is the one has extremely high momentum
# note: the host do care about the particle has large momentum (>3GeV/c)

# In[18]:


tgroups = truth.groupby("hasweight")

unrel = tgroups.get_group(False)
unrel_ = unrel.groupby('particle_id').first()

rel = tgroups.get_group(True)
rel_ = rel.groupby('particle_id').first()


# In[19]:


fig, axs = plt.subplots(1,2, figsize=(18,6))
axs[0].set_title('zero weight')
sns.distplot(unrel.tp, hist=True, kde=False, ax = axs[0] )
axs[1].set_title('has weight')
sns.distplot(rel.tp, hist=True, kde=False, ax= axs[1])


# ### Discovey 4: the data is pretty noisy 
# 15% of the data is the noise
# 
# 
# the paricle_id is the one with generally high momentum

# In[20]:


par0 = truth[truth.particle_id==0]
len(truth), len(par0), len(par0)/len(truth)    


# In[21]:


par0.tp.describe()


# ## Let's plot the initial location of the particle
# At here the marker color indicate the magnitude of the momentum 

# In[186]:


stdlayout3d = dict(
    width=800,
    height=700,
        
    autosize=False,
    title= 'unknown',
    scene=dict(
        xaxis=dict(
            title = "unknown x",
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        yaxis=dict(
            title = "unknown y",
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        zaxis=dict(
            title = "unknown z",
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        camera=dict(
            up=dict(x=0, y=0, z=1),
            eye=dict(x=-1.7428, y=1.0707, z=0.7100,)
        ),
        aspectratio = dict(x=1, y=1, z=0.7),
        aspectmode = 'manual'
    ),
)

def layout_costom3d(xtitle, ytitle, ztitle, title, xrange=None, yrange=None, zrange=None):
    layout = copy.deepcopy(stdlayout3d)
    layout['scene']['xaxis']['title'] = xtitle 
    layout['scene']['yaxis']['title'] = ytitle
    layout['scene']['zaxis']['title'] = ztitle
    if xrange is not None: layout['scene']['xaxis']['range'] = xrange
    if yrange is not None: layout['scene']['yaxis']['range'] = yrange
    if zrange is not None: layout['scene']['zaxis']['range'] = zrange
    layout['title'] = title
    return layout


# In[187]:


stdlayout2d = dict(
    height = 800,
    width = 800,
    title = 'unknown',
    yaxis = dict(),
        #zeroline = False,

    xaxis = dict(),
        #zeroline = False,
)    
def layout_costom2d(xtitle, ytitle, title, xrange=None, yrange=None, zrange=None):
    layout = copy.deepcopy(stdlayout2d)
    layout['xaxis']['title'] = xtitle 
    layout['yaxis']['title'] = ytitle    
    if xrange is not None: layout['xaxis']['range'] = xrange
    if yrange is not None: layout['yaxis']['range'] = yrange
    layout['title'] = title
    return layout   


# In[188]:


xyzlayout = layout_costom3d('z axis(mm)', 'x axis(mm)', 'y axis(mm)', 'sample trajectories')
xylayout = layout_costom2d('x axis(mm)', 'y axis(mm)', 'sample trajectories', [-1000, 1000], [-1000, 1000])


# In[89]:


def set_marker(pp, pp_name, isnoise, ms, cmin, cmax):
    marker = dict(
        size=ms,
        symbol= "square" if isnoise else "circle",
    )
    #pdb.set_trace()
    if pp is not None:
        marker['color'] = pp
        marker['colorscale']='Rainbow'
        marker['colorbar']=dict(
                title = pp_name,
                x = 1.20
            )
        marker['cmin'] = cmin
        marker['cmax'] = cmax
    return marker

def set_line( width=1):
    line = dict(
        width=1
    )


# In[136]:


def plotly_3d(x, y, z, pp=None, pp_name=None, isnoise=False, visible="legendonly", mode=None, ms=4, cmin=0, cmax=1):     
    marker = set_marker(pp, pp_name, isnoise, ms, cmin, cmax)        
    trace = go.Scatter3d(
        mode=mode,
        visible=visible,
        x=x, y=y, z=z,
        marker= marker,
        line = dict(width=1)
    )
    return [trace]

def plot_df(df, xyzcols, n=10, pproperty=None, pids=None, visible="legendonly", mode=None, ms=2, cmin=0, cmax=1):
    particlegroup = df.groupby('particle_id')
    if pids is None:
        pids = np.random.choice(df.particle_id.unique(), n)
    
    particles = [particlegroup.get_group(pid) for pid in pids]
    
    data = []
    xc, yc, zc = xyzcols
    for particle in particles:
        trace=plotly_3d(
            x=particle[xc],
            y=particle[yc],
            z=particle[zc],
            pp=particle[pproperty] if pproperty is not None else None,
            pp_name=pproperty,
            isnoise=particle.weight.values[0] == 0,
            visible=visible,
            mode=mode,
            ms=ms, cmin=cmin, cmax=cmax
        )
        data+=trace
    return data


# In[149]:


def plotly_2d(x, y, pp=None, pp_name=None, isnoise=False, visible="legendonly", mode=None, ms=4, cmin=0, cmax=1):
        
    marker = set_marker(pp, pp_name, isnoise, ms, cmin, cmax)       
    trace = go.Scatter(
        x = x,
        y = y,
        mode = mode,
        marker = marker,
        line = dict(width=1)
    )
    return [trace]
        
def plot_df2d(df, xycols, n=10, pproperty=None, pids=None, visible="legendonly", mode=None, ms=4, cmin=0, cmax=1):
    particlegroup = df.groupby('particle_id')
    if pids is None:
        pids = np.random.choice(df.particle_id.unique(), n)
    
    particles = [particlegroup.get_group(pid) for pid in pids]
        
    data = []
    xc, yc = xycols
    for particle in particles:
        trace=plotly_2d(
            x=particle[xc],
            y=particle[yc],
            pp=particle[pproperty] if pproperty is not None else None,
            pp_name=pproperty,
            isnoise=particle.weight.values[0] == 0,
            visible=visible,
            mode=mode,
            ms=ms, cmin=cmin, cmax=cmax
        )
        data+=trace
    return data 


# In[104]:


data = plotly_3d(particles.vz, particles.vx, particles.vy, particles.p, 'momentum', visible=True, ms=1, mode='markers')
layout = xyzlayout.copy()
layout['title'] = 'initial position'
iplot(dict(data=data, layout=layout), filename='local')


# ## Let's plot the trajectory

# ### plot 10 randomly picked particles 

# In[105]:


data = plot_df(truth, ['tz','tx','ty'], n=10, visible=True)
iplot(dict(data=data, layout=xyzlayout))


# ### plot the noise

# In[109]:


data = plot_df(truth, ['tz','tx','ty'], pids=[0], visible=True, mode='markers', ms=1)
layout = xyzlayout.copy()
layout['title'] = 'Noise'
iplot(dict(data=data, layout=layout))


# ### plot the trajectory projection on xy plane

# In[127]:


data = plot_df2d(truth, ['tx', 'ty'], n=100, mode=None)
iplot(dict(data=data, layout=xylayout))


# ## the hits data
# Here we have: 
# * $x, y, z$ global coordinates of the initial position of the particles
# * volume_id, layer_id and the module_id claim the arrgement information of each device

# In[25]:


hits.head()


# ## the cells data
# Here we have: 
# * $ch0, ch1$ global coordinates of the initial position of the particles
# * volume_id, layer_id and the module_id claim the arrgement information of each device
# 

# In[26]:


cells.head()


# For one hit_id, there are more than two recorded data. We take the max to find the most sensitive cell.

# In[27]:


cells_ = cells.set_index('hit_id')
cells_.drop(['ch0','ch1'], axis=1, inplace=True)
cells_ = cells_.groupby('hit_id').agg('sum')
cells_.head()


# ### We merge hits, cells, particles, and truth into one dataframe

# In[28]:


hits_ = hits.set_index('hit_id')
info = cells_.join(hits_)
cheat = info.join(fulltruth)
cheat.dropna(axis=0)
cheat.R.replace(np.inf, 10000, inplace=True)
cheat.head()


# ## The distribution of signal value for different particle 

# In[29]:


pids = cheat.particle_id.unique()
samples = np.random.choice(pids, 2)
for sample in samples:
    if sample != 0:   
        ax = sns.distplot( cheat[cheat.particle_id==sample].value , kde=False, bins=np.linspace(0,1,10) )
ax.set_xlim([0,1])


# In[30]:


fig, axs =plt.subplots(1,3,figsize=(18,4))

ax = sns.distplot(cheat.value, ax=axs[0], kde=False, bins=np.linspace(0,1,20))
ax.set_xlim([0,1])
ax.set_title('overall')
ax = sns.distplot(cheat[cheat.particle_id == 0].value, ax=axs[1], kde=False, bins=np.linspace(0,1,20))
ax.set_xlim([0,1])
ax.set_title('noise')
ax = sns.distplot(cheat[~(cheat.particle_id == 0)].value, ax=axs[2], kde=False, bins=np.linspace(0,1,20))
ax.set_xlim([0,1])
ax.set_title('not noise');


# ## Let's plot the trajectory and vary the marker's color with respect to the particle property

# ### w.r.t the signal value (i.e. value column)

# In[129]:


data = plot_df(cheat, ['tx', 'ty', 'tz'], n=30 ,pproperty='value', visible=True)
iplot(dict(data=data, layout=xyzlayout))


# #### the trajectory with high signal value (>0.5) 

# In[34]:


highsig = cheat.value > 0.5
highsigpart = cheat.particle_id[highsig].unique()
lowsigpart = np.setdiff1d(cheat.particle_id,highsigpart)
cheat.particle_id.nunique(), len(highsigpart), len(lowsigpart)


# In[139]:


data = plot_df(cheat, ['tx', 'ty', 'tz'], pids=np.random.choice(highsigpart, 10), pproperty='value', visible=True)
iplot(dict(data=data, layout=xyzlayout))


# #### trajectory with generally low signal value (>0.5) 

# In[140]:


data = plot_df(cheat, ['tx', 'ty', 'tz'], pids=np.random.choice(lowsigpart, 10), pproperty='value', visible=True)
iplot(dict(data=data, layout=xyzlayout))


# ### w.r.t the magnitude of momentum (i.e. tp column)

# In[145]:


data = plot_df(cheat, ['tx', 'ty', 'tz'], pproperty='tp', visible=True, cmin=0, cmax=2, ms=2)
iplot(dict(data=data, layout=xyzlayout))


# ### w.r.t particle relative charge (i.e. q)
# We could clearly see charge affect the bending direction in the second plot.

# In[146]:


data = plot_df(cheat, ['tx', 'ty', 'tz'], pproperty='q', visible=True, cmin=-1, cmax=1, ms=4)
iplot(dict(data=data, layout=xyzlayout))


# #### Projections on xy plane

# In[156]:


data = plot_df2d(cheat, ['tx', 'ty'], pproperty='q', visible=True, cmin=-1, cmax=1, ms=6, mode='markers')
iplot(dict(data=data, layout=xylayout))


# ### w.r.t relative rotation radius on xy plane

# In[158]:


data = plot_df2d(cheat, ['tx', 'ty'], pproperty='R', visible=True, cmin=-1, cmax=1, ms=6, mode='markers')
iplot(dict(data=data, layout=xylayout))


# ## Convert the Cartesian Coordinatea to Cylindrical Coordinates
# It provide a more straight forward visualization (trajactory: helix curve -> straight line)

# In[35]:


def xyz2c(df, cols, newcols):
    x, y, z = df[cols[0]], df[cols[1]], df[cols[2]]
    #pdb.set_trace()
    cr, cpsi, cz = newcols
    r = np.sqrt( x**2 + y**2 )
    psi = np.arctan(y/x)
    df[cr] = r
    df[cpsi] = psi
    df[cz] = z


# In[36]:


def xyz2s(df, cols, newcols):
    x, y, z = df[cols[0]], df[cols[1]], df[cols[2]]
    cr, cpsi, ctheta = newcols
    r = np.sqrt( x**2 + y**2 + z**2 )
    psi = np.arctan(y/x)
    theta = np.arccos(z/r)
    df[cr] = r
    df[cpsi] = psi
    df[ctheta] = theta


# In[163]:


xyz2c(cheat, ['tx','ty','tz'], ['trc','tpsic','tzc'])
xyz2s(cheat, ['tx','ty','tz'], ['trs','tpsis','tthetas'])


# In[ ]:


## Plotting the trajectory in new coordinates


# In[ ]:


### in Spherical Coordinates


# In[181]:


spherelayout = layout_costom3d('theta (radians)','radius (mm)', 'psi (radians)', 'sample trajectories in spherical coor',xrange=[0, np.pi], zrange=[-np.pi, np.pi])


# In[169]:


data = plot_df(cheat, ['tthetas', 'trs', 'tpsis'], visible=True)
iplot(dict(data=data, layout=spherelayout))


# In[ ]:


### in Cylindricall Coordinates


# In[189]:


cylindricallayout = layout_costom3d('z (mm)','radius (mm)', 'psi (radians)', 'sample trajectories in cylindrical coor', zrange=[-np.pi, np.pi])


# In[191]:


data = plot_df(cheat, ['tz', 'trc', 'tpsic'], visible=True)
iplot(dict(data=data, layout=cylindricallayout))


# ### in the r-z Plane of the Cylinderical Coordinate + detectors' location
# According to several discussions about transformation, visualizes the trajectory in this plane is seemingly the most promisible way. (links at the top of the notebook)
# 
# We could also plot trajectories with the location of detectors to create a better picture.

# In[49]:


xyz2c(device, ['cx','cy','cz'], ['rc','psic','zc'])
xyz2s(device, ['cx','cy','cz'], ['rs','psis','thetas'])


# In[194]:


zrlayout = layout_costom2d('z (mm)','radius (mm)', 'sample trajectories on rz plane')


# In[197]:


data = plotly_2d(x=device.zc, y=device.rc, isnoise=True, ms=5, mode='markers')
data += plot_df2d(cheat, ['tzc','trc'])
iplot(dict(data=data, layout=zrlayout))

