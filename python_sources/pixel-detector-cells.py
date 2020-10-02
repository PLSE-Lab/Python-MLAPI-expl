#!/usr/bin/env python
# coding: utf-8

# # Motivation: Measurements in the Pixel detector
# 
# The innermost componets of tracking detectors (``volume_id = 7,8,9``)  for high energy physics are often Silicon Pixel detector, as it is is modelled in this dataset.  A pixelated readout structure allows to detect the *single* or *group of* pixels that are traversed by the particle and thus receive a signal. Evidently, this is strongly dependent on the incident angle of the particle into the detection sensor, as illustrated below:
# 
# <img src="https://asalzbur.web.cern.ch/asalzbur/work/tml/PixelModule.png" width=600>
# 
# The markers here are:
# - **black** dots : pixel center positions
# - **magenta** dot: reconstructed cluster position
# - **red** dot: true intersection of particle with sensor mid-surface
# 
# 
# Each single module has thus a channel system which is two-dimensional to the local coordinates. These are the channel indizes ``ch0`` and ``ch1``, which indicate which pixel(s) on the module have been crossed by a particle. When a particle crosses a pixel, it induces charge by *ionisation*, this does not alter the charge of the traversing particle, but reduces its energy (and thus momentum) slightly. The value of the charge can be read out in the pixel detector and is stored for each channel identifiec by ``(ch0,ch1)`` as the approprated ``value``. The following shows such a pixel module channel schema with the came cluster as above, the coloring of the traversed pixels indicates their ``value`` (charge).
# 
# <img src="https://asalzbur.web.cern.ch/asalzbur/work/tml/PixelChannels.png" width=600>
# 
# ## Position information
# 
# In a first stage, pixels that are adjunct are grouped together into *clusters*, an operation that has already been done in the presented dataset. The particle intersection with the module can then be rather precisely determined by taking the pixel position - and, as it it the case in the dataset - using the charge information of the individual pixels that contribute to the cluster. 
# 

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Import the `trackml` library for convenient data loading.

# In[ ]:


import trackml
from trackml.dataset import load_event, load_dataset
from trackml.randomize import shuffle_hits
from trackml.score import score_event


# Load a specific event ...

# In[ ]:


# One event of 8850
event_id = 'event000001000'
# "All methods either take or return pandas.DataFrame objects"
hits, cells, particles, truth = load_event('../input/train_1/'+event_id)


# Let us now pick a single *Pixel cluster*, I've picked a rather large one ... 

# In[ ]:


h_id = 19144 
pixel_cluster = cells[ cells['hit_id']==h_id ]


# In[ ]:


len(pixel_cluster)


# This cluster has  1 to many individual pixels contributing, let's display the cluster 

# In[ ]:


# a function that calculates the cluster size and makes a pixel matrix
def pixel_matrix(pixel_cluster, show=False):
    # cluster size
    min0 = min(pixel_cluster['ch0'])
    max0 = max(pixel_cluster['ch0'])
    min1 = min(pixel_cluster['ch1'])
    max1 = max(pixel_cluster['ch1'])
    # the matrix
    matrix = np.zeros(((max1-min1+3),(max0-min0+3)))
    for pixel in pixel_cluster.values :
        i0 = int(pixel[1]-min0+1)
        i1 = int(pixel[2]-min1+1)
        value = pixel[3]
        matrix[i1][i0] = value 
    # return the matris
    if show :
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.YlOrRd)
        plt.colorbar()
        plt.show()
    return matrix, max0-min0+1, max1-min1+1


# In[ ]:


cluster,width,length = pixel_matrix(pixel_cluster,True)


# The cluster size in `u` and `v` direction is:

# In[ ]:


print(width,length)


# You can almost *see* the particle's trajectory through the silicon, the coloring corresonds to the charge of the individual pixel. As you can see, the edge pixels which are not fully traversed by the particle have only little charge, which is what you expect, because the particle did traverse less of Silicon and thus induces less charge.
# 
# Let us now see how well the position is estamated for this measurement:

# In[ ]:


pixel_hit = hits[ hits['hit_id']==h_id ]


# In[ ]:


print(pixel_hit)


# In the truth file we can find the **truth** position of the intersection:

# In[ ]:


truth_pixel_hit = truth[ truth['hit_id']==h_id]


# In[ ]:


print(truth_pixel_hit)


# The reconstruction did really well!! Compare ``(x,y,z)`` with ``(tx,ty,tz)`` , the values are **really** close, that's a good detector.
# How did we come from the local cluster position on the surface, which was made of the local positions of the individual cells - to the global positions ? 
# 
# This is with the help of the geometry description. 
# 
# Each module is unequile positioned in space through a **center** position and a **rotation** that transforms the local coordinate system ``(u,v,w)`` to the global coordinate system ``(x,y,z)``. For the dataset, ``ch0`` is measured in ``u`` and ``ch1`` is measured in the ``v`` direction, while the ``w`` direction is the ``thickness`` of the module as shown above.
# 
# <img src="https://asalzbur.web.cern.ch/asalzbur/work/tml/localToGlobal.png" width=600>
# 
# It's time to load the detector descrption. 
# We load the full detector and then picke the module our pixel cluster in question as measured at:

# In[ ]:


detector = pd.read_csv('./../input/detectors.csv')
# method to retrieve the according module associated to a hit
def retrieve_module(detector,hit) :
    volume = detector[ detector['volume_id']==hit.volume_id.data[0] ]
    layer  = volume[ volume['layer_id']==hit.layer_id.data[0] ]
    module = layer[ layer['module_id']== hit.module_id.data[0] ]
    return module
# get the one for our example
module = retrieve_module(detector,pixel_hit)


# We now have the full detector information of this detection module:
# - its *translation* given by the module center position `(cx,cy,cz)`
# - its *rotation* with reference to the global coordinate system `(rot_xy, ..., rot_zw)`
# - the  module thicknes  `thickness = 2 * module_t`
# - the module dimension (rectangular): `2 * module_minh` in local `u` and `2 * mondule_hv` in locl `v`
# - the measurement segmentation, i.e. the pixel size `pitch_u` in `u` and `pitch_v` in `v`
# 
# ## Direction information
# 
# In the pixel cluster shape there's obviously some directional information decoded about the particle, to access this, we can compare the *expected* cluster size and the *measured* ones.
# To calculate the expected cluster shape of a track hypothesis on a sensor, you need to express the *momentum direction* in the local coordinate system of the module. This is done by applying the inverse rotation to the *global direction*:
# 
# `direction_local(_hit) = rotation.inverse() * direction_global_(hit)`
# 
# We will take the truth direction here to demonstrate - at the ``hit`` position, in global coordinates.

# In[ ]:


# method to build and nomralize direction vector from the 
def direction_vector(ipx, ipy, ipz) :
    # the absolute momentum for normalization
    p = np.sqrt(ipx*ipx+ipy*ipy+ipz*ipz)
    # build the direction vector - to be used with the matrix 
    direction = [[ipx/p], [ipy/p], [ipz/p]]
    return direction


# In[ ]:


# get the truth direction at the module, it's more accurate than the starting position
direction_global_hit = direction_vector(truth_pixel_hit.tpx.data[0],truth_pixel_hit.tpy.data[0],truth_pixel_hit.tpz.data[0])
print(direction_global_hit)


# Let's do a comparison how the global momentum direction at particle creation:

# In[ ]:


# get the truth particle information
particle = particles[ particles['particle_id'] == truth_pixel_hit.particle_id.data[0] ]
print(particle)


# In[ ]:


# build the direction vector with the start momentum
direction_global_start = direction_vector(particle.px.data[0],particle.py.data[0],particle.pz.data[0])
print(direction_global_start)


# In global coordinates, the *polar* angle ``theta_global(_hit)`` is (from truth information):

# In[ ]:


# extract phi and theta from a direciton vector
def phi_theta(dx,dy,dz) :
    dr  = np.sqrt(dx*dx+dy*dy)
    phi = np.arctan2(dy,dx)
    theta = np.arctan2(dr,dz)
    return phi, theta
# get thet and phi
phi_hit, theta_hit = phi_theta(direction_global_hit[0][0],
                               direction_global_hit[1][0],
                               direction_global_hit[2][0])
print(theta_hit)


# From this, the expected cluster length is given from trigenometry, we using the module thicknes ( = `2*module_t`) to calucate the path in the Silicon wafer.
# When staying in the global system, we can only assume that the module is parallel to the ``z``-axis, so we can only get an estimate. There's little we can say about the `u` direction, if we do not know how the moudle is oriented.

# In[ ]:


# get the length of the cluster in v direction
cluster_length_v_hit = np.abs(2.*module.module_t.data[0]/np.tan(theta_hit))
cluster_size_v_hit   = cluster_length_v_hit/module.pitch_v.data[0]
print(cluster_length_v_hit,cluster_size_v_hit)


# Not bad, we estimated the cluster size rather accurately only from the global information of the particle and the knowledge that the detector element is from the barrel detector (this, you can see from the `volume_id`).
# 
# 
# ### Refining with geometry information
# 
# So far, we have not used the `module` information in order to get the actual incident angle into the module (which is the direct cause of the cluster size), hence this information should help to increase the accuracy of our prediction.

# In[ ]:


# function to extract the rotation matrix (and its inverse) from module dataframe
def extract_rotation_matrix(module) :
    rot_matrix = np.matrix( [[ module.rot_xu.data[0], module.rot_xv.data[0], module.rot_xw.data[0]],
                            [  module.rot_yu.data[0], module.rot_yv.data[0], module.rot_yw.data[0]],
                            [  module.rot_zu.data[0], module.rot_zv.data[0], module.rot_zw.data[0]]])
    return rot_matrix, np.linalg.inv(rot_matrix)


# Let's extract the rotation matrix and it's inverse from the module then, and transform the global direction into a local direction:

# In[ ]:


module_matrix, module_matrix_inv = extract_rotation_matrix(module)
print (module_matrix)


# And let's have a look at the inverse martrix as well:

# In[ ]:


direction_local_hit =  module_matrix_inv*direction_global_hit
print(direction_local_hit)


# As you can see, the local momentum direction has the biggest component along the local `v` axis, this is not surprising giving that the cluster size in `v` is about 
# Now let's see, what *predicted cluster sizes we get*, first we need to calculate the `phi` and `theta` in the local coordinate frame:

# In[ ]:


# theta is defined as the arctan of the radial vs the longitudinal components
# phi is defined as the acran of the two transvese components
phi_local,theta_local = phi_theta(direction_local_hit[0][0],
                                  direction_local_hit[1][0],
                                  direction_local_hit[2][0])
print(phi_local,theta_local)


# From the module thickness, we can get the full path length in the silicon:

# In[ ]:


path_in_silicon = 2*module.module_t.data[0]/np.cos(theta_local)
print(path_in_silicon)


# Which finally allows us to calculate the path length in `u` and `v` and thus the cluster sizes:
# 

# In[ ]:


# calculate the component in u and v
path_component_u = path_in_silicon*np.sin(theta_local)*np.cos(phi_local)
path_component_v = path_in_silicon*np.sin(theta_local)*np.sin(phi_local)
cluster_size_in_u = path_component_u/module.pitch_u.data[0]
cluster_size_in_v = path_component_v/module.pitch_v.data[0]
# print the cluster size 
print(cluster_size_in_u, cluster_size_in_v)


# 
# We have `reconstructed` the cluster size quite accurately in `v` and somewhat okish in `u` - why is this ?
# The cluster is rather long in `v` and thus we do not really converned if the particle entered really at the outermost extend in the `v` direction, while for the `u` direction this makes some difference, indeed. 
# 

# In[ ]:




