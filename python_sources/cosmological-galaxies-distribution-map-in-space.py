#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Galaxy Distribution in Space
# 
# Galaxy was a favourite object in cosmological research about expanding universe. Long time ago, galaxies soar in narrow space, resulting in collision between galaxies and forms elliptical galaxy. 
# 
# Redshift value shows the lookup on how early was the galaxy formed in universe. Edwin Hubble formulate the redshift into how shifted the wavelength of galaxy, thus indicating the radial velocity
# 
# 
# \begin{align}
# z = \frac{v}{c} = \frac{\lambda_{v}-\lambda_{0}}{\lambda_{0}}
# \end{align}
# 
# Hubble's law could be stated in
# \begin{align}
# v = \frac{H_{0}}{d}
# \end{align}
# 
# ### Reference: 
# * Ryden, Barbara. Introduction to cosmology. Cambridge University Press, 2017.

# In[ ]:


df = pd.read_csv('/kaggle/input/sloan-digital-sky-survey/Skyserver_SQL2_27_2018 6_51_39 PM.csv')
df.head()


# We need to locate galaxies within the dataset, and take their attribute

# In[ ]:


dfgal = df.loc[(df["class"]) == 'GALAXY']
dfgal.head()


# For map the location of galaxy, we need to plot the cartesian space coordinate from the equatorial coordinate. This one formulated from the galactic coordinate, known distance, galactic longitude, and galactic latitude. 
# 
# 
# ![Galactic Coordinate](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Galactic_coordinates.JPG/220px-Galactic_coordinates.JPG)
# 
# \begin{align}
# Image: Wikipedia
# \end{align}
# 
# Galactic coordinate could be formulated into
# 
# \begin{align}
# sin(b) = sin(\delta_{NGP}) \ cos(i_{g})  -  cos(\delta) \ sin(\alpha - \alpha_{NGP}) \ sin(i_{g})
# \end{align}
# 
# \begin{align}
# cos(b) \ cos(l-l_{0}) = cos(\delta) \ cos(\alpha - \alpha_{NG})
# \end{align}
# 
# \begin{align}
# cos(b) \ sin(l-l_{0}) = sin(\delta) \ sin(i_{g}) + cos (\delta) \ sin(\alpha - \alpha_{NGP})\ cos(i_{g})
# \end{align}
# 
# Known $i = 62.6^{o}, \alpha_{N} = 282.5^{o}, l_{0} = 33.0^{o}$
# 
# Source: https://www.ucl.ac.uk/~ucapsj0/galcor.pdf
# 
# For the ease, thanks to kaggle, they provide astropy

# In[ ]:


from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP9 as cosmo

radec = SkyCoord(ra=dfgal['ra']*u.degree, dec=dfgal['dec']*u.degree, frame='icrs')
#radec.ra.value
#radec.dec.value
galactic = radec.galactic

dfgal['l'] = galactic.l.value
dfgal['b'] = galactic.b.value


r = cosmo.comoving_distance(dfgal['redshift'])
dfgal['distance']= r.value

dfgal.head()


# In[ ]:


def cartesian(dist,alpha,delta):
    x = dist*np.cos(np.deg2rad(delta))*np.cos(np.deg2rad(alpha))
    y = dist*np.cos(np.deg2rad(delta))*np.sin(np.deg2rad(alpha))
    z = dist*np.sin(np.deg2rad(delta))
    return x,y,z

cart = cartesian(dfgal['distance'],dfgal['ra'],dfgal['dec'])
dfgal['x_coord'] = cart[0]
dfgal['y_coord'] = cart[1]
dfgal['z_coord'] = cart[2]

dfgal.head()


# ## Plot the Galaxies Location
# 
# After we know the cartesian coordinate of galaxies, we could plot it in space. Zero indicates our galaxy (exactly our sun, which is negligible distance in mpc scale).
# 
# The lack of dots in 

# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dfgal['x_coord'],dfgal['y_coord'],dfgal['z_coord'], s = 0.7)
ax.set_xlabel('X (mpc)')
ax.set_ylabel('Y (mpc)')
ax.set_zlabel('Z (mpc)')
ax.set_title('Galactic Distribution from SDSS',fontsize=18)
plt.show()


# In[ ]:


fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)
ax.scatter(dfgal['x_coord'],dfgal['y_coord'], s = 0.5)
ax.set_xlabel('X (mpc)')
ax.set_ylabel('Y (mpc)')
ax.set_title('Galactic Distribution from SDSS in X and Y Space',fontsize=18)
plt.show()


# In[ ]:


fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)
ax.scatter(dfgal['x_coord'],dfgal['z_coord'], s = 0.5)
ax.set_xlabel('X (mpc)')
ax.set_ylabel('Z (mpc)')
ax.set_title('Galactic Distribution from SDSS in X and Z Space',fontsize=18)
plt.show()


# In[ ]:


fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)
ax.scatter(dfgal['y_coord'],dfgal['z_coord'], s = 0.5)
ax.set_xlabel('Y (mpc)')
ax.set_ylabel('Z (mpc)')
ax.set_title('Galactic Distribution from SDSS in Y and Z Space',fontsize=18)
plt.show()


# In[ ]:


import seaborn as sb

fig = plt.figure(figsize=(12,10))
sb.distplot(dfgal['redshift'])
plt.title('Redshift Distribution',fontsize=18)
plt.show()

fig = plt.figure(figsize=(12,10))
sb.distplot(dfgal['distance'])
plt.title('Distance Distribution (MPC)',fontsize=18)
plt.show()


# In[ ]:


dfgal['redshift'].describe()


# In[ ]:


dfgal['distance'].describe()

