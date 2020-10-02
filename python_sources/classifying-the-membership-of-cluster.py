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


# ## Classify the cluster membership
# 
# Galaxies Cluster is a network made from many galaxies. Each object gravitationally bounded to others. Galaxies would transform into a structure.
# 
# This is the image of Abell 426 from NASA
# 
# ![Abell 426](https://apod.nasa.gov/apod/image/1107/abell426_franke_1800.jpg)
# 
# To filter the membershiop, it could be done by filtering the redshift. I will assume there will be no stars in such far redshift, because the stars lied in the field should came from inside our galaxy. 

# In[ ]:


df = pd.read_csv('/kaggle/input/perseus-galaxies-cluster-in-sdss-dr15/Skyserver_SQL11_25_2019 9_13_47 AM.csv')


# Let's check, there should be several field. 
# * **ra** is the right ascension of the object, and **dec** is the declination. Those are related to the object position in equatorial coordinate. 
# * **bestObjID** is the unique object ID
# * **z** is the redshift value; how far the wavelength shifted. Larger value related to larger distance. 
# * **zErr** is the error of the redshift. 
# * **psfMag_u, psfMag_g, psfMag_r, psfMag_i, and psfMag_z** is the PSF magnitude of object in UGRIZ color system. 

# In[ ]:


df.head()


# ### I. Redshift of Cluster
# 
# Well, we need some information about the galaxy. First, make histogram of redshifts. 
# 

# In[ ]:


import seaborn as sb

sb.distplot(df['z']).set_title('Redshift')
print('mean: ',df['z'].mean())


# Too bad to recognize all of them as the cluster, so we need to filter the redshift range. We could filter it with range beyond the known redshift at 0.0179. 
# 
# Beware that the nearly zero redshift should be belong to close objects such as stars. 

# In[ ]:


df = df.loc[df['z'] < 0.050]
df = df.loc[df['z'] > 0.010]

df.describe()


# In[ ]:


sb.distplot(df['z'], bins = 5, kde=False).set_title('Redshift')
print('mean distance: ',df['z'].mean())

df['z'].size


# Alright, we got the **redshift as 0.01723**. It was not so accurate, because the redshift data was not complete for all cluster member. Instead it could be calculated from photo z, but It would be covered later because there should be some tuning done, 

# ### II. Radius of Cluster
# 
# We will make the limit of cluster radius by plottin the Regos-Geller relation. 
# 
# First, we need the radius from the centre of the cluster, named r_deg. 

# In[ ]:


df['r_deg'] = np.sqrt((df['ra'] - 49.9467)**2 + (df['dec'] - 41.5131)**2)
df


# In[ ]:


sb.scatterplot(x="r_deg", y="z", data=df).set_title('Redshift and distance from centre')


# We found that the cluster spread as wide as 0.9 degree, and we could convert it to parsecs. 
# 
# $sin  \theta = \frac{R}{d}$
# 
# with $l = 0.8$ degree and $d = 73$ Mpc, we got the **radius of cluster as 1.019 Mpc**
# 

# ### III. Color Magnitude Diagram
# 
# We will see how red or blue are the galaxies in this cluster, so we could prompt the CMD. Above 2.2, the galaxies could be attributed to elliptical, and below it should be spiral. It was related to bluer star forming region in galaxies which often occured in spiral galaxies. 

# In[ ]:


df['u-r'] = df['psfMag_u']-df['psfMag_r']


# In[ ]:


blue = df[df['u-r'] >= 2.2]
red = df[df['u-r'] < 2.2]

sb.scatterplot(x="psfMag_r", y="u-r", data=red).set_title('CMD')
sb.scatterplot(x="psfMag_r", y="u-r", data=blue)


# ### IV. Schechter Luminosity Function
# 
# We will make the luminosity function of the cluster. First, we should consider where the luminosity falls. It is around 19.

# In[ ]:


sb.distplot(df['psfMag_r']);


# In[ ]:


import matplotlib.pyplot as plt 

alpha=-1.35
log_M0=19
phi=5.96E-11
def schechter_fit(logM):
    schechter = phi*(10**((alpha+1)*(logM-log_M0)))*(np.e**(-pow(10,logM-log_M0)))
    return schechter


# In[ ]:


y = schechter_fit(df['psfMag_r'].sort_values(ascending = True))
y_i = schechter_fit(df['psfMag_r'])

plt.plot(df['psfMag_r'].sort_values(ascending = True), np.log(y.real), 'b', label="Re Part")
plt.scatter(df['psfMag_r'], np.log(y_i.real))
plt.title('Schechter Luminosity Function')
plt.xlabel('psfMag_r')
plt.ylabel('log $\phi *$')


# ## References
# 
# * Schneider, Peter. Extragalactic astronomy and cosmology: an introduction. Springer, 2014.
# * Regos, Eniko, and Margaret J. Geller. "Infall patterns around rich clusters of galaxies." The Astronomical Journal 98 (1989): 755-765.
# * Chang, Ruixiang, et al. "The colours of elliptical galaxies." Monthly Notices of the Royal Astronomical Society 366.3 (2006): 717-726.
# 
