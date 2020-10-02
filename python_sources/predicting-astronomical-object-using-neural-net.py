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


# In[ ]:


df = pd.read_csv('/kaggle/input/sloan-digital-sky-survey/Skyserver_SQL2_27_2018 6_51_39 PM.csv')


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# In[ ]:


df['class'].value_counts()


# ## Distance Sparse
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


from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP9 as cosmo

radec = SkyCoord(ra=df['ra']*u.degree, dec=df['dec']*u.degree, frame='icrs')
#radec.ra.value
#radec.dec.value
galactic = radec.galactic

df['l'] = galactic.l.value
df['b'] = galactic.b.value


r = cosmo.comoving_distance(df['redshift'])
df['distance']= r.value

df.head()


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


def cartesian(dist,alpha,delta):
    x = dist*np.cos(np.deg2rad(delta))*np.cos(np.deg2rad(alpha))
    y = dist*np.cos(np.deg2rad(delta))*np.sin(np.deg2rad(alpha))
    z = dist*np.sin(np.deg2rad(delta))
    return x,y,z

cart = cartesian(df['distance'],df['ra'],df['dec'])
df['x_coord'] = cart[0]
df['y_coord'] = cart[1]
df['z_coord'] = cart[2]

df.head()


# In[ ]:


df['u-r'] = df['u']-df['r']


# In[ ]:


galaxy = df[df['class']=='GALAXY']
star = df[df['class']=='STAR']
quasar = df[df['class']=='QSO']


# In[ ]:


fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(galaxy['ra'],galaxy['dec'],galaxy['redshift'], s = 0.7, color = 'b', label = 'galaxy')
ax.scatter(star['ra'],star['dec'],star['redshift'], s = 0.7, color = 'y', label = 'star')
ax.scatter(quasar['ra'],quasar['dec'],quasar['redshift'], s = 0.7, color = 'r', label = 'quasar')
ax.set_xlabel('ra')
ax.set_ylabel('dec')
ax.set_zlabel('z')
ax.set_title('Object Distribution from SDSS',fontsize=18)
plt.legend()
plt.show()


# ## Color Magnitude Diagram

# In[ ]:


plt.scatter(galaxy['r'], galaxy['u-r'], s = 0.9, color = 'b')
plt.scatter(star['r'], star['u-r'], s = 0.9, color = 'y')
plt.scatter(quasar['r'], quasar['u-r'], s = 0.9, color = 'r')
plt.xlabel('r')
plt.ylabel('u-r')
plt.title('CMD')


# ## Plotting the Galaxy

# In[ ]:


galaxy.head()


# In[ ]:


fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(galaxy['ra'],galaxy['dec'],galaxy['redshift'], s = 0.7)
ax.set_xlabel('ra')
ax.set_ylabel('dec')
ax.set_zlabel('z')
ax.set_title('Galactic Distribution from SDSS',fontsize=18)
plt.show()


# In[ ]:


fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(galaxy['x_coord'],galaxy['y_coord'],galaxy['z_coord'], s = 0.7, color = 'b')
ax.set_xlabel('x_coord')
ax.set_ylabel('y_coord')
ax.set_zlabel('z_coord')
ax.set_title('Galoaxy Distribution from SDSS',fontsize=18)
plt.show()

# z is the position from galaxy in cartesian coordinate, not to be confused with redshift


# In[ ]:


sns.distplot(galaxy['redshift'], kde = False)


# In[ ]:


sns.distplot(galaxy['distance'], kde = False)
plt.title('Distance (Mpc)')


# ## Plotting The Stars

# In[ ]:


star.head()


# In[ ]:


fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(star['ra'],star['dec'],star['redshift'], s = 0.7, color = 'y')
ax.set_xlabel('ra')
ax.set_ylabel('dec')
ax.set_zlabel('z')
ax.set_title('Star Distribution from SDSS',fontsize=18)
plt.show()


# In[ ]:


fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(star['x_coord'],star['y_coord'],star['z_coord'], s = 0.7, color = 'y')
ax.set_xlabel('x_coord')
ax.set_ylabel('y_coord')
ax.set_zlabel('z_coord')
ax.set_title('Distribution from SDSS',fontsize=18)
plt.show()

# z is the position from galaxy in cartesian coordinate, not to be confused with redshift


# In[ ]:


sns.distplot(star['redshift'], kde = False)


# ### Stars appeared to be centered at 0 redshift since it is too close

# ## Plotting the Quasi-Stellar Objects

# In[ ]:


quasar.head()


# In[ ]:


fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(quasar['ra'],quasar['dec'],quasar['redshift'], s = 0.7, color = 'r')
ax.set_xlabel('ra')
ax.set_ylabel('dec')
ax.set_zlabel('z')
ax.set_title('QSO Distribution from SDSS',fontsize=18)
plt.show()


# In[ ]:


fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(quasar['x_coord'],quasar['y_coord'],quasar['z_coord'], s = 0.7, color = 'r')
ax.set_xlabel('x_coord')
ax.set_ylabel('y_coord')
ax.set_zlabel('z_coord')
ax.set_title('Distribution from SDSS',fontsize=18)
plt.show()

# z is the position from galaxy in cartesian coordinate, not to be confused with redshift


# ## Summing up

# In[ ]:


df.head()


# In[ ]:


fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(galaxy['x_coord'],galaxy['y_coord'],galaxy['z_coord'], s = 0.7, color = 'b')
ax.scatter(star['x_coord'],star['y_coord'],star['z_coord'], s = 0.7, color = 'y')
ax.scatter(quasar['x_coord'],quasar['y_coord'],quasar['z_coord'], s = 0.7, color = 'r')
ax.set_xlabel('x_coord')
ax.set_ylabel('y_coord')
ax.set_zlabel('z_coord')
ax.set_title('Distribution from SDSS',fontsize=18)
plt.show()

# z is the position from galaxy in cartesian coordinate, not to be confused with redshift


# ### As expected, we could see distant quasars because they are too bright for a galaxy

# ### Okay, back to the primary dataset

# In[ ]:


display(df.head())
display(df.columns)


# In[ ]:


df['class'] = df['class'].astype('category').cat.codes


# In[ ]:


df['class'].value_counts()


# ### We need to change the object class into numerical label

# ### 0s are galaxies, 2s are stars, and 1s are QSOs.

# In[ ]:


df.columns


# ### I will not dropping RA and dec because a group of stars or galaxies could entangled in mutual gravitation, so they're located in a same area.

# In[ ]:


X_df = df.drop(['objid','class'], axis=1).values
y_df = df['class'].values


# In[ ]:


display(X_df)
display(y_df)


# ### We use RA and dec first (note that some galaxies and stars grouping in same areas thanks to mutual gravitation)

# ## Neural Network Model

# ### We will try to use standard scaler first
# 
# Standardize features by removing the mean and scaling to unit variance
# 
# The standard score of a sample x is calculated as:
# 
# $ z = (x - u) / s $

# In[ ]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler

#ss = StandardScaler()
#X_df = ss.fit_transform(X_df)
minmax = MinMaxScaler()
X_df = minmax.fit_transform(X_df)


# In[ ]:


y_df = y_df.reshape(-1,1)


# In[ ]:


display(X_df)
display(y_df)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
y_df = enc.fit_transform(y_df).toarray()


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_df,y_df, test_size=0.25)


# In[ ]:


X_train.shape


# In[ ]:


y_train.shape


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


model = Sequential()
model.add(Dense(64, input_dim=23, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


history = model.fit(X_train, y_train, validation_split = 0.1, epochs=30, batch_size=32)


# In[ ]:


prediction = []
test = []


# In[ ]:


y_predict = model.predict(X_test)


# In[ ]:


for i in range(len(y_test)): 
    prediction.append(np.argmax(y_predict[i]))
    test.append(np.argmax(y_test[i])) 


# In[ ]:


from sklearn.metrics import accuracy_score
acc = accuracy_score(prediction,test) 
print('Accuracy is:', acc*100, '%')


# ### Yea, and now we gonna save the output data

# In[ ]:


compare = pd.DataFrame(prediction, columns = ['prediction'])
compare['test'] = test


# In[ ]:


result = pd.DataFrame(X_test, columns = ['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'run', 'rerun', 'camcol',
       'field', 'specobjid', 'redshift', 'plate', 'mjd', 'fiberid',
       'l', 'b', 'distance', 'x_coord', 'y_coord', 'z_coord', 'u-r'])


# In[ ]:


result['class'] = compare['test']
result['prediction'] = compare['prediction']

result.to_csv('object_prediction.csv', index = False)

