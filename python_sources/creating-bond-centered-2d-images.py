#!/usr/bin/env python
# coding: utf-8

# In a molecule, the coupling constants will depend, on a first order, on the geometry of the molecules: once the positions of the nuclei are fixed in space, 
# the electronic orbitals can be calculated resolving the shrodinger equation in the scope of the Born Oppenheiner approximation. From the potential energy surface obtained that way, coupling constants can be derived. Hence, as a first guess, I thought that an approach that would capture the global view of the geometry of the molecule could provide good results. In order to simplify the problem, I thought of reducing the complexity of the problem by integrating over a dimension of space, thus reducing the 3D geometrical structure of the molecule to a 2D image. Doing so, the problem could then be treated as a regression task on images. 
# 
# In this notebook, I describe the way how such images are obtained.
# 

# ## 1. Description of the method

# In[ ]:


get_ipython().system('pip install chart_studio ')


# Considering a vector $\vec{d}$ along the molecular bond, we define a coordinate system aligned with $\vec{d}$:

# In[ ]:


import numpy as np

d = np.array([1, 2, 1])
print(f'with d = {d}')

#_______________________
norm = np.linalg.norm(d)
d = d / norm
#______________________
if d[2] != 0:
    u = np.ones(3)
    u[2] = -(d[0]*u[0] + d[1]*u[1]) / d[2]
    norm = np.linalg.norm(u)
    u = u / norm
elif d[0] != 0:
    u = np.ones(3)
    u[0] = - d[1]*u[1] / d[0]
    norm = np.linalg.norm(u)
    u = u / norm
else:
    u = np.ones(3)
    u[1] = 0
    norm = np.linalg.norm(u)
    u = u / norm
#____________________________________
v = np.cross(d, u)
norm = np.linalg.norm(v)
v = v / norm
print('the coordinate system will be defined through the vectors:'
    '\n', d,'\n', u,'\n', v)


# Using this coordinate system, we define 2 distances. The first, **R** corresponds to the distance along $\vec{d}$ from the centroid of the molecular bond. The scond, **r**, defines the radius of a circle in a plane perpendicular to $\vec{d}$. Defining a set of **(r, R)** points:

# In[ ]:


values = [
    [1, 5],
    [2, 5],
    [3, 5],
    [1, 7],
    [2, 7],
    [3, 7]]


# In[ ]:


new_data = []
for r, R in values:
    center_circle = R * d  # origine du cercle
    #______________________________________________________________
    val = []
    for theta in np.linspace(0, 2 * np.pi):
        xp = r * np.cos(theta)
        yp = r * np.sin(theta)
        val.append([xp, yp])
    #________________________________
    new_pt = []
    for pt in val:
        new_pt.append(center_circle + (pt[0] * u + pt[1] * v ))
    new_data.append(np.array(new_pt))


# We obtain the following set of points in space:

# In[ ]:


from plotly.offline import init_notebook_mode, iplot
import pandas as pd
import chart_studio.plotly as py
import plotly.graph_objects as go

data = []
clusters = []
colors = ['rgb(228,26,28)','rgb(55,126,184)','rgb(77,175,74)',
          'rgb(155,126,184)','rgb(177,175,74)', 'rgb(77,175,174)',]

for i, pt in enumerate(new_data):
    x = pt[:, 0]
    y = pt[:, 1]
    z = pt[:, 2]
    trace = dict(
            name = f'r={values[i][0]} R={values[i][1]}',
            x = x, y = y, z = z,
            type = "scatter3d",    
            mode = 'markers',
            marker = dict( size=3, color=colors[i], line=dict(width=0) ) )
    data.append( trace )
 

layout = dict(
    width=800,
    height=550,
    autosize=False,
    title='Cone',
    scene=dict(
        xaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        yaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        zaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        aspectratio = dict( x=1, y=1, z=1 ),
        aspectmode = 'data'        
    ),
)

fig = dict(data=data, layout=layout)
url = iplot(fig, filename='pandas-3d-iris', validate=False)


# For every position defined that way, we calculate the density of atoms through
# $ 
#  \sum_i 1/d_i^2
# $
#     where the sum if over all the atoms of a single type in the molecule. At this stage, in order to create the 2D images, we integrate over the circumference of the circles the density calculated above. At the end of the process, we thus have 5 *pseudo images* whose pixels are in the **(r, R)** coordinate system. 

# ## 2. Application to a chunk of 1JHN bonds 
# Below, I use the scheme defined above to some 1JHN bonds.

# In[ ]:


import pandas as pd
import numpy as np
import pickle
import time

structures = pd.read_csv('../input/champs-scalar-coupling/structures.csv')
train = pd.read_csv('../input/champs-scalar-coupling/test.csv')


# In[ ]:


symetry = '1JHN'
train = train[train['type'] == symetry]
print(f'selecting {train.shape[0]} bonds from {symetry}')


# In[ ]:


def map_atom_info(df, atom_idx):
    df = pd.merge(df, structures, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

train = map_atom_info(train, 0)
train = map_atom_info(train, 1)

train_p_0 = train[['x_0', 'y_0', 'z_0']].values
train_p_1 = train[['x_1', 'y_1', 'z_1']].values
d_bond = train_p_0 - train_p_1
nbonds = len(d_bond)

train_p_0 = train[['x_0', 'y_0', 'z_0']].values
train_p_1 = train[['x_1', 'y_1', 'z_1']].values
center = (train_p_0 + train_p_1) / 2


# we define a function that performs the integration along the circles:

# In[ ]:


def density_integration(df, nb_angles):
    density_integrale = []
    for icouple in range(df.shape[0]): 

        step_length = length[icouple]
        p = df[icouple, :, :].copy()   
        density = np.zeros(nb_angles)

        for iatom in range(len(atoms)):
            val = p - atoms[iatom]
            dist = np.linalg.norm(val, axis=1)
            density += 1 / dist**2

        integrale = density.sum() * step_length
        density_integrale.append(integrale)
    return density_integrale


# We define images of size 35 x 35:

# In[ ]:


nb_r = 35
nb_R = 35
nb_angles = 45

values = [(r, R) for r in np.linspace(5, 0.01, nb_r) for R in np.linspace(-5, 5, nb_R)]


# In[ ]:


start_time = time.time()

image_file = 'images.pkl'
bond_file = 'bond.pkl'

bond_list = []
image_set = []

ids = train['id'].values
for ibond in range(nbonds): 
    #__________________
    d = d_bond[ibond]
    orig = center[ibond]

    molec = train.iloc[ibond]['molecule_name']
    bond_list.append(molec+'_'+str(ids[ibond]))

    norm = np.linalg.norm(d)
    d = d / norm
    #______________________
    if d[2] != 0:
        u = np.ones(3)
        u[2] = -(d[0]*u[0] + d[1]*u[1]) / d[2]
        norm = np.linalg.norm(u)
        u = u / norm
    elif d[0] != 0:
        u = np.ones(3)
        u[0] = - d[1]*u[1] / d[0]
        norm = np.linalg.norm(u)
        u = u / norm
    else:
        u = np.ones(3)
        u[1] = 0
        norm = np.linalg.norm(u)
        u = u / norm
    #____________________________________
    v = np.cross(d, u)
    norm = np.linalg.norm(v)
    v = v / norm
    #_______________________________
    new_data = []
    length = []
    for r, R in values:
        center_circle = R * d  + orig 
        #______________________________________________________________
        val = []
        for theta in np.linspace(0, 2 * np.pi, nb_angles):
            xp = r * np.cos(theta)
            yp = r * np.sin(theta)
            val.append([xp, yp])
    #         print(theta)
        #________________________________
        new_pt = []
        for pt in val:
            new_pt.append(center_circle + (pt[0] * u + pt[1] * v ))
        new_data.append(np.array(new_pt))

        length.append(r * (2 * np.pi) / (nb_angles - 1))
    #__________________________________
    new_data = np.array(new_data)
    #__________________________________
    image = []
    for atom in ['C', 'N', 'O', 'H', 'F']:
        atoms = structures[(structures['molecule_name'] == molec) &
                           (structures['atom'] == atom)][['x', 'y', 'z']].values

        data = density_integration(new_data, nb_angles)
        image.append(np.array(data).reshape(nb_r, nb_R))

    image_set.append(image)
    
    # limiting the number of images 
    if ibond > 20: break
    if (time.time() - start_time) / 3600 > 8: break

with open(image_file, 'wb') as fp:
    pickle.dump(image_set, fp)

with open(bond_file, 'wb') as fp:
    pickle.dump(bond_list, fp)
        
time.time() - start_time


# Below, we see how the 5 images looks like for a sample of bonds: 

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
    
for i, num_image in enumerate([1, 5, 7, 8]):
    
    image = image_set[num_image]
    
    fig = plt.figure(figsize=(16, 3))

    clip_val = 255

    ax1 = fig.add_subplot(151)
    ax1.imshow(image[0], cmap=plt.cm.BrBG, interpolation='nearest', origin='lower')
    plt.imshow(np.clip(image[0], 0, clip_val) / clip_val)
    if i == 0: ax1.set_title('C')

    ax1 = fig.add_subplot(152)
    ax1.imshow(image[1], cmap=plt.cm.BrBG, interpolation='nearest', origin='lower')
    plt.imshow(np.clip(image[1], 0, clip_val)/ clip_val)
    if i == 0: ax1.set_title('N')

    ax1 = fig.add_subplot(153)
    ax1.imshow(image[2], cmap=plt.cm.BrBG, interpolation='nearest', origin='lower')
    plt.imshow(np.clip(image[2], 0, clip_val)/ clip_val)
    if i == 0: ax1.set_title('O')

    ax1 = fig.add_subplot(154)
    ax1.imshow(image[3], cmap=plt.cm.BrBG, interpolation='nearest', origin='lower')
    plt.imshow(np.clip(image[3], 0, clip_val)/ clip_val)
    if i == 0: ax1.set_title('H')
    
    ax1 = fig.add_subplot(155)
    ax1.imshow(image[3], cmap=plt.cm.BrBG, interpolation='nearest', origin='lower')
    plt.imshow(np.clip(image[4], 0, clip_val)/ clip_val)
    if i == 0: ax1.set_title('F')
    
    plt.show()


# The next steps then consists in introducing these images in a CNN in order to perform the regression task. An example for the 1JHN symetry is given in [this kernel](https://www.kaggle.com/fabiendaniel/cnn-on-1jhn-2d-images)

# In[ ]:




