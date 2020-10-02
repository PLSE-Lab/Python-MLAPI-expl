#!/usr/bin/env python
# coding: utf-8

# # Overview
# The notebook shows how to download and process a measurement from the panoptic dataset from CMU. 
# 
# ## Notice 
# - Please read the [original page](http://domedb.perception.cs.cmu.edu/index.html) and license details before continuing to examine the script 

# # Setup
# Clone the panoptic toolbox and download a few datasets

# In[ ]:


get_ipython().system('cd /tmp; git clone https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox')


# We download the first haggling dataset: http://domedb.perception.cs.cmu.edu/170407_haggling_a1.html

# In[ ]:


get_ipython().system('cd /tmp/panoptic-toolbox; ./scripts/getData.sh 170407_haggling_a1 2 0; cd ..')


# # Move/Extract Data

# In[ ]:


from pathlib import Path
import shutil
import sys
pano_dir = Path('/tmp') / 'panoptic-toolbox' 
haggle_dir = pano_dir / '170407_haggling_a1'
sys.path.append(str(pano_dir / 'python'))


# In[ ]:


EXTRACT = False
if EXTRACT:
    for c_tar in haggle_dir.glob('*.tar'):
        get_ipython().system('tar -xf {c_tar}')
        get_ipython().system('rm {c_tar}')
for c_file in haggle_dir.glob('*'):
    try:
        shutil.move(str(c_file), str(Path('.') / c_file.name))
    except:
        shutil.move(str(c_file), str(Path('.') / c_file.name))
get_ipython().system('rm -rf /tmp/panoptic-toolbox/170407_haggling_a1')


# In[ ]:


haggle_dir = Path('.')
list(haggle_dir.glob('*'))


# # Process Data

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
plt.rcParams["figure.figsize"] = (10, 10)
plt.rcParams["figure.dpi"] = 125
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
sns.set_style("whitegrid", {'axes.grid': False})
from itertools import cycle
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


# In[ ]:


import numpy as np
import pandas as pd
import doctest
import copy
import functools
import json
from skimage.io import imread
from skimage.util import montage as montage2d
from skimage.color import label2rgb
from tqdm import tqdm_notebook
# tests help notebooks stay managable

def autotest(func):
    globs = copy.copy(globals())
    globs.update({func.__name__: func})
    doctest.run_docstring_examples(
        func, globs, verbose=True, name=func.__name__)
    return func


# In[ ]:


import panutils # tools for processing and reprojecting panoptic data


# ## Face Mesh

# In[ ]:


get_ipython().system('tar -xf hdFace3d.tar')
face_frames_df = pd.concat([
    pd.read_json(c_path).assign(path=c_path) 
    for c_path in tqdm_notebook(list(Path('hdFace3d').glob('*.json')))
]).reset_index(drop=True)
get_ipython().system('rm -rf hdFace3d')


# In[ ]:


face_frames_df['frame'] = face_frames_df['path'].map(lambda x: x.stem.split('_')[-1][2:])
face_frames_df['landmark_70'] = face_frames_df['people'].map(
    lambda x: 
    np.reshape(x['face70']['landmarks'], (-1, 3))
)
face_frames_df.head(4)


# In[ ]:


landmark_vec = np.reshape(face_frames_df['people'].iloc[0]['face70']['landmarks'], (-1, 3))
fig = plt.figure(figsize=(10, 10), dpi=300)
ax1 = fig.add_subplot(111, projection='3d')
for _, c_row in face_frames_df.head(20).iterrows():
    ax1.plot(c_row['landmark_70'][:, 0], c_row['landmark_70'][:, 1], c_row['landmark_70'][:, 2], 's')


# In[ ]:


face_frames_df.drop('path', 1).to_json('face_frames.json')


# # Cameras

# In[ ]:


with open('calibration_170407_haggling_a1.json') as cfile:
    calib = json.load(cfile)

# Cameras are identified by a tuple of (panel#,node#)
cameras = {(cam['panel'],cam['node']):cam for cam in calib['cameras']}

# Convert data into numpy arrays for convenience
for k,cam in cameras.items():    
    cam['K'] = np.matrix(cam['K'])
    cam['distCoef'] = np.array(cam['distCoef'])
    cam['R'] = np.matrix(cam['R'])
    cam['t'] = np.array(cam['t']).reshape((3,1))
    
# Select the first 10 VGA cameras in a uniformly sampled order
cams = list(panutils.get_uniform_camera_order())[0:10]
sel_cameras = [cameras[cam].copy() for cam in cams]


# In[ ]:


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(50,90)

# Draw all cameras in black
for k,cam in cameras.items():
    cc = (-cam['R'].transpose()*cam['t'])
    ax.scatter(cc[0], cc[1], cc[2], '.', color=[0,0,0])

# Selected camera subset in green
for cam in sel_cameras:
    cc = (-cam['R'].transpose()*cam['t'])
    ax.scatter(cc[0], cc[1], cc[2], '.', color=[0,1,0])

ax.set_aspect('equal')
ax.set_xlim3d([-300, 300])
ax.set_ylim3d([-300, 300])
ax.set_zlim3d([-300, 300]);


# In[ ]:




