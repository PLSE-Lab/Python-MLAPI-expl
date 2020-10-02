#!/usr/bin/env python
# coding: utf-8

# Main goal here is to learn to work with the data as-is (because not all needed features are presnet in the SDK), join all lidars into one point cloud and apply annotations to make sure we're not missing anything.
# 
# **EDIT** as @alexamadori pointed out in comments, most scenes have data only from one lidar, so merging all 3 lidars might not make much sense. Still writing this kernel was useful to me to understand the data and coordinate systems better.
# 
# Overview of what we're doing below:
# 
# - we load lidar data (3d points), these points are in coordinate system of the lidar, rotated and translated relative to the car
# - using sensor information of the lidar, we translate points from lidar coordinate frame to car coordinate frame, this allows us to merge data from all 3 lidars
# - annotations and submission are in global coordinates. We translate annotations into the car coordinates, which allows to have both the data and annotations in the same (car) coordinate frame, which can then be used for training
# 
# We learn:
# 
# - how the raw data looks like, so that we understand how SDK works better and can extend it if needed (also if you see any missing features, file issues at https://github.com/lyft/nuscenes-devkit/issues/)
# - how to translate objects between various coordinate frames
# 
# In terms of custom packages, we'll use only pyquaternion which is also used by Lyft SDK.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import json
import os.path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pyquaternion import Quaternion


# Some small helpers for loading the json data. We won't load all data here.

# In[ ]:


class Table:
    def __init__(self, data):
        self.data = data
        self.index = {x['token']: x for x in data}


DATA_ROOT = '/kaggle/input/3d-object-detection-for-autonomous-vehicles/'


def load_table(name, root=os.path.join(DATA_ROOT, 'train_data')):
    with open(os.path.join(root, name), 'rb') as f:
        return Table(json.load(f))

    
scene = load_table('scene.json')
sample = load_table('sample.json')
sample_data = load_table('sample_data.json')
ego_pose = load_table('ego_pose.json')
calibrated_sensor = load_table('calibrated_sensor.json')


# Like in official tutorial, let's check the first scene

# In[ ]:


my_scene = scene.data[0]
my_scene


# And the first sample from that scene

# In[ ]:


sample.index[my_scene['first_sample_token']]


# We'll use ``sample_data`` to fetch lidar images related to this sample. First check what is inside ``sample_data``

# In[ ]:


sample_data.data[0]


# Now fetch lidar images related to the sample (note that to make it efficient you'll want to add an index like in official SDK)

# In[ ]:


lidars = []
for x in sample_data.data:
    if x['sample_token'] == my_scene['first_sample_token'] and 'lidar' in x['filename']:
        lidars.append(x)
lidars


# All lidars happen to have the same ego_pose (because they are on the same car?)

# In[ ]:


{x['ego_pose_token'] for x in lidars}


# Now let's load lidar's point data, we'll keep only first 3 columns (point coordinates)

# In[ ]:


lidars_data = [
    # here, sorry
    np.fromfile(os.path.join(DATA_ROOT, x['filename']).replace('/lidar/', '/train_lidar/'), dtype=np.float32)
    .reshape(-1, 5)[:, :3] for x in lidars]
lidars_data[0].shape


# In[ ]:


lidars_data[0]


# In[ ]:


lidars_data[0].min(axis=0), lidars_data[0].max(axis=0)


# Apparently, we're mostly underground but that's ok.
# 
# Now to the interesting stuff - translating all lidars into car coordinate system. For that we need sensor info for each lidar

# In[ ]:


[calibrated_sensor.index[x['calibrated_sensor_token']] for x in lidars]


# Some helpers

# In[ ]:


def rotate_points(points, rotation, inverse=False):
    assert points.shape[1] == 3
    q = Quaternion(rotation)
    if inverse:
        q = q.inverse
    return np.dot(q.rotation_matrix, points.T).T
    
def apply_pose(points, cs):
    """ Translate (lidar) points to vehicle coordinates, given a calibrated sensor.
    """
    points = rotate_points(points, cs['rotation'])
    points = points + np.array(cs['translation'])
    return points

def inverse_apply_pose(points, cs):
    """ Reverse of apply_pose (we'll need it later).
    """
    points = points - np.array(cs['translation'])
    points = rotate_points(points, np.array(cs['rotation']), inverse=True)
    return points


# And let's vizualize all lidars. First let's see what happens if we don't use poses from the lidars

# In[ ]:


def viz_all_lidars(lidars, lidars_data, clip=50, skip_apply_pose=False):
    all_points = []
    all_colors = []
    for color, points, lidar in zip([[1, 0, 0, 0.5], [0, 1, 0, 0.5], [0, 0, 1, 0.5]], lidars_data, lidars):
        cs = calibrated_sensor.index[lidar['calibrated_sensor_token']]
        if not skip_apply_pose:
            points = apply_pose(points, cs)
        all_points.append(points)
        all_colors.append(np.array([color] * len(points)))
    all_points = np.concatenate(all_points)
    all_colors = np.concatenate(all_colors)
    perm = np.random.permutation(len(all_points))
    all_points = all_points[perm]
    all_colors = all_colors[perm]

    plt.figure(figsize=(12, 12))
    plt.axis('equal')
    plt.grid()
    plt.scatter(np.clip(all_points[:, 0], -clip, clip), np.clip(all_points[:, 1], -clip, clip), s=1, c=all_colors)

viz_all_lidars(lidars, lidars_data, clip=20, skip_apply_pose=True)


# Each lidar has different color, and we permute points to make them more visible (there is probably a better way to do this). Also note that we clip all points which fall outside of a small area, for training you most likely won't do this.
# 
# Above looks wrong, doesn't it? This is because each lidar is rotated and also slightly translated relative to the car, and lidar points are in lidar coordinates. Now let's enable translation to the car coorinatates:

# In[ ]:


viz_all_lidars(lidars, lidars_data, clip=20)


# We see that data from different lidars agrees, so all translations were right, yasss. Note that curvy lines come from the ground and don't need to aling, we look mostly at cars to check alignment.
# 
# Let's zoom out a bit:

# In[ ]:


viz_all_lidars(lidars, lidars_data, clip=50)


# Next we load annotated data. We'll  only use annotations available from train.csv because they are more compact and faster to load. They are indexed by sample token.

# In[ ]:


train_df = pd.read_csv(os.path.join(DATA_ROOT, 'train.csv')).set_index('Id')
train_df.loc[my_scene['first_sample_token']]


# A little helper to convert annotation from string

# In[ ]:


def get_annotations(token):
    annotations = np.array(train_df.loc[token].PredictionString.split()).reshape(-1, 8)
    return {
        'point': annotations[:, :3].astype(np.float32),
        'wlh': annotations[:, 3:6].astype(np.float32),
        'rotation': annotations[:, 6].astype(np.float32),
        'cls': np.array(annotations[:, 7]),
    }

get_annotations(my_scene['first_sample_token']).keys()


# Now, this annotations are in global coordinates. Let's translate them into the frame of the car, for that we'll use the ego_pose of the lidars (they are all the same)

# In[ ]:


ego_pose.index[lidars[0]['ego_pose_token']]


# We'll use inverse_apply_pose which we defined above, will show only the car class, and will deal only with point centers

# In[ ]:


def viz_annotation_centers(token, lidars, clip=50):
    # translate annotation points to the car frame
    ego_pose_token, = {x['ego_pose_token'] for x in lidars}
    ep = ego_pose.index[ego_pose_token]
    annotations = get_annotations(token)
    car_points = annotations['point'][annotations['cls'] == 'car']
    car_points = inverse_apply_pose(car_points, ep)
    
    plt.scatter(np.clip(car_points[:, 0], -clip, clip),
                np.clip(car_points[:, 1], -clip, clip),
                s=30,
                color='black')
    
viz_all_lidars(lidars, lidars_data, clip=50)
viz_annotation_centers(my_scene['first_sample_token'], lidars, clip=50)


# Car markers look aligned to some blips in the lidar data, so probably above is correct. Note that here again we clipped the data which you'd need to remove for actual training.
# 
# Let's zoom in a bit

# In[ ]:


viz_all_lidars(lidars, lidars_data, clip=20)
viz_annotation_centers(my_scene['first_sample_token'], lidars, clip=20)


# Thanks for reading, if you liked it, please give an upvote.
