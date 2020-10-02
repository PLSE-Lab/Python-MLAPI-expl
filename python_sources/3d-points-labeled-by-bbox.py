#!/usr/bin/env python
# coding: utf-8

# # This notebook contains several visualizations where points are labeled by category of a particular bounding box

# Preprocessing based on https://www.kaggle.com/fartuk1/3d-segmentation-approach

# In[ ]:


import numpy as np
import pandas as pd
import os
print(os.listdir('/kaggle/input/3d-object-detection-for-autonomous-vehicles'))


# In[ ]:


get_ipython().system('pip install pyquaternion')


# In[ ]:


import json
import os.path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pyquaternion import Quaternion

from mpl_toolkits.mplot3d import Axes3D
import random
import itertools
from skimage.morphology import convex_hull_image


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


# In[ ]:


train_df = pd.read_csv(os.path.join(DATA_ROOT, 'train.csv')).set_index('Id')
CLASSES = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]


# Translations of coordinates from https://www.kaggle.com/lopuhin/lyft-3d-join-all-lidars-annotations-from-scratch

# In[ ]:


def rotate_points(points, rotation, inverse=False):
    assert points.shape[1] == 3
    q = Quaternion(rotation)
    if inverse:
        q = q.inverse
    return np.dot(q.rotation_matrix, points.T).T
    
def apply_pose(points, cs, inverse=False):
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

def get_annotations(token):
    annotations = np.array(train_df.loc[token].PredictionString.split()).reshape(-1, 8)
    return {
        'point': annotations[:, :3].astype(np.float32),
        'wlh': annotations[:, 3:6].astype(np.float32),
        'rotation': annotations[:, 6].astype(np.float32),
        'cls': np.array(annotations[:, 7]),
    }


# Helpers to rotate bounding box points

# In[ ]:


import copy

import math

def rotate(origin, point, angle):
    ox, oy, _ = origin
    px, py, pz = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return [qx, qy, pz]


def make_box_coords(center, wlh, rotation, ep):

    planar_wlh = copy.deepcopy(wlh)
    planar_wlh = planar_wlh[[1,0,2]]

    bottom_center = copy.deepcopy(center)
    bottom_center[-1] = bottom_center[-1] - planar_wlh[-1] / 2

    bottom_points = []
    bottom_points.append(bottom_center + planar_wlh * [1, 1, 0] / 2)
    bottom_points.append(bottom_center + planar_wlh * [-1, -1, 0] / 2)
    bottom_points.append(bottom_center + planar_wlh * [1, -1, 0] / 2)
    bottom_points.append(bottom_center + planar_wlh * [-1, 1, 0] / 2)
    bottom_points = np.array(bottom_points)

    rotated_bottom_points = []
    for point in bottom_points:
        rotated_bottom_points.append(rotate(bottom_center, point, rotation))

    rotated_bottom_points = np.array(rotated_bottom_points)
    rotated_top_points = rotated_bottom_points + planar_wlh * [0,0,1]

    box_points = np.concatenate([rotated_bottom_points, rotated_top_points], axis=0)

    box_points = inverse_apply_pose(box_points, ep)
    
    return box_points


# In[ ]:


def get_sample_data(sample_token):
    lidars = []
    for x in sample_data.data:
        if x['sample_token'] == sample_token and 'lidar' in x['filename']:
            print(x['filename'])
            lidars.append(x)

    lidars_data = [
        # here, sorry
        np.fromfile(os.path.join(DATA_ROOT, x['filename'].replace('lidar/', 'train_lidar/')), dtype=np.float32)
        .reshape(-1, 5)[:, :3] for x in lidars]


    all_points = []
    for points, lidar in zip(lidars_data, lidars):
        cs = calibrated_sensor.index[lidar['calibrated_sensor_token']]
        points = apply_pose(points, cs)
        all_points.append(points)
    all_points = np.concatenate(all_points)


    ego_pose_token, = {x['ego_pose_token'] for x in lidars}
    ep = ego_pose.index[ego_pose_token]
    annotations = get_annotations(sample_token)
    
    all_boxes = {}
    for class_name in CLASSES:
        obj_centers = annotations['point'][annotations['cls'] == class_name]
        obj_wlhs = annotations['wlh'][annotations['cls'] == class_name]
        obj_rotations = annotations['rotation'][annotations['cls'] == class_name]
        obj_boxes = []
        for k in range(len(obj_centers)):
            center = obj_centers[k]
            wlh = obj_wlhs[k]
            rotation = obj_rotations[k]
            box_coords = make_box_coords(center, wlh, rotation, ep)
            obj_boxes.append(box_coords)
        all_boxes[class_name] = np.array(obj_boxes)
        
    car_centers = annotations['point'][annotations['cls'] == 'car'] 
    car_centers = inverse_apply_pose(car_centers, ep)
    return all_points, all_boxes


# ## Data example

# In[ ]:


train_df


# In[ ]:


sample_token = train_df.reset_index()['Id'].values[20]
all_points, all_boxes = get_sample_data(sample_token)
print(all_points.shape)
for obj in all_boxes:
    print(obj, all_boxes[obj].shape)


# In[ ]:


# plot top view with bbox corners
plt.figure(figsize=(25,15))
plt.scatter(all_points[:, 0], all_points[:, 1],s=[0.1]*len(all_points))
for obj in all_boxes:
    if all_boxes[obj].shape[0] == 0:
        continue
    boxes_coords = np.concatenate(all_boxes[obj], axis=0)
    plt.scatter(boxes_coords[:, 0], boxes_coords[:, 1],s=[15]*len(boxes_coords))


# In[ ]:


def points_in_cuboid(bbox, cloud, bbox_idx=[0, 2, 3, 4]):
    """Get points within rectangular cuboid. Make sure the cuboid coordinates are in correct order.
    
    bbox: 8 points as corneres of cuboid.
    bbox_idx: indices of bbox which should be taken for vector creation.
            First and last index are bottom and upper points on the same edge. 
            2nd and 3th index are bottom points, but not diagonal to 1st index.
    """
    i = bbox[bbox_idx[1]] - bbox[bbox_idx[0]]
    j = bbox[bbox_idx[2]] - bbox[bbox_idx[0]]
    k = bbox[bbox_idx[3]] - bbox[bbox_idx[0]]
    v_s = cloud - bbox[bbox_idx[0]]
    ii_dot = np.dot(i, i) 
    jj_dot = np.dot(j, j) 
    kk_dot = np.dot(k, k) 
    vi_dot = np.dot(v_s, i)
    vj_dot = np.dot(v_s, j) 
    vk_dot = np.dot(v_s, k)
    mask = (0 <= vi_dot) & (vi_dot <= ii_dot) & (0 <= vj_dot) & (vj_dot <= jj_dot) & (0 <= vk_dot) & (vk_dot <= kk_dot)
    return mask


# In[ ]:


def plot_small_region_masked(ann_idx):
    center_point = all_boxes['car'][ann_idx].mean(axis=0)
    x_min = center_point[0] - 5
    x_max = center_point[0] + 5
    y_min = center_point[1] - 5
    y_max = center_point[1] + 5
    z_min= center_point[2] - 5
    z_max = center_point[2] + 5

    area_mask = (all_points[:, 0] > x_min) * (all_points[:, 0] < x_max) * (all_points[:, 1] > y_min) * (all_points[:, 1] < y_max) * (all_points[:, 2] > z_min) * (all_points[:, 2] < z_max)
    area_mask = np.where(area_mask)[0]

    fig = plt.figure(figsize=(25,15))
    ax = Axes3D(fig)
    
    mask_in = points_in_cuboid(all_boxes['car'][ann_idx], all_points[area_mask, 0:3], bbox_idx=[0, 2, 3, 4])
    
    ax.scatter(all_points[area_mask, 0][mask_in], all_points[area_mask, 1][mask_in], all_points[area_mask, 2][mask_in])
    ax.scatter(all_points[area_mask, 0][~mask_in], all_points[area_mask, 1][~mask_in], all_points[area_mask, 2][~mask_in])

    ax.scatter(all_boxes['car'][ann_idx][:, 0], all_boxes['car'][ann_idx][:, 1], all_boxes['car'][ann_idx][:, 2], color='r', s=[100])
    plt.show()


# ### Small regions around bbox of a car class
# Blue - points inside chosen bbox, orange - points outside chosen bbox, red - bbox corners

# In[ ]:


for i in range(all_boxes['car'].shape[0]):
    plot_small_region_masked(i)


# In[ ]:


def get_points_inside_boxes(points, boxes):
    """return mask of all points which are in any bbox """
    all_masks = []
    for box_i in range(boxes.shape[0]):
        mask_in = points_in_cuboid(boxes[box_i], points, bbox_idx=[0, 2, 3, 4])
        all_masks.append(mask_in)
    return np.array(all_masks).any(axis=0)


# ## Examples of labeled surrounding points
# Some plots use more lidar sensors, so they are more dense, especially first one looks noisy on this figure.

# In[ ]:


colors = ['#CBD6F166', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
map_dict = { i: color for i, color in enumerate(colors)}

import matplotlib.patches as mpatches


for i in range(0,train_df.shape[0], 1000):
    print(i)
    sample_token = train_df.reset_index()['Id'].values[i]
    all_points, all_boxes = get_sample_data(sample_token)

    seg_lables = np.zeros((all_points.shape[0],))
    for i, (obj, boxes) in enumerate(all_boxes.items()):
        if not boxes.shape[0] == 0:
            mask_in = get_points_inside_boxes(all_points, boxes)
            seg_lables[mask_in] = i+1


    fig = plt.figure(figsize=(25,15))
    ax = Axes3D(fig)
    mask = (all_points[:, 0] < 20) & (all_points[:, 0] > -20) & (all_points[:, 1] < 20) & (all_points[:, 1] > -20)
    ax.scatter(all_points[mask, 0], all_points[mask, 1], all_points[mask, 2], c=np.vectorize(map_dict.get)(seg_lables[mask]))
    ax.legend()
    ax.set_zlim(-6, 6)
    
    patches = []
    for col, cla in zip(colors, ['']+CLASSES):
        patch = mpatches.Patch(color=col, label=cla)
        patches.append(patch)
    plt.legend(handles=patches)
        
    plt.show()


# ## Create labels for all points in train dataset
# work in progress

# In[ ]:


# points_train = []
# seg_labels_train = []
# for i in range(100):
#     sample_token = train_df.reset_index()['Id'].values[i]
#     all_points, all_boxes, car_centers = get_sample_data(sample_token)

#     seg_lables = np.zeros((all_points.shape[0],))
#     for i, (obj, boxes) in enumerate(all_boxes.items()):
#         if not boxes.shape[0] == 0:
#             mask_in = get_points_inside_boxes(all_points, boxes)
#             seg_lables[mask_in] = i+1
    
#     points_train.append(all_points)
#     seg_labels_train.append(seg_lables)

