#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install lyft-dataset-sdk')


# In[ ]:


get_ipython().system('pip install squaternion')


# In[ ]:


import pdb
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

# Load the SDK
from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer
from lyft_dataset_sdk.utils.data_classes import Box, LidarPointCloud, RadarPointCloud 
from lyft_dataset_sdk.utils.geometry_utils import BoxVisibility, box_in_image, view_points
from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D, recall_precision
import math

# from source.utilities import print_progress
from tqdm import tqdm_notebook as tqdm

get_ipython().run_line_magic('matplotlib', 'inline')
import string 
from pyquaternion import Quaternion
import matplotlib.patches as patches

from squaternion import euler2quat, quat2euler
import time
from multiprocessing import Process
from sklearn.decomposition import PCA


# In[ ]:


# gotta do this for LyftDataset SDK, it expects folders to be named as `images`, `maps`, `lidar`

get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images images')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_maps maps')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_lidar lidar')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_data data')


# In[ ]:


level5data = LyftDataset(data_path='.', json_path='data/', verbose=True)


# In[ ]:


train_pred = pd.read_csv('../input/lyft3d-inference-kernel-train-dataset/lyft3d_pred_train.csv')
train_pred.head()


# In[ ]:


def get_pred_boxes_list_3d(pred_boxes):
    if not pd.isna(pred_boxes):
        pred_boxes = pred_boxes.split(' ')
        pred_boxes = pred_boxes[:-1]
        pred_boxes_list = np.reshape(pred_boxes, (len(pred_boxes)//9, 9))
    else:
        pred_boxes_list = []
    return pred_boxes_list


# In[ ]:


pred_boxes = train_pred.iloc[0]['PredictionString']
pred_boxes_list = get_pred_boxes_list_3d(pred_boxes)
pred_boxes_list


# In[ ]:


train = pd.read_csv('../input/different-classes/pred_animal_3D.csv')
train.head()


# In[ ]:


def get_gth_boxes_list_3d(pred_boxes):
    if not pd.isna(pred_boxes):
        pred_boxes = pred_boxes.split(' ')
        pred_boxes = pred_boxes[1:-1]
        pred_boxes_list = np.reshape(pred_boxes, (len(pred_boxes)//10, 10))
    else:
        pred_boxes_list = []
    return pred_boxes_list


# In[ ]:


pred_boxes = train.iloc[0]['3D']
gth_list = get_gth_boxes_list_3d(pred_boxes)
gth_list


# In[ ]:


df_pred = pd.read_csv("../input/rcnn-pytorch-animal-detection/rcnn_pred_animal_2d.csv")
df_pred.head()


# In[ ]:


def get_boxes_list(pred_boxes):
    pred_boxes = pred_boxes.split(' ')
    pred_boxes_list = pred_boxes[:-1]
    pred_boxes_list = np.reshape(pred_boxes_list, (len(pred_boxes_list)//6, 6))
    return pred_boxes_list


# In[ ]:


title = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_ZOOMED']


# In[ ]:


train_num = len(train)
total_num_pred = 0
total_num = 0
for idx in tqdm(range(train_num)):
    total_num += int(train.iloc[idx]['3D'][0])
    for i in range(7):
        pred_boxes = df_pred.iloc[idx][title[i]]
        if not pd.isna(pred_boxes):
            pred_boxes_list = get_boxes_list(pred_boxes)
            num = pred_boxes_list.shape[0]
            total_num_pred += num


# In[ ]:


print(total_num, total_num_pred, total_num_pred/total_num)


# In[ ]:


columns = ['Id', '2D']
df_pred_2d = pd.DataFrame(columns=columns)
df_pred_2d.head()


# In[ ]:


train_num = len(train)
for idx in tqdm(range(train_num)):
    token = train.iloc[idx]['Id']
    row = []
    row.append(token)
    result = ''
    for i in range(7):
        pred_boxes = df_pred.iloc[idx][title[i]]
        if not pd.isna(pred_boxes):
            pred_boxes_list = get_boxes_list(pred_boxes)
            x1 = pred_boxes_list[0][0]
            y1 = pred_boxes_list[0][1]
            x2 = pred_boxes_list[0][2]
            y2 = pred_boxes_list[0][3]
            cls_conf = pred_boxes_list[0][4]
            label = pred_boxes_list[0][5]
            cam = title[i]
            result += str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + ' ' + str(cls_conf) + ' ' + str(label) + ' '  + str(cam) + " "
    row.append(result)  
    if not result == '':
        df_row = pd.DataFrame([row], columns=columns)
        df_pred_2d = df_pred_2d.append(df_row)


# In[ ]:


df_pred_2d.shape[0]


# In[ ]:


df_pred_2d.head()


# In[ ]:


def detection_show(pred_boxes_list):
    for pred_box in pred_boxes_list:
        coor = pred_box[:4]
        x1 = int(float(coor[0]))
        y1 = int(float(coor[1]))
        x2 = int(float(coor[2]))
        y2 = int(float(coor[3]))
        cls_conf = pred_box[4]
        label = pred_box[5]
        cam = pred_box[6]
        photo_filename = level5data.get('sample_data', my_sample['data'][cam])['filename']
        img = cv2.imread(photo_filename) # Read image with cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB   
        cv2.rectangle(img, (x1, y1),(x2, y2),255, 3) # Draw Rectangle with the coordinates
        cv2.putText(img, cls_conf, (x1, y1),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),thickness=3) # Write the prediction class
        plt.figure(figsize=(15,10)) # display the output image
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()


# In[ ]:


def get_2d_boxes_list(pred_boxes):
    pred_boxes = pred_boxes[:-1]
    pred_boxes = pred_boxes.split(' ')
    pred_boxes_list = np.reshape(pred_boxes, (len(pred_boxes)//7, 7))
    return pred_boxes_list


# In[ ]:


train_num = len(df_pred_2d)
for idx in range(10):
    token = df_pred_2d.iloc[idx]['Id']
    my_sample = level5data.get('sample', token)
    pred_boxes = df_pred_2d.iloc[idx]['2D']
    pred_boxes_list = get_2d_boxes_list(pred_boxes)
    detection_show(pred_boxes_list)
    print(idx)


# In[ ]:


class_heights = {'animal':0.51,'bicycle':1.44,'bus':3.44,'car':1.72,'emergency_vehicle':2.39,'motorcycle':1.59,
                'other_vehicle':3.23,'pedestrian':1.78,'truck':3.44}
class_lens = {'animal':0.73,'bicycle':1.76,'bus':12.34,'car':4.76,'emergency_vehicle':6.52,'motorcycle':2.35,
                'other_vehicle':8.20,'pedestrian':0.81,'truck':10.24}
class_widths = {'animal':0.36,'bicycle':0.63,'bus':2.96,'car':1.93,'emergency_vehicle':2.45,'motorcycle':0.96,
                'other_vehicle':2.79,'pedestrian':0.77,'truck':2.84}


# In[ ]:


pred_boxes = df_pred_2d.iloc[18]['2D']
pred_boxes_list = get_2d_boxes_list(pred_boxes)
print(pred_boxes_list)
for pred_box in pred_boxes_list:
    coor = pred_box[:4]
    x1 = int(float(coor[0]))
    y1 = int(float(coor[1]))
    x2 = int(float(coor[2]))
    y2 = int(float(coor[3]))
    cls_conf = pred_box[4]
    label = pred_box[5]
    cam = pred_box[6]
    print(x1)


# In[ ]:


def project_2D_to_3D(points, Z, intrinsic, quaternion, translation):
    f_x = intrinsic[0, 0]
    f_y = intrinsic[1, 1]
    c_x = intrinsic[0, 2]
    c_y = intrinsic[1, 2]
    result = []
    for idx in range(points.shape[0]):
        point = []
        z = Z
        x = z * (points[idx, 0] - c_x) / f_x 
        y = z * (points[idx, 1] - c_y) / f_y
        point.append([x, y, z])
        point_rot = np.dot(quaternion.rotation_matrix, point[0]) + np.array(translation)
        result.append(point_rot)
    return result


# In[ ]:


def render_pointcloud_xyz(point_cloud, color_channel, ax, x_lim, y_lim, marker_size):
    colors = np.ones(len(point_cloud[:,2]))*100
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], c = colors, s=marker_size)
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1]) 


# In[ ]:


def render_pointcloud_xz(point_cloud, color_channel, ax, x_lim, z_lim, marker_size):
    colors = np.ones(len(point_cloud[:,2]))*100
    ax.scatter(point_cloud[:, 0], point_cloud[:, 2], c = colors, s=marker_size)
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(z_lim[0], z_lim[1]) 


# In[ ]:


def render_pointcloud_yz(point_cloud, color_channel, ax, y_lim, z_lim, marker_size):
    colors = np.ones(len(point_cloud[:,2]))*100
    ax.scatter(point_cloud[:, 1], point_cloud[:, 2], c = colors, s=marker_size)
    ax.set_xlim(y_lim[0], y_lim[1])
    ax.set_ylim(z_lim[0], z_lim[1]) 


# In[ ]:


pred_box3ds = []
pred_box_point_cloud = []
num_samples = len(df_pred_2d)
for idx in tqdm(range(num_samples)):
    token0 = df_pred_2d.iloc[idx]['Id']
    my_sample = level5data.get('sample', token0)
    pred_boxes = df_pred_2d.iloc[idx]['2D']
    pred_boxes_list = get_2d_boxes_list(pred_boxes)
    #print(token0, pred_boxes_list)
    for pred_box in pred_boxes_list:
        coor = pred_box[:4]
        x1 = int(float(coor[0]))
        y1 = int(float(coor[1]))
        x2 = int(float(coor[2]))
        y2 = int(float(coor[3]))
        cls_conf = float(pred_box[4])
        if cls_conf < 0.7:
            break
        cls_pred = pred_box[5]
        cam_type = pred_box[6]
        box_h = abs(y2 - y1)
        box_w = abs(x2 - x1)
        #print(cam)
        sample_cam_token = my_sample["data"][cam_type]
        data_path, boxes, camera_intrinsic = level5data.get_sample_data(sample_cam_token, box_vis_level=BoxVisibility.ANY)
        sample_lidar_token = my_sample["data"]["LIDAR_TOP"]
        pcl_path, boxes, lidar_intrinsic = level5data.get_sample_data(sample_lidar_token)
        pointsensor = level5data.get("sample_data", sample_lidar_token)
        pc = LidarPointCloud.from_file(pcl_path)
        #print(pcl_path)

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = level5data.get("calibrated_sensor", pointsensor["calibrated_sensor_token"])
        pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
        pc.translate(np.array(cs_record["translation"]))

        # Second step: transform to the global frame.
        poserecord = level5data.get("ego_pose", pointsensor["ego_pose_token"])
        pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
        pc.translate(np.array(poserecord["translation"]))

        cam = level5data.get("sample_data", sample_cam_token)
        # Third step: transform into the ego vehicle frame for the timestamp of the image.
        poserecord = level5data.get("ego_pose", cam["ego_pose_token"])
        pc.translate(-np.array(poserecord["translation"]))
        pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

        # Fourth step: transform into the camera.
        cs_record = level5data.get("calibrated_sensor", cam["calibrated_sensor_token"])
        pc.translate(-np.array(cs_record["translation"]))
        pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc.points[2, :]

        # Retrieve the color from the depth.
        coloring = depths

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(pc.points[:3, :], np.array(cs_record["camera_intrinsic"]), normalize=True)

        im = Image.open(str(level5data.data_path / cam["filename"]))
        # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 0)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)

        row_mask1 = np.logical_and(points[0, :] >= x1, points[0, :] <= x2)
        row_mask2 = np.logical_and(points[1, :] >= y1, points[1, :] <= y2)
        row_mask = np.logical_and(row_mask1, row_mask2)

        input_2Dbbox = np.array([x1, x2, y1, y2])
        base_points_cam = []
        base_points_cam.append((x1, y2, 1))
        base_points_cam.append((x2, y2, 1))
        base_points_cam = np.array(base_points_cam)
        cs_record = level5data.get("calibrated_sensor", cam["calibrated_sensor_token"])
        quaternion = Quaternion(cs_record["rotation"])
        translation = cs_record["translation"]
        for i in range(80):
            points_projected = project_2D_to_3D(base_points_cam, i, camera_intrinsic, quaternion, translation)
            z = points_projected[0][2]
            if z <= 0:
                break
              
        #print(i, cam)
        #z1 = i-1
        #z2 = z1 + 1



        pc0 = LidarPointCloud.from_file(pcl_path)
        pointsensor_token = sample_lidar_token
        pointsensor = level5data.get("sample_data", pointsensor_token)
        cs_record = level5data.get("calibrated_sensor", pointsensor["calibrated_sensor_token"])
        pc0.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
        pc0.translate(np.array(cs_record["translation"]))
        
        # Second step: transform to the global frame.
        poserecord = level5data.get("ego_pose", pointsensor["ego_pose_token"])
        pc0.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
        pc0.translate(np.array(poserecord["translation"]))

        cam = level5data.get("sample_data", sample_cam_token)
        # Third step: transform into the ego vehicle frame for the timestamp of the image.
        poserecord = level5data.get("ego_pose", cam["ego_pose_token"])
        pc0.translate(-np.array(poserecord["translation"]))
        pc0.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)
        
        
        point_cloud = pc0.points.transpose((1, 0))

        #depth_mask = np.logical_and((pc0.points[0, :]**2 + pc0.points[1, :]**2) >= z1**2, \
        #                            (pc0.points[0, :]**2 + pc0.points[1, :]**2) <= z2**2)

        mask = np.logical_and(mask, row_mask)
        #all_mask = np.logical_and(mask, depth_mask)

        box_point_cloud = point_cloud[mask, :]
        
        if box_point_cloud.shape[0] > 0:
            z_box_center = np.mean(box_point_cloud[:, 2])
            mask = np.ones(box_point_cloud.shape[0], dtype=bool)
            h_mask = np.logical_and(mask, box_point_cloud[:, 2] > (z_box_center+0.1))
            h_point_cloud = box_point_cloud[h_mask,:]
            if h_point_cloud.shape[0] > 0:
                x_box_center = np.mean(h_point_cloud[:,0])
                y_box_center = np.mean(h_point_cloud[:,1])
                box_mask1 = np.logical_and(box_point_cloud[:, 0]>=(x_box_center-1), box_point_cloud[:, 0]<=(x_box_center+1))
                box_mask2 = np.logical_and(box_point_cloud[:, 1]>=(y_box_center-1), box_point_cloud[:, 1]<=(y_box_center+1)) 
                box_mask = np.logical_and(box_mask1, box_mask2)
                box_point_cloud = box_point_cloud[box_mask,:]
                if box_point_cloud.shape[0] > 2:
                    pca = PCA(n_components=2)
                    pca.fit(box_point_cloud[:,:2])
                    vector = pca.components_[0]
                    yaw = np.arctan(vector[1]/vector[0])
                    q = euler2quat(0, 0, yaw)
                    pred_box_point_cloud.append([token0,cam_type, box_point_cloud])
                    x_box_center = np.mean(box_point_cloud[:,0])
                    y_box_center = np.mean(box_point_cloud[:, 1])
                    #z_box_center = np.mean(box_point_cloud[:, 2])
                    #z_box_center = min(box_point_cloud[:, 2]) + class_heights[cls_pred]/2
                    
                    dist = np.sqrt(x_box_center*x_box_center + y_box_center*y_box_center)
                    points_projected = project_2D_to_3D(base_points_cam, dist, camera_intrinsic, quaternion, translation)
                    l1 = abs(points_projected[0][0] - points_projected[1][0])
                    l2 = abs(points_projected[0][1] - points_projected[1][1])
                    l = np.sqrt(l1**2 + l2**2)
                    w = 0.5 * l
                    h = abs(0.8 * l * (y2-y1)/(x2-x1)) 
                    z_box_center = np.mean(box_point_cloud[:, 2])
                    #print(box_center)
                    #w = class_widths[cls_pred]
                    #h = class_heights[cls_pred]
                    #l = class_lens[cls_pred]
                    box_wlh = [w, l, h]
                    box_center = [x_box_center, y_box_center, z_box_center]
                    q = Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
                    box1 = Box(
                           center=box_center,
                           size=box_wlh,
                           orientation=q,
                           name=cls_pred,
                           score=cls_conf,
                           token=token0
                        )


                    poserecord = level5data.get("ego_pose", cam["ego_pose_token"])
                    box1.rotate(Quaternion(poserecord["rotation"]))
                    box1.translate(np.array(poserecord["translation"]))
                    q = box1.orientation.elements
                    box_center = box1.center

                    box3d = Box3D(
                            sample_token=token0,
                            translation=box_center,
                            size=box_wlh,
                            rotation=[q[0],q[1],q[2],q[3]],
                            name=cls_pred,
                            score=cls_conf
                        )

                    pred_box3ds.append(box3d)


# In[ ]:


len(pred_box3ds)


# In[ ]:


len(pred_box_point_cloud)


# In[ ]:


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',linewidth=2,shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


# In[ ]:


def show_lidar_cloud(box_list_3d_pred, pred_box3ds, point_cloud, sample_lidar_token, token, cam): 
    fig, ax = plt.subplots(1,3, figsize=(18, 6))
    
    sample_cam_token = my_sample["data"][cam]
    sd_record = level5data.get("sample_data", sample_cam_token)
    cs_record = level5data.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = level5data.get("sensor", cs_record["sensor_token"])
    pose_record = level5data.get("ego_pose", sd_record["ego_pose_token"])
    #print(pose_record)
    

    print("predicted from 3d boxes: ") 
    for box in box_list_3d_pred:
        x, y, z = box.center
        w, l, h = box.wlh
        name = box.name
        if name == 'animal':
            x_min = x - l/2
            y_min = y - w/2
            rect = patches.Rectangle((x_min,y_min),l,w,linewidth=3,edgecolor='r',facecolor='none')

            c = np.array([255, 61, 99]) / 255.0 #red
            #box.render(ax[0], view=np.eye(4), colors=(c, c, c))
            #print(x,y,z, name)
    
    
    
    
    boxes = level5data.get_boxes(sample_cam_token)
    box_list = []
    #_, boxes, _ = level5data.get_sample_data(sample_lidar_token, box_vis_level=BoxVisibility.ANY, flat_vehicle_coordinates=True)
    for box in boxes:
        if box.name == 'animal':
            ypr = Quaternion(pose_record["rotation"]).yaw_pitch_roll
            yaw = ypr[0]
            box.translate(-np.array(pose_record["translation"]))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
            box_list.append(box)
    print("gth boxes: ")
    for box in box_list:
        x, y, z = box.center
        w, l, h = box.wlh
        name = box.name
        print(x,y,z, name)
        c = np.array(level5data.explorer.get_color(box.name)) / 255.0
        box.render(ax[0], view=np.eye(4), colors=(c, c, c))
    print("predicted from 2d boxes: ") 
    for box3d in pred_box3ds:
        token0 = box3d.sample_token
        if token0 == token:
            x0, y0, z0 = box3d.translation
            w, l, h = box3d.size
            q0, q1, q2, q3 = box3d.rotation
            cls_pred = box3d.name
            cls_conf = box3d.score
            q = Quaternion(w=q0, x=q1, y=q2, z=q3)
            box1 = Box(
                    center=[x0, y0, z0],
                    size=[w, l, h],
                    orientation=q,
                    name=cls_pred,
                    score=cls_conf,
                    token=token
                    )

            box1.translate(-np.array(pose_record["translation"]))
            box1.rotate(Quaternion(pose_record["rotation"]).inverse)
            c = [0.5, 0, 1]
            box1.render(ax[0], view=np.eye(4), colors=(c, c, c))
            x, y, z = box1.center
            x_min = x - l/2
            y_min= y - w/2
            rect = patches.Rectangle((x_min,y_min),l,w,linewidth=5,edgecolor='green',facecolor='none')
            #ax[0].add_patch(rect)  
            print(x, y, z)
    x_lim = (x-2, x+2)
    y_lim = (y-2, y+2)
    z_lim = (-2, 2)
    render_pointcloud_xyz(point_cloud, 2, ax[0], x_lim, y_lim, 1)
    pca.fit(point_cloud[:,:2])
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = 5*vector * np.sqrt(length)
        draw_vector(pca.mean_ + [1,1], pca.mean_+ [1,1] + v, ax[0])
    vector = pca.components_[0]
    yaw = np.degrees(np.arctan(vector[1]/vector[0]))
    print("yaw angle: ", yaw)    
    render_pointcloud_xz(point_cloud, 2, ax[1], x_lim, z_lim, 1)
    render_pointcloud_yz(point_cloud, 2, ax[2], y_lim, z_lim, 1)


# In[ ]:


def get_box_list_3d_gth(boxes_list_3d, sample_cam_token):
    sd_record = level5data.get("sample_data", sample_cam_token)
    cs_record = level5data.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = level5data.get("sensor", cs_record["sensor_token"])
    pose_record = level5data.get("ego_pose", sd_record["ego_pose_token"])
    box_list_3d_gth = []

    for j, pred_box in enumerate(boxes_list_3d):
        x, y, z, w, l, h, q0, q1,q2,q3 = pred_box
        #print(x, y, z)
        x = float(x)
        y = float(y)
        z = float(z)
        w = float(w)
        h = float(h)
        l = float(l)
        q = Quaternion(w=q0, x=q1, y=q2, z=q3)
        box1 = Box(
                center=[x, y, z],
                size=[w, l, h],
                orientation=q,
                name='animal',
                score=1.0,
                token=token
                )
        ypr = Quaternion(pose_record["rotation"]).yaw_pitch_roll
        yaw = ypr[0]
        box1.translate(-np.array(pose_record["translation"]))
        box1.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        box_list_3d_gth.append(box1)
    return box_list_3d_gth


# In[ ]:


def get_box_list_3d_pred(pred_boxes_list_3d, sample_cam_token):
    sd_record = level5data.get("sample_data", sample_cam_token)
    cs_record = level5data.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = level5data.get("sensor", cs_record["sensor_token"])
    pose_record = level5data.get("ego_pose", sd_record["ego_pose_token"])
    box_list_3d_gth = []

    for j, pred_box in enumerate(pred_boxes_list_3d):
        cls_conf, x, y, z, w, l, h, yaw, cls_pred = pred_box
        name = cls_pred
        if name != 'animal':
            cls_conf = float(cls_conf)
            x = float(x)
            y = float(y)
            z = float(z)
            w = float(w)
            h = float(h)
            l = float(l)
            yaw = float(yaw)
            #print(x, y, z, w, l, h, yaw)
            q = euler2quat(0, 0, yaw)
            q = Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
            box1 = Box(
                    center=[x, y, z],
                    size=[w, l, h],
                    orientation=q,
                    name=cls_pred,
                    score=cls_conf,
                    token=token
                    )
            #print(box1)
            ypr = Quaternion(pose_record["rotation"]).yaw_pitch_roll
            yaw = ypr[0]
            box1.translate(-np.array(pose_record["translation"]))
            box1.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
            #print(box1)
            box_list_3d_gth.append(box1)
    return box_list_3d_gth


# In[ ]:


token, cam_type, box_point_cloud = pred_box_point_cloud[13]     
my_sample = level5data.get('sample', token)
sample_lidar_token = my_sample["data"]["LIDAR_TOP"]

df = train_pred.loc[train_pred['Id'] == token]
pred_boxes_list_3d = np.array(df['PredictionString'])[0]
pred_boxes_list_3d = get_pred_boxes_list_3d(pred_boxes_list_3d)
box_list_3d_pred = get_box_list_3d_pred(pred_boxes_list_3d, sample_cam_token)
show_lidar_cloud(box_list_3d_pred, pred_box3ds, box_point_cloud, sample_lidar_token, token, cam_type)


# In[ ]:


for idx in range(10):
    print("Index: ", idx)
    token = df_pred_2d.iloc[idx]['Id']

    pred_boxes = df_pred_2d.iloc[idx]['2D']
    pred_boxes_list = get_2d_boxes_list(pred_boxes)
    for pred_box in pred_boxes_list:
        coor = pred_box[:4]
        x1 = int(float(coor[0]))
        y1 = int(float(coor[1]))
        x2 = int(float(coor[2]))
        y2 = int(float(coor[3]))
        cls_conf = pred_box[4]
        cls_pred = pred_box[5]
        cam_type = pred_box[6]
        
    my_sample = level5data.get('sample', token)
    sample_cam_token = my_sample["data"][cam_type]
    sample_lidar_token = my_sample["data"]["LIDAR_TOP"]

    pcl_path, boxes, lidar_intrinsic = level5data.get_sample_data(sample_lidar_token)
    #print(token, pcl_path)
    pcd = LidarPointCloud.from_file(pcl_path)

    df = train.loc[train['Id'] == token]
    gth_boxes = np.array(df['3D'])[0]
    gth_boxes_list_3d = get_gth_boxes_list_3d(gth_boxes)
    box_list_3d_gth = get_box_list_3d_gth(gth_boxes_list_3d, sample_cam_token)
    
    df = train_pred.loc[train_pred['Id'] == token]
    pred_boxes_list_3d = np.array(df['PredictionString'])[0]
    pred_boxes_list_3d = get_pred_boxes_list_3d(pred_boxes_list_3d)
    box_list_3d_pred = get_box_list_3d_pred(pred_boxes_list_3d, sample_cam_token)

    pointsensor_token = sample_lidar_token
    pointsensor = level5data.get("sample_data", pointsensor_token)
    cs_record = level5data.get("calibrated_sensor", pointsensor["calibrated_sensor_token"])
    pcd.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
    pcd.translate(np.array(cs_record["translation"]))
    
    poserecord = level5data.get("ego_pose", pointsensor["ego_pose_token"])
    pcd.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
    pcd.translate(np.array(poserecord["translation"]))
    
    cam = level5data.get("sample_data", sample_cam_token)
    # Third step: transform into the ego vehicle frame for the timestamp of the image.
    poserecord = level5data.get("ego_pose", cam["ego_pose_token"])
    pcd.translate(-np.array(poserecord["translation"]))
    pcd.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)
    
    point_cloud = pcd.points.transpose((1, 0))
    #show_lidar_cloud(box_list_3d_pred, pred_box3ds, point_cloud, sample_lidar_token, token, cam_type)


# In[ ]:


for idx in range(0,10):
    token = df_pred_2d.iloc[idx]['Id']
    pred_boxes = df_pred_2d.iloc[idx]['2D']
    pred_boxes_list = get_2d_boxes_list(pred_boxes)
    for pred_box in pred_boxes_list:
        coor = pred_box[:4]
        x1 = int(float(coor[0]))
        y1 = int(float(coor[1]))
        x2 = int(float(coor[2]))
        y2 = int(float(coor[3]))
        cls_conf = pred_box[4]
        cls_pred = pred_box[5]
        cam = pred_box[6]
    my_sample = level5data.get('sample', token)
    sample_cam_token = my_sample["data"][cam]
    sample_lidar_token = my_sample["data"]["LIDAR_TOP"]
    # Retrieve sensor & pose records
    sd_record = level5data.get("sample_data", sample_cam_token)
    cs_record = level5data.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = level5data.get("sensor", cs_record["sensor_token"])
    pose_record = level5data.get("ego_pose", sd_record["ego_pose_token"])
    # Get gth boxes
    data_path, box_list_gth, cam_intrinsic = level5data.get_sample_data(sample_cam_token)
    # Get predicted boxes
    boxes = pred_box3ds
    box_list_pred = []
    for box in boxes:
        token0 = box.sample_token
        if token0 == token:
            q = [0,0,0,1]
            x,y,z = box.translation
            w,l,h = box.size
            cls_pred = box.name
            score = box.score

            #print(q,x,y,z,w,l,h)
            box1 = Box(
                center=[x, y, z],
                size=[w, l, h],
                orientation=Quaternion(q),
                name=cls_pred,
                score=score,
                token=token
            )

            #print(box1)    
            #  Move box to ego pose
            box1.translate(-np.array(pose_record["translation"]))
            box1.rotate(Quaternion(pose_record["rotation"]).inverse)
            #  Move box to sensor coord system
            box1.translate(-np.array(cs_record["translation"]))
            box1.rotate(Quaternion(cs_record["rotation"]).inverse)
            #print(box1)
            box_list_pred.append(box1)

    data = Image.open(data_path)
    _, ax = plt.subplots(1, 2, figsize=(18, 9))
    ax[0].imshow(data)

    for box in box_list_gth:
        if box.name == 'animal':
            #print(box)
            c = np.array(level5data.explorer.get_color(box.name)) / 255.0
            box.render(ax[0], view=cam_intrinsic, normalize=True, colors=(c, c, c))

    # Limit visible range.
    ax[0].set_xlim(0, data.size[0])
    ax[0].set_ylim(data.size[1], 0)
    ax[0].set_title('Ground Truth Boxes')

    ax[1].imshow(data)       
    for box in box_list_pred:
        c = np.array(level5data.explorer.get_color(box.name)) / 255.0
        box.render(ax[1], view=cam_intrinsic, normalize=True, colors=(c, c, c))

    # Limit visible range.
    ax[1].set_xlim(0, data.size[0])
    ax[1].set_ylim(data.size[1], 0)
    ax[1].set_title('Predicted Boxes')


# In[ ]:


train_num = len(train)
print(train_num)


# In[ ]:


from lyft_dataset_sdk.eval.detection.mAP_evaluation import  Box3D, recall_precision, get_class_names, get_average_precisions
sample_tokens = []

for idx in range(train_num):
    sample_tokens.append(train.iloc[idx]['Id'])


# In[ ]:


def load_groundtruth_boxes(nuscenes, sample_tokens):
    gt_box3ds = []

    # Load annotations and filter predictions and annotations.
    for sample_token in tqdm(sample_tokens):

        sample = nuscenes.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']

        sample_lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = level5data.get("sample_data", sample_lidar_token)
        ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])
        ego_translation = np.array(ego_pose['translation'])
        
        for sample_annotation_token in sample_annotation_tokens:
            sample_annotation = nuscenes.get('sample_annotation', sample_annotation_token)
            sample_annotation_translation = sample_annotation['translation']
            
            class_name = sample_annotation['category_name']
            if class_name =="animal":
                box3d = Box3D(
                    sample_token=sample_token,
                    translation=sample_annotation_translation,
                    size=sample_annotation['size'],
                    rotation=sample_annotation['rotation'],
                    name=class_name
                )
                gt_box3ds.append(box3d)
            
    return gt_box3ds

gt_box3ds = load_groundtruth_boxes(level5data, sample_tokens)


# In[ ]:


len(gt_box3ds)


# In[ ]:


import json
ARTIFACTS_FOLDER = "./artifacts"
os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)
gt = [b.serialize() for b in gt_box3ds]
pr = [b.serialize() for b in pred_box3ds]


# In[ ]:


gt[4], pr[0]


# In[ ]:


iou_th_range = np.linspace(0.5, 0.95, 10)
metric = {}
processes = []
output_dir = 'tmp/'
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)


# In[ ]:


class_names = ['animal', 'bicycle', 'bus', 'car', 'emergency_vehicle',
                    'motorcycle', 'other_vehicle', 'pedestrian', 'truck']


# In[ ]:


def save_AP(gt, predictions, class_names, iou_threshold, output_dir):
    #computes average precisions (AP) for a given threshold, and saves the metrics in a temp file 
    # use lyft's provided function to compute AP
    AP = get_average_precisions(gt, predictions, class_names, iou_threshold)
    # create a dict with keys as class names and values as their respective APs
    metric = {c:AP[idx] for idx, c in enumerate(class_names)}

    # save the dict in a temp file
    summary_path = str(output_dir) + f'metric_summary_{iou_threshold}.json'
    with open(str(summary_path), 'w') as f:
        json.dump(metric, f)


# In[ ]:


def get_metric_overall_AP(iou_th_range, output_dir, class_names):
    ''' reads temp files and calculates overall per class APs.
    returns:
        `metric`: a dict with key as iou thresholds and value as dicts of class and their respective APs,
        `overall_AP`: overall AP of each class
    '''

    metric = {}
    overall_AP = np.zeros(len(class_names))
    for iou_threshold in iou_th_range:
        summary_path = str(output_dir) + f'metric_summary_{iou_threshold}.json'
        with open(str(summary_path), 'r') as f:
            data = json.load(f) # type(data): dict
            metric[iou_threshold] = data
            overall_AP += np.array([data[c] for c in class_names])
    overall_AP /= len(iou_th_range)
    return metric, overall_AP


# In[ ]:


time_start = time.time()
for iou_threshold in iou_th_range:
    process = Process(target=save_AP, args=(gt, pr, class_names, iou_threshold, output_dir))
    process.start()
    processes.append(process)

for process in processes:
    process.join()
print("Time to evaluate = ", time.time() - time_start)      


# In[ ]:


# get overall metrics
metric, overall_AP = get_metric_overall_AP(iou_th_range, output_dir, class_names)
metric['overall'] = {c: overall_AP[idx] for idx, c in enumerate(class_names)}
metric['mAP'] = np.mean(overall_AP)
for th in iou_th_range:
    print("IOU threshold = ", th)
    average_precisions = list(metric[th].values())
    mAP = np.mean(average_precisions)
    print("Average per class mean average precision = ", mAP)
    for class_id in sorted(list(zip(class_names, average_precisions))):
        print(class_id)
    print("_______________________________________________")
print("Overall mean average precision = ", metric['mAP'])


# In[ ]:


time_start = time.time()
average_precisions = get_average_precisions(gt, pr, class_names, 0.01)
mAP = np.mean(average_precisions)
print("Average per class mean average precision = ", mAP)
for class_id in sorted(list(zip(class_names, average_precisions.flatten().tolist()))):
    print(class_id)
print("Time to evaluate = ", time.time() - time_start)  


# In[ ]:


sub = {}
for i in tqdm(range(len(pred_box3ds))):
    q0 = pred_box3ds[i].rotation[0]
    q1 = pred_box3ds[i].rotation[1]
    q2 = pred_box3ds[i].rotation[2]
    q3 = pred_box3ds[i].rotation[3]
    yaw = np.arctan2(2.0 * (q3 * q0 + q1 * q2) , - 1.0 + 2.0 * (q0 * q0 + q1 * q1))
    pred =  str(pred_box3ds[i].score) + ' ' + str(pred_box3ds[i].center_x)  + ' '  +     str(pred_box3ds[i].center_y) + ' '  + str(pred_box3ds[i].center_z) + ' '  +     str(pred_box3ds[i].width) + ' '     + str(pred_box3ds[i].length) + ' '  + str(pred_box3ds[i].height) + ' ' + str(yaw) + ' '     + str(pred_box3ds[i].name) + ' ' 
        
    if pred_box3ds[i].sample_token in sub.keys():     
        sub[pred_box3ds[i].sample_token] += pred
    else:
        sub[pred_box3ds[i].sample_token] = pred        
    
sample_sub = pd.read_csv('../input/3d-object-detection-for-autonomous-vehicles/train.csv')
for token in set(sample_sub.Id.values).difference(sub.keys()):
    sub[token] = ''


# In[ ]:


sub = pd.DataFrame(list(sub.items()))
sub.columns = sample_sub.columns
sub.head()


# In[ ]:


sub.tail()


# In[ ]:


sub.to_csv('lyft3d_train_pred_animal.csv',index=False)

