#!/usr/bin/env python
# coding: utf-8

# # Rendering LIDAR points inside a bounding box

# ### This is part of a larger notebook to attempt feeding data into PointRCNN

# In[ ]:


# Operating system
import sys
import os
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "1"

# math
import numpy as np

#progress bar
from tqdm import tqdm, tqdm_notebook
tqdm.pandas()

# data analysis
import pandas as pd

#plotting 2D
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import animation, rc
from PIL import Image
# data analysis
import pandas as pd


# In[ ]:


#machine learning
import sklearn
import h5py
import sklearn.metrics
import tensorflow as tf


# Lyft dataset SDK
get_ipython().system('pip install lyft-dataset-sdk')
from lyft_dataset_sdk.utils.map_mask import MapMask
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion


# science
import scipy
import scipy.ndimage
import scipy.special
from scipy.spatial.transform import Rotation as R


# In[ ]:


DATA_PATH = '.'
ARTIFACTS_FOLDER = "./artifacts"
CWD = os.getcwd()
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images images')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_maps maps')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_lidar lidar')
lyft_dataset = LyftDataset(data_path='.', json_path='/kaggle/input/3d-object-detection-for-autonomous-vehicles/train_data', verbose=True)


# In[ ]:


selected_log_entries = list(filter(lambda l: l['token'] == '71dfb15d2f88bf2aab2c5d4800c0d10a76c279b9fda98720781a406cbacc583b' , lyft_dataset.log))


# In[ ]:


def set_lidar_pointcloud(row, df):
    lidar_filepath = lyft_dataset.get_sample_data_path(row['sampledata_token'])
    lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
    return lidar_pointcloud.points


# In[ ]:


def delete_images():
    get_ipython().system("find ../data/train_lidar/. -name '*.bin' | xargs rm -f")
    get_ipython().system("find ../data/train_images/. -name '*.jpeg' | xargs rm -f")


# In[ ]:



def extract_data(selected_log_entries):
    log_df = pd.DataFrame(selected_log_entries)
    log_df.rename(columns={'token':'log_token'}, inplace=True)


    scene_df =  pd.DataFrame(lyft_dataset.scene)
    scene_df = pd.merge(log_df, scene_df, left_on='log_token', right_on='log_token',how='inner')
    scene_df.rename(columns={'token':'scene_token'}, inplace=True)
    sample_df = pd.DataFrame(lyft_dataset.sample)
    sample_df = pd.merge(sample_df, scene_df[['scene_token','vehicle','log_token']], left_on='scene_token', right_on='scene_token',how='inner')
    sample_df.rename(columns={'token':'sample_token'}, inplace=True)
    sample_df = sample_df[['log_token','vehicle','sample_token','data']]

    sampledata_df = pd.DataFrame(lyft_dataset.sample_data)
    sampledata_df.rename(columns={'token':'sampledata_token'}, inplace=True)
    sampledata_df = pd.merge(sample_df, sampledata_df, left_on='sample_token', right_on='sample_token',how='inner')
    sampledata_df = sampledata_df[[
        'log_token',
        'vehicle',
        'sample_token', 
        'sampledata_token',
        'ego_pose_token', 
        'channel',
        'calibrated_sensor_token',
        'fileformat',
        'filename']]


    ann_df = pd.DataFrame(lyft_dataset.sample_annotation)
    ann_df.rename(columns={'token':'ann_token', 'rotation': 'ann_rotation', 'translation': 'ann_translation'}, inplace=True)
    ann_df = pd.merge(ann_df, sample_df, left_on='sample_token', right_on='sample_token', how='inner')
    ann_df.sort_values(by=['sample_token'], axis=0, inplace=True)
    ann_df = ann_df[['ann_token',
                     'sample_token',
                     'size',
                     'ann_rotation',
                     'ann_translation',
                     'data',
                     'category_name']]
    # sampledata_df = pd.merge(sampledata_df, ann_df, left_on='sample_token', right_on='sample_token',how='inner')

    ep_df = pd.DataFrame(lyft_dataset.ego_pose)
    ep_df.rename(columns={'token':'ego_pose_token', 'rotation': 'ep_rotation', 'translation': 'ep_translation'}, inplace=True)
    ep_df = ep_df[['ego_pose_token',
                     'ep_rotation',
                     'ep_translation']]
    sampledata_df = pd.merge(sampledata_df, ep_df, left_on='ego_pose_token', right_on='ego_pose_token',how='inner')


    cs_df = pd.DataFrame(lyft_dataset.calibrated_sensor)
    cs_df.rename(columns={'token':'calibrated_sensor_token', 'rotation': 'cs_rotation', 'translation': 'cs_translation'}, inplace=True)
    cs_df = cs_df[['calibrated_sensor_token',                     'cs_rotation',                     'cs_translation',                     'camera_intrinsic'
                  ]]
    sampledata_df = pd.merge(sampledata_df, cs_df, left_on='calibrated_sensor_token', right_on='calibrated_sensor_token',how='inner')
    files_df = sampledata_df[['sample_token', 'fileformat', 'filename']]
    files_df['filename'] = 'train_' + files_df['filename'].astype(str)
    lidar_df = files_df[files_df['fileformat'] == 'bin']
    images_df = files_df[files_df['fileformat'] == 'jpeg']
    
    # delete_images()

#     for index, entry in tqdm(lidar_df.iterrows()):
#         zip_command = "unzip ../3d-object-detection-for-autonomous-vehicles.zip "\
#             + entry['filename']\
#             + " -d "\
#             + CWD + "/../data/"
        
    #     os.system(zip_command)
#     for index, entry in tqdm(images_df.iterrows()):
#         zip_command = "unzip ../3d-object-detection-for-autonomous-vehicles.zip "\
#             + entry['filename']\
#             + " -d "\
#             + CWD + "/../data/"
        
#         os.system(zip_command)

    samplelidardata_df = sampledata_df[sampledata_df['fileformat'] == 'bin']
    samplelidardata_df['lidar_pointcloud'] = samplelidardata_df.apply(lambda row: set_lidar_pointcloud(row, sampledata_df), axis=1)
    
    return (sampledata_df, samplelidardata_df, ann_df, lidar_df)


# In[ ]:


(sampledata_df, lidardata_df, ann_df, lidar_df) = extract_data(selected_log_entries)


# In[ ]:


def car_to_sensor (coords, translation, rotation):
    return  Quaternion(rotation).inverse.rotate(np.add(coords, np.negative(translation)))

def world_to_car (coords, translation, rotation):
    return  Quaternion(rotation).inverse.rotate(np.add(coords, np.negative(translation)))

def car_to_world (coords, translation, rotation):
    return  np.add(translation, Quaternion(rotation).rotate(coords))

def sensor_to_car (coords, translation, rotation):
    return  np.add(translation, Quaternion(rotation).rotate(coords))


# In[ ]:


def map_points_to_image(points, camera_token: str, camera_front_token: str, camera_back_token: str):
        # based on devkit map_pointcloud_to_image

        cam = lyft_dataset.get("sample_data", camera_token)
        cam_back = lyft_dataset.get("sample_data", camera_back_token)
        cam_front = lyft_dataset.get("sample_data", camera_front_token)


        cs_record = lyft_dataset.get("calibrated_sensor", cam['calibrated_sensor_token'])
        # print(cs_record)
        image = Image.open(str(lyft_dataset.data_path / cam["filename"]))

        
        points_t = points.T
        
        poserecord_front = lyft_dataset.get("ego_pose", cam_front["ego_pose_token"])
        poserecord_back = lyft_dataset.get("ego_pose", cam_back["ego_pose_token"])
        poserecord = lyft_dataset.get("ego_pose", cam["ego_pose_token"])

        ep_t = np.array(poserecord["translation"])
        ep_r = poserecord_front["rotation"]
        print(ep_t, ep_r)
        
        cs_record = lyft_dataset.get("calibrated_sensor", cam["calibrated_sensor_token"])
        cs_t = np.array(cs_record["translation"])
        cs_r = cs_record["rotation"]

        world_to_camera = lambda coords: [car_to_sensor(world_to_car(xyz, ep_t, ep_r), cs_t, cs_r) for xyz in coords]
        points_t = world_to_camera(points_t)
        points = np.array(points_t).T

        depths = points[2, :]

        # Retrieve the color from the depth.
        coloring = depths
        
        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(points[:3, :], np.array(cs_record["camera_intrinsic"]), normalize=True)

        # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 0)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < image.size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < image.size[1] - 1)
        points = points[:, mask]
        coloring = coloring[mask]

        return points, coloring, image


# In[ ]:


def render_points_in_image(
      points,
      sample_token: str,
      dot_size: int = 2,
      camera_channel: str = "CAM_FRONT",
      out_path: str = None,
  ):
      # based on devkit render_pointcloud_to_image

      sample_record = lyft_dataset.get("sample", sample_token)

      camera_token = sample_record["data"][camera_channel]
      camera_front_token = sample_record["data"]['CAM_FRONT']
      camera_back_token = sample_record["data"]['CAM_BACK']



      points, coloring, im = map_points_to_image(points, camera_token, camera_front_token, camera_back_token)
      plt.figure(figsize=(9, 16))
      plt.imshow(im)
      plt.scatter(points[0, :], points[1, :], c=coloring, s=dot_size)
      plt.axis("off")

      if out_path is not None:
          plt.savefig(out_path)


# In[ ]:


def map_pointcloud_to_box(points, corners):
        corners = np.array(corners)
        minx = np.min(corners[:,0])
        maxx = np.max(corners[:,0])
        miny = np.min(corners[:,1])
        maxy = np.max(corners[:,1])
        minz = np.min(corners[:,2])
        maxz = np.max(corners[:,2])
        
        # Remove points that are outside the bounding box
        mask = np.ones(points.shape[1], dtype=bool)
        mask = np.logical_and(mask, points[0, :] >= minx)
        mask = np.logical_and(mask, points[0, :] <= maxx)
        mask = np.logical_and(mask, points[1, :] >= miny)
        mask = np.logical_and(mask, points[1, :] <= maxy)
        mask = np.logical_and(mask, points[2, :] >= minz)
        mask = np.logical_and(mask, points[2, :] <= maxz)
        box_points = points[:, mask]

        return box_points


# In[ ]:


def render_box_points_in_image(ds, s_df, a_df, sample_token, ann_token, camera_channel):
    s_df = s_df[s_df['sample_token'] == sample_token]
    s_df = s_df[s_df['channel'].str.match('LIDAR')]
    
    all_points = np.zeros((3,0))
    for i in range(len(s_df)):
        print(s_df.iloc[i]['channel'])
        row = s_df.iloc[i]
        cs_t = row['cs_translation']
        cs_r = row['cs_rotation']
        ep_t = row['ep_translation']
        ep_r = row['ep_rotation']


        pointcloud = s_df.iloc[i]['lidar_pointcloud']
        box = lyft_dataset.get_box(ann_token)
        # box corners are in world coordinates
        corners = box.corners().T

        # so transform point cloud to world coordinates
        sensor_to_world = lambda coords: [car_to_world(sensor_to_car(xyz, cs_t, cs_r), ep_t, ep_r) for xyz in coords]
        pc_points_t = pointcloud.T
        pc_points_t = pc_points_t[:,:3]
        pc_points_t = sensor_to_world(pc_points_t)
        pc_points_t = np.array(pc_points_t).T
        all_points = np.concatenate([all_points, pc_points_t], axis=1)
    box = lyft_dataset.get_box(ann_token)
    # box corners are in world coordinates
    corners = box.corners().T

    box_points = map_pointcloud_to_box(all_points, corners)
    print(box_points.shape)


    render_points_in_image(box_points,sample_token,
        5,
        camera_channel)
    return box_points


# In[ ]:


box_points = render_box_points_in_image(lyft_dataset, lidardata_df, ann_df, ann_df.iloc[19]['sample_token'], ann_df.iloc[19]['ann_token'], 'CAM_FRONT_RIGHT')


# In[ ]:


box_points = render_box_points_in_image(lyft_dataset, lidardata_df, ann_df, ann_df.iloc[2]['sample_token'], ann_df.iloc[2]['ann_token'], 'CAM_BACK_RIGHT')


# In[ ]:


lyft_dataset.render_annotation(ann_df.iloc[19]["ann_token"])


# In[ ]:


len(ann_df)


# In[ ]:


lyft_dataset.render_pointcloud_in_image(sample_token = ann_df.iloc[2]['sample_token'],
                                        dot_size = 0.1,
                                        camera_channel = 'CAM_BACK_RIGHT')

