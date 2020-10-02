#!/usr/bin/env python
# coding: utf-8

# <h1>Dataset class for use with Torch</h1>

# In[ ]:


get_ipython().system('pip install -qqq lyft-dataset-sdk')
get_ipython().system('pip install -qqq pyquaternion')


# In[ ]:


import os
import cv2
from pathlib import Path
from functools import reduce

import numpy as np
from pyquaternion import Quaternion
import torch
from torch.utils.data import Dataset, DataLoader
np.random.seed(42)
torch.manual_seed(42)

import matplotlib.pyplot as plt

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud


# In[ ]:


input_dir = '/kaggle/input/3d-object-detection-for-autonomous-vehicles/'


# In[ ]:


get_ipython().system('ln -sf {input_dir}/train_maps maps')
get_ipython().system('ln -sf {input_dir}/train_images images')
get_ipython().system('ln -sf {input_dir}/train_lidar lidar')


# I use the SDK for now since it makes it very easy to access all the properties. Might want to find a better way

# In[ ]:


level5data = LyftDataset(
    data_path='.',
    json_path=os.path.join(input_dir + 'train_data'),
    verbose=False
)


# <h1>Utility functions</h1>

# In[ ]:


def data2itemlist(data):
    items = []
    for scene in data.scene:
        sample = data.get("sample", scene["first_sample_token"])
        while sample:
            data_dict = {k:data.get("sample_data", v) for k, v in sample["data"].items()}
            for v in data_dict.values(): v["calibrated_sensor"] = data.get("calibrated_sensor", v["calibrated_sensor_token"])
            items.append({
                **data_dict,
                "ego_pose": data.get("ego_pose", data_dict["CAM_FRONT"]["ego_pose_token"]),
                "anns": [data.get("sample_annotation", x) for x in sample["anns"]]
            })
            sample = data.get("sample", sample["next"]) if sample["next"] else None
    return items

def apply_rotation(coords, rotation):
    q = Quaternion(rotation)
    return np.dot(q.rotation_matrix, coords.T).T

def load_and_rotate(sensor):
    data = LidarPointCloud.from_file(Path(sensor["filename"])).points.T[:, :3]
    calibration = sensor["calibrated_sensor"]
    data = apply_rotation(data, calibration["rotation"]) + calibration["translation"]
    return data


# <h1>Collate function for padding</h1>
# 
# This part needs to be thought out. For now it pads everything by adding zeros at the end (0 vectors at the end for LIDAR data, black pixels in the bottom/left part of images).
# 
# Not only does every sample have a different number of LIDAR points, but the videos recorded by the cameras have a different shape.
# Possible solutions include cropping everything to the same size, or somehow unifying all the camera into some kind of 360 degrees view.

# In[ ]:


def collate_fn(samples):
    data = {}
    # list of dicts to dict of lists
    samples = reduce(lambda x, y: {k:[v]+(x[k] if k in x else []) for k, v in y.items()}, samples, {})
    for k, v in samples.items():
        if v[0].shape:
            max_length = [max([dp.shape[dim] for dp in v]) for dim in range(len(v[0].shape))]
            padded = np.zeros(max_length)
            for dp in v:
                padded[tuple(slice(0, dp.shape[dim]) for dim in range(len(dp.shape)))] = dp
            data[k] = torch.from_numpy(padded)
        else:
            data[k] = torch.from_numpy(np.array(v))
    return data


# <h1>Dataset class</h1>

# In[ ]:


class LyftTorchDataset(Dataset):
    def __init__(self, data, use_cache=False):
        self.data = data
        # To add or remove sensors
        self.image_sensors = [
            "CAM_FRONT_RIGHT",
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ]
        self.lidar_sensors = [
            "LIDAR_TOP",
        ]
        self.items = [
            item for item in data2itemlist(data)
            if reduce(lambda x, y: x and (y in item), self.image_sensors + self.lidar_sensors, True)
        ]
        self.use_cache = use_cache
        if use_cache:
            self.cache_dict = {}
    def __len__(self):
        return len(self.items)
    def __getitem__(self, index):
        if self.use_cache and (index in self.cache):
            return self.cache[index]    
        image_data = {
            sensor: cv2.imread(self.items[index][sensor]["filename"]).astype(np.float32)
            for sensor in self.image_sensors
        }
        # TODO group by anchor?
        lidar_data = {
            sensor: load_and_rotate(self.items[index][sensor]).astype(np.float32)
            for sensor in self.lidar_sensors
        }
        data = {
            **image_data,
            **lidar_data,
            "index": np.array(index)
        }
        if self.use_cache:
            self.cache[index] = data
        return data


# Usage

# In[ ]:


dataset = LyftTorchDataset(level5data)
dataloader = DataLoader(dataset, batch_size=6, shuffle=True, collate_fn=collate_fn)


# In[ ]:


example_batch = next(iter(dataloader))


# In[ ]:


{k:v.shape for k, v in example_batch.items()}


# <h1>Stats</h1>
# Let's sample 100 random indeces and plot some stats on the data format

# In[ ]:


shapes = np.array([(*dataset[idx]["CAM_FRONT"].shape, *dataset[idx]["CAM_BACK"].shape, *dataset[idx]["LIDAR_TOP"].shape) for idx in np.random.choice(len(dataset), size=100, replace=False)])


# In[ ]:


plt.figure(figsize=(18, 4))
plt.subplot(1, 5, 1)
plt.title("CAM_FRONT height")
plt.hist(shapes[:,  0])
plt.subplot(1, 5, 2)
plt.title("CAM_FRONT width")
plt.hist(shapes[:, 1])
plt.subplot(1, 5, 3)
plt.title("CAM_BACK height")
plt.hist(shapes[:, 3])
plt.subplot(1, 5, 4)
plt.title("CAM_BACK height")
plt.hist(shapes[:, 4])
plt.subplot(1, 5, 5)
plt.title("LIDAR_TOP points")
plt.hist(shapes[:, 6])
plt.show()


# It seems about 80% of the data consists of 1224x1024 videos, and 20% of 1920x1080 videos. 
# There is significant overlap in the field of view recorded by CAM_FRONT and CAM_FRONT_RIGHT when they record in 1920x1080, but the overlap is quite small for 1224x1024.
# I assume they switched to 1224x1024 recording because their 45 degrees FOV is enough to capture 360 degrees. The 1920x1080 recordings can probably be cut to 1224x1024 without losing any information. I show an example below.
# 
# We see a similar distribution for the number of Lidar points, suggesting there's 2 different sensors that record each ~70000 points and ~110000 points.
# Lidar points may be grouped into voxels with size that can be specified when creating the dataset.
# These two techniques would produce tensors of fixed size, allowing us to use the default `collate_fn` function that doesn't apply any padding.

# In[ ]:


level5data.render_pointcloud_in_image(level5data.scene[0]["first_sample_token"], camera_channel="CAM_FRONT")
level5data.render_pointcloud_in_image(level5data.scene[0]["first_sample_token"], camera_channel="CAM_FRONT_RIGHT")

