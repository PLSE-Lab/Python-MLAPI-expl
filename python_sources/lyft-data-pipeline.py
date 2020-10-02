#!/usr/bin/env python
# coding: utf-8

# Self-driving are the future :)

# ### Installing lyft_datset_sdk

# In[ ]:


get_ipython().system(' pip install git+https://github.com/lyft/nuscenes-devkit')


# In[ ]:


from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer, Quaternion, view_points
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud

import pandas as pd
import matplotlib.pyplot as plt
import os


# In[ ]:


train = pd.read_csv("/kaggle/input/3d-object-detection-for-autonomous-vehicles/train.csv")
train.head(2)


# In[ ]:


train[train["Id"] == "fd5f1c634b59e3b4e0f7a5c48c768a7d334a63221fced363a2ebac41f465830d"]["PredictionString"]


# **PredictionString format** - center_x | center_y | center_z | width | length | height | yaw | class_name 

# In[ ]:


single_object = train.iloc[0]["PredictionString"].split()[:8]
single_object


# In[ ]:


single_object


# In[ ]:


int_single_object = [ float(_) for _ in single_object[:6]]
int_single_object.insert(8 , single_object[7] )
int_single_object


# ## Let's checkout train images

# In[ ]:


from PIL import Image
import matplotlib.pyplot as plt

w=50
h=50
fig=plt.figure(figsize=(20, 20))
columns = 4
rows = 5
for image_index , train_image in enumerate(os.listdir("/kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images/")):
    img = Image.open("/kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images/" + train_image )
    fig.add_subplot(rows, columns, image_index+1)
    plt.title(train_image)
    plt.imshow(img)
    if image_index == 19: break
plt.show()


# ### attribute.json 
# <br>
# 1:{} 3 items <br>
# token:1b388c1f5e5149ae173ad3d674e7ad7f1847e213173d14ce5ecf431ad697ca17 <br>
# description: <br>
# name:object_action_running <br>
# <br>
# 2:{} 3 items <br>
# description: <br>
# token:17d61007ee69782e0ad8ffa5f8cd4c075f18b4b09e11f0e966bc27026c7929ea <br>
# name:object_action_lane_change_left <br>

# ### category.json
# <br>
# 1:{} 3 items <br>
# description: <br>
# token:73e8de69959eb9f5b4cd2859e74bec4b5491417336cad63f27e8edb8530ffbf8 <br>
# name:pedestrian  <br>
# <br>
# 2:{} 3 items <br>
# description: <br>
# token:f81f51e1897311b55c0c6247c3db825466733e08df687c0ea830b026316a1c12 <br>
# name:animal <br>

# ### sample_data.json
# <br>
# 0:{} 10 items <br>
# 1:{} 10 items<br>
# is_key_frame:true <br>
# fileformat:bin <br>
# <br>
# prev:83f6f61cb6f2fc9b985cc23bd5613219eee7da390cd083ebdfa52c3521bff499 <br>
# calibrated_sensor_token:2dba2d3171d7c60b847d65d96babfd7efb487428fa5a3cb20de23712f66f0b50 <br>
# ego_pose_token:97c348ec54ca0cfc892361595513d34d19e53583f97832380f24247f5aca7401 <br>
# timestamp:1551742021601281.5 <br>
# next:c291c5e9c1c70363496e304e964a33036cace6d14c177eaacdf1cf6ae8b59aea <br>
# token:ea079138d89f3887b2bfcddfe16b747e2ec7a366ce66b3b672039956aee2bc82 <br>
# sample_token:c60dc70e93949cbefdda68813cf024fe8e9103bc8a68d691f2070f586104c103 <br>
# filename:lidar/host-a008_lidar1_1235777221601281486.bin <br>

# In[ ]:


import json

with open("/kaggle/input/3d-object-detection-for-autonomous-vehicles/train_data/sample_data.json",encoding='utf-8', errors='ignore') as json_data:
     data = json.load(json_data, strict=False)

data[0]       
        


# In[ ]:


def parse_string_list(single_object):  
   int_single_object = [ float(_) for _ in single_object[:6]]
   int_single_object.insert(8 , single_object[7] )
   return int_single_object


# In[ ]:


for image_index , train_image in enumerate(os.listdir("/kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images/")):
    img = Image.open("/kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images/" + train_image )
    
    for each_image_data in data:
        filename = each_image_data["filename"]
        filename = filename.split("/")[-1]
        sample_token = each_image_data["sample_token"]
        
        if train_image == filename:
            prediction_string = train[train["Id"] == sample_token]["PredictionString"].values
            prediction_string = prediction_string[0].split()
            for each_object_data in range(0 , len(prediction_string) , 8):
                each_object = prediction_string[each_object_data:each_object_data+8]
                each_object =  parse_string_list(each_object)
                
                plt.imshow(img)
                plt.title(each_object[-1])
                print(each_object)

            break
    break   


# ## Using lyft_dataset_sdk

# In[ ]:


get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images images')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_maps maps')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_lidar lidar')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_data data')


# In[ ]:


lyftdata = LyftDataset(data_path='.', json_path='data/', verbose=True)


# In[ ]:


token0 = "fd5f1c634b59e3b4e0f7a5c48c768a7d334a63221fced363a2ebac41f465830d" 
my_sample = lyftdata.get('sample', token0)
my_sample


# In[ ]:


my_sample.keys()


# In[ ]:


my_annotation = lyftdata.get('sample_annotation', my_sample['anns'][0])
my_annotation


# In[ ]:


my_box = lyftdata.get_box(my_annotation['token'])
my_box # Box class instance


# In[ ]:


my_box.center, my_box.wlh # center coordinates + width, length and height


# In[ ]:


lyftdata.render_annotation(my_annotation['token'], margin=10)


# In[ ]:


lyftdata.render_sample(token0)

