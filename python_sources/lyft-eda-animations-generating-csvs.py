#!/usr/bin/env python
# coding: utf-8

# # About this kernel
# 
# This kernel takes a look at the dataset for the Lyft competition, with some visual exploration (display images, animations, etc.). I also convert some of the JSON files into CSV, so feel free to use this kernel output as supplementary data.
# 
# ## Updates
# 
# V3: Added Animations! Please go to the "Animating the images" section at the end!
# 
# ## References
# 
# * Starter Devkit Lyft3D: https://www.kaggle.com/jesucristo/starter-devkit-lyft3d
# * Devkit for the public 2019 Lyft Level 5 AV Dataset: https://github.com/lyft/nuscenes-devkit

# Thanks for [Nanashi's notebook](https://www.kaggle.com/jesucristo/starter-devkit-lyft3d) for going over the setup for the sdk!

# In[ ]:


get_ipython().system('pip install lyft-dataset-sdk --quiet')
# Load the dataset
# Adjust the dataroot parameter below to point to your local dataset path.
# The correct dataset path contains at least the following four folders (or similar): images, lidar, maps, v1.0.1-train
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images images')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_maps maps')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_lidar lidar')


# In[ ]:


import os
import json
from pprint import pprint

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML

from lyft_dataset_sdk.lyftdataset import LyftDataset


# In[ ]:


BASE_PATH = '/kaggle/input/3d-object-detection-for-autonomous-vehicles/'


# In[ ]:


# Thanks to Nanashi!
lyft_data = LyftDataset(
    data_path='.',
    json_path='/kaggle/input/3d-object-detection-for-autonomous-vehicles/train_data', 
    verbose=True
)


# # Preliminary Exploration
# 
# ## Load CSV files

# In[ ]:


train = pd.read_csv('/kaggle/input/3d-object-detection-for-autonomous-vehicles/train.csv')
sub = pd.read_csv('/kaggle/input/3d-object-detection-for-autonomous-vehicles/sample_submission.csv')
print(train.shape)
train.head()


# ## Taking a look at train_images

# In[ ]:


os.listdir(BASE_PATH + "/train_images")[:10]


# ## Exploring JSON files

# In[ ]:


os.listdir(BASE_PATH + "/train_data")


# ### sample_data.json
# Let's load `sample_data.json`, since it contains information about our training data.

# In[ ]:


with open(BASE_PATH + '/train_data/sample_data.json') as f:
    data_json = json.load(f)

print("There are", len(data_json), "records in sample_data.json")

print("\nBelow is a record containing lidar data:")
pprint(data_json[0])

print('\n This one contains information about image data:')
pprint(data_json[2])


# ### Exploring scene.json

# In[ ]:


with open(BASE_PATH + '/train_data/scene.json') as f:
    scene_json = json.load(f)

print("There are", len(scene_json), "records in sample_data.json")

pprint(scene_json[0])


# # Creating Useful Files
# 
# ## Converting sample_data.json to dataframes

# We can safely store the two types into separate dataframes:

# In[ ]:


lidar_data = []
image_data = []

for record in data_json:
    if record['fileformat'] == 'jpeg':
        image_data.append(record)
    else:
        lidar_data.append(record)


# In[ ]:


lidar_df = pd.DataFrame(lidar_data)
image_df = pd.DataFrame(image_data)

print(lidar_df.shape)
print(image_df.shape)


# Let's take a look at what they look like:

# In[ ]:


lidar_df.head()


# In[ ]:


image_df.head()


# ## Augment image_df
# Finally, we will augment the image_df with some information about its host and camera:

# In[ ]:


image_df['host'] = image_df['filename'].apply(lambda st: st.strip('images/host-').split('_')[0])
image_df['cam'] = image_df['filename'].apply(lambda st: st.split('_')[1])
image_df['timestamp'] = image_df['filename'].apply(lambda st: st.split('_')[2].strip('.jpeg'))


# In[ ]:


image_df.head()


# Let's store the dataframes we just created:

# In[ ]:


image_df.to_csv("sample_data_images.csv")
lidar_df.to_csv("lidar_data_images.csv")


# # Exploring images

# In[ ]:


image_df['host'].value_counts()


# In[ ]:


image_df['cam'].value_counts()


# It seems like there is only 12 hosts, of which there are up to 6 cameras. We also notice that each of the cameras are getting an equal amount of pictures (therefore there are little to no defects, or pictures being "thrown away"). We also notice that each host contains a different number of pictures, ranging from 882 to 44k. This could be caused by some of the cars being used on the road for longer than others.

# ## Displaying sample image by host
# 
# In the sample below, we display all the images in a given timeframe (10 pictures taken from the first timestamp), in the order they were taken. Each column are the images given by each camera. We do so for a few hosts.
# 
# Unhide the cell below to see the definition of `display_host_sample(host, n_images, jumps=1)`.

# In[ ]:


def display_host_sample(host, n_images, jumps=1):
    cams = list(sorted(image_df['cam'].unique()))
    
    fig, axs = plt.subplots(
        n_images, len(cams), figsize=(3*len(cams), 3*n_images), 
        sharex=True, sharey=True, gridspec_kw = {'wspace':0.1, 'hspace':0.1}
    )
    
    for i in range(n_images):
        for c, cam in enumerate(cams):
            if i == 0:
                axs[i, c].set_title(cam)
            
            mask1 = image_df.cam == cam
            mask2 = image_df.host == host
            image_path = image_df[mask1 & mask2]
            image_path = image_path.sort_values('timestamp')['filename'].iloc[i*jumps]
            
            img = cv2.imread(BASE_PATH + '/train_' + image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (200, 200))
            
            axs[i, c].imshow(img)
            axs[i, c].axis('off')


# ### Host 017, no frame skip

# In[ ]:


display_host_sample('017', 5, jumps=1)


# ### Host 009, skip 5 frames

# In[ ]:


display_host_sample('009', 5, jumps=5)


# ### Host 012, skips 10 frames

# In[ ]:


display_host_sample('012', 5, jumps=10)


# # Animating the images
# 
# Note: Please use the "-" button to slow down the animation speed.
# 
# Unhide below for the definition of `animate_images(scene, frames, pointsensor_channel='LIDAR_TOP', interval=1)`

# In[ ]:


def generate_next_token(scene):
    scene = lyft_data.scene[scene]
    sample_token = scene['first_sample_token']
    sample_record = lyft_data.get("sample", sample_token)
    
    while sample_record['next']:
        sample_token = sample_record['next']
        sample_record = lyft_data.get("sample", sample_token)
        
        yield sample_token

def animate_images(scene, frames, pointsensor_channel='LIDAR_TOP', interval=1):
    cams = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_FRONT_LEFT',
    ]

    generator = generate_next_token(scene)

    fig, axs = plt.subplots(
        2, len(cams), figsize=(3*len(cams), 6), 
        sharex=True, sharey=True, gridspec_kw = {'wspace': 0, 'hspace': 0.1}
    )
    
    plt.close(fig)

    def animate_fn(i):
        for _ in range(interval):
            sample_token = next(generator)
            
        for c, camera_channel in enumerate(cams):    
            sample_record = lyft_data.get("sample", sample_token)

            pointsensor_token = sample_record["data"][pointsensor_channel]
            camera_token = sample_record["data"][camera_channel]
            
            axs[0, c].clear()
            axs[1, c].clear()
            
            lyft_data.render_sample_data(camera_token, with_anns=False, ax=axs[0, c])
            lyft_data.render_sample_data(camera_token, with_anns=True, ax=axs[1, c])
            
            axs[0, c].set_title("")
            axs[1, c].set_title("")

    anim = animation.FuncAnimation(fig, animate_fn, frames=frames, interval=interval)
    
    return anim


# In[ ]:


get_ipython().run_cell_magic('time', '', 'anim = animate_images(scene=0, frames=100, interval=1)\nHTML(anim.to_jshtml(fps=8))')


# In[ ]:


anim = animate_images(scene=10, frames=100, interval=1)
HTML(anim.to_jshtml(fps=8))


# In[ ]:


anim = animate_images(scene=50, frames=100, interval=1)
HTML(anim.to_jshtml(fps=8))


# In[ ]:


anim = animate_images(scene=100, frames=100, interval=1)
HTML(anim.to_jshtml(fps=8))


# # Animating LIDAR

# Unhide below to see the definition of `animate_lidar(scene, frames, pointsensor_channel='LIDAR_TOP', with_anns=True, interval=1)`.

# In[ ]:


def animate_lidar(scene, frames, pointsensor_channel='LIDAR_TOP', with_anns=True, interval=1):
    generator = generate_next_token(scene)

    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    plt.close(fig)

    def animate_fn(i):
        for _ in range(interval):
            sample_token = next(generator)
        
        axs.clear()
        sample_record = lyft_data.get("sample", sample_token)
        pointsensor_token = sample_record["data"][pointsensor_channel]
        lyft_data.render_sample_data(pointsensor_token, with_anns=with_anns, ax=axs)

    anim = animation.FuncAnimation(fig, animate_fn, frames=frames, interval=interval)
    
    return anim


# In[ ]:


get_ipython().run_cell_magic('time', '', 'anim = animate_lidar(scene=0, frames=100, interval=1)\nHTML(anim.to_jshtml(fps=8))')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'anim = animate_lidar(scene=10, frames=100, interval=1)\nHTML(anim.to_jshtml(fps=8))')


# In[ ]:


anim = animate_lidar(scene=50, frames=100, interval=1)
HTML(anim.to_jshtml(fps=8))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'anim = animate_lidar(scene=100, frames=100, interval=1)\nHTML(anim.to_jshtml(fps=8))')

