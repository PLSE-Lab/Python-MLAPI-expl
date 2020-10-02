#!/usr/bin/env python
# coding: utf-8

# ## In this kernel we convert LEVEL5 Lyft data (NuScenes format) to KITTI format, which is usually used in public repositories. After this you can search for repos, that solve KITTI 3d-detection task.

# In[ ]:


get_ipython().system('pip install -qqq -U git+https://github.com/stalkermustang/nuscenes-devkit.git')


# In[ ]:


from pathlib import Path
from PIL import Image


# In[ ]:


# dir with all input data from Kaggle
INP_DIR = Path('/kaggle/input/3d-object-detection-for-autonomous-vehicles/')


# In[ ]:


# dir with index json tables (scenes, categories, logs, etc...)
TABLES_DIR = INP_DIR.joinpath('train_data')


# In[ ]:


# Adjust the dataroot parameter below to point to your local dataset path.
# The correct dataset path contains at least the following four folders (or similar): images, lidar, maps
get_ipython().system('ln -s {INP_DIR}/train_images images')
get_ipython().system('ln -s {INP_DIR}/train_maps maps')
get_ipython().system('ln -s {INP_DIR}/train_lidar lidar')


# In[ ]:


DATA_DIR = Path().absolute() 
# Empty init equals '.'.
# We use this because we link train dirs to current dir (cell above)


# In[ ]:


# dir to write KITTY-style dataset
STORE_DIR = DATA_DIR.joinpath('kitti_format')


# In[ ]:


get_ipython().system('python -m lyft_dataset_sdk.utils.export_kitti nuscenes_gt_to_kitti -h')


# In[ ]:


# convertation to KITTY-format
get_ipython().system('python -m lyft_dataset_sdk.utils.export_kitti nuscenes_gt_to_kitti         --lyft_dataroot {DATA_DIR}         --table_folder {TABLES_DIR}         --samples_count 20         --parallel_n_jobs 2         --get_all_detections True         --store_dir {STORE_DIR}')


# In[ ]:


# check created (converted) files. velodyne = LiDAR poinclouds data (in binary)
get_ipython().system('ls {STORE_DIR}/velodyne | head -2')


# In[ ]:


# render converted data for check. Currently don't support multithreading :(
get_ipython().system('python -m lyft_dataset_sdk.utils.export_kitti render_kitti         --store_dir {STORE_DIR}')


# In[ ]:


# Script above write images to 'render' folder
# in store_dir (where we have converted dataset)
RENDER_DIR = STORE_DIR.joinpath('render')


# In[ ]:


# get all rendered files
all_renders = list(RENDER_DIR.glob('*'))
all_renders.sort()


# In[ ]:


# render radar data (bird view) and camera data with bboxes


# In[ ]:


Image.open(all_renders[0])


# In[ ]:


Image.open(all_renders[1])


# ## I'm use rendering only for check success converting. 
# 
# ## Can be used to visualize NN predictions for test lyft set (visual metric estimation :D)

# In[ ]:


get_ipython().system('rm -rf {STORE_DIR}')

