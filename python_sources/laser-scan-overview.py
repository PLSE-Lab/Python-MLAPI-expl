#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from glob import glob


# In[ ]:


all_files_df=pd.DataFrame({'path': glob('../input/newlaser/Laser_Data/*/laser_data/*.csv*')})
all_files_df['date']=pd.to_datetime(all_files_df['path'].map(lambda x: x.split('/')[-3]), 
                                    yearfirst=True)
all_files_df['file_id']=all_files_df['path'].map(lambda x: x.split('/')[-1].split('.')[0])
all_files_df['timecode']=pd.to_datetime(
    all_files_df['file_id'].map(lambda x: x.split('_')[1]),
    format='%Y-%m-%d-%H-%M-%S')
all_files_df['idx']=all_files_df['file_id'].map(lambda x: int(x.split('_')[-1]))
all_files_df['size']=all_files_df['path'].map(lambda x: os.stat(x).st_size/1024)
all_files_df = all_files_df[all_files_df['size']>50]
all_files_df.sort_values('timecode', inplace=True)
all_files_df.head(3)


# In[ ]:


all_files_df['size'].plot.hist(100)


# In[ ]:


all_files_df.plot(x='timecode', y='idx')


# In[ ]:


def read_block(in_paths, as_mat_list=False):
    df_list = []
    for c_path in in_paths:
        in_df = pd.read_csv(c_path,header=None,index_col=0)
        df_list+=[in_df]
    if as_mat_list:
        return [x.values for x in df_list]
    else:
        return pd.concat(df_list)


# # Read all scans from one day
# 

# In[ ]:


first_day_df = all_files_df[all_files_df['date']==all_files_df['date'].iloc[0]]
print(first_day_df.shape[0], 'scans')
test_df=read_block(first_day_df['path'].values)
first_day_df.head(3)


# In[ ]:


plt.plot(test_df.index)


# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize=(20, 20))
ax1.matshow(test_df.values)
ax1.set_aspect(0.01)


# # Read the first scan from every time-step

# In[ ]:


first_index_df = all_files_df[all_files_df['idx']==all_files_df['idx'].iloc[0]].copy()
print(first_index_df.shape[0], 'scans')
test_df=read_block(first_index_df['path'].values)
first_index_df.head(3)


# In[ ]:


plt.plot(test_df.index)


# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize=(20, 20))
ax1.matshow(test_df.values)
ax1.set_aspect(0.01)


# # Read scans as stacked array

# In[ ]:


test_mat_list=read_block(first_index_df['path'].values, as_mat_list=True)


# In[ ]:


from skimage.util.montage import montage2d
min_stack_height = np.min([x.shape[0] for x in test_mat_list])
print(min_stack_height)
stack_scans = np.stack([x[:min_stack_height] for x in test_mat_list], 0)
fig, ax1 = plt.subplots(1, 1, figsize=(20, 20))
ax1.imshow(montage2d(stack_scans))


# In[ ]:




