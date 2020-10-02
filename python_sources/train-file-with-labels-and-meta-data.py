#!/usr/bin/env python
# coding: utf-8

# # This kernel prepares the csv file needed for another kernel : [EDA - Weird Images](https://www.kaggle.com/chewzy/eda-weird-images-with-new-updates/)
# 
# ### The csv contains:
# * Image level meta data (width, height, pixel statistics etc)
# * One-hot encoded image labels

# In[ ]:


import numpy as np
import pandas as pd
import PIL
import gc

from PIL import ImageOps, ImageFilter
from multiprocessing import Pool


# In[ ]:


def get_img_properties(img_id, path):
    im = PIL.Image.open(f'{path}{img_id}.png')
    
    width = im.size[0]
    height = im.size[1]
    
    r, g, b = im.split()
    r_arr, g_arr, b_arr = np.array(r), np.array(g), np.array(b)
    r_mean, r_std = np.mean(r_arr), np.std(r_arr)
    g_mean, g_std = np.mean(g_arr), np.std(g_arr)
    b_mean, b_std = np.mean(b_arr), np.std(b_arr)
    
    edges_arr = np.array(im.filter(ImageFilter.FIND_EDGES))
    r_edge_arr, g_edge_arr, b_edge_arr = edges_arr[:,:,0], edges_arr[:,:,1], edges_arr[:,:,2]
    r_edge_mean, r_edge_std = np.mean(r_edge_arr), np.std(r_edge_arr)
    g_edge_mean, g_edge_std = np.mean(g_edge_arr), np.std(g_edge_arr)
    b_edge_mean, b_edge_std = np.mean(b_edge_arr), np.std(b_edge_arr)
    
    hist = im.histogram()
    peak_index = np.argmax(hist)
    peak_val = np.max(hist) / (width * height) # normalize this as images have different size
    
    return np.array([width, height,                      r_mean, r_std, g_mean, g_std, b_mean, b_std,                      r_edge_mean, r_edge_std, g_edge_mean, g_edge_std, b_edge_mean, b_edge_std,                      peak_index, peak_val])


# In[ ]:


df_train = pd.read_csv('../input/train.csv')


# In[ ]:


meta_cols = ['width', 'height',              'r_mean', 'r_std', 'g_mean', 'g_std', 'b_mean', 'b_std',              'r_edge_mean', 'r_edge_std', 'g_edge_mean', 'g_edge_std', 'b_edge_mean', 'b_edge_std',              'peak_index', 'peak_val']

# allocate some memory first
for col in meta_cols:
    df_train[col] = 0


# In[ ]:


n_partitions = 12
n_workers = 12
train_path = '../input/train/'

def parallelize_dataframe(df, func):
    df_split = np.array_split(df, n_partitions)
    pool = Pool(n_workers)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def get_meta_data(data):
    
    meta_data = np.zeros((data.shape[0], 16))
    
    for index, file_id in enumerate(data['id'].values):
        meta_data[index] = get_img_properties(file_id, train_path)
    
    data[meta_cols] = meta_data
    
    return data


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndf_train = parallelize_dataframe(df_train, get_meta_data)')


# In[ ]:


label_df = pd.read_csv('../input/labels.csv')
label_names = label_df['attribute_name'].values


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_labels = np.zeros((df_train.shape[0], len(label_names)))\n\nfor row_index, row in enumerate(df_train['attribute_ids']):\n    for label in row.split():\n        train_labels[row_index, int(label)] = 1")


# In[ ]:


for col in label_names:
    df_train[col] = 0


# In[ ]:


gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndf_train[label_names] = train_labels')


# In[ ]:


df_train.head()


# In[ ]:


df_train.to_csv('weird_images_w_labels.csv', index=False)


# In[ ]:




