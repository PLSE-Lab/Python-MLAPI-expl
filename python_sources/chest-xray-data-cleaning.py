#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import os
from glob import glob
from tqdm import tqdm

from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras


# In[ ]:


tf.test.is_gpu_available()


# > # Prepare training data

# ## Identify path

# In[ ]:


ls ../input/data/


# In[ ]:


all_xray_df = pd.read_csv('../input/data/Data_Entry_2017.csv')
print('Total Headers', all_xray_df.shape[0])
all_xray_df.tail()


# In[ ]:


# filter unimportant columns
all_xray_df = all_xray_df[['Image Index', 'Finding Labels']]
all_xray_df


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Name -> full path mapping\nall_image_paths = {os.path.basename(x): x for x in glob('../input/data/images*/*/*.png')}\nprint('Scans found:', len(all_image_paths))")


# In[ ]:


# add pull path to dataframe
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df.tail()


# # Clean-up labels

# In[ ]:


all_xray_df['Finding Labels'].value_counts()


# In[ ]:


all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))

from itertools import chain
all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x)>0]
len(all_labels), all_labels


# In[ ]:


# 0-1 encoding
for c_label in all_labels:
    all_xray_df[c_label] = all_xray_df['Finding Labels'].map(
        lambda finding: 1 if c_label in finding else 0)


# In[ ]:


df_label = all_xray_df[all_labels]
df_label


# > # Read raw images and format training data

# In[ ]:


def read_training_data(df, target_size=(128, 128)):
    
    X_raw = []
    y_raw = []
    
    n_rows = df.shape[0]
    for i in tqdm(range(n_rows)):
        row = df.iloc[i]
        image = plt.imread(row['path'])
        
        if image.shape != (1024, 1024):  # a few samples has shape (1024, 1024, 4) instead of (1024, 1024)
            continue

        image = resize(image, target_size)  # downsample to reduce dataset size
        X_raw.append(image)
        y_raw.append(row[all_labels].values.astype(np.int8))
        
    X_raw = np.array(X_raw)
    y_raw = np.array(y_raw)
    
    print(X_raw.shape, y_raw.shape)
    
    ds = xr.Dataset({
        'image': (('sample', 'x', 'y'), X_raw),
        'label': (('sample', 'feature'), y_raw)},
        coords = {'feature': all_labels}
    )
    return ds


# In[ ]:


get_ipython().run_cell_magic('time', '', 'ds_sample = read_training_data(all_xray_df[0:40000])\nds_sample')


# In[ ]:


ds_sample.nbytes / 1e9  # GB


# In[ ]:


ds_sample['image'].isel(sample=slice(0, 12)).plot(col='sample', col_wrap=4, cmap='gray')


# # Save to disk as output

# In[ ]:


get_ipython().run_line_magic('time', "ds_sample.to_netcdf('chest_xray.nc')")


# In[ ]:


ls -lh ./chest_xray.nc


# # Fit simple CNN

# In[ ]:


# def make_model(filters=32, input_shape=(128, 128, 1), num_output=14):
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Conv2D(filters, (3, 3), input_shape=input_shape, activation='relu'),
#         tf.keras.layers.Conv2D(filters, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
#         tf.keras.layers.Conv2D(filters * 2, (3, 3), activation='relu'),
#         tf.keras.layers.Conv2D(filters * 2, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(num_output, activation='sigmoid')  
#         # not softmax, as here is independent binary classification, not multi-label 
#         ])
#     return model


# In[ ]:


# model = make_model()


# In[ ]:


# %%time
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
# model.fit(ds_sample['image'].values[..., np.newaxis], ds_sample['label'].values, epochs=10)


# In[ ]:




