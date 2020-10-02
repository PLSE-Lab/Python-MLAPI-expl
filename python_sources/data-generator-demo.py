#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import glob
import tqdm
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial

import pydicom as dicom
import nibabel as nib

from keras import utils as kutils

from matplotlib import pyplot as plt


# In[ ]:


def load_dicom_volume(src_dir, suffix='*.dcm'):
    """Load DICOM volume and get meta data.
    """
    encode_name = src_dir.split('/')[-1]
    # Read dicom files from the source directory
    # Sort the dicom slices in their respective order by slice location
    dicom_scans = [dicom.read_file(sp)                    for sp in glob.glob(os.path.join(src_dir, suffix))]
    # dicom_scans.sort(key=lambda s: float(s.SliceLocation))
    dicom_scans.sort(key=lambda s: float(s[(0x0020, 0x0032)][2]))

    # Convert to int16, should be possible as values should always be low enough
    # Volume image is in z, y, x order
    volume_image = np.stack([ds.pixel_array                              for ds in dicom_scans]).astype(np.int16)
    return encode_name, volume_image

def load_label(label_fpath, transpose=False):
    encode_name = label_fpath[-39: -7]
    label_data = nib.load(label_fpath)
    label_array = label_data.get_fdata()
    if transpose:
        label_array = np.transpose(label_array, axes=(2, 1, 0))
    return encode_name, label_array


# ## DICOM to npz

# In[ ]:


train_image_folder = "../input/train-images/image"
train_label_folder = "../input/train-labels/label"

train_list = os.listdir(train_image_folder)

# Ignore this data
if 'hmvsa0loxh3ek2y8rzmcyb6zrrh9mwyp' in train_list:
    train_list.remove('hmvsa0loxh3ek2y8rzmcyb6zrrh9mwyp')


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


train_image_npz_folder = './npz/train_images/'

if not os.path.exists(train_image_npz_folder):
    os.makedirs(train_image_npz_folder)


# In[ ]:


for encode in tqdm.tqdm(train_list):
    _, volume_image = load_dicom_volume(os.path.join(train_image_folder, encode))
    npz_folder = os.path.join(train_image_npz_folder, encode)
    if not os.path.exists(npz_folder):
        os.mkdir(npz_folder) 
        
    num_slice = volume_image.shape[0]
    for _z in range(0, num_slice):
        npz_path = os.path.join(npz_folder, "%03d.npz"%(_z))
        np.savez_compressed(npz_path, image=volume_image[_z])
        
    del volume_image


# In[ ]:


get_ipython().system("ls './npz/train_images/'")


# ## NIFTI to npz

# In[ ]:


train_label_npz_folder = './npz/train_labels/'

if not os.path.exists(train_label_npz_folder):
    os.makedirs(train_label_npz_folder)


# In[ ]:


for encode in tqdm.tqdm(train_list):
    _, label_array = load_label(os.path.join(train_label_folder, encode + '.nii.gz'), transpose=True)
    npz_folder = os.path.join(train_label_npz_folder, encode)
    if not os.path.exists(npz_folder):
        os.mkdir(npz_folder) 
        
    num_slice = label_array.shape[0]
    for _z in range(0, num_slice):
        npz_path = os.path.join(npz_folder, "%03d.npz"%(_z))
        np.savez_compressed(npz_path, label=label_array[_z])
        
    del label_array


# ## Data generator

# In[15]:


map_image_list = sorted(glob.glob(os.path.join(train_image_npz_folder, '*/*.npz')))
map_label_list = sorted(glob.glob(os.path.join(train_label_npz_folder, '*/*.npz')))

map_df = pd.DataFrame(data={'image': map_image_list, 'label': map_label_list})
map_df.head()


# In[16]:


class LungSliceModelGenerator(kutils.Sequence):
    'Generates data for Keras'
    def __init__(self, mapping_df, batch_size, shuffle=True):
        'Initialization'
        self.mapping_df = mapping_df
        self.data_num   = mapping_df.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.data_num / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_mapping_df =             self.mapping_df.iloc[index*self.batch_size: (index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(batch_mapping_df)
        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.mapping_df = self.mapping_df.sample(frac=1).reset_index(drop=True)
            
    def __data_generation(self, batch_mapping_df):
        'Generates data containing batch_size samples' 
        # Initialization
        X = np.zeros((self.batch_size, 512, 512, 1))
        y = np.zeros((self.batch_size, 512, 512, 1))

        # Generate data
        cnt = 0
        for i, row in batch_mapping_df.iterrows():
            X[cnt, :, :, 0] = np.load(row['image'])['image']
            y[cnt, :, :, 0] = np.load(row['label'])['label']
            cnt += 1
        return X, y


# In[17]:


batch_size = 16
slice_generator = LungSliceModelGenerator(map_df, batch_size=batch_size)


# In[18]:


def check_gen_images(index, batch_size):
    X, y = slice_generator.__getitem__(index)
    ncols = 4
    nrows = batch_size // 2
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(16, 32))
    for i in range(nrows * ncols // 2):
        axes[i // 2, (i % 2) * 2].imshow(X[i, ..., 0], cmap='gray')
        axes[i // 2, (i % 2) * 2 + 1].imshow(y[i, ..., 0], cmap='gray')
    plt.show()


# In[19]:


check_gen_images(10, 16)


# In[ ]:




