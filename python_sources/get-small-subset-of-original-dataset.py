#!/usr/bin/env python
# coding: utf-8

# The datasets are too big in many kaggle competitions. For example, the RSNA intracranial hemorrhage detection dataset is 180G. Sometimes I just want to download a very small subset of the original dataset so I can play with it in my local computer. Below is an example of how I achieve this goal. The idea is to create a Kaggle kernel to copy some data files into an output folder, then zip the folder and download the generated .zip file from Kaggle kernel interface.

# In[ ]:


import os
import random
from shutil import copy, make_archive


data_root = '/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection'
k = 100 # randomly select 100 images from both train and test data folder
os.makedirs('./dataset', exist_ok=True) # create a dataset folder to hold all the files that I wanted to download
copy(os.path.join(data_root, 'stage_2_sample_submission.csv'), 'dataset/stage_2_sample_submission.csv')
copy(os.path.join(data_root, 'stage_2_train.csv'), 'dataset/stage_2_train.csv')
for d in ['stage_2_train', 'stage_2_test']:
    # list all images in train/test folder
    dir_path = os.path.join(data_root, d)
    files = os.listdir(dir_path)
    
    # copy images to target folder
    target_dir = os.path.join('dataset', d)
    os.makedirs(target_dir, exist_ok=True) 
    for f in random.choices(files, k=k): # randomly select k images and copy them to the target folder
        src_file = os.path.join(dir_path, f)
        copy(src_file, target_dir)
        
# zip generated files
make_archive(base_name='download_dataset', format='zip', root_dir='dataset')


# Now you can download the `download_dataset.zip` from the kernel interface.
# 
# ![](https://user-images.githubusercontent.com/1262709/69744216-c2e11780-110d-11ea-82b4-88006cc6d0aa.png)

# In[ ]:




