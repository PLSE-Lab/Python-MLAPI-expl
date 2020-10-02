#!/usr/bin/env python
# coding: utf-8

# <a id="toc"></a>
# # Table of Contents
# 1. [Introduction](#introduction)
# 1. [Configure hyper-parameters](#configure_hyper_parameters)
# 1. [Import libraries](#import_libraries)
# 1. [Define useful classes](#define_useful_classes)
# 1. [Get all train directories](#get_all_train_directories)
# 1. [Construct DataFrame for training data](#construct_dataframe_for_training_data)
# 1. [Split training data](#split_training_data)
# 1. [Create datasets and dataloaders](#create_datasets_and_dataloaders)
# 1. [Test the dataloaders](#test_the_dataloaders)
# 1. [Conclusion](#conclusion)

# <a id="introduction"></a>
# # Introduction
# In this kernel, I'll provide a simple example to demonstrate how to load data from multiple Kaggle datasets and feed all of these to a PyTorch Dataset. The custom Dataset created in this kernel is just a demo, you can modify it or create a new one to fit your own purpose.
# 
# All the datasets used in this demo was listed in this dicussion -> [*Other useful datasets*](https://www.kaggle.com/c/deepfake-detection-challenge/discussion/128954).
# 
# ---
# [Back to Table of Contents](#toc)

# <a id="configure_hyper_parameters"></a>
# # Configure hyper-parameters
# [Back to Table of Contents](#toc)

# In[ ]:


INPUT_DIR = '/kaggle/input/'

TEST_SIZE = 0.3
RANDOM_STATE = 128

BATCH_SIZE = 8
NUM_WORKERS = 0


# <a id="import_libraries"></a>
# # Import libraries
# [Back to Table of Contents](#toc)

# In[ ]:


import torch
from torch.utils.data import Dataset, DataLoader
from albumentations import Normalize, Compose
import numpy as np
import pandas as pd
import cv2
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import os
import glob


# <a id="define_useful_classes"></a>
# # Define useful classes
# [Back to Table of Contents](#toc)

# In[ ]:


class RandomFaceDataset(Dataset):
    def __init__(self, img_dirs, labels, preprocess=None):
        '''
        Parameters:
            img_dirs: The directories that contain face images.
                Each directory coresponding to a video in the original training data.
            labels: Corresponding labels {'FAKE': 1, 'REAL', 0} of videos
            
        '''
        self.img_dirs = img_dirs
        self.labels = labels
        self.preprocess = preprocess

    def __len__(self):
        return len(self.img_dirs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir = self.img_dirs[idx]
        label = self.labels[idx]
        face_paths = glob.glob(f'{img_dir}/*.png')

        sample = face_paths[np.random.choice(len(face_paths))]
        
        face = cv2.imread(sample, 1)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        if self.preprocess is not None:
            augmented = self.preprocess(image=face)
            face = augmented['image']
        
        return {'face': face, 'label': np.array([label], dtype=float)}


# <a id="get_all_train_directories"></a>
# # Get all train directories
# Return all directories that match a specific pattern
# 
# [Back to Table of Contents](#toc)

# In[ ]:


all_train_dirs = glob.glob(INPUT_DIR + 'deepfake-detection-faces-*')
for i, train_dir in enumerate(all_train_dirs):
    print('[{:02}]'.format(i), train_dir)


# <a id="construct_dataframe_for_training_data"></a>
# # Construct DataFrame for training data
# [Back to Table of Contents](#toc)

# In[ ]:


all_dataframes = []
for train_dir in all_train_dirs:
    df = pd.read_csv(os.path.join(train_dir, 'metadata.csv'))
    df['path'] = df['filename'].apply(lambda x: os.path.join(train_dir, x.split('.')[0]))
    all_dataframes.append(df)

train_df = pd.concat(all_dataframes, ignore_index=True, sort=False)


# In[ ]:


train_df


# In[ ]:


# Remove videos that don't have any face
train_df = train_df[train_df['path'].map(lambda x: os.path.exists(x))]


# In[ ]:


train_df


# In[ ]:


train_df['label'].replace({'FAKE': 1, 'REAL': 0}, inplace=True)


# In[ ]:


train_df


# In[ ]:


label_count = train_df.groupby('label').count()['filename']
print(label_count)


# <a id="split_training_data"></a>
# # Split training data
# [Back to Table of Contents](#toc)

# In[ ]:


X = train_df['path'].to_numpy()
y = train_df['label'].to_numpy()


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


# <a id="create_datasets_and_dataloaders"></a>
# # Create datasets and dataloaders
# [Back to Table of Contents](#toc)

# In[ ]:


preprocess = Compose([
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1)
])

train_dataset = RandomFaceDataset(
    img_dirs=X_train,
    labels=y_train,
    preprocess=preprocess
)
val_dataset = RandomFaceDataset(
    img_dirs=X_val,
    labels=y_val,
    preprocess=preprocess
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# <a id="test_the_dataloaders"></a>
# # Test the dataloaders
# [Back to Table of Contents](#toc)

# In[ ]:


for batch in tqdm(train_dataloader):
    face_batch = batch['face']
    label_batch = batch['label']
    
    print(type(face_batch), face_batch.shape)
    print(type(label_batch), label_batch.shape)

    break


# In[ ]:


for batch in tqdm(val_dataloader):
    face_batch = batch['face']
    label_batch = batch['label']
    
    print(type(face_batch), face_batch.shape)
    print(type(label_batch), label_batch.shape)

    break


# <a id="conclusion"></a>
# # Conclusion
# It's quite easy to load a bunch of datasets into a Kaggle kernel, isn't it?
# Next, let feed these faces and labels to your hungry classifier to see whether it can learn something from these FAKE and REAL faces :3
# 
# I'll daily update the list of preprocessed datasets in this discussion topic -> [*Other useful datasets*](https://www.kaggle.com/c/deepfake-detection-challenge/discussion/128954).
# 
# Do we truly get better classifiers when we have more data? Let's try =]]
# 
# ---
# [Back to Table of Contents](#toc)
