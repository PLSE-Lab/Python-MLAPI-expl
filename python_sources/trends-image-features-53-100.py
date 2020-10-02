#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import h5py
import random
from tqdm.notebook import tqdm
import nibabel as nib
from glob import glob
from scipy.stats import ks_2samp
from nilearn import plotting, image, input_data, datasets
import torch
import os
from skimage.feature import greycomatrix
from joblib import Parallel, delayed


# In[ ]:


brain_mask = nib.load('../input/trends-assessment-prediction/fMRI_mask.nii')


# In[ ]:


scores = pd.read_csv('../input/trends-assessment-prediction/train_scores.csv', index_col='Id')
sample_submission = pd.read_csv('../input/trends-assessment-prediction/sample_submission.csv')
# Test indices
test_index = sample_submission.Id.str.split('_', expand=True)[0].unique().astype('int')
# Train indices
train_index = scores.index.astype('int')


# In[ ]:


try:
    schaefer_400_data = datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=1, resume=True, verbose=1)
except:
    print("Probably time out")


# In[ ]:


train_path = '../input/trends-assessment-prediction/fMRI_train/'
test_path = '../input/trends-assessment-prediction/fMRI_test/'


# In[ ]:


from nilearn import input_data
# We also use the brain_mask from the beginning
schaefer_400_masker = input_data.NiftiLabelsMasker(schaefer_400_data.maps, mask_img=brain_mask, smoothing_fwhm=None,
                                                  standardize=False, detrend=False)


# In[ ]:


def load_matlab(participant_id, path, masker=schaefer_400_masker, apply_mask=False):
    mat = np.array(h5py.File(os.path.join(path, f'{participant_id}.mat'), mode='r').get('SM_feature')).transpose([3,2,1,0])
    if apply_mask:
        mat = masker.fit_transform(nib.Nifti1Image(mat, affine=masker.mask_img.affine))
    return mat.astype('float32')


# In[ ]:


def extract_features(idx, path):
    img = load_matlab(idx, path, apply_mask=True)
    features = img.flatten()
    # assert (img == features.reshape((53,100))).all()
    return features


# In[ ]:


extract_features(train_index[0], train_path).shape


# In[ ]:


train_sm_data = []
train_sm_data = Parallel(n_jobs=-1)(delayed(extract_features)(ii, train_path) for ii in tqdm(list(train_index)))


# In[ ]:


test_sm_data = []
test_sm_data = Parallel(n_jobs=-1)(delayed(extract_features)(ii, test_path) for ii in tqdm(list(test_index)))


# In[ ]:


train_df = {}
train_sm_data = np.array(train_sm_data)
for i in range(train_sm_data[0].shape[0]):
    train_df[f'feature_{i}'] = train_sm_data[:,i]

train_df = pd.DataFrame(train_df)
train_df['Id'] = train_index
col = list(train_df.columns)
col = ['Id'] + col[:-1]
train_df = train_df[col]
train_df.head(2)


# In[ ]:


test_df = {}
test_sm_data = np.array(test_sm_data)
for i in range(test_sm_data[0].shape[0]):
    test_df[f'feature_{i}'] = test_sm_data[:,i]

test_df = pd.DataFrame(test_df)
test_df['Id'] = test_index

col = list(test_df.columns)
col = ['Id'] + col[:-1]
test_df = test_df[col]

test_df.head(2)


# In[ ]:


train_df.to_csv('train_features.csv', index=False)
test_df.to_csv('test_features.csv', index=False)


# In[ ]:




