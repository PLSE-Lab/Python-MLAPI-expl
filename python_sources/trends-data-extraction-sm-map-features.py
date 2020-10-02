#!/usr/bin/env python
# coding: utf-8

# # Data Reduction / Extraction

# I thought it might help to make the 'SM' features more acessible to some people by reducing the amount of data (I'm also casting the values down to floats32 to reduce data - hopefully not losing too much information). 
# 
# In neuroimaging parcellations of the brain are often used for this and average values per parcellation are extracted. 
# 
# This way we can get to lower resolutions (i.e. 53 x 400 in this case). 
# 
# Using nilearn masker objects it is also possible to back-project the data into 3D / 4D space, but at the much lower resolution. 
# 
# For simplicity - and to add the participant Id, I am also flattening the data you can recover the structure by reshaping to (53, 400). 
# 
# Using joblib, we get a datasize of around 1GB - which is nice. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nilearn import plotting, image, input_data, datasets
import nibabel as nb
import h5py
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from tqdm.auto import tqdm
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


brain_mask = nb.load('../input/trends-assessment-prediction/fMRI_mask.nii')


# In[ ]:


scores = pd.read_csv('../input/trends-assessment-prediction/train_scores.csv', index_col='Id')
sample_submission = pd.read_csv('../input/trends-assessment-prediction/sample_submission.csv')
# Test indices
test_index = sample_submission.Id.str.split('_', expand=True)[0].unique().astype('int')
# Train indices
train_index = scores.index.astype('int')


# In[ ]:


try:
    schaefer_400_data = datasets.fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2, resume=True, verbose=1)
except:
    print("Probably time out")


# 400 regions seem to provide a robust solution for feature extraction ([c.f. this article ](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5911196/))

# In[ ]:


plotting.plot_roi(schaefer_400_data.maps, title='Schaefer 2018, 400 rois parcellation');


# In[ ]:


from nilearn import input_data
# We also use the brain_mask from the beginning
schaefer_400_masker = input_data.NiftiLabelsMasker(schaefer_400_data.maps, mask_img=brain_mask, smoothing_fwhm=None,
                                                  standardize=False, detrend=False)

def load_matlab(participant_id, masker, path='../input/trends-assessment-prediction/fMRI_train/'):
    mat = np.array(h5py.File(f'{path}{participant_id}.mat', mode='r').get('SM_feature'))
    mat = masker.fit_transform(nb.Nifti1Image(mat.transpose([3,2,1,0]), affine=masker.mask_img.affine))
    return mat.astype('float32').flatten()


# In[ ]:


tmp = load_matlab(list(train_index)[0], schaefer_400_masker)


# In[ ]:


# Parallelization
from joblib import Parallel, delayed

train_sm_data = []

train_sm_data = Parallel(n_jobs=-1)(delayed(load_matlab)(ii, schaefer_400_masker) for ii in tqdm(list(train_index)))


# In[ ]:


train_sm_data = np.stack(train_sm_data)
train_sm_data = pd.DataFrame(train_sm_data, index=train_index)


# In[ ]:


train_sm_data.to_csv('training_data_schaefer18_400.csv.gz', compression='gzip')


# In[ ]:


joblib.dump(train_sm_data, 'training_data_schaefer18_400.pkl', compress=3)


# In[ ]:


import gc
train_sm_data = []
gc.collect()


# In[ ]:


test_sm_data = []

test_sm_data = Parallel(n_jobs=-1)(delayed(load_matlab)(ii, schaefer_400_masker, '../input/trends-assessment-prediction/fMRI_test/') for ii in tqdm(list(test_index)))
test_sm_data = np.stack(test_sm_data)


# In[ ]:



test_sm_data = pd.DataFrame(test_sm_data, index=test_index)

joblib.dump(test_sm_data, 'test_data_schaefer18_400.pkl', compress=3)
test_sm_data.to_csv('test_data_schaefer18_400.csv.gz', compression='gzip')

