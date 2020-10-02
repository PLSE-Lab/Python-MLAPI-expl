#!/usr/bin/env python
# coding: utf-8

# Tutorial adapted from [NiMARE examples](https://github.com/neurostuff/NiMARE/blob/98dd69b7b637ab61c90a8de0f45b36cdadc8154a/examples/nidm_pain_meta-analyses.ipynb)  prepared by Taylor Salo. 

# **Before you begin please make sure your kernel is [Internet connected](https://www.kaggle.com/product-feedback/63544)**

# In[ ]:


get_ipython().system('pip install git+git://github.com/neurostuff/NiMARE.git@608516ec3034e356326dfe70df5e9ed77efd2be8')
import os
import json
import numpy as np
import nibabel as nb
import tempfile
from glob import glob
from os.path import basename, join, dirname, isfile

import pandas as pd
import nibabel as nib
import pylab as plt
from scipy.stats import t
from nilearn.masking import apply_mask
from nilearn.plotting import plot_stat_map

import nimare
from nimare.meta.ibma import (stouffers, fishers, weighted_stouffers,
                              rfx_glm, ffx_glm)
from nimare.utils import t_to_z


# # Exploring the data
# This tutorial assumes that you did literature review and extracted coordinates of peaks of activation from relevant papers. Those could be saved in spreadsheeta (or CSV files). In our case we have to CSV files. One with all of the coordinates:

# In[ ]:


pd.read_csv('../input/coordinates.csv').head()


# ...and one with information about the studies:

# In[ ]:


pd.read_csv('../input/studies.csv').head()


# **Excercise: how many coordinates are there per study?**

# Before using NiMARE we need to prepare the data in a format the library expects

# In[ ]:


dset_dict = {}
coords_df = pd.read_csv('../input/coordinates.csv')
for i, row in pd.read_csv('../input/studies.csv').iterrows():
    this_study_coords = coords_df[coords_df['study_id'] == row[0]]
    contrast = {"sample_sizes": [row[1]],
                "coords": { "space": this_study_coords['space'].unique()[0],
                            "x": list(this_study_coords['x']),
                            "y": list(this_study_coords['y']),
                            "z": list(this_study_coords['z'])}}
    dset_dict[row[0]] = {"contrasts": {"1": contrast }}
with tempfile.NamedTemporaryFile(mode='w', suffix=".json") as fp:
    json.dump(dset_dict, fp)
    fp.flush()
    db = nimare.dataset.Database(fp.name)
    dset = db.get_dataset()
mask_img = dset.mask


# In[ ]:


dset.data['pain_01']


# # Coordinate based meta-analysis

# ## ALE

# We will start with using coordinates of peaks and convolving them with gaussians. This is also known as the ALE method. First image is before statistical inference and the second is showing only statistically significant regions.

# In[ ]:


ale = nimare.meta.cbma.ALE(dset, ids=dset.ids)
ale.fit(n_iters=10)
plot_stat_map(ale.results.images['ale'], cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r', figure=plt.figure(figsize=(18,4)))
plot_stat_map(ale.results.images['z_vfwe'], cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r', figure=plt.figure(figsize=(18,4)))


# Note that we are only using 10 interations in permutation testing to speed things up. Normally you would use ~10,000 permutations to accuratelly establish the null distribution.

# ## MKDA

# An alternative to ALE is convolving with solid spheres and accounting for overlaps. This is known as the MKDA method

# In[ ]:


mkda = nimare.meta.cbma.MKDADensity(dset, ids=dset.ids, kernel__r=10)
mkda.fit(n_iters=10)
plot_stat_map(mkda.results.images['vfwe'], cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r', figure=plt.figure(figsize=(18,4)))


# The results in this particular example are vey similar.

# Want to explore more coordinate based data? Check out https://www.kaggle.com/chrisfilo/neurosynth

# # Image based meta-analysis

# So far we used only coordinates of peaks. This is a massive data reduction if you take into account the amount of data acquired by the scanner. How would the results of the analysis look if we had access to the underlying statistical maps?

# ## Get z-maps
# Here'we are going to prepare the maps for analysis. All maps will be masked and converted to numpy arrays. T maps will be converted to Z maps.

# In[ ]:


z_imgs = []
sample_sizes = []
for study in dset_dict.keys():
    z_map_path = "../input/stat_maps/%s.nidm/ZStatistic_T001.nii.gz"%study
    t_map_path = "../input/stat_maps/%s.nidm/TStatistic.nii.gz"%study
    sample_size = dset_dict[study]["contrasts"]["1"]["sample_sizes"][0]
    if os.path.exists(z_map_path):
        z_imgs.append(nb.load(z_map_path))
        sample_sizes.append(sample_size)
    elif os.path.exists(t_map_path):
        t_map_nii = nb.load(t_map_path)
        # assuming one sided test
        z_map_nii = nb.Nifti1Image(t_to_z(t_map_nii.get_fdata(), sample_size-1), t_map_nii.affine)
        z_imgs.append(z_map_nii)
        sample_sizes.append(sample_size)
        
z_data = apply_mask(z_imgs, mask_img)
sample_sizes = np.array(sample_sizes)


# Lets have a look at some of the  individual maps.

# In[ ]:


for z_img in z_imgs[:5]:
    plot_stat_map(z_img, threshold=0, cut_coords=[0, 0, -8], 
                  draw_cross=False, figure=plt.figure(figsize=(18,4)))


# ## Fisher's

# In[ ]:


result = fishers(z_data, mask_img)
plot_stat_map(result.images['ffx_stat'], threshold=0,
              cut_coords=[0, 0, -8], draw_cross=False,
              figure=plt.figure(figsize=(18,4)))
plot_stat_map(result.images['log_p'], threshold=-np.log(.05),
              cut_coords=[0, 0, -8], draw_cross=False,
              cmap='RdBu_r', figure=plt.figure(figsize=(18,4)))


# **Excercise: does this method distinguish positive and negative activation? Could you plot such map?**

# ## Weighted Stouffer's
# We can use sample_sizes to perform a weighted version of the above analysis.

# In[ ]:


result = weighted_stouffers(z_data, sample_sizes, mask_img)
plot_stat_map(result.images['log_p'], threshold=-np.log(.05),
              cut_coords=[0, 0, -8], draw_cross=False,
              cmap='RdBu_r', figure=plt.figure(figsize=(18,4)))


# **Excercise: adjust the sample sizes to influence the weighted analysis.**
