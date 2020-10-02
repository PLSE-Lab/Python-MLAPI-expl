#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">Medical Images Visualization</font></center></h1>
# 
# <img src="https://kaggle2.blob.core.windows.net/datasets-images/515/1026/5aed8d13079d6ff6f733d70bde17001d/dataset-card.png" width=400></img>  
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>Introduction</a>  
# - <a href='#2'>Load packages</a>  
# - <a href='#3'>Read the data</a> 
# - <a href='#4'>Plot low resolution images</a>
# - <a href='#5'>Plot PET/CT images</a>
# - <a href='#6'>References</a>

# # <a id="1">Summary</a>
# 
# The data is a preprocessed subset of the TCIA Study named Soft Tissue Sarcoma. The data have been converted from DICOM folders of varying resolution and data types to 3D HDF5 arrays with isotropic voxel size. This should make it easier to get started and test out various approaches (NN, RF, CRF, etc) to improve segmentations.

# # <a id="1">Load packages</a>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os
PATH="../input/"
print(os.listdir(PATH))


# # <a id="3">Read the data</a>

# In[ ]:


study_df = pd.read_csv(os.path.join(PATH, 'study_list.csv'))
print("Study list: %2d rows, %2d columns" % (study_df.shape[0], study_df.shape[1]))
study_df


# # <a id="4">Plot low resolution images</a>   
# 
# Let's start by verifying what is the number of images per patient and what is the largest number of images for a patient.

# In[ ]:


maxImgSet = 0
with h5py.File(os.path.join(PATH, 'patient_images_lowres.h5'), 'r') as patient_data:
    for (patient_id, patient_img) in patient_data['ct_data'].items():
        maxImgSet = max(maxImgSet, len(patient_img))
        print("Patient:", patient_id, " Images:", len(patient_img))
print("\nLargest number of images:",maxImgSet)      


# We retrieve the images and represent them grouped by `Patient ID`. For each patient data, we prepare to represent maximum **156** images (the maximum number of images in a image set). In most of the cases will be less, we will show 12 images / row.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
with h5py.File(os.path.join(PATH, 'patient_images_lowres.h5'), 'r') as patient_data:
    for (patient_id, patient_img) in patient_data['ct_data'].items():
        crt_row_df = study_df[study_df['Patient ID']==patient_id]
        print(list(crt_row_df.T.to_dict().values()))
        fig, ax = plt.subplots(13,12, figsize=(13,12), dpi = 250)
        for i, crt_patient_img in enumerate(patient_img):
            ax[i//12, i%12].imshow(crt_patient_img, cmap = 'bone')
            ax[i//12, i%12].axis('off')
        plt.subplots_adjust(hspace = .1, wspace = .1)
        plt.show()


# # <a id="5">Plot PET/CT images</a>   
# 
# Let's start by verifying what is the number of images per patient and what is the largest number of images for a patient.

# In[ ]:


maxImgSet = 0
with h5py.File(os.path.join(PATH, 'lab_petct_vox_5.00mm.h5'), 'r') as patient_data:
    for (patient_id, patient_img) in patient_data['ct_data'].items():
        maxImgSet = max(maxImgSet, len(patient_img))
        print("Patient:", patient_id, " Images:", len(patient_img))
print("\nLargest number of images:",maxImgSet)      


# We retrieve the images and represent them grouped by `Patient ID`. For each patient data, we prepare to represent maximum **203** images (the maximum number of images in a image set). In most of the cases will be less, we will show 12 images / row.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
with h5py.File(os.path.join(PATH, 'lab_petct_vox_5.00mm.h5'), 'r') as patient_data:
    for (patient_id, patient_img) in patient_data['ct_data'].items():
        crt_row_df = study_df[study_df['Patient ID']==patient_id]
        print(list(crt_row_df.T.to_dict().values()))
        fig, ax = plt.subplots(17,12, figsize=(13,12), dpi = 250)
        for i, crt_patient_img in enumerate(patient_img):
            ax[i//12, i%12].imshow(crt_patient_img, cmap = 'bone')
            ax[i//12, i%12].axis('off')
        plt.subplots_adjust(hspace = .1, wspace = .1)
        plt.show()


# # <a id="6">References</a>
# 
# [1] Segmenting Soft Tissue Sarcomas, https://www.kaggle.com/4quant/soft-tissue-sarcoma/    
# [2] Visualize CT DICOM Data, https://www.kaggle.com/gpreda/visualize-ct-dicom-data   
# [3] Viewing the data, https://www.kaggle.com/kmader/viewing-the-data  
# 
