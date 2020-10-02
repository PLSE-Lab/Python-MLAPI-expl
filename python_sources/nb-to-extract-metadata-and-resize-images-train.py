#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import seaborn as sns
import os
import sys
import tqdm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pydicom
import glob
import os
from typing import Dict, List


# In[ ]:


def visualize_one_image(image_files: List[str]) -> None:
    # Take only the first 12 images in the list for ID00165637202237320314458
    
    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    axes = axes.flatten()
    image_index=0
    for image_file in image_files:
        # Load the DICOM image and convert to pixel array
        if 'ID00165637202237320314458' in image_file:
            image_data = pydicom.read_file(image_file).pixel_array
            axes[image_index].imshow(image_data, cmap=plt.cm.bone)
            image_name = '-'.join(image_file.split('/')[-2:])
            axes[image_index].set_title(f'{image_name}')
            image_index+=1
train_image_path = '/kaggle/input/osic-pulmonary-fibrosis-progression/train'
train_image_files = sorted(glob.glob(os.path.join(train_image_path, '*', '*.dcm')))


# In[ ]:


list_files=glob.glob(os.path.join(train_image_path, '*','*.dcm'))
df_im=pd.DataFrame()
df_im['files']=list_files
df_im['folder_name']=df_im['files'].apply(lambda x: x.split('/')[-2])


# ### distribution of images in each folder

# In[ ]:


df_im['folder_name'].value_counts().hist()


# In[ ]:


df_im['folder_name'].value_counts()


# In[ ]:


visualize_one_image(train_image_files)


# In[ ]:


sorted(train_image_files)


# In[ ]:


import gc
gc.collect()


# ### Load the images and resize the image to 512 x 512 preserving the range
# this is memory intensive, we can change this to load into batches 

# In[ ]:


import pydicom
import os
import numpy
from skimage.transform import resize
IMG_PX_SIZE = 512


lstFilesDCM = train_image_files  # create an empty list            
# Get ref file
RefDs = pydicom.read_file(lstFilesDCM[0])

# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

print(ConstPixelDims)
# Load spacing values (in mm)
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]))

# The array is sized based on 'ConstPixelDims'
ArrayDicom = []

# loop through all the DICOM files
for index,filenameDCM in tqdm.tqdm(enumerate(lstFilesDCM)):
    try:
        # read the filet
        ds = pydicom.read_file(filenameDCM)
        img=np.array(ds.pixel_array)
        # store the raw image data
        resized_img = resize(img, (IMG_PX_SIZE, IMG_PX_SIZE), anti_aliasing=True,preserve_range=True)
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = resized_img
        del ds,resized_img,img
    except Exception as e:
        continue


# ### Load the metadata of each file

# In[ ]:


def extract_dicom_meta_data(filename: str) -> Dict:
    # Load image
    
    image_data = pydicom.read_file(filename)
    img=np.array(image_data.pixel_array).flatten()
    row = {
        'Patient': image_data.PatientID,
        'body_part_examined': image_data.BodyPartExamined,
        'image_position_patient': image_data.ImagePositionPatient,
        'image_orientation_patient': image_data.ImageOrientationPatient,
        'photometric_interpretation': image_data.PhotometricInterpretation,
        'rows': image_data.Rows,
        'columns': image_data.Columns,
        'pixel_spacing': image_data.PixelSpacing,
        'window_center': image_data.WindowCenter,
        'window_width': image_data.WindowWidth,
        'modality': image_data.Modality,
        'StudyInstanceUID': image_data.StudyInstanceUID,
        'SeriesInstanceUID': image_data.StudyInstanceUID,
        'StudyID': image_data.StudyInstanceUID, 
        'SamplesPerPixel': image_data.SamplesPerPixel,
        'BitsAllocated': image_data.BitsAllocated,
        'BitsStored': image_data.BitsStored,
        'HighBit': image_data.HighBit,
        'PixelRepresentation': image_data.PixelRepresentation,
        'RescaleIntercept': image_data.RescaleIntercept,
        'RescaleSlope': image_data.RescaleSlope,
        'img_min': np.min(img),
        'img_max': np.max(img),
        'img_mean': np.mean(img),
        'img_std': np.std(img)}

    return row


# In[ ]:


meta_data_df = []
for filename in tqdm.tqdm(train_image_files):
    try:
        meta_data_df.append(extract_dicom_meta_data(filename))
    except Exception as e:
        continue


# In[ ]:


296226


# In[ ]:


meta_data_df = pd.DataFrame.from_dict(meta_data_df)
meta_data_df.head()


# In[ ]:


meta_data_df.to_csv('meta_data.csv',index=False)


# In[ ]:


Conclusion:
Out of 33026 images provides, 296226 images are proper.

