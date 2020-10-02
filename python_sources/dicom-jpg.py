#!/usr/bin/env python
# coding: utf-8

# ### Full Dataset - https://www.kaggle.com/rashmibanthia/osic-pulmonary-jpg 
# 
# 
# This kernel only has a sample, but I used this locally to create the full dataset. Hopefully it will be easier to work with on Colab. Please let me know if I missed something. 
# 
# 
# Quite a bit of code is from - 
# https://www.kaggle.com/aakashnain/dicom-to-jpg-png-on-steroids
# 
# 
# -----------
# 
# Ver 2: Fixed files for patient ID00011637202177653955184 and ID00052637202186188008618/4.dcm 
# 
# 
# Ver 1: 
# (All files for patient - ID00011637202177653955184 seems corrupt, 
# Also this file - ID00052637202186188008618/4.dcm) 
# 

# In[ ]:


get_ipython().system('conda install -c conda-forge gdcm -y')


# In[ ]:


import warnings
warnings.filterwarnings("ignore")

import os
# import cv2
from PIL import Image
import glob
import time
import gdcm
import pydicom
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path, PosixPath

import dask as dd
import dask.array as da


# ## Get list of all training DCM files

# In[ ]:


# Path to the data
data_dir = Path('../input/osic-pulmonary-fibrosis-progression/train')

# get the list of all the dcm files recursively
train_all_files = list(data_dir.glob("**/*.dcm"))

train_all_dirs = list(data_dir.glob("**"))[1:] #Excluding this directory - ../input/osic-pulmonary-fibrosis-progression/train

print("Number of train dcm files found: ", len(train_all_files), " in ", len(train_all_dirs), " directories.")


# One patient per directory - We have 176 patients in training data.

# In[ ]:


# Define the path to output directory
outdir = "./processed_images/train"

# Make the directory
if not os.path.exists(outdir):
    os.makedirs(outdir)


# In[ ]:



# Convert DICOM to JPG/PNG  
def convert_images(filename,outdir,img_type='jpg'):
    """Reads a dcm file and saves the files as png/jpg
    
    Args:
        filename: path to the dcm file
        img_type: format of the processed file (jpg or png)
        
    """
    
    # extract the name of the file
    name = filename.parts[-1]
    
    # read the dcm file
    try:
        ds = pydicom.read_file(str(filename)) 
        img = ds.pixel_array

        outdir = outdir + "/" + str(filename).split("/")[4] #PatientID
        # Make the directory
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        #https://stackoverflow.com/questions/56956198/pil-converting-from-i16-to-jpeg-produce-white-image
        norm = (img.astype(np.float)-img.min())*255.0 / (img.max()-img.min())
        # save the image as jpg/png
        if img_type=="jpg":
            Image.fromarray(norm.astype(np.uint8)).save(outdir + "/" + name.replace('.dcm','.jpg'))
        else:
            Image.fromarray(norm.astype(np.uint8)).save(outdir + "/" + name.replace('.dcm','.png'))
            
    except:
        print(filename)


# In[ ]:


print(len(train_all_files))

train_all_files = train_all_files[0:10] #sample only 10 files


# In[ ]:


# # First using the simple way: the for loop
# t = time.time()
# for f in train_all_files:
#     convert_images(f,outdir)
# print("Time taken : ", time.time() - t)


# In[ ]:


# Using dask and Running only for 10 samples
all_images = [dd.delayed(convert_images)(train_all_files[x],outdir) for x in range(len(train_all_files))]

t = time.time()
dd.compute(all_images)
print("Time taken when using all cores: ", time.time()-t)


# ### Check to ensure every directory has the same set of files as input

# In[ ]:


# Path to the data
data_dir = Path('./processed_images/train')

# get the list of all the dcm files recursively
train_all_files_output = list(data_dir.glob("**/*.jpg"))

train_all_dirs_output = list(data_dir.glob("**"))[1:] #Excluding this directory - ../input/osic-pulmonary-fibrosis-progression/train

print("Number of train dcm files found: ", len(train_all_files_output), " in ", len(train_all_dirs_output), " directories.")

train_all_files_output = [str(i) for  i in train_all_files_output]
train_all_files_output = [i.split("/")[1] + "/" + i.split("/")[2]+"/" + i.split("/")[3][:-4]  for  i in train_all_files_output]
train_all_files_output = sorted(train_all_files_output)


# In[ ]:


train_all_files_tmp = [str(i) for  i in train_all_files]
train_all_files_tmp =  [i.split("/")[3] + "/" + i.split("/")[4]+"/" + i.split("/")[5][:-4]  for  i in train_all_files_tmp]
train_all_files_tmp = sorted(train_all_files_tmp)


# In[ ]:


for i in range(len(train_all_files_tmp)):
    if train_all_files_tmp[i]!=train_all_files_output[i]:
        print(train_all_files_output[i],train_all_files_tmp[i])


# ### Visualize Dicom and Visualize JPG
# 
# (Sanity check)

# In[ ]:


print(train_all_files_tmp[1],train_all_files_output[1])


# In[ ]:


pyd = pydicom.read_file("../input/osic-pulmonary-fibrosis-progression/" + train_all_files_tmp[1]+".dcm")
# pyd = pydicom.read_file('../input/osic-pulmonary-fibrosis-progression/train/ID00052637202186188008618/4.dcm')
image_data = pyd.pixel_array
plt.imshow(image_data, cmap=plt.cm.bone);


# In[ ]:


pyd


# In[ ]:


plt.imshow(Image.open('./processed_images/' + train_all_files_output[1] + '.jpg'));


# # Process Test

# In[ ]:


# Path to the data
data_dir = Path('../input/osic-pulmonary-fibrosis-progression/test')

# get the list of all the dcm files recursively
test_all_files = list(data_dir.glob("**/*.dcm"))

test_all_dirs = list(data_dir.glob("**"))[1:] #Excluding this directory - ../input/osic-pulmonary-fibrosis-progression/test

print("Number of test dcm files found: ", len(test_all_files), " in ", len(test_all_dirs), " directories.")


# One patient per directory - We have only 5 patients in test set.

# In[ ]:


# Define the path to output directory
outdir = "./processed_images/test"

# Make the directory
if not os.path.exists(outdir):
    os.makedirs(outdir)


# In[ ]:


# Using dask 
all_images = [dd.delayed(convert_images)(test_all_files[x],outdir) for x in range(len(test_all_files[0:10]))]

t = time.time()
dd.compute(all_images)
print("Time taken when using all cores: ", time.time()-t)


# In[ ]:


data_dir = Path('./processed_images/test')


# get the list of all the dcm files recursively
test_all_files_output = list(data_dir.glob("**/*.jpg"))
len(test_all_files_output)


# In[ ]:


get_ipython().system('ls -R ./processed_images')


# In[ ]:




