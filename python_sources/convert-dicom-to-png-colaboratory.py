#!/usr/bin/env python
# coding: utf-8

# The original code is on colaboratory so the below will not run on Kaggle.
# 
# I combined a few functions and some code from the earlier EDA kernels so if you see your code, just let me know and I'll source you.
# 
# This is my first public kernel so sorry for any obscurity.
# 
# https://colab.research.google.com/drive/1reOc-bBi2CNBZ94rHsy86dMJbm9KwJGU

# In[ ]:


#!pip install -q pydicom
#!pip3 install -q tqdm 
#!pip3 install -q imgaug
#!pip3 install -q kaggle
#!pip3 install -q pypng
#!pip3 install -q pillow
#!pip3 install PyDrive


# In[ ]:


import os 
import sys
import shutil
import glob
import png
import itertools
import pydicom # for reading dicom files
import os # for doing directory operations 
import pandas as pd # for some simple data analysis (right now, just to load in the labels data and quickly reference it)

get_ipython().run_line_magic('pylab', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import cv2


# In[ ]:


# enter your Kaggle credentionals here ,  you can create tokens from you profile page
os.environ['KAGGLE_USERNAME']="username"
os.environ['KAGGLE_KEY']="key"


# In[ ]:


# run from 'content' directory
# Root directory of the project
os.mkdir('../content/RSNAdata')
ROOT_DIR = os.path.abspath('../content/RSNAdata')
os.chdir(ROOT_DIR)


# In[ ]:


#!kaggle competitions download -c rsna-pneumonia-detection-challenge


# In[ ]:


# unzipping takes a few minutes
#!unzip -q -o stage_1_test_images.zip -d stage_1_test_images
#!unzip -q -o stage_1_train_images.zip -d stage_1_train_images
#!unzip -q -o stage_1_train_labels.csv.zip


# In[ ]:


os.chdir('../')
data_dir = '../content/RSNAdata/stage_1_train_images/'
patients = os.listdir(data_dir)
labels_df = pd.read_csv('../content/RSNAdata/stage_1_train_labels.csv')

labels_df.head()


# In[ ]:


#merges the train labels and detailed class

class_info_df = pd.read_csv('../content/RSNAdata/stage_1_detailed_class_info.csv.zip')
train_labels_df = pd.read_csv('../content/RSNAdata/stage_1_train_labels.csv')

train_class_df = train_labels_df.merge(class_info_df, left_on='patientId', right_on='patientId', how='inner')

train_class_df.head()


# In[ ]:


os.mkdir('../content/Target_0NN_PNGs')    #create folder to store png files, here 0NN = No Lung Opacity / Not Normal


# In[ ]:


#batch conversion using PyPNG

def convertDCMtoPNG():

    img_data = list(train_class_df.T.to_dict().values())
     
    for i ,data_row in enumerate(itertools.islice(img_data, 0, None)):
      try:
        if data_row['class']=='No Lung Opacity / Not Normal':
            patientImage = data_row['patientId']+'.dcm'
            imagePath = os.path.join("../content/RSNAdata/stage_1_train_images/", patientImage)
            data_row_img_data = pydicom.read_file(imagePath)
            data_row_img = pydicom.dcmread(imagePath)
            shape = data_row_img.pixel_array.shape
        
   # Convert to float to avoid overflow or underflow losses.
            image_2d = data_row_img.pixel_array.astype(float)
    
       # Rescaling grey scale between 0-255
            image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

    # Convert to uint
            image_2d_scaled = np.uint8(image_2d_scaled)

    # Write the PNG file
            download_location = os.path.join('..content/Target_0NN_PNGs', data_row['patientId'] + '.png')
            with  open(data_row['patientId'] + '.png','wb') as png_file:
                  w = png.Writer(shape[1], shape[0], greyscale=True)
                  w.write(png_file, image_2d_scaled)
            shutil.move(data_row['patientId'] + '.png', '../content/Target_0NN_PNGs/')
        else:
            continue
      except:
        
        if os.path.exists(download_location):
            copy = False
            continue


# In[ ]:


#convertDCMtoPNG()


# In[ ]:


#adds a zip with all the pngs to your drive

filename = "target_0_NN" #@param {type:"string"}
folders_or_files_to_save = "Target_0NN_PNGs" #@param {type:"string"}
from google.colab import files
from google.colab import auth
from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build

def save_file_to_drive(name, path):
    file_metadata = {
    'name': name,
    'mimeType': 'application/octet-stream'
    }

    media = MediaFileUpload(path, 
                  mimetype='application/octet-stream',
                  resumable=True)

    created = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    print('File ID: {}'.format(created.get('id')))

    return created

extension_zip = ".zip"
zip_file = filename + extension_zip

get_ipython().system('zip -r $zip_file {folders_or_files_to_save} # FOLDERS TO SAVE INTO ZIP FILE')

auth.authenticate_user()
drive_service = build('drive', 'v3')

destination_name = zip_file
path_to_file = zip_file
save_file_to_drive(destination_name, path_to_file)

