#!/usr/bin/env python
# coding: utf-8

# ## Objective
# 
# The objective of this notebook is to go through the **SIIM-ACR Pneumothorax Segmentation** competition to identify Pneumothorax disease in chest x-rays.

# ## Data set for the competition
# 
# For the purpose of this competition, chest x-rays are given in DICOM format. 
# 
# **What is DICOM format?**
# 
# DICOM is an abbreviation for Digital Imaging and Communications in Medicine. It is a standard format for storing, printing and transmitting information in medical imaging. This format is used worldwide. DICOM format is represented as ".dcm".
# 
# Source: https://www.dicomlibrary.com/dicom/
# 
# **What does DICOM format contain?**
# 
# DICOM contains image data along with other important information about the patent demographics, and study parameters such as patient name, patient id, patient age and patient weight. For confidentiality purpose, all information that identifies the patient are removed before transmitting it for educational or research purpose.
# 
# Source: Varma DR. Managing DICOM images: Tips and tricks for the radiologist. Indian J Radiol Imaging [serial online] 2012 [cited 2019 Aug 23];22:4-13. Available from: http://www.ijri.org/text.asp?2012/22/1/4/95396
# 
# **How to read and extract data from DICOM format in python?**
# 
# You can read DICOM file in python using pydicom library. 
# 
# Source https://pydicom.github.io/
# 

# In[ ]:


# Code from https://www.kaggle.com/schlerp/getting-to-know-dicom-and-the-data/data
# This sample code will read a .dcm file, display some information, and plot the image.

import os
from matplotlib import cm
from matplotlib import pyplot as plt
import pydicom

def show_dcm_info(dataset):
    print("Filename.........:", file_path)
    print("Storage type.....:", dataset.SOPClassUID)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name......:", display_name)
    print("Patient id..........:", dataset.PatientID)
    print("Patient's Age.......:", dataset.PatientAge)
    print("Patient's Sex.......:", dataset.PatientSex)
    print("Modality............:", dataset.Modality)
    print("Body Part Examined..:", dataset.BodyPartExamined)
    print("View Position.......:", dataset.ViewPosition)
    
    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)

def plot_pixel_array(dataset, figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()
    
# Path of the dicom file
file_name = 'siim-acr-pneumothorax-segmentation/sample images/1.2.276.0.7230010.3.1.4.8323329.4982.1517875185.837576.dcm'
file_path = os.path.join('../input/', file_name)

# Read dicom format file
dataset = pydicom.dcmread(file_path)

# Display information
show_dcm_info(dataset)

# Plot pixel
plot_pixel_array(dataset)


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:




