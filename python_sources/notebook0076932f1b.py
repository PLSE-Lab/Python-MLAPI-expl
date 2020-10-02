#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

INPUT_FOLDER = '../input/sample_images/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()


# ## Displaying patients' features ##

# In[ ]:


type(patients)


# In[ ]:


len(patients)


# In[ ]:


patients


# ## Load_scan function ##
# 
#  - define the heterogeneous attribute SliceThickness 

# In[ ]:


# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


# ## Peek into each patients' folder ##

# In[ ]:


slices = [dicom.read_file(INPUT_FOLDER + patients[0] + '/' + s) for s in os.listdir(INPUT_FOLDER + patients[0])]


# In[ ]:


type(slices)


# In[ ]:


len(slices)


# In[ ]:


type(slices[0])


# In[ ]:


slices[0]


# In[ ]:


slices[1]


# In[ ]:


slices[2]


# In[ ]:


slices1 = [dicom.read_file(INPUT_FOLDER + patients[1] + '/' + s) for s in os.listdir(INPUT_FOLDER + patients[1])]
len(slices)


# In[ ]:


slices2 = [dicom.read_file(INPUT_FOLDER + patients[2] + '/' + s) for s in os.listdir(INPUT_FOLDER + patients[2])]
len(slices)


# In[ ]:


slices1[0]


# In[ ]:


slices2[0]


# In[ ]:


slices3 = [dicom.read_file(INPUT_FOLDER + patients[3] + '/' + s) for s in os.listdir(INPUT_FOLDER + patients[3])]
slices3[0]


# In[ ]:


slices4 = [dicom.read_file(INPUT_FOLDER + patients[4] + '/' + s) for s in os.listdir(INPUT_FOLDER + patients[4])]
slices4[0]


# Each patients' folder has a set of DICOM files.  The number of DICOM files vary with patients.

# ## Understanding dicom dataset structure ##

# In[ ]:


slices[0]


# List out all the data elements containing the specified string

# In[ ]:


slices[0].dir('s')


# Accessing the data element by
# 
#  -  'tag number'

# In[ ]:


slices[0][0x28,0x1054]


#  -  'data_element()' function

# In[ ]:


slices[0].data_element('PixelSpacing')


#  - data element

# In[ ]:


slices[0].ImagePositionPatient[2]


# In[ ]:


slices[0].PatientName


# 'in' operator

# In[ ]:


'ImageOrientationPatient' in slices[0]


# 'del' operator - delete an element in dicom.dataset

# To work with pixel data intelligently, use 'pixel_array' tag

# In[ ]:


pix_data = slices[0].pixel_array
pix_data


# In[ ]:


import pylab
pylab.imshow(slices[0].pixel_array, cmap=pylab.cm.bone)
pylab.show()


# ## get_pixels_hu function ##
# 
#  - convert to Hounsfield Unit (HU)

# In[ ]:


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


# In[ ]:


first_patient = load_scan(INPUT_FOLDER + patients[0])
first_patient_pixels = get_pixels_hu(first_patient)
plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()

# Show some slice in the middle
plt.imshow(first_patient_pixels[80], cmap=plt.cm.gray)
plt.show()


# In[ ]:


for i in range(len(slices)):
    print(i,'\t',slices[i].PixelSpacing)


# In[ ]:


for i in range(len(slices)):
    print(i,'\t',slices[i].ImagePositionPatient)


# In[ ]:


slices[1].ImagePositionPatient


# In[ ]:


slices[1].pixel_array


# In[ ]:


ss = [dicom.read_file(INPUT_FOLDER + patients[10] + '/' + s) for s in os.listdir(INPUT_FOLDER + patients[10])]


# In[ ]:


ss[0].PixelSpacing


# In[ ]:


ss[0]


# In[ ]:


int(1.5)


# In[ ]:




