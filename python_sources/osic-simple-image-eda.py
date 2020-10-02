#!/usr/bin/env python
# coding: utf-8

# # OSIC Simple Image EDA
# 
# Just a very simple notebook to help understand the image data for potential modeling use :)

# # View Directories

# In[ ]:


import matplotlib.pyplot as plt
import pydicom
import json


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
i = 0
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        i+= 1
        if i>30:
            break

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # View Single Image

# In[ ]:


filename = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140/137.dcm"
ds = pydicom.dcmread(filename)
plt.imshow(ds.pixel_array, cmap=plt.cm.bone) 


# # View Examples of Image Size

# In[ ]:


filename = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140/137.dcm"
pydicom.dcmread(filename).pixel_array.shape


# In[ ]:


filename = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00136637202224951350618/353.dcm"
pydicom.dcmread(filename).pixel_array.shape


# # View Example of Image Meta-Data

# In[ ]:


filename = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140/137.dcm"
pydicom.dcmread(filename)


# # Example Extracting Meta-Data

# In[ ]:


dir(pydicom.dcmread(filename))


# In[ ]:


dir(pydicom.dcmread(filename)['ImageOrientationPatient'])


# In[ ]:


pydicom.dcmread(filename)['ImageOrientationPatient'].to_json()


# In[ ]:


json.loads(pydicom.dcmread(filename)['ImageOrientationPatient'].to_json())['Value']


# # View First Few Images in Order

# In[ ]:


# directory for a patient
imdir = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140"
print("total images for patient ID00123637202217151272140: ", len(os.listdir(imdir)))


# In[ ]:


print("images for patient ID00123637202217151272140 in a rough order:")
mylist = os.listdir(imdir)
mylist.sort()
print(mylist)


# In[ ]:


# view first (columns*rows) images in order
w=10
h=10
fig=plt.figure(figsize=(12, 12))
columns = 4
rows = 5
imglist = os.listdir(imdir)
for i in range(1, columns*rows +1):
    filename = imdir + "/" + str(i) + ".dcm"
    ds = pydicom.dcmread(filename)
    fig.add_subplot(rows, columns, i)
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
plt.show()


# # Number of Patients and Images in Training Images Folder

# In[ ]:


files = folders = 0

path = "/kaggle/input/osic-pulmonary-fibrosis-progression/train"

for _, dirnames, filenames in os.walk(path):
  # ^ this idiom means "we won't be using this value"
    files += len(filenames)
    folders += len(dirnames)

print("{:,} files/images, {:,} folders/patients".format(files, folders))


# In[ ]:


files = []
for _, dirnames, filenames in os.walk(path):
  # ^ this idiom means "we won't be using this value"
    files.append(len(filenames))

print("{:,} average files/images per patient".format(round(np.mean(files))))
print("{:,} max files/images per patient".format(round(np.max(files))))
print("{:,} min files/images per patient".format(round(np.min(files))))


# # Number of Patients and Images in Test Images Folder

# In[ ]:


files = folders = 0

path = "/kaggle/input/osic-pulmonary-fibrosis-progression/test"

for _, dirnames, filenames in os.walk(path):
  # ^ this idiom means "we won't be using this value"
    files += len(filenames)
    folders += len(dirnames)

print("{:,} files/images, {:,} folders/patients".format(files, folders))


# In[ ]:


files = []
for _, dirnames, filenames in os.walk(path):
  # ^ this idiom means "we won't be using this value"
    files.append(len(filenames))

print("{:,} average files/images per patient".format(round(np.mean(files))))
print("{:,} max files/images per patient".format(round(np.max(files))))
print("{:,} min files/images per patient".format(round(np.min(files))))


# # Resources
# 
# https://pydicom.github.io/pydicom/stable/old/viewing_images.html
# 
# https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly/46616645
# 
# https://stackoverflow.com/questions/29769181/count-the-number-of-folders-in-a-directory-and-subdirectories

# In[ ]:




