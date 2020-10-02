#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import cv2
import json
import os
import keras
import sklearn
import pydicom
import matplotlib.pyplot as plt
from glob import glob
print(os.listdir("../input"))


# In[ ]:


img_files = glob("../input/sample images/*.dcm")
print(len(img_files))


# ### What is DISCOM?
# 
# Anyone in the medical image processing or diagnostic imaging field, will have undoubtedly dealt with the infamous Digital Imaging and Communications in Medicine (DICOM) standard the de-facto solution to storing and exchanging medical image-data.
# 
# Let's read uncompressed DICOM files through pydicom package

# In[ ]:


fig, axs = plt.subplots(2, 5, figsize=(25, 10))

for ax, img in zip(axs.flatten(), img_files):
    img = pydicom.dcmread(img)
    ax.set_title("Sex {}, Age {}, {}".format(img.PatientSex, img.PatientAge, img.BodyPartExamined))
    ax.imshow(img.pixel_array, cmap=plt.cm.bone)
    ax.grid(True)

plt.show()


# In[ ]:


sample_img = pydicom.dcmread(img_files[0])
plt.figure(figsize = (10, 10))
plt.imshow(sample_img.pixel_array, cmap=plt.cm.bone)


# In[ ]:


# Extra information present inside single image
sample_img.fix_meta_info

