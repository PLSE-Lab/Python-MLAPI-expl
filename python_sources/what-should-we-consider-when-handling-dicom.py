#!/usr/bin/env python
# coding: utf-8

# When you handle medical images, most images have format called DICOM(.dcm).
# 
# DICOM is global standard of medical imaging and has some unique properties compared to usual imaging formats, such as .png, .jpeg.
# 
# In this notebook, we will see what factors should we concern, and how to handle them.
# 
# First, let's load some packages.

# In[ ]:


get_ipython().system('pip install natsort')

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from natsort import natsorted


# I know pydicom is more famous than SimpleITK, yet I prefer SimpleITK.
# 
# You can use whatever package you want.
# 
# Natsort is natural sort package in python.

# In[ ]:


path_train = os.path.join('../input/osic-pulmonary-fibrosis-progression', 'train')
ptns_train = [os.path.join(path_train, _) for _ in os.listdir(path_train)]


# In[ ]:


ptns_train[:10]


# # Number of Slices
# 
# First, let's see how much slices each patient have. This can be counted by reading number of files in one folder (=patient).

# In[ ]:


tot_len = []

for ptn in ptns_train:
    dcmlist = os.listdir(ptn)
    tot_len.append(len(dcmlist))


# In[ ]:


plt.hist(tot_len, bins=20)
plt.show()


# Oops! There is high discrepancy between patients!!!

# # Slice Thickness
# 
# When you handle DICOM, especially CT or MRI data, data are stored in bulk of one shooting.
# 
# Therefore, it can be considered as 3-dimensional data.
# 
# However, computers can only handle discrete data, thus there is a concept called "Slice Thickness" in CT images.
# 
# Usually, CT machine takes image of the patient by rotating spirally. Therefore, original CT data can be re-constructed with any slice thickness bigger than 0.6mm (Usually. This can depend on CT machine).
# 
# To see how thick train images are, we can check slice thickness metadata. 
# 
# This can be done with .GetSpacing() method or .GetMetaData("0018|0050") method on SimpleITK Image class.

# In[ ]:


slice_thickness = []

for ptn in ptns_train:
    exdcm = [os.path.join(ptn, _) for _ in os.listdir(ptn)][0]
    spacing = sitk.ReadImage(exdcm).GetSpacing()
    slice_thickness.append(spacing[2])
plt.hist(slice_thickness, bins=20)
plt.show()


# Every patient has slice thickness with 1mm! How happy we are!!
# 
# However, for the cross-check, let's see how far between slices are.
# 
# # Slice Distance, Pixel Spacing
# 
# By subtracting absolute location of each DICOM files, we can acquire distance between slices.
# 
# This can be done with .GetMetaData("0020|0032") method on SimpleITK Image class.
# 
# And also, we can know physical length of each pixel by .GetSpacing() method.

# In[ ]:


slice_interval = []
spacing_list = []

import pydicom

for ptn in ptns_train:
    try:
        exdcm1 = natsorted([os.path.join(ptn, _) for _ in os.listdir(ptn)])[0]
        exdcm2 = natsorted([os.path.join(ptn, _) for _ in os.listdir(ptn)])[1]
        location1 = sitk.ReadImage(exdcm1).GetMetaData('0020|0032').split('\\')[2]
        location2 = sitk.ReadImage(exdcm2).GetMetaData('0020|0032').split('\\')[2]
        spacing = sitk.ReadImage(exdcm1).GetSpacing()[0]
        spacing_list.append(spacing)
        interval = np.abs(float(location2) - float(location1))
        slice_interval.append(interval)
    except:
        print(ptn)
    slice_thickness.append(interval)

print(len(slice_interval))


# In[ ]:


plt.title("Physical size of each pixel")
plt.hist(spacing_list, bins=20)
plt.show()


# In[ ]:


plt.title("Distance between slices")
plt.hist(slice_interval, bins=20)
plt.show()


# Well, pixel size is very different from patient to patient, as well as distance between slices.
# 
# Why distance has discrepancy though slice thickness is 1mm?
# 
# This can happen when protocols of hospital is "Reconstruct CT image with 1mm. Then save 1 slice of consecutive 10 slices" for various reasons.
# 
# For example, slict distance with 20mm is that only one slice is selected of consecutive 20 slices.
# 
# Happiness does not go so far...

# In[ ]:


ptn1_path = os.path.join(path_train, 'ID00078637202199415319443')
ptn1_dcm = [os.path.join(path_train, 'ID00078637202199415319443', _) for _ in natsorted(os.listdir(ptn1_path))][0]
npy1 = sitk.GetArrayFromImage(sitk.ReadImage(ptn1_dcm)).squeeze()
plt.title('Patient 1')
plt.imshow(npy1, 'gray')
plt.show()

ptn2_path = os.path.join(path_train, 'ID00128637202219474716089')
ptn2_dcm = [os.path.join(path_train, 'ID00128637202219474716089', _) for _ in natsorted(os.listdir(ptn2_path))][0]
npy2 = sitk.GetArrayFromImage(sitk.ReadImage(ptn2_dcm)).squeeze()
plt.title("Patient 2")
plt.imshow(npy2, 'gray')
plt.show()


# In the above images, there is a significant difference - Patient 1 is surrounded by a circle, yet Patient 2 is not.
# 
# Why is this important?
# 
# When CT gantry rotates, it has round shape and gets image with circle shape.
# 
# However, what we really see is square image, therefore what we really see is not exactly what CT machine takes.
# 
# Some CT manufacturers consider area outside of the gantry as uncredible area thus masking it with values less than -1024(Hounsfield unit of air), like -3072.
# 
# Other CT manufacturers consider area outside of the gantry as noise area and just fills that area with air Hounsfield unit.
# 
# Therefore, if we see the min value of npy1, npy2 and draws histogram,

# In[ ]:


print(npy1.min(), npy2.min())


# In[ ]:


plt.title("Histogram of Patient 1")
plt.hist(npy1.flatten(), bins=20)
plt.show()

plt.title("Histogram of Patient 2")
plt.hist(npy2.flatten(), bins=20)
plt.show()


# As you can see, the circled image (Patient 1) has three peaks on histogram - the left peak is outside of circle, middle peak is air area, right peak is body area
# 
# However, the non-circled image (Patient 2) has two peaks on histogram - the left peak is air area, right peak is body area.

# In[ ]:


less_than_3000 = []
bigger_than_3000 = []

for ptn in ptns_train:
    exdcm = natsorted([os.path.join(ptn, _) for _ in os.listdir(ptn)])[0]
    if sitk.GetArrayFromImage(sitk.ReadImage(exdcm)).min()<-3000:
        less_than_3000.append(ptn)
    else:
        bigger_than_3000.append(ptn)
    
print("Number of DICOMS that have HU less than -3000:", len(less_than_3000))
print("Else:", len(bigger_than_3000))


# As we count CT images that has circle area, there seems to be 44 cases for this case

# In conclusion, DICOM is not that easy as png, jpeg image formats. There are lots to consider when preprocessing.
# 
# This is just start of DICOM handling!
# 
# I hope this notebook would be helpful for begineers who start handling DICOM.

# In[ ]:




