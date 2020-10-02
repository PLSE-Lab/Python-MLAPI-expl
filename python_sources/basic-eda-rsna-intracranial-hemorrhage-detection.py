#!/usr/bin/env python
# coding: utf-8

# https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview/hemorrhage-types
# 
# # Hemorrhage Types
# 
# Hemorrhage in the head (intracranial hemorrhage) is a relatively common condition that has many causes ranging from trauma, stroke, aneurysm, vascular malformations, high blood pressure, illicit drugs and blood clotting disorders. The neurologic consequences also vary extensively depending upon the size, type of hemorrhage and location ranging from headache to death. The role of the Radiologist is to detect the hemorrhage, characterize the hemorrhage subtype, its size and to determine if the hemorrhage might be jeopardizing critical areas of the brain that might require immediate surgery.
# 
# While all acute (i.e. new) hemorrhages appear dense (i.e. white) on computed tomography (CT), the primary imaging features that help Radiologists determine the subtype of hemorrhage are the location, shape and proximity to other structures (see table).
# 
# Intraparenchymal hemorrhage is blood that is located completely within the brain itself; intraventricular or subarachnoid hemorrhage is blood that has leaked into the spaces of the brain that normally contain cerebrospinal fluid (the ventricles or subarachnoid cisterns). Extra-axial hemorrhages are blood that collects in the tissue coverings that surround the brain (e.g. subdural or epidural subtypes). ee figure.) Patients may exhibit more than one type of cerebral hemorrhage, which c may appear on the same image. While small hemorrhages are less morbid than large hemorrhages typically, even a small hemorrhage can lead to death because it is an indicator of another type of serious abnormality (e.g. cerebral aneurysm). 
# 
# ![Types](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F603584%2F56162e47358efd77010336a373beb0d2%2Fsubtypes-of-hemorrhage.png?generation=1568657910458946&alt=media)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import warnings
warnings.filterwarnings('ignore')

print(os.listdir("../input/rsna-intracranial-hemorrhage-detection"))
import glob

import pydicom

from matplotlib import cm
from matplotlib import pyplot as plt

from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K

import tensorflow as tf

from tqdm import tqdm_notebook
# Any results you write to the current directory are saved as output.


# # What is DICOM?
# 
# Dicom is a format that has metadata, as well as Pixeldata attached to it. Below I extract some basic info with an image. You will know about the gender and age of the patient, as well as info how the image is sampled and generated. Quite useful to programatically read. Here's the Wikipedia article for it.
# From - https://www.kaggle.com/jesperdramsch/intro-chest-xray-dicom-viz-u-nets-full-data

# In[ ]:


path = "../input/rsna-intracranial-hemorrhage-detection"
dataset = pydicom.dcmread("../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_231d901c1.dcm")
print("Finding the details that can be fetched from a dcm file")
print("")
print(dataset)


# **We dont have information on patient's age, sex, etc. which could have been some helpful features.**

# In[ ]:


# https://www.kaggle.com/jesperdramsch/intro-chest-xray-dicom-viz-u-nets-full-data

def show_dcm_info(dataset):
    print("Storage type.....:", dataset.SOPInstanceUID)
    print()

    print("Photometric.........:", dataset.PhotometricInterpretation)
    print("Patient id..........:", dataset.PatientID)
    print("Modality............:", dataset.Modality)
    print("Image Position......:", dataset.ImagePositionPatient)
    print("Image Orient........:", dataset.ImageOrientationPatient)
    
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


# In[ ]:


show_dcm_info(dataset)
plot_pixel_array(dataset)


# In[ ]:


start = 5   # Starting index of images
num_img = 10 # Total number of images to show

fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10))
for q, file_path in enumerate(glob.glob('../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/*.dcm')[start:start+num_img]):
    dataset = pydicom.dcmread(file_path)
    #show_dcm_info(dataset)
    ax[q].imshow(dataset.pixel_array, cmap=plt.cm.bone)


# In[ ]:


train_df = pd.read_csv(path + '/stage_1_train.csv')
test_df = pd.read_csv(path + '/stage_1_sample_submission.csv')


# In[ ]:


print('Train -', len(train_df))
print('Test -', len(test_df))


# In[ ]:


train_df.head(12)


# Patients go through six tests **(ID_63eb1e259_epidural, ID_63eb1e259_intraparenchymal, ID_63eb1e259_intraventricular, ID_63eb1e259_subarachnoid, ID_63eb1e259_subdural, ID_63eb1e259_any)** and the results are tabulated as boolean values.

# In[ ]:


train_df[['PID','Test']] = train_df.ID.str.rsplit("_", n=1, expand=True)


# In[ ]:


epidural = train_df[train_df.Test == 'epidural']
intraparenchymal = train_df[train_df.Test == 'intraparenchymal']
intraventricular = train_df[train_df.Test == 'intraventricular']
subarachnoid = train_df[train_df.Test == 'subarachnoid']
subdural = train_df[train_df.Test == 'subdural']
anyy = train_df[train_df.Test == 'any']


# In[ ]:


print('EPIDURAL')
display(epidural['Label'].value_counts())
print('INTRAPARENCHYMAL')
display(intraparenchymal['Label'].value_counts())
print('INTRAVENTRICULAR')
display(intraventricular['Label'].value_counts())
print('SUBARACHNOID')
display(subarachnoid['Label'].value_counts())
print('SUBDURAL')
display(subdural['Label'].value_counts())
print("ANY")
display(anyy['Label'].value_counts())

