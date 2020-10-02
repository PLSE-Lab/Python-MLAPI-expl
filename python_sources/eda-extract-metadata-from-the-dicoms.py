#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import csv
import pandas as pd
import pydicom as dicom


# ## Get list of images for train and test sets

# In[ ]:


get_ipython().run_cell_magic('time', '', "train_folder_dcm = '../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/'\ntrain_images_path = os.listdir(train_folder_dcm)\n\ntest_folder_dcm = '../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/'\ntest_images_path = os.listdir(test_folder_dcm)")


# ### List of attributes in training set dicom image

# In[ ]:


get_ipython().run_cell_magic('time', '', 'dataset = dicom.dcmread(train_folder_dcm + train_images_path[0])\ndataset')


# ### List of attributes in test set dicom image

# In[ ]:


get_ipython().run_cell_magic('time', '', 'dataset = dicom.dcmread(test_folder_dcm + test_images_path[0])\ndataset')


# In total, there are 22 attributes for each dcm, I am creating a list of the attributes.

# In[ ]:


# list of attributes in dicom image
attributes = ['SOPInstanceUID', 'Modality', 'PatientID', 'StudyInstanceUID',
              'SeriesInstanceUID', 'StudyID', 'ImagePositionPatient',
              'ImageOrientationPatient', 'SamplesPerPixel', 'PhotometricInterpretation',
              'Rows', 'Columns', 'PixelSpacing', 'BitsAllocated', 'BitsStored', 'HighBit',
              'PixelRepresentation', 'WindowCenter', 'WindowWidth', 'RescaleIntercept',
              'RescaleSlope', 'PixelData']


# ## Training set attribute extraction

# In[ ]:


get_ipython().run_cell_magic('time', '', 'with open(\'train_patient_detail.csv\', \'w\', newline =\'\') as csvfile:\n    writer = csv.writer(csvfile, delimiter=\',\')\n    writer.writerow(attributes)\n    for image in train_images_path:\n        ds = dicom.dcmread(os.path.join(train_folder_dcm, image))\n        rows = []\n        for field in attributes:\n            if ds.data_element(field) is None:\n                rows.append(\'\')\n            else:\n                x = str(ds.data_element(field)).replace("\'", "")\n                y = x.find(":")\n                x = x[y+2:]\n                rows.append(x)\n        writer.writerow(rows)')


# ## Test set attribute extraction

# In[ ]:


get_ipython().run_cell_magic('time', '', 'with open(\'test_patient_detail.csv\', \'w\', newline =\'\') as csvfile:\n    writer = csv.writer(csvfile, delimiter=\',\')\n    writer.writerow(attributes)\n    for image in test_images_path:\n        ds = dicom.dcmread(os.path.join(test_folder_dcm, image))\n        rows = []\n        for field in attributes:\n            if ds.data_element(field) is None:\n                rows.append(\'\')\n            else:\n                x = str(ds.data_element(field)).replace("\'", "")\n                y = x.find(":")\n                x = x[y+2:]\n                rows.append(x)\n        writer.writerow(rows)')


# ### There is a bad dicom and it is skipped.

# In[ ]:




