#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pylab
import numpy as np
import pydicom
import pandas as pd
from glob import glob
import os

datapath = '../input/'


# This notebook is a short version of my more detailed data exploration notebook here: [beginner-intro-to-lung-opacity-s1](https://www.kaggle.com/giuliasavorgnan/start-here-beginner-intro-to-lung-opacity-s1). I believe I identified a data leakage and wanted to share it asap. 

# # Data Leakage explanation
# 
# The "view position" of the radiograph image appears to be really important in terms of target split. This information is extracted from the metadata header of the radiograph images. 
# 
# For ViewPosition=='AP' the Target [0,1] split is 0.62 - 0.37
# For ViewPosition=='PA' the Target [0,1] split is 0.91 - 0.09
# 
# AP means Anterior-Posterior, whereas PA means Posterior-Anterior. This [webpage](https://www.med-ed.virginia.edu/courses/rad/cxr/technique3chest.html) explains that "Whenever possible the patient should be imaged in an upright PA position.  AP views are less useful and should be reserved for very ill patients who cannot stand erect". One way to interpret this target unbalance is that patients that are imaged in an AP position are those that are more ill, and therefore more likely to have contracted pneumonia. Note that the absolute split between AP and PA images is about 50-50, so the above consideration is extremely significant. 
# 

# In[ ]:


df_box = pd.read_csv(datapath+'stage_1_train_labels.csv')
print('Number of rows (unique boxes per patient) in main train dataset:', df_box.shape[0])
print('Number of unique patient IDs:', df_box['patientId'].nunique())
df_box.head(6)


# In[ ]:


df_aux = pd.read_csv(datapath+'stage_1_detailed_class_info.csv')
print('Number of rows in auxiliary dataset:', df_aux.shape[0])
print('Number of unique patient IDs:', df_aux['patientId'].nunique())
df_aux.head(6)


# In[ ]:


assert df_box['patientId'].values.tolist() == df_aux['patientId'].values.tolist(), 'PatientId columns are different.'
df_train = pd.concat([df_box, df_aux.drop(labels=['patientId'], axis=1)], axis=1)
df_train.head(6)


# In[ ]:


def get_dcm_data_per_patient(pId):
    '''
    Given one patient ID, 
    return the corresponding dicom data.
    '''
    return pydicom.read_file(datapath+'stage_1_train_images/'+pId+'.dcm')


# In[ ]:


def get_metadata_per_patient(pId, attribute):
    '''
    Given a patient ID, return useful meta-data from the corresponding dicom image header.
    Return: 
    attribute value
    '''
    # get dicom image
    dcmdata = get_dcm_data_per_patient(pId)
    # extract attribute values
    attribute_value = getattr(dcmdata, attribute)
    return attribute_value


# In[ ]:


# create list of attributes that we want to extract (manually edited after checking which attributes contained valuable information)
attributes = ['PatientSex', 'PatientAge', 'ViewPosition']
for a in attributes:
    df_train[a] = df_train['patientId'].apply(lambda x: get_metadata_per_patient(x, a))
# convert patient age from string to numeric
df_train['PatientAge'] = df_train['PatientAge'].apply(pd.to_numeric, errors='coerce')
# remove a few outliers
df_train['PatientAge'] = df_train['PatientAge'].apply(lambda x: x if x<120 else np.nan)
df_train.head()


# In[ ]:


# look at age statistics between positive and negative target groups
df_train.drop_duplicates('patientId').groupby('Target')['PatientAge'].describe()


# In[ ]:


# look at gender statistics between positive and negative target groups
df_train.drop_duplicates('patientId').groupby(['PatientSex', 'Target']).size() / df_train.drop_duplicates('patientId').groupby(['PatientSex']).size()


# In[ ]:


# look at patient position statistics between positive and negative target groups
df_train.drop_duplicates('patientId').groupby(['ViewPosition', 'Target']).size() / df_train.drop_duplicates('patientId').groupby(['ViewPosition']).size()


# In[ ]:


# absolute split of view position
df_train.groupby('ViewPosition').size()

