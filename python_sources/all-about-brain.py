#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import os


# ### DCM extention 
# 
# Files that contain the .dcm file extension are most commonly associated with image files that have been saved in the DICOM image format.
# 
# DICOM stands for Digital Imaging and Communications in Medicine. This file format was created as a way to distribute and view medical images using a standardized file format.
# 
# The DCM file format was developed by the National Electrical Manufacturers Association. In addition to medical images, DCM files may also contain patient information.
# 
# The DiskCatalogMaker software application has also been known to use the .dcm file suffix. This program uses the .dcm file extension when saving catalog files that have been created with the software.

# In[ ]:


df_sample=pd.read_csv("/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv")


# In[ ]:


df_train=pd.read_csv("/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv")


# In[ ]:


df_sample.head()


# In[ ]:


df_train.head()


# In[ ]:


df_train['Label'].unique()


# In[ ]:


import seaborn as sns


# In[ ]:


sns.countplot(df_train.Label)


# In[ ]:





# In[ ]:





# Ack: https://www.kaggle.com/mobassir/keras-efficientnetb4-for-intracranial-hemorrhage
