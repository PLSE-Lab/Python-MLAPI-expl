#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install fastai2 -q')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
#Load the dependancies
from fastai2.basics import *
from fastai2.callback.all import *
from fastai2.vision.all import *
from fastai2.medical.imaging import *
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
from torchvision.models import resnet18
import pydicom.pixel_data_handlers.gdcm_handler as gdcm_handler 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import pydicom
import os
from torch.utils.data import DataLoader, Dataset
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # There seem to be a bunch of dicom files where the pixels are corrupted

# In[ ]:


train_df=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test_df=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')


# ## Ive extracted the good and bad files into dictionaries of lists: key=subject, value=[list of filenames]

# In[ ]:


import gc

good_file_dict={}
bad_file_dict={}
i=0
for subject in train_df.Patient.unique():
    subject_path='../input/osic-pulmonary-fibrosis-progression/train/' +subject
    all_subject_files = os.listdir(subject_path)
    good_file_list=[]
    bad_file_list=[]
    for file in all_subject_files:
        try:
            im=dcmread(os.path.join(subject_path, file)).pixels
            good_file_list.append(file)
        except ValueError:
            bad_file_list.append(os.path.join(subject_path, file))
            continue
        except RuntimeError:
            bad_file_list.append(os.path.join(subject_path, file))
            continue
    good_file_dict[subject]=good_file_list
    bad_file_dict[subject]=bad_file_list
    i+=1
    print(i/len(train_df.Patient.unique())*100,'%')


i=0
for subject in test_df.Patient.unique():
    subject_path='../input/osic-pulmonary-fibrosis-progression/test/' +subject
    all_subject_files = os.listdir(subject_path)
    good_file_list=[]
    bad_file_list=[]
    for file in all_subject_files:
        try:
            im=dcmread(os.path.join(subject_path, file)).pixels
            good_file_list.append(file)
        except ValueError:
            bad_file_list.append(os.path.join(subject_path, file))
            continue
        except RuntimeError:
            bad_file_list.append(os.path.join(subject_path, file))
            continue
    good_file_dict[subject]=good_file_list
    bad_file_dict[subject]=bad_file_list
    i+=1
    print(i/len(test_df.Patient.unique())*100,'%')


# In[ ]:


good_file_dict['ID00007637202177411956430']


# In[ ]:


import pickle

with open('good_files.pickle', 'wb') as handle:
    pickle.dump(good_file_dict, handle)

with open('good_files.pickle', 'rb') as handle:
    file_dict = pickle.load(handle)
    
with open('bad_files.pickle', 'wb') as handle:
    pickle.dump(bad_file_dict, handle)

