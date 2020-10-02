#!/usr/bin/env python
# coding: utf-8

# **Quick Note**
# 
# The purpose of this notebook is simply to create the tabular data that will be used in Part 2. At the end of this notebook the dataframes are saved as pickled files. For the details of how the Generators were built and how the Keras cnn was set up please go straight to Part 2.
# 
# However, you will find this notebook useful if you would like to know how to extract meta data from the image files or see how to find and fix patient age errors.

# <hr>

# In[ ]:


from numpy.random import seed
seed(101)
from tensorflow import set_random_seed
set_random_seed(101)

import pandas as pd
import numpy as np
import pydicom
import pylab
import os
import pickle

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Don't Show Warning Messages
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df_train = pd.read_csv('../input/stage_1_train_labels.csv')
df_test = pd.read_csv('../input/stage_1_sample_submission.csv')
df_info = pd.read_csv('../input/stage_1_detailed_class_info.csv')

print(df_train.shape)
print(df_test.shape)
print(df_info.shape)


# ### Create new columns using df_info

# In[ ]:


# check df_info for missing values
df_info.isnull().sum()


# In[ ]:


# New Col: num_bounding_boxes
df_train['num_bounding_boxes'] = 1


# ### Extract the meta data from the images

# In[ ]:


# create a dataframe with unique patientId's
df_group = df_train.drop(['x','y','width','height','Target'], axis=1).groupby('patientId').sum()

# reset the index
df_group.reset_index(inplace=True)

df_group.head()


# In[ ]:


# check
print(df_train['patientId'].nunique())
print(len(df_group))


# In[ ]:


# create new columns
df_group['PatientAge'] = 0
df_group['PatientSex'] = 0
df_group['ViewPosition'] = 0


# In[ ]:


# extract the meta data and store in df_group

for i in range(0,len(df_group)):
    patientId = df_group.loc[i,'patientId']
    
    path = '../input/stage_1_train_images/%s.dcm' % patientId
    
    # get the meta data
    dcm_data = pydicom.read_file(path)
    
    df_group.loc[i,'PatientAge'] = dcm_data.PatientAge
    df_group.loc[i,'PatientSex'] = dcm_data.PatientSex
    df_group.loc[i,'ViewPosition'] = dcm_data.ViewPosition
    


# ### Extract meta data from the test images and store in df_test

# In[ ]:


# create new columns
df_test['PatientAge'] = 0
df_test['PatientSex'] = 0
df_test['ViewPosition'] = 0

for i in range(0,len(df_test)):
    patientId = df_test.loc[i,'patientId']
    
    path = '../input/stage_1_test_images/%s.dcm' % patientId
    
    # get the meta data
    dcm_data = pydicom.read_file(path)
    
    df_test.loc[i,'PatientAge'] = dcm_data.PatientAge
    df_test.loc[i,'PatientSex'] = dcm_data.PatientSex
    df_test.loc[i,'ViewPosition'] = dcm_data.ViewPosition
    


# In[ ]:


# change the datatype to int16. now it is string
df_group['PatientAge'] = df_group['PatientAge'].astype(np.int16)
df_test['PatientAge'] = df_test['PatientAge'].astype(np.int16)


# ### Add the class column to df_group

# In[ ]:


df = df_info

df.drop_duplicates(inplace=True)

# reset the index or NaN's will be produced when trying to add the class col to df_group
df.reset_index(inplace=True)

df_group['class'] = df['class']


# ### Fix errors in the age feature

# The images all look like adults. Therefore, it could be that 1 was added by mistake to the beginning of the age. 

# In[ ]:


# check for age errors in the train set
df_group[df_group['PatientAge'] > 100]

# 5 age errors found


# In[ ]:


# check for age errors in the test set
df_test[df_test['PatientAge'] > 100]

# no age errors found


# In[ ]:


# view the x-rays
# load a patient's file
patientId = df_train.loc[24537,'patientId']
path = '../input/stage_1_train_images/%s.dcm' % patientId

dcm_data = pydicom.read_file(path)

# convert the image to a numpy array
im = dcm_data.pixel_array

# view an image
pylab.imshow(im, cmap=pylab.cm.gist_gray)
pylab.axis('off')


# In[ ]:


# remove the 1 at the start of the age
# assumes 1 was added by mistake to these ages
age_errors = ['3b8b8777-a1f6-4384-872a-28b95f59bf0d', '73aeea88-fc48-4030-8564-0a9d7fdecac4',
             'a4e8e96d-93a6-4251-b617-91382e610fab', 'ec3697bd-184e-44ba-9688-ff8d5fbf9bbc',
             'f632328d-5819-4b29-b54f-adf4934bbee6']

df_group.loc[3175, 'PatientAge'] = 48
df_group.loc[9708, 'PatientAge'] = 51
df_group.loc[15273, 'PatientAge'] = 53
df_group.loc[23374, 'PatientAge'] = 50
df_group.loc[24537, 'PatientAge'] = 55


# In[ ]:


# check the changes
df_group.loc[23374,:]


# ### Add a col to df_group for noopacity_but_not_normal

# In[ ]:


df_group['noopacity_but_not_normal'] = df_group['class']

def noopacity_but_not_normal(x):
    if x == 'No Lung Opacity / Not Normal':
        return 1
    else:
        return 0

df_group['noopacity_but_not_normal'] = df_group['noopacity_but_not_normal'].apply(noopacity_but_not_normal)


# ### Map the 3 class labels to 0 and 1

# In[ ]:


# New Col: target
df_group['target'] = df_group['class'].map({'Lung Opacity':1, 'Normal':0, 'No Lung Opacity / Not Normal':0})


# ### If no pneumonia set num_bounding_boxes to 0

# In[ ]:


for i in range(0,len(df_group)):
    if df_group.loc[i,'target'] == 0:
        df_group.loc[i,'num_bounding_boxes'] = 0


# ### Create the train set bounding boxes

# In[ ]:


# Source: https://www.kaggle.com/peterchang77/exploratory-data-analysis

def parse_data(df):
    """
    Method to read a CSV file (Pandas dataframe) and parse the 
    data into the following nested dictionary:

      parsed = {
        
        'patientId-00': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        },
        'patientId-01': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        }, ...

      }

    """
    # --- Define lambda to extract coords in list [y, x, height, width]
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]

    parsed = {}
    for n, row in df.iterrows():
        # --- Initialize patient entry into parsed 
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom': '../input/stage_1_train_images/%s.dcm' % pid,
                'label': row['Target'],
                'boxes': []}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed


# run the function

df_boxes = pd.read_csv('../input/stage_1_train_labels.csv')

parsed = parse_data(df_boxes)


# In[ ]:


# check the bounding boxes for one patientId
box = parsed['00436515-870c-4b36-a041-de91049b9ab4']['boxes']

box


# In[ ]:


# extract the boxes for each patient

df_group['bounding_boxes'] = df_group['patientId']

def bounding_boxes(x):
    # get the dictionary value
    box = parsed[x]['boxes']
    
    return box

df_group['bounding_boxes'] = df_group['bounding_boxes'].apply(bounding_boxes)


# In[ ]:


# drop the PredictionString col from df_test
df_test = df_test.drop('PredictionString', axis=1)


# ### TAIL CHECKS

# Make sure that NaN's have not been introduced during pre processing.

# In[ ]:


df_group.isnull().sum()        


# In[ ]:


df_test.isnull().sum()


# ### Save the dataframes

# In[ ]:


# note: we are saving df_group with the name dftrain for easy reference later
pickle.dump(df_group,open('dftrain.pickle','wb'))
pickle.dump(df_test,open('dftest.pickle','wb'))


# In[ ]:


# check if the pickled files exist
get_ipython().system('ls')


# <hr>
# **Continued in Part 2...**

# In[ ]:




