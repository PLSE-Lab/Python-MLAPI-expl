#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
import re
import os
from pathlib import Path

plt.style.use('grayscale')


# ## Directory Structure & Files

# In[ ]:


ROOT_DIR = Path('../input/rsna-intracranial-hemorrhage-detection')


# In[ ]:


get_ipython().system(' ls {ROOT_DIR}')


# In[ ]:


TRAIN_DIR = ROOT_DIR/'stage_1_train_images'
TEST_DIR = ROOT_DIR/'stage_1_test_images'


# In[ ]:


get_ipython().system(' ls {TRAIN_DIR} | wc -l')


# In[ ]:


get_ipython().system(' ls {TRAIN_DIR} | head -n 5')


# In[ ]:


train_df = pd.read_csv(ROOT_DIR/'stage_1_train.csv')
print(train_df.shape)
train_df.head(10)


# ## Label Exploration
# 
# Start by pivoting the DataFrame to explore the label distribution over slices

# In[ ]:


train_df[['ID', 'Subtype']] = train_df['ID'].str.rsplit(pat='_', n=1, expand=True)
print(train_df.shape)
train_df.head()


# In[ ]:


def fix_id(img_id, img_dir=TRAIN_DIR):
    if not re.match(r'ID_[a-z0-9]+', img_id):
        sop = re.search(r'[a-z0-9]+', img_id)
        if sop:
            img_id_new = f'ID_{sop[0]}'
            return img_id_new
        else:
            print(img_id)
    return img_id

# test
assert(fix_id('ID_63eb1e259') == fix_id('ID63eb1e259'))
test = 'ID_dbdedfada'
assert(fix_id(test) == 'ID_dbdedfada')


# In[ ]:


train_df['ID'] = train_df['ID'].apply(fix_id)


# In[ ]:


# this method also handles duplicates gracefully
train_new = train_df.pivot_table(index='ID', columns='Subtype').reset_index()
print(train_new.shape)
train_new.head()


# In[ ]:


subtype_ct = train_new['Label'].sum(axis=0)
print(subtype_ct)


# In[ ]:


sns.barplot(x=subtype_ct.values, y=subtype_ct.index);


# As a Neuroradiologist, this distribution looks pretty true to daily practice.

# In[ ]:


def id_to_filepath(img_id, img_dir=TRAIN_DIR):
    filepath = f'{img_dir}/{img_id}.dcm' # pydicom doesn't play nice with Path objects
    if os.path.exists(filepath):
        return filepath
    else:
        return 'DNE'


# In[ ]:


img_id = train_new['ID'][0]
img_filepath = id_to_filepath(img_id)
print(img_filepath)


# In[ ]:


train_new['filepath'] = train_new['ID'].apply(id_to_filepath)
train_new.head()


# ## DICOM Tags
# - `Study Instance UID`, `Series Instance UID` and `Patient ID` will be helpful in organizing our data later.
# - `Rows` and `Columns` give us the image resolution (512 x 512 in this case)
# - `Window Center` and `Window Width` tell us the window settings applied to the image at acquisition
# - `Rescale Intercept` and `Rescale Slope` tell us how to rescale the pixel values to match the standard Hounsfield Unit (HU) scale

# In[ ]:


dcm_data = pydicom.dcmread(img_filepath)
print(dcm_data)


# ## Unique patients, studies & series

# In[ ]:


def get_patient_data(filepath):
    if filepath != 'DNE':
        dcm_data = pydicom.dcmread(filepath, stop_before_pixels=True)
        return dcm_data.PatientID, dcm_data.StudyInstanceUID, dcm_data.SeriesInstanceUID


# In[ ]:


patient, study, series = get_patient_data(img_filepath)
print(patient, study, series)


# In[ ]:


# quick test to make sure our df.apply syntax is working, since the next cell takes a long time to run
test = train_new[:5].copy()
test['PatientID'], test['StudyID'], test['SeriesID'] = zip(*test['filepath'].map(get_patient_data))
test.head()


# __Warning__: This next cell takes a very long time to run (>> 10 min).

# In[ ]:


train_new['PatientID'], train_new['StudyID'], train_new['SeriesID'] = zip(*train_new['filepath'].map(get_patient_data))


# In[ ]:


print(train_new.shape[0])
print(len(train_new['PatientID'].unique()))
print(len(train_new['StudyID'].unique()))
print(len(train_new['SeriesID'].unique()))


# So, we have ~ 17,000 unique patients in our dataset with > 670,000 images.

# ## Windowing the image using DICOM metadata

# In[ ]:


type(dcm_data.WindowWidth)


# In[ ]:


def window_img(dcm, width=None, level=None):
    pixels = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    if not width:
        width = dcm.WindowWidth
        if type(width) != pydicom.valuerep.DSfloat:
            width = width[0]
    if not level:
        level = dcm.WindowCenter
        if type(level) != pydicom.valuerep.DSfloat:
            level = level[0]
    lower = level - (width / 2)
    upper = level + (width / 2)
    return np.clip(pixels, lower, upper)

def load_one_image(idx, df=train_new, width=None, level=None):
    assert('filepath' in df.columns)
    dcm_data = pydicom.dcmread(df['filepath'][idx])
    pixels = window_img(dcm_data, width, level)
    return pixels


# In[ ]:


# standard brain window
pixels = load_one_image(0)
plt.imshow(pixels);


# In[ ]:


# subdural window
pixels_new = load_one_image(0, width=200, level=80)
plt.imshow(pixels_new);


# ## Examples of different hemorrhage subtypes

# In[ ]:


def show_examples(subtype='epidural', df=train_new):
    df_new = df.set_index('ID')
    filt = df_new['Label'][subtype] == 1
    df_new = df_new[filt]
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for i in range(16):
        idx = df_new.index[i]
        pixels = load_one_image(idx, df_new)
        a = i // 4
        b = i % 4
        axes[a, b].imshow(pixels)


# In[ ]:


show_examples('epidural')


# In[ ]:


show_examples('subdural')


# In[ ]:


show_examples('intraventricular')


# In[ ]:


show_examples('intraparenchymal')


# In[ ]:


show_examples('subarachnoid')


# In[ ]:




