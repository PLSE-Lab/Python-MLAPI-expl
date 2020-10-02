#!/usr/bin/env python
# coding: utf-8

# # Overview
# Here we show the files in dataset to make loading and figuring out paths easier. The dicoms do not consist of all the dicom images, they are just the ones where a tumor has been marked by one of the annotators (the full dataset is 25GB). The image paths are formed as `../input/annotated_dicoms/`_patientID_`/`_seriesUID_`/`_sliceIndex_` to make them easier to load. The original format had lots of confusing folders and subfolders and required parsing dicom headers to match the files.

# In[1]:


import os
data_dir = '../input'
get_ipython().system('ls -R {data_dir} | head')


# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pydicom import read_file
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
base_dcm_dir = '../input/annotated_dicoms/'


# # Sample the Data
# Here we sample some of the data to see how it looks and try and visualize the lesions well. For this we pick some lung regions segmented multiple times by different annotators

# In[3]:


ann_df = pd.read_csv(os.path.join(data_dir, 'CrowdsCureCancer2017Annotations.csv'))
test_rows = ann_df.sort_values(['anatomy','seriesUID']).head(501).tail(6)
test_rows.T


# # Show the ROIs
# Here we show the ROI on each slice, using a slightly hacky reverse-engineered code that seems to appear reasonable (someone should check that this is really how the data are intended to be read in)
# 
# ## as bounding boxes
# 

# In[4]:


fig, m_axs = plt.subplots(2,3, figsize = (20, 12))
for (_, c_row), c_ax in zip(test_rows.iterrows(), m_axs.flatten()):
    t_dicom = read_file(os.path.join(base_dcm_dir, 
                                     c_row['patientID'], # format for dicom files
                                     c_row['seriesUID'], 
                                     str(c_row['sliceIndex'])))
    c_ax.imshow(t_dicom.pixel_array, cmap = 'bone')
    rect = Rectangle((min(c_row['start_x'], c_row['end_x']), 
                      min(c_row['start_y'], c_row['end_y'])), 
                     np.abs(c_row['end_x']-c_row['start_x']),
                     np.abs(c_row['end_y']-c_row['start_y'])
                     )
    c_ax.add_collection(PatchCollection([rect], alpha = 0.25, facecolor = 'red'))
    c_ax.set_title('{patientID} {anatomy}\n{annotator}\n{radiologist_status}'.format(**c_row))
    c_ax.axis('off')


# As circles

# In[5]:


test_rows = ann_df.sort_values(['anatomy','seriesUID']).head(11).tail(6)
fig, m_axs = plt.subplots(2,3, figsize = (20, 12))
for (_, c_row), c_ax in zip(test_rows.iterrows(), m_axs.flatten()):
    t_dicom = read_file(os.path.join(base_dcm_dir, 
                                     c_row['patientID'], # format for dicom files
                                     c_row['seriesUID'], 
                                     str(c_row['sliceIndex'])))
    c_ax.imshow(t_dicom.pixel_array, cmap = 'bone')
    circle = Circle((c_row['start_x'], c_row['start_y']), 
                     radius = c_row['length']/2
                     )
    c_ax.add_collection(PatchCollection([circle], alpha = 0.25, facecolor = 'red'))
    c_ax.set_title('{patientID} {anatomy}\n{annotator}\n{radiologist_status}'.format(**c_row))
    c_ax.axis('off')


# # Statistics
# Now we focus on the statistics to get a better feel for what is actually in the entire dataset

# In[6]:


from IPython.display import display, Markdown
# view annotators
dmark = lambda x: display(Markdown(x))
sum_view = lambda x, rows = 10: ann_df.groupby(x).count()[['order']].reset_index().sort_values('order', ascending = False).head(rows)
dmark('# Annotators')
display(sum_view(['annotator', 'radiologist_status']))


# In[7]:


dmark('# Anatomy')
display(sum_view('anatomy'))
dmark('# Patient')
display(sum_view('patientID'))


# In[8]:


sns.violinplot(x='anatomy', y = 'length', data = ann_df)


# In[9]:


sns.violinplot(x='anatomy', y = 'sliceIndex', data = ann_df)


# In[10]:


top_annotations = ann_df.groupby('anatomy').apply(
    lambda x: x.groupby('seriesUID').count()[['order']].reset_index().sort_values('order', ascending = False).head(5)).reset_index(drop = True)


# In[11]:


#fig, ax1 = plt.subplots(1,1, figsize = (25, 8))
g = sns.factorplot(x = 'patientID', 
              y = 'length', 
              hue = 'radiologist_status',
               col = 'anatomy',
               kind = 'swarm',
                   sharex = False,
                   sharey = False,
              data = ann_df[ann_df['seriesUID'].isin(top_annotations['seriesUID'])])
g.set_xticklabels(rotation=90)


# # Comparing Radiologists to Not Radiologists
# See if there are any clear trends when comparing them

# In[12]:


length_summary_df = ann_df.pivot_table(values = 'length', 
                   columns='radiologist_status', 
                   index = ['anatomy', 'seriesUID'],
                  aggfunc='mean').reset_index().dropna()
display(length_summary_df.groupby('anatomy').agg('mean').reset_index())
length_summary_df['mean'] = length_summary_df.apply(lambda x: 0.5*x['not_radiologist']+0.5*x['radiologist'], 1)
length_summary_df['not_radiologist'] = length_summary_df['not_radiologist'] / length_summary_df['mean']
length_summary_df['radiologist'] = length_summary_df['radiologist'] / length_summary_df['mean']
length_summary_df.sample(3)


# In[13]:


sns.factorplot(
    x = 'anatomy',
    y = 'value',
    hue = 'radiologist_status',
    kind = 'swarm',
    data = pd.melt(length_summary_df, 
        id_vars = ['anatomy'], 
        value_vars = ['radiologist', 'not_radiologist']))


# In[14]:


sns.factorplot(
    x = 'anatomy',
    y = 'value',
    hue = 'radiologist_status',
    kind = 'violin',
    data = pd.melt(length_summary_df, 
        id_vars = ['anatomy'], 
        value_vars = ['radiologist', 'not_radiologist']))


# In[ ]:




