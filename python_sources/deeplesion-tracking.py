#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from glob import glob
import os, pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
# make the necessary conversion
read_hu = lambda x: imread(x).astype(np.float32)-32768
def create_boxes(in_row):
    box_list = []
    for (start_x, start_y, end_x, end_y) in in_row['bbox']:
        box_list += [Rectangle((start_x, start_y), 
                         np.abs(end_x-start_x),
                         np.abs(end_y-start_y)
                         )]
    return box_list
base_img_dir = '../input/minideeplesion/'


# In[ ]:


patient_df = pd.read_csv('../input/DL_info.csv')
patient_df['kaggle_path'] = patient_df.apply(lambda c_row: os.path.join(base_img_dir, 
                                                                        '{Patient_index:06d}_{Study_index:02d}_{Series_ID:02d}'.format(**c_row),
                                                                        '{Key_slice_index:03d}.png'.format(**c_row)), 1)

print('Loaded', patient_df.shape[0], 'cases')
patient_df['exists'] = patient_df['kaggle_path'].map(os.path.exists)
patient_df = patient_df[patient_df['exists']].drop('exists', 1)
# extact the bounding boxes
patient_df['bbox'] = patient_df['Bounding_boxes'].map(lambda x: np.reshape([float(y) for y in x.split(',')], (-1, 4)))
patient_df['norm_loc'] = patient_df['Normalized_lesion_location'].map(lambda x: np.reshape([float(y) for y in x.split(',')], (-1)))
patient_df['Spacing_mm_px_'] = patient_df['Spacing_mm_px_'].map(lambda x: np.reshape([float(y) for y in x.split(',')], (-1)))
patient_df['Lesion_diameters_Pixel_'] = patient_df['Lesion_diameters_Pixel_'].map(lambda x: np.reshape([float(y) for y in x.split(',')], (-1)))
patient_df['Radius_x'] = patient_df.apply(lambda x: x['Lesion_diameters_Pixel_'][0]*x['Spacing_mm_px_'][0], 1)
for i, ax in enumerate('xyz'):
    patient_df[f'{ax}_loc'] = patient_df['norm_loc'].map(lambda x: x[i])
print('Found', patient_df.shape[0], 'patients with images')
patient_df.sample(3)


# In[ ]:


sns.pairplot(hue='Patient_gender', data=patient_df[['Patient_age', 'Patient_gender', 'Key_slice_index', 'Radius_x']])


# # Group the patient scans together

# In[ ]:


freq_flyers_df = patient_df.groupby('Patient_index')[['Patient_age']].apply(
    lambda x: pd.Series({'counts': x.shape[0], 
                         'Age_Range': np.max(x['Patient_age'])-np.min(x['Patient_age'])})).reset_index().sort_values('Age_Range', ascending = False)
sns.pairplot(freq_flyers_df[['counts', 'Age_Range']])
freq_flyers_df.head(5)


# # Draw Image and  Bounding Box
# Here we use basic code to draw the image and the bounding box. We use the Lung window for the CT to make the views as consistent as possible

# In[ ]:


fig, m_axs = plt.subplots(5, 8, figsize = (25, 25))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
join_df = pd.merge(patient_df, freq_flyers_df.head(5))
for n_axs, (_, c_df) in zip(m_axs, join_df.groupby('Patient_index')):
    _, t_row = next(c_df.iterrows())
    n_axs[0].scatter(c_df['Patient_age'], c_df['Radius_x'])
    n_axs[0].set_xlabel('Age')
    n_axs[0].set_ylabel('Tumor Size (mm)')
    n_axs[0].set_title('ID:{Patient_index}-SEX:{Patient_gender}'.format(**t_row))
    n_axs[0].axis('on')
    for c_ax, (_, c_row) in zip(n_axs[1:], c_df.sort_values('Study_index').iterrows()):
        c_img = read_hu(c_row['kaggle_path'])
        c_ax.imshow(c_img, vmin = -1200, vmax = 600, cmap = 'gray')
        c_ax.add_collection(PatchCollection(create_boxes(c_row), alpha = 0.25, facecolor = 'red'))
        c_ax.set_title('Age:{Patient_age:1.0f}'.format(**c_row))
        c_ax.axis('off')
fig.savefig('overview.png', dpi = 300)


# # Overall Statistics
# Here we try and look at more patients and the relationship between the lesion size and the age of the patient

# In[ ]:


join_df = pd.merge(patient_df, freq_flyers_df.head(15))
ax = sns.lmplot(x='Patient_age', y='Radius_x', ci=50,
                sharex=False, sharey=False, x_jitter=0.5,
                 col='Patient_index', col_wrap=5,
                data = join_df)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'sns.factorplot')


# In[ ]:




