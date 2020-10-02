#!/usr/bin/env python
# coding: utf-8

# # Install dicom-csv, deep-pipe.

# In[ ]:


# Install dicom-csv, utils for gathering, aggregation and handling metadata from DICOM files.
# This allows us to join DICOMs into volumes in a correct order without relying on file names.

get_ipython().system('pip install git+https://github.com/neuro-ml/dicom-csv.git')


# In[ ]:


# install deep-pipe, collection of tools for dl experiments,
# here just for the vizualisation

get_ipython().system('pip install deep-pipe')


# # Collect metadata from all DICOMs.

# In[ ]:


from dicom_csv import join_tree, aggregate_images, load_series


# In[ ]:


# Single row of df corresponds to a single DICOM file (unique SOPInstanceUID in DICOM's slang)
# First run takes ~3-4 mins, since it actually opens all available DICOMs inside a directory,
# so I recommend you to store the resulting DataFrame

df = join_tree('/kaggle/input/osic-pulmonary-fibrosis-progression/train', verbose=True, relative=False, force=True)
df.head(3)


# In[ ]:


df['NoError'].value_counts()


# In[ ]:


# Single row of df_images corresponds to a single volume (unique SeriesInstanceUID in DICOM's slang)


df_images = aggregate_images(df.query('NoError == True'))
df_images.head(3)


# # Load images and visualize
# 
# ## RescaleSlope, RescaleIntercept
# 
# Recall that there are two important parameters `RescaleSlope` and `RescaleIntercept`, which you would
# completely loose if you simply did pydicom.dcmread('image.dcm'), see for example https://stackoverflow.com/questions/10193971/rescale-slope-and-rescale-intercept .
# 
# `load_series` takes them into account.

# In[ ]:


df_images.RescaleSlope.value_counts().sort_index()


# In[ ]:


df_images.RescaleIntercept.value_counts().sort_index()


# In[ ]:


image = load_series(df_images.iloc[0], orientation=False)


# In[ ]:


from dpipe.im.visualize import slice3d


# In[ ]:


# There is a slide bar you could move to go over different slices (run notebook to see it).

slice3d(image)


# # DICOMs metadata
# 
# Pixel space (mm) distribution, SliceCount's distribution

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_images.columns


# In[ ]:





# In[ ]:


df_images['PixelArrayShape'].value_counts()


# In[ ]:


df_images['PatientSex'].value_counts()


# In[ ]:


df_images['PixelSpacing0'].value_counts().sort_index()


# In[ ]:


plt.hist(df_images['PixelSpacing0'], bins=15)
plt.xlabel('Pixel spacing, mm.')
plt.ylabel('Number of images.');


# In[ ]:


df_images['SlicesCount'].value_counts().sort_index()


# In[ ]:


plt.hist(df_images['SlicesCount'], bins=15)
plt.xlabel('Number of slices in a single 3D image.')
plt.ylabel('Number of images.');


# In[ ]:




