#!/usr/bin/env python
# coding: utf-8

# The notebook shows a simple overview of the data and some of the basic analysis possible

# In[ ]:


import numpy as np # matrix tools
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for basic plots
import seaborn as sns # for nicer plots


# In[ ]:


from glob import glob
glob('../input/*')


# In[ ]:


from glob import glob
glob('../input/dicom_dir/*')


# In[ ]:


overview_df = pd.read_csv('../input/overview.csv')
overview_df.columns = ['idx']+list(overview_df.columns[1:])
overview_df['Contrast'] = overview_df['Contrast'].map(lambda x: 'Contrast' if x else 'No Contrast')
overview_df.sample(3)


# # Show a histogram of the age distribution

# In[ ]:


overview_df['Age'].hist()


# In[ ]:


with np.load('../input/full_archive.npz') as im_data:
    # make a dictionary of the data vs idx
    full_image_dict = dict(zip(im_data['idx'], im_data['image']))


# Show a single slice

# In[ ]:


plt.matshow(full_image_dict[0])


# # Feature Calculation
# So now we calculate a simple feature like the mean intensity in the image and show how it relates to age and contrast

# In[ ]:


overview_df['MeanHU'] = overview_df['idx'].map(lambda x: np.mean(full_image_dict.get(x, np.zeros((512,512)))))
overview_df['StdHU'] = overview_df['idx'].map(lambda x: np.std(full_image_dict.get(x, np.zeros((512,512)))))
overview_df.sample(3)


# Here we show a pair plot of all variables to see if there are any interesting overlaps

# In[ ]:


sns.set()
_ = sns.pairplot(overview_df[['Age', 'Contrast', 'MeanHU', 'StdHU']], hue="Contrast")

