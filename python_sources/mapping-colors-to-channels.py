#!/usr/bin/env python
# coding: utf-8

# The notebook is trying to look at a basic analysis of the hyperspectral data and see if there is any information in the wavelengths

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import matplotlib.pyplot as plt # making plots
from skimage.util.montage import montage2d # showing a montage
import os
import seaborn as sns
from warnings import warn
base_path = '../input/MultiSpectralImages'


# In[ ]:


all_files = glob(os.path.join(base_path, '*'))
label_df = pd.read_csv(os.path.join(base_path, 'Labels.csv'))
label_df = label_df[['Label', 'FileName']] # other columns are NaNs
label_df['Number'] = label_df['Label'].map(lambda x: int(x.split(' ')[-1]))
label_df['Color'] = label_df['Label'].map(lambda x: x.split(' ')[0])
print('Number of numbers',label_df.shape)
label_df.sample(3)


# In[ ]:


label_df['Number'].plot.hist()


# Read in a large number of files and calculate the mean value for each channel

# In[ ]:


get_ipython().run_cell_magic('time', '', "out_results = []\nfor i, i_row in label_df.iterrows():\n    cur_image_path = os.path.join(base_path, i_row['FileName'])\n    if os.path.exists(cur_image_path):\n        cur_df = pd.read_csv(cur_image_path)\n        cur_df = cur_df[cur_df.columns[:-1]] # drop the last column\n        out_results += [dict(list(i_row.items())+\n             list(dict(cur_df.query('Channel0<255').apply(np.mean,axis = 0)).items()))]\n    else:\n        warn('File is missing {}'.format(cur_image_path), RuntimeWarning)\n\nsummary_df = pd.DataFrame(out_results)\nsummary_df.sample(3)")


# # Show Results
# Here we can show a pair plot showing how the different color names correspond with different channels

# In[ ]:


sns.pairplot(summary_df, hue = 'Color', vars = ['Channel{}'.format(i) for i in range(10)])


# Here we can show a pair plot showing how the different numbers correspond with different channels

# In[ ]:


sns.pairplot(summary_df, hue = 'Number', vars = ['Channel{}'.format(i) for i in range(10)])


# In[ ]:




