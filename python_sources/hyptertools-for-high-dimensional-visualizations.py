#!/usr/bin/env python
# coding: utf-8

# The notebook takes the visualizations shown in the 'No Free Hunch' blog and applies it to these data (http://blog.kaggle.com/2017/04/10/exploring-the-structure-of-high-dimensional-data-with-hypertools-in-kaggle-kernels/)

# In[ ]:


import os
from skimage.io import imread
from glob import glob
import pandas as pd
import hypertools as hyp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Load in and organize the data as a dataframe

# In[ ]:


time_df = pd.read_csv('../input/data141110.csv')
time_df['path'] = time_df['Image.No.'].map(lambda x: "141110A3.%04d" % (x))
with np.load('../input/flo_image_1.npz') as im_data:
    image_dict = dict(zip(im_data['image_ids'], im_data['image_stack']))
time_df['loaded'] = time_df['path'].map(lambda x: x in image_dict)
valid_time_df = time_df.query('loaded')    
ordered_im_stack = np.stack([image_dict[c_path] for c_path in valid_time_df.sort_values('Time.hrs.')['path'].values],0)
print('Loaded',len(image_dict), 'images')


# In[ ]:


# unwrap the pixel positions so we have just the time observations
time_vec = ordered_im_stack.reshape(ordered_im_stack.shape[0], -1).T
print(time_vec.shape)


# # Preview
# The data has now been unwrapped by pixel and we have 262144 points in an 800 dimensional space (each dimension covering a time-point in the measurement). Since this is very difficult to visualize we can use the hyperplot tools to help show all of these points. We downsample the data massively to make the visualizations quicker

# In[ ]:


time_vec=time_vec[::10,:]


# In[ ]:


hyp.plot(time_vec, 'o')


# In[ ]:


hyp.plot(time_vec, 'o', n_clusters=10)


# In[ ]:


hyp.plot(time_vec, 'o', n_clusters=30)


# In[ ]:


hyp.plot(time_vec, normalize='across')


# # Reversed Dimensions
# Rather than looking at an 800-dimensional object where each dimension is a point in time we can look at the 800 time steps as points and each dimension is a position in the image to see how they fit together

# In[ ]:





# In[ ]:


hyp.plot(time_vec.T, 'o', n_clusters=10)


# In[ ]:


hyp.plot(time_vec.T, normalize='across')


# Now, for the grand finale. In addition to creating static plots, HyperTools can also create animated plots, which can sometimes reveal additional patterns in the data. To create an animated plot, simply pass animate=True to hyp.plot when visualizing timeseries data. If you also pass chemtrails=True, a low-opacity trace of the data will remain in the plot:

# In[ ]:


hyp.plot(time_vec, normalize='across', animate=True, chemtrails=True)

