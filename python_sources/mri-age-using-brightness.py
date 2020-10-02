#!/usr/bin/env python
# coding: utf-8

# # Overview
# In this notebook we load the data and view different images to get a better idea about the challenge we are facing. This is always a very helpful first step. It is also important that you can see and try to make some of your own predictions about the data. If you cannot see differences between the groups it is going to be difficult for a biomarker to capture that (but not necessarily impossible)

# In[ ]:


import numpy as np # for manipulating 3d images
import pandas as pd # for reading and writing tables
import h5py # for reading the image files
import skimage # for image processing and visualizations
import sklearn # for machine learning and statistical models
import os # help us load files and deal with paths


# ### Plot Setup Code
# Here we setup the defaults to make the plots look a bit nicer for the notebook

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["figure.dpi"] = 125
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
sns.set_style("whitegrid", {'axes.grid': False})


# # Load the Training Data
# We start with the training data since we have labels for them and can look in more detail

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
train_df.head(5) # show the first 5 lines


# # Read Image

# In[ ]:


def read_scan(in_filename, folder='train'):
    full_scan_path = os.path.join('..', 'input',folder, in_filename)
    # load the image using hdf5
    with h5py.File(full_scan_path, 'r') as h:
        return h['image'][:][:, :, :, 0] # we read the data from the file


# # Load a Scan
# - the data on kaggle are located in a parent folder called input. 
# - Since the files have been organized into train and test we use the train folder

# In[ ]:


sample_scan = train_df.iloc[0] # just take the first row
print(sample_scan)
# turn the h5_path into the full path
image_data = read_scan(sample_scan['h5_path'])
print('Image Shape:', image_data.shape)


# # Calculate Brightness
# We can calculate the average brightness by just taking the average of all the pixels

# In[ ]:


def calc_brightness(in_image_data):
    return np.mean(in_image_data)
print(calc_brightness(image_data))


# ### Run over all scans
# We use the `.map` function from pandas to calculate the brightness for all the scans

# In[ ]:


get_ipython().run_cell_magic('time', '', "train_df['brightness'] = train_df['h5_path'].map(lambda c_filename: calc_brightness(read_scan(c_filename)))")


# # Visualize the Values
# Using a simple linear model we see that brightness goes down with age

# In[ ]:


sns.lmplot('age_years', 'brightness', data=train_df)


# We see that the linear model was correct and that we lose 9 brightness points per year

# In[ ]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(train_df['brightness'].values.reshape((-1, 1)), train_df['age_years'].values)
lin_reg.coef_


# # Apply to Test Data
# We can now load the test data and make a prediction

# In[ ]:


test_df = pd.read_csv('../input/test_sample_submission.csv')[['scan_id']]
test_df['h5_path'] = test_df['scan_id'].map(lambda s_id: 'mri_{:08d}.h5'.format(s_id))
test_df['brightness'] = test_df['h5_path'].map(lambda c_filename: calc_brightness(read_scan(c_filename, folder='test')))
test_df.head(5)


# We reuse the trained model to generate predictions of the age based on the brightness

# In[ ]:


test_df['age_years'] = lin_reg.predict(test_df['brightness'].values.reshape((-1, 1)))
sns.lmplot('age_years', 'brightness', data=test_df)


# In[ ]:


# save the output
test_df[['scan_id', 'age_years']].to_csv('linear_brightness_prediction.csv', index=False)


# In[ ]:




