#!/usr/bin/env python
# coding: utf-8

# # Some test plots

# ## Load the data
# 
# For this notebook, we'll only need the metadata.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.interpolate as itp


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# Set seasborn style

# In[ ]:


sns.set()


# In[ ]:


meta_data = pd.read_csv('../input/training_set_metadata.csv')
data = pd.read_csv('../input/training_set.csv')

target = meta_data['target']
object_id = meta_data['object_id']


# In[ ]:


plt.scatter(meta_data['gal_l'], meta_data['gal_b'], s=1)
plt.xlabel('l')
plt.ylabel('b')


# In[ ]:


colours = np.array(('purple', 'blue', 'green', 'orange', 'red', 'brown'))
for i in range(10):
    plt.figure(i, figsize=(20, 15))
    mask_u = (data['object_id']==object_id[i]) & (data['passband']==0)
    mask_g = (data['object_id']==object_id[i]) & (data['passband']==1)
    mask_r = (data['object_id']==object_id[i]) & (data['passband']==2)
    mask_i = (data['object_id']==object_id[i]) & (data['passband']==3)
    mask_z = (data['object_id']==object_id[i]) & (data['passband']==4)
    mask_y = (data['object_id']==object_id[i]) & (data['passband']==5)
    plt.scatter(data['mjd'][mask_u], data['flux'][mask_u], s=1, color=colours[0])
    plt.scatter(data['mjd'][mask_g], data['flux'][mask_g], s=1, color=colours[1])
    plt.scatter(data['mjd'][mask_r], data['flux'][mask_r], s=1, color=colours[2])
    plt.scatter(data['mjd'][mask_i], data['flux'][mask_i], s=1, color=colours[3])
    plt.scatter(data['mjd'][mask_z], data['flux'][mask_z], s=1, color=colours[4])
    plt.scatter(data['mjd'][mask_y], data['flux'][mask_y], s=1, color=colours[5])
    plt.plot(data['mjd'][mask_u],itp.UnivariateSpline(data['mjd'][mask_u], data['flux'][mask_u], s=1000.*np.mean(data['flux_err'][mask_u]))(data['mjd'][mask_u]),lw=1, color=colours[0])
    plt.plot(data['mjd'][mask_g],itp.UnivariateSpline(data['mjd'][mask_g], data['flux'][mask_g], s=1000.*np.mean(data['flux_err'][mask_g]))(data['mjd'][mask_g]),lw=1, color=colours[1])
    plt.plot(data['mjd'][mask_r],itp.UnivariateSpline(data['mjd'][mask_r], data['flux'][mask_r], s=1000.*np.mean(data['flux_err'][mask_r]))(data['mjd'][mask_r]),lw=1, color=colours[2])
    plt.plot(data['mjd'][mask_i],itp.UnivariateSpline(data['mjd'][mask_i], data['flux'][mask_i], s=1000.*np.mean(data['flux_err'][mask_i]))(data['mjd'][mask_i]),lw=1, color=colours[3])
    plt.plot(data['mjd'][mask_z],itp.UnivariateSpline(data['mjd'][mask_z], data['flux'][mask_z], s=1000.*np.mean(data['flux_err'][mask_z]))(data['mjd'][mask_z]),lw=1, color=colours[4])
    plt.plot(data['mjd'][mask_y],itp.UnivariateSpline(data['mjd'][mask_y], data['flux'][mask_y], s=1000.*np.mean(data['flux_err'][mask_y]))(data['mjd'][mask_y]),lw=1, color=colours[5])
    plt.title("Class = " + str(target[i]))


# In[ ]:


colours = np.array(('purple', 'blue', 'green', 'orange', 'red', 'brown'))

for i, class_id in enumerate(set(target)):
    plt.figure(i, figsize=(20, 15))
    mask_u = (class_id == target) & (data['passband']==0)
    mask_g = (class_id == target) & (data['passband']==1)
    mask_r = (class_id == target) & (data['passband']==2)
    mask_i = (class_id == target) & (data['passband']==3)
    mask_z = (class_id == target) & (data['passband']==4)
    mask_y = (class_id == target) & (data['passband']==5)
    plt.scatter(data['mjd'][mask_u], data['flux'][mask_u], s=1, color=colours[0])
    plt.scatter(data['mjd'][mask_g], data['flux'][mask_g], s=1, color=colours[1])
    plt.scatter(data['mjd'][mask_r], data['flux'][mask_r], s=1, color=colours[2])
    plt.scatter(data['mjd'][mask_i], data['flux'][mask_i], s=1, color=colours[3])
    plt.scatter(data['mjd'][mask_z], data['flux'][mask_z], s=1, color=colours[4])
    plt.scatter(data['mjd'][mask_y], data['flux'][mask_y], s=1, color=colours[5])
    #plt.plot(data['mjd'][mask_u],itp.UnivariateSpline(data['mjd'][mask_u], data['flux'][mask_u], s=1000.*np.mean(data['flux_err'][mask_u]))(data['mjd'][mask_u]),lw=1, color=colours[0])
    #plt.plot(data['mjd'][mask_g],itp.UnivariateSpline(data['mjd'][mask_g], data['flux'][mask_g], s=1000.*np.mean(data['flux_err'][mask_g]))(data['mjd'][mask_g]),lw=1, color=colours[1])
    #plt.plot(data['mjd'][mask_r],itp.UnivariateSpline(data['mjd'][mask_r], data['flux'][mask_r], s=1000.*np.mean(data['flux_err'][mask_r]))(data['mjd'][mask_r]),lw=1, color=colours[2])
    #plt.plot(data['mjd'][mask_i],itp.UnivariateSpline(data['mjd'][mask_i], data['flux'][mask_i], s=1000.*np.mean(data['flux_err'][mask_i]))(data['mjd'][mask_i]),lw=1, color=colours[3])
    #plt.plot(data['mjd'][mask_z],itp.UnivariateSpline(data['mjd'][mask_z], data['flux'][mask_z], s=1000.*np.mean(data['flux_err'][mask_z]))(data['mjd'][mask_z]),lw=1, color=colours[4])
    #plt.plot(data['mjd'][mask_y],itp.UnivariateSpline(data['mjd'][mask_y], data['flux'][mask_y], s=1000.*np.mean(data['flux_err'][mask_y]))(data['mjd'][mask_y]),lw=1, color=colours[5])
    plt.title("Class = " + str(class_id))

