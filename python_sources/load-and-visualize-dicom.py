#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('conda install -c conda-forge pydicom gdcm -y')


# In[ ]:


import numpy as np # linear algebra
import os
import pydicom
import matplotlib.pyplot as plt


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Load one of the files

# In[ ]:


filepath = "/kaggle/input/covid19-ct-scans/Case_002/coronacases_002_171.dcm"
dcmfile = pydicom.dcmread(filepath)


# In[ ]:


dcmfile


# In[ ]:


dcm_numpy = dcmfile.pixel_array


# In[ ]:


dcm_numpy.shape


# ### Visualize

# In[ ]:


plt.imshow(dcm_numpy, cmap=plt.cm.bone)


# In[ ]:




