#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from skimage import feature
from sklearn import preprocessing
import seaborn as sns

from subprocess import check_output
print(check_output(["ls",'-lsa', "../input/"]).decode("utf8"))


# In[ ]:


data = np.load('../input/img_array_train_6k_22.npy')


# In[ ]:


data.shape


# In[ ]:


image_1 = data[0,:,:]
plt.imshow(image_1)


# In[ ]:


edge_img = feature.canny(data[1,:,:]).astype(int)


# In[ ]:


plt.imshow(edge_img)


# In[ ]:


df = pd.read_csv('../input/adni_demographic_master_kaggle.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:



sns.kdeplot(image_1.flatten())


# In[ ]:


bin_im1 = preprocessing.binarize(image_1,600)
plt.imshow(bin_im1)


# In[ ]:




