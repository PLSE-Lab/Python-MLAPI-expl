#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import load_sample_image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# **Img charging**

# In[ ]:


my_img = load_sample_image('china.jpg')


# In[ ]:


plt.imshow(my_img)
plt.xticks([])
plt.yticks([])
plt.show()


# **Model and training**

# In[ ]:


my_img.reshape(-1, 3).shape


# In[ ]:


model = KMeans(n_clusters=64).fit(my_img.reshape(-1, 3))


# **Prediction**

# In[ ]:


predicted_img = model.cluster_centers_[model.labels_].reshape(my_img.shape)


# In[ ]:


plt.imshow(predicted_img.reshape(my_img.shape)/255.0)


# In[ ]:




