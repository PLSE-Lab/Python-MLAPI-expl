#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('tar -xvf /kaggle/input/diabeticretinopathydetectionpretraineddata/DiabeticRetinopathyDetection.tar.xz ')


# In[ ]:


import os
os.chdir('/kaggle/working/DiabeticRetinopathyDetection')


# In[ ]:


get_ipython().system('bash classify.sh sample/Level-4.jpeg')


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread("sample/Level-4.jpeg")
plt.imshow(image)
plt.show()

