#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


train_df=pd.read_csv("../input/train.csv")


# In[ ]:


train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


from PIL import Image
import matplotlib.pyplot as plt

im = Image.open("../input/train/0000e88ab.jpg")


# In[ ]:


plt.imshow(im)


# In[ ]:


im.size


# In[ ]:


whales = train_df.groupby('Id')['Image'].nunique()
whales.sort_values(ascending=False)


# In[ ]:


whales.sort_values(ascending=False)[1:100].hist()


# In[ ]:


test_df=pd.read_csv("../input/sample_submission.csv")
test_df.head()


# In[ ]:


im_test = Image.open("../input/test/001a4d292.jpg")
plt.imshow(im_test)

