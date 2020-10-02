#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import PIL.Image


# Parquet files can be read directly into pandas

# In[ ]:


df = pd.read_parquet('/kaggle/input/bengaliai-ocr-2019/train_image_data_1.parquet')


# There's one row per image, plus an image_id column

# In[ ]:


df.shape


# In[ ]:


flattened_image = df.iloc[123].drop('image_id').values.astype(np.uint8)


# In[ ]:


unpacked_image = PIL.Image.fromarray(flattened_image.reshape(137, 236))


# In[ ]:


unpacked_image


# 
