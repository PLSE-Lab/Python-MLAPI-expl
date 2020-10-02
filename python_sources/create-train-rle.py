#!/usr/bin/env python
# coding: utf-8

# # Create train-rle for the 2nd stage

# In[ ]:


import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


df = pd.read_csv('../input/siim-acr-pneumothorax-segmentation/stage_2_train.csv')
len(df)


# In[ ]:


ndf = df.drop_duplicates(['ImageId'])
ndf['EncodedPixels'] = ' ' + ndf.loc[:,['EncodedPixels']]
ndf = ndf.rename(columns={"EncodedPixels":" EncodedPixels"})


# In[ ]:


len(ndf)


# In[ ]:


ndf.to_csv('train-rle.csv', index=False)


# In[ ]:




