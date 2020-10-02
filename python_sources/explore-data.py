#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from PIL import Image
path = '/kaggle/input/yandextoloka-water-meters-dataset/WaterMeters/'
df = pd.read_csv(path + 'data.csv')
display(df.head())


# In[ ]:


image = Image.open(path + '/images/id_53_value_595_825.jpg')
image


# In[ ]:


mask = Image.open(path + '/masks/id_53_value_595_825.jpg')
mask


# In[ ]:


collage = Image.open(path + '/collage/id_53_value_595_825.jpg')
collage


# In[ ]:




