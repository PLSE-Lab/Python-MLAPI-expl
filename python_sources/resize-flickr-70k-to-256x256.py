#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
from tqdm.notebook import tqdm


# In[ ]:


get_ipython().system('mkdir ../resized')


# In[ ]:


paths = []

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if 'png' in filename:
            path = os.path.join(dirname, filename)
            paths.append([path, filename.strip('.png')])


# In[ ]:


for path, filename in tqdm(paths):
    im = cv2.imread(path)
    resized = cv2.resize(im, (256, 256))
    cv2.imwrite(f'/kaggle/resized/{filename}.jpg', resized)


# In[ ]:


get_ipython().system('tar -zcf resized.tar.gz ../resized/')
get_ipython().system('ls')

