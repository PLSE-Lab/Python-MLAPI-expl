#!/usr/bin/env python
# coding: utf-8

# Current notebook is empiric improvment current [top decision](https://www.kaggle.com/naivelamb/alaska2-srnet-baseline-inference) by [Xuan Cao](https://www.kaggle.com/naivelamb).

# In[ ]:


import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.io import imread
from tqdm.notebook import tqdm


# In[ ]:


PATH = '../input/alaska2-image-steganalysis'
SRNet = '../input/alaska2-srnet-baseline-inference'
sub = pd.read_csv(os.path.join(SRNet, 'submission.csv'))


# ## Compression Rate
# May we have an RGB image with height H and width W. The theoretical number bytes B for this image is $$B = H * W * 3$$
# Therefor if the real size of image is S, then compression rate C is $$ C = (B - S) / B$$

# In[ ]:


class JPEGImageCompressionRateDeterminer:
    def __call__(self, image_path):
        image = imread(image_path)
        w, h, c = image.shape
        
        # theoretical image size
        b = w*h*3
        
        # real image file size in bytes
        s = os.stat(image_path).st_size
        return (b - s) / b 


# In[ ]:


compression_rate_determiner = JPEGImageCompressionRateDeterminer()

compressions = {}

dir_path = os.path.join(PATH, 'Test')
for impath in tqdm(sub.Id.values):
    c = compression_rate_determiner(os.path.join(dir_path, impath))
    compressions[impath] = c
    if c > 0.95:
        sub.loc[sub.Id == impath, 'Label'] = 1. - 1e-3


# ## Empiric improvement
# Empiric improvement based on punct 4 in data description.
# > "4.The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75."
# 
# So images with compression rate more then 0.95 are prohibited.

# In[ ]:


plt.figure(figsize=(10,10))

plt.axvline(0.75, color='orange')
plt.axvline(0.90, color='orange')
plt.axvline(0.95, color='orange')
plt.axvspan(0., 0.95, color='green', alpha=0.25)
plt.axvspan(0.95, 1.0, color='red', alpha=0.25)
sns.distplot(list(compressions.values()));


# ## Submission

# In[ ]:


sub.to_csv('submission.csv', index=None)


# In[ ]:


sub.head()


# ## Reference
# * [ALASKA2: SRNet baseline inference](https://www.kaggle.com/naivelamb/alaska2-srnet-baseline-inference)
