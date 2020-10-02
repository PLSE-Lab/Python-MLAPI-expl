#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# I found out that image size is related to the rate (probability) of `new_whale`. Check below example if you are interested.
# 
# # Example
# ## Import Packages
# 
# cv2 is faster, but PIL is easy to distinguish gray scale images and RGB color images.
# 

# In[ ]:


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image


# ## Make function to get image shapes
# make the list of image shapes. Gray scale images are two dimensional array, Color images are three dimensional array.

# In[ ]:


def get_size_list(targets, dir_target):

    result = list()

    for target in tqdm(targets):

        img = np.array(Image.open(os.path.join(dir_target, target)))
        result.append(str(img.shape))

    return result


# ## Get image shape for each train image
# load `train.csv` and add a column which represents size of images.

# In[ ]:


data = pd.read_csv('../input/train.csv')
data['size_info'] = get_size_list(data.Image.tolist(), dir_target='../input/train')
data.to_csv('./size_train.csv', index=False)


# ## Group by shape and summerize
# Summerizing number of samples and rate of `new_whale, we can see unnatual bias.
# (700, 1050, 3), (600, 1050, 3) includes 23-25% of new_whales. On the other hand, (600, 1050) includes 89% of new_whale.

# In[ ]:


counts = data.size_info.value_counts()

agg = data.groupby('size_info').Id.agg({'number_sample': len,
                                        'rate_new_whale': lambda g: np.mean(g == 'new_whale')})

agg = agg.sort_values('number_sample', ascending=False)
agg.to_csv('result.csv')
print(agg.head(20))


# # Conclusion
# It seems that image size is related to the rate of `new_whale`. Does this feature help us ? Please your comment.
