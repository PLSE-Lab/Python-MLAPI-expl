#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.data import imread
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))


# In[ ]:


train_images = os.listdir('../input/train_images')
print(len(train_images))

test_images = os.listdir('../input/test_images')
print(len(test_images))


# In[ ]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')
submission.head()


# In[ ]:


labels = sorted(list(set(train['Image_Label'].apply(lambda x: x.split('_')[1]))))
print(labels)


# In[ ]:


def rle_decode(mask_rle, shape=(1400, 2100)):
    '''
    mask_rle: run-length as string formatted (start length)
    shape: (height, width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')  # Needed to align to RLE direction


# In[ ]:


image_name = '0011165.jpg'
img = imread('../input/train_images/' + image_name)

fig, ax = plt.subplots(2, 2, figsize=(15, 10))

for e, label in enumerate(labels):
    axarr = ax.flat[e]
    image_label = image_name + '_' + label
    mask_rle = train.loc[train['Image_Label'] == image_label, 'EncodedPixels'].values[0]
    try: # label might not be there!
        mask = rle_decode(mask_rle)
    except:
        mask = np.zeros((1400, 2100))
    axarr.axis('off')
    axarr.imshow(img)
    axarr.imshow(mask, alpha=0.5, cmap='gray')
    axarr.set_title(label, fontsize=24)
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.show()


# In[ ]:


# Let's make a "predict everything is a Flower" submission.
# Remember, masks need to be scaled 1/4 per side for predictions, so 350 * 525 = 183750 pixels to cover the entire image.

submission['EncodedPixels'] = submission['Image_Label'].map(lambda x: '1 183750' if x[-6:]=='Flower' else '')
display(submission.head())


# In[ ]:


submission.to_csv('all_flower_submission.csv', index=False)

