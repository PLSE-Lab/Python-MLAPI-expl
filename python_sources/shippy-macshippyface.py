#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from skimage.morphology import closing, disk, label
from scipy.ndimage.morphology import binary_closing
from PIL import Image, ImageFilter
from multiprocessing import *
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import cv2

path = '../input/'
train = pd.read_csv('../input/train_ship_segmentations.csv')
test = pd.read_csv('../input/sample_submission.csv')
print(train.shape, test.shape)


# In[ ]:


img = np.random.choice(train.ImageId.values)
im = Image.open(path + 'train/' + img).convert('RGB')
plt.imshow(im)


# In[ ]:


im = im.filter(ImageFilter.EMBOSS).convert('L')
im = (np.array(im)> 150).astype(np.uint8)
im = closing(im, selem=disk(5))
plt.imshow(im)


# In[ ]:


get_ipython().run_cell_magic('time', '', "#https://www.kaggle.com/rakhlin/fast-run-length-encoding-python\ndef rl_encoding(img):\n    im = Image.open('../input/test/' + img).convert('RGB')\n    im = im.filter(ImageFilter.EMBOSS).convert('L')\n    im = (np.array(im)> 150).astype(np.uint8)\n    im = closing(im, selem=disk(5))\n    dots = np.where(im.T.flatten()==1)[0]\n    run_lengths = []\n    prev = -2\n    for b in dots:\n        if (b>prev+1): run_lengths.extend((b+1, 0))\n        run_lengths[-1] += 1\n        prev = b\n    run_lengths = np.array(run_lengths).reshape(len(run_lengths)//2,2)\n    run_lengths = [' '.join(map(str, [x, y])) for x, y in run_lengths if y > 2] #limit\n    run_lengths = ' '.join(map(str, run_lengths))\n    return [img, run_lengths]\n\np = Pool(cpu_count())\nresults = p.map(rl_encoding, test['ImageId'].values[:8000])\np.close(); p.join()")


# In[ ]:


sub = pd.DataFrame(results)
sub.columns = ['ImageId', 'EncodedPixels']
sub1 = pd.read_csv('../input/sample_submission.csv')
sub1 = pd.DataFrame(np.setdiff1d(sub1['ImageId'].unique(), sub['ImageId'].unique(), assume_unique=True), columns=['ImageId'])
sub1['EncodedPixels'] = None #'1 2'
print(len(sub1), len(sub))
sub = pd.concat([sub, sub1])
print(len(sub))
sub.to_csv('submission.csv', index=False)

