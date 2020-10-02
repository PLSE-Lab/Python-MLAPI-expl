#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


split_num=3
split_index=[0,1,3]


# In[ ]:


def rle_decode(rle_mask):
    '''
    rle_mask: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    if type(rle_mask)==float:
        return np.zeros([101,101])
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101*101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101,101)
"""
used for converting the decoded image to rle mask

"""
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# In[ ]:


test_df=pd.read_csv("../input/tgs-salt-identification-challenge/sample_submission.csv",index_col='id')


# In[ ]:


test_img=np.zeros([18000,101,101])
for fold in split_index:
    test_fold=pd.read_csv("../input/fork-of-result-visualization-split-"+str(fold)+"/submission.csv",index_col='id')
    for index,ids in enumerate(test_fold.index):
        img=rle_decode(test_fold.loc[ids,'rle_mask'])
        test_img[index]+=img/split_num
test_img=np.round(test_img)


# In[ ]:


import matplotlib.pyplot as plt
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.imshow(test_img[i])
    
    


# In[ ]:


test_fold=pd.read_csv("../input/fork-of-result-visualization-split-0/submission.csv",index_col='id')


# In[ ]:


show=[]
for index,ids in enumerate(test_fold.index):
    if index==20:
        break
    show.append(rle_decode(test_fold.loc[ids,'rle_mask']))
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.imshow(show[i])


# In[ ]:


for index,ids in enumerate(test_df.index):
    test_df.loc[ids,'rle_mask']=rle_encode(test_img[index])
    
    


# In[ ]:


test_df.head()


# In[ ]:


test_df.to_csv('submission.csv')


# In[ ]:




