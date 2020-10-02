#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import seaborn as sns
from scipy import misc
import matplotlib.pyplot as plt

from tqdm import tqdm
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Initialize global variables
SAMPLE_SIZE = 10000
BATCH_SIZE = 32
TEST_PERC = 0.2


# In[ ]:


segmentations = pd.read_csv("../input/train_ship_segmentations_v2.csv")


# In[ ]:


segmentations['path'] = '../input/train/' + segmentations['ImageId']
segmentations.shape


# In[ ]:


segmentations.head()


# In[ ]:


segmentations = segmentations.sample(n=SAMPLE_SIZE)


# In[ ]:


def has_ship(encoded_pixels):
    hs = [0 if pd.isna(n) else 1 for n in tqdm(encoded_pixels)]
    return hs


# In[ ]:


# This function takes a list of pixels, and returns them in run-length encoded format.
def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


# In[ ]:


conv = lambda l: ' '.join(map(str, l)) # list -> string


# In[ ]:


segmentations['HasShip'] = has_ship(segmentations['EncodedPixels'].values)
segmentations['HasShip'].head()


# In[ ]:


segmentations


# In[ ]:


sns.countplot(segmentations['HasShip'])


# In[ ]:


segmentations.head()


# In[ ]:


segmentationsTest = pd.read_csv("../input/sample_submission_v2.csv")


# In[ ]:


#segmentationsTest = segmentationsTest.sample(n=50)


# In[ ]:


segmentationsTest['path'] = '../input/test/' + segmentationsTest['ImageId']
segmentationsTest.shape


# In[ ]:


from PIL import Image


# In[ ]:


sLength = len(segmentationsTest['ImageId'])
segmentationsTest['HasShip'] = pd.Series('', index=segmentationsTest.index)
segmentationsTest['EncodedPixels'] = pd.Series('', index=segmentationsTest.index)
segmentationsTest.head()


# In[ ]:


for index, row in segmentationsTest.iterrows():
   img = Image.open('../input/test_v2/' + row['ImageId'], 'r') 
   x = np.array(img.getdata(), dtype=np.uint8)
   x = x // 255
   val = rle_encoding(x)
   result = conv(val) 
   print('val = ',index, conv(val))
   if not val:
        segmentationsTest.at[index, 'EncodedPixels'] = float('nan')
   else:
        segmentationsTest.at[index, 'EncodedPixels'] = str(result)


# In[ ]:


segmentationsTest['HasShip'] = has_ship(segmentationsTest['EncodedPixels'].values)
segmentationsTest['HasShip'].head()


# In[ ]:


segmentationsTest


# In[ ]:


segmentationsTest = segmentationsTest.drop(['path','HasShip'], axis=1)


# In[ ]:


segmentationsTest.to_csv('submission.csv', index=False)


# In[ ]:




