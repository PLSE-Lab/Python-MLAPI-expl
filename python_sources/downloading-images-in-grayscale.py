#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('apt-get install zip')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from skimage.transform import resize

import os
print(os.listdir("../input/train_images"))
IMG_SIZE = 512

# Any results you write to the current directory are saved as output.


# In[ ]:


list = os.listdir("../input/train_images")
np.size(list)


# In[ ]:


get_ipython().system('mkdir grayscale')
#!mkdir graham
#!mkdir graham_color


# In[ ]:


import cv2
import matplotlib.pyplot as plt
# input image dimensions

for file in list:
    base = os.path.basename("../input/train_images/" + file)
    fileName = os.path.splitext(base)[0]
    img = cv2.imread("../input/train_images/" + file, cv2.IMREAD_GRAYSCALE)   
    #img = cv2.resize(img, (512,512))
    cv2.imwrite("./grayscale/" +  fileName + ".jpeg",img)


# In[ ]:


#for file in list:
#    base = os.path.basename("../input/aptos2019-blindness-detection/train_images/" + file)
#    fileName = os.path.splitext(base)[0]
#    img = cv2.imread("../input/aptos2019-blindness-detection/train_images/" + file, cv2.IMREAD_GRAYSCALE)   
#    #img = cv2.resize(img, (512,512))
#    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , IMG_SIZE/10) ,-4 ,128) # the trick is to add this line
#    cv2.imwrite("./graham/" + file,img)


# In[ ]:


#def load_ben_color(path, sigmaX=10 ):
  #  image = cv2.imread(path)
 #   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   # image = crop_image_from_gray(image)
    #image = cv2.resize(image, (512,512))
    #image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    #return image

#for file in list:
 #   base = os.path.basename("../input/aptos2019-blindness-detection/train_images/" + file)
  #  fileName = os.path.splitext(base)[0]
   # image = load_ben_color("../input/aptos2019-blindness-detection/train_images/" + file,sigmaX=30)
    #cv2.imwrite("./graham_color/" + file,img)


# In[ ]:


#!zip grayscale.zip grayscale/


# In[ ]:


#!rm -r grayscale


# In[ ]:


get_ipython().system('ls grayscale')


# In[ ]:




