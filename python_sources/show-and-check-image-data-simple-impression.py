#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### Print some images, and simple describe
# Print some images for training data. 
# These image data (documents) are various types of documents.  
# For example, novel(#12), Cooking recipes(#1), Seasonal food list(#6), picture book(#10), and so on. And, some document has kana pronouncing(#1, #2, #4, ...).  
# In this competition, the kana pronouncing letters shall be ignored. Normally, the kana pronouncing letters are write smaller size than main letters (Some people write in large letters ...). #10 document have kuzusiji letter, so this image should no label in submission data. 
# And, I think, there are beautiful letters and dirty letters. For example, #12 document is easy to read because the letters are written carefully (or Woodblock printing document), but #1 is a litter difficult beacuse the letters are durty ( perhaps, this document is handwritten document).  
#   
# We shall try to recognise these variety of documents.

# In[ ]:


# Show some image data
imageList = os.listdir("../input/train_images/")[:12]
print(imageList)
plt.figure(figsize=(27,27))
for index, imageFile in enumerate(imageList):
    filePath = "../input/train_images/" + imageFile
    image = cv2.imread(filePath)
    plt.subplot(4,3,index+1)
    plt.title("#{}".format(index+1))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.show()


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
train_df.head()


# In[ ]:


unicode_df = pd.read_csv("../input/unicode_translation.csv")
unicode_df


# In[ ]:




