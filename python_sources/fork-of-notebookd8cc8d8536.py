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
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


plt.figure(figsize=(20,10))

img2 = cv2.imread("../input/multi_line_text.png");
img = cv2.imread("../input/test1.jpg");
print (img.shape)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_thresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
tiff.imshow(img_thresh)

kernel = np.ones((3,10),np.uint8)
eroded = cv2.dilate(img_thresh,kernel,iterations = 1)
tiff.imshow(eroded)

im2, contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros(img_gray.shape, dtype=np.uint8)

for idx in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[idx])
    print (x,y,w,h)
    mask[y:y+h, x:x+w] = 0
    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
    #cv2.drawContours(mask,contours[idx],0,255,-1)
    #pixelpoints = np.transpose(np.nonzero(mask))
    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
   
    print (r)
    if h >10 and w > 10 and r > 0.5:
        cv2.rectangle(img, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)

#
tiff.imshow( img)

