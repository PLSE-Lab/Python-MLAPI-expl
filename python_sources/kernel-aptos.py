#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


traincsv=pd.read_csv('/kaggle/input/aptos2019-blindness-detection/train.csv')
trainImgsPath='/kaggle/input/aptos2019-blindness-detection/train_images'
testImgsPath='/kaggle/input/aptos2019-blindness-detection/test_images'


# In[ ]:


import cv2 as cv2
from tqdm import tqdm 
import matplotlib.pyplot as plt


# In[ ]:


def getNImageFromFolder(folderName,N):
    imgs=[]
    folder=os.listdir(folderName)
    for img in tqdm(folder):
        if(len(imgs)<N):
            imgPath=os.path.join(folderName,img)
            tempImg=cv2.imread(imgPath)
            tempImg=cv2.cvtColor(tempImg,cv2.COLOR_BGR2RGB)
            if tempImg is not None:
                imgs.append(tempImg)
        else :
            break
    return imgs


# In[ ]:


def plotNImages(imgs,r,c):
    row,col=r,c
    fig=plt.figure(figsize=(row*col,row+col))
    for i in range(0,len(imgs)):
        fig.add_subplot(row,col,i+1)
        plt.imshow(imgs[i])
    plt.show()   
    


# In[ ]:


trainImgs=getNImageFromFolder(trainImgsPath,20)
plotNImages(trainImgs,4,5)


# In[ ]:


#Concept:Unsharp Masking...first blur an image and subtract the blur in the original to get sharpened version
def preprocessImgs(imgs,size,sigmaX):
    ppImgs=[]
    imgSize=size
    for i in range(0,len(imgs)):
        tempImg=imgs[i]
        #tempImg=cv2.cvtColor(tempImg,cv2.COLOR_RGB2GRAY)        
        tempImg=cv2.resize(tempImg,(imgSize,imgSize))       
        blurredImg=cv2.GaussianBlur(tempImg,(0,0),sigmaX)
        tempImg=cv2.addWeighted(tempImg,4,blurredImg,-4 ,128)
        ppImgs.append(tempImg)
    return ppImgs


# In[ ]:


ppImgs=preprocessImgs(trainImgs,size=500,sigmaX=38)
plotNImages(ppImgs,r=4,c=5)


# In[ ]:


def cropImagesFromCircleCenter():
    

