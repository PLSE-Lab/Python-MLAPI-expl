#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image
from PIL import ImageFilter
import multiprocessing
import random; random.seed(2016);
import cv2
import re
import os, glob

sample_sub = pd.read_csv('../input/sample_submission.csv')
train_files = pd.DataFrame([[f,f.split("/")[3].split(".")[0].split("_")[0],f.split("/")[3].split(".")[0].split("_")[1]] for f in glob.glob("../input/train_sm/*.jpeg")])
train_files.columns = ['path', 'group', 'pic_no']
test_files = pd.DataFrame([[f,f.split("/")[3].split(".")[0].split("_")[0],f.split("/")[3].split(".")[0].split("_")[1]] for f in glob.glob("../input/test_sm/*.jpeg")])
test_files.columns = ['path', 'group', 'pic_no']
print(len(train_files),len(test_files),len(sample_sub))
train_images = train_files[train_files["group"]=='set107']
train_images = train_images.sort_values(by=["pic_no"], ascending=[1]).reset_index(drop=True)


# In[ ]:


brisk = cv2.BRISK_create()
dm = cv2.DescriptorMatcher_create("BruteForce")

def c_resize(img, ratio):
    wh = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
    img = cv2.resize(img, wh, interpolation = cv2.INTER_AREA)
    return img
    
def im_stitcher(imp1, imp2, pcntDownsize = 1.0, withTransparency=False):
    
    #Read image1
    image1 = cv2.imread(imp1)
    
    # perform the resizing of the image by pcntDownsize and create a Grayscale version
    dim1 = (int(image1.shape[1] * pcntDownsize), int(image1.shape[0] * pcntDownsize))
    img1 = cv2.resize(image1, dim1, interpolation = cv2.INTER_AREA)
    img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
    #Read image2
    image2 = cv2.imread(imp2)
    
    # perform the resizing of the image by pcntDownsize and create a Grayscale version
    dim2 = (int(image2.shape[1] * pcntDownsize), int(image2.shape[0] * pcntDownsize))
    img2 = cv2.resize(image2, dim2, interpolation = cv2.INTER_AREA)
    img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    #use BRISK to create keypoints in each image
    brisk = cv2.BRISK_create()
    kp1, des1 = brisk.detectAndCompute(img1Gray,None)
    kp2, des2 = brisk.detectAndCompute(img2Gray,None)
    
    # use BruteForce algorithm to detect matches among image keypoints 
    dm = cv2.DescriptorMatcher_create("BruteForce")
    
    matches = dm.knnMatch(des1,des2, 2)
    matches_ = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            matches_.append((m[0].trainIdx, m[0].queryIdx))
    
    kp1_ = np.float32([kp1[m[1]].pt for m in matches_]).reshape(-1,1,2)
    kp2_ = np.float32([kp2[m[0]].pt for m in matches_]).reshape(-1,1,2)
    
    
    H, mask = cv2.findHomography(kp2_,kp1_, cv2.RANSAC, 4.0)
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    
    t = [-xmin,-ymin]
    
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
    
    #warp the colour version of image2
    im = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    
    #overlay colur version of image1 to warped image2
    if withTransparency == True:
        h3,w3 = im.shape[:2]
        bim = np.zeros((h3,w3,3), np.uint8)
        bim[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
        
        #imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #imColor = cv2.applyColorMap(imGray, cv2.COLORMAP_JET)
        
        im =(im[:,:,2] - bim[:,:,2])
        #im = cv2.addWeighted(im,0.6,bim,0.6,0)
    else:
        im[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return(im)


# In[ ]:


img = im_stitcher(train_images.path[0], train_images.path[4], 0.4, True)
plt.rcParams['figure.figsize'] = (12.0, 12.0)
plt.imshow(img); plt.axis('off')


# In[ ]:


plt.rcParams['figure.figsize'] = (20, 20)

for j in range(1,5):
    for i in range(j+1,6):
        img = im_stitcher(train_images.path[j-1], train_images.path[i-1], 0.4, True)
        plt.subplot(4,4,i-1+(j-1)*4)
        plt.imshow(img);plt.axis('off')

