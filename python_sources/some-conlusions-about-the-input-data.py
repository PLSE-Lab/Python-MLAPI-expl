#!/usr/bin/env python
# coding: utf-8

# Hey, I am new to kaggle and I am struggle to make model which make some decent predictions, but as geophysicist I would like to share some of my findings about the input data.
# 
# During revision of the images I was thinking a lot about process of data preparation. As they said in the competiotion overview and data description we are dealing here with 3d seismic data, which means that images could have been cutted out from x, y and z planes or In-lines, Cross-lines and time-slices using the geophysical vocabulary. 
# 
# 
# [0](https://drive.google.com/file/d/1m9Cu6bAvOzWkAaUbVl_ooSP-w-Jrp9Wx/view?usp=sharing)
# 
# In this case we are probably not dealing with time slices so two planes remains to cut out the images. As i saw a lot of images which looked similiar I decided to try to recombine them at least partially. The competition rules says that labeling by-hand is forbidden but there is nothing about image reconstruction.
# 
# First step: reading the images and depth info
# 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import os



f = open("../input/depths.csv")
data=f.readlines()
f.close()

fils=[]
dep=[]

for i in data[1:]:
    fils.append(i.split(',')[0])
    dep.append(float(i.split(',')[1]))
    
    


# As a second step we need to make lists of images, masks, depths and lists of the leftmost and rightmost seismic traces (left and right column of the image).
# I also generated numpy zeros array with 255 in the topmost row to put them as a test masks. 
# This makes it easier to plot later and allows to clearly differentiate between test and train images.

# In[ ]:


dir_='../input/'

fnames=[]
imag=[]
r=[]
l=[]
u=[]
d=[]
dept=[]
mask=[]

#No mask generation
ema=np.zeros((101,101))
ema[0,:]=255

for filename in os.listdir(dir_+'train/images'):
    fnames.append(filename)
    fimg=cv2.imread(dir_+'train/images/'+filename,0)
    mask.append(cv2.imread(dir_+'/train/masks/'+filename,0))
    imag.append(fimg)
    r.append(fimg[:,-1])
    l.append(fimg[:,0])
    dept.append(dep[fils.index(filename.split('.')[0])])

for filename in os.listdir(dir_+'test/images'):
    fnames.append(filename)
    fimg=cv2.imread(dir_+'test/images/'+filename,0)
    mask.append(ema)
    imag.append(fimg)
    r.append(fimg[:,-1])
    l.append(fimg[:,0])
    dept.append(dep[fils.index(filename.split('.')[0])])
    


# The next step is finding the correlation coefficients between the rightmost column of given image and the leftmost columns of every other image. 
# Index of maximum correlation coeff gives us the most probable neighbouring image to the right. I find that in general correlation coeffs of >0.9 % gives true neighbours. 
# 
# You can choose random image to look for neighbour.

# In[ ]:


m=np.random.randint(22000,size=1)


# Here are some of the interesting indexes to check: 
# 20449,4439, 18152, 6156, 16461,3535,8944,15799,21609,453

# In[ ]:


m=[20449,4439, 18152, 6156, 16461,3535,8944,15799,21609,453]


# In[ ]:


for x in m:

    corr_r=[]
    corr_l=[]

    for i in range(0,len(l)):
        if i==x:
            corr_r.append(0)
        else:
            corr_r.append(np.corrcoef(r[x],l[i])[0,1])

    # Finding index of max correlation
    ri=corr_r.index(max(corr_r))

    # Normalization of images
    a=imag[x]-np.mean(imag[x])
    b=imag[ri]-np.mean(imag[ri])

    mina=min([np.amin(a),np.amin(b)])
    maxa=min([np.amax(a),np.amax(b)])

    plt.figure(figsize=(10,7))
    plt.subplot(121)
    plt.imshow(a,cmap='gray',vmin=mina,vmax=maxa)
    plt.imshow(mask[x],cmap='Greens',vmin=0,vmax=255,alpha=0.2)
    plt.axis('off')
    plt.text(50, 100, dept[x], fontsize=12)
    plt.subplot(122)
    plt.imshow(b,cmap='gray',vmin=mina,vmax=maxa)
    plt.imshow(mask[ri],cmap='Greens',vmin=0,vmax=255,alpha=0.2)
    plt.axis('off')
    plt.text(0, 10, str(max(corr_r))[:5], fontsize=12)
    plt.text(50, 100, dept[ri], fontsize=12)

    # Tight layout of images
    plt.suptitle(str(x))
    plt.subplots_adjust(wspace=0, hspace=0)


# Now you can compute correlations for all of the images and make patches of highly correlated images.
# Here are some examples. The values at the bottom of every picture is the depth.
# 
# [1](https://drive.google.com/file/d/1kyKBrk6GzZk8nMeIg1IRP6WUQ5SYj-G1/view?usp=sharing)
# [2](https://drive.google.com/file/d/1fIcIuVis4lqB3HW2QPIM_mhqsAVZ97f-/view?usp=sharing)
# [3](https://drive.google.com/file/d/1N_NxGyemLoiEKTnsK4ImFXrHTX97JyzW/view?usp=sharing)
# [4](https://drive.google.com/file/d/1QzlZVgBUSlWjJ-Lu95Yh4LKW08Dbc421/view?usp=sharing)
# [5](https://drive.google.com/file/d/1mVGERNTAdKCfCINyhzfnGajMQ1Hekb4P/view?usp=sharing)
# [6](https://drive.google.com/file/d/1QMFwpP8-uUpmD4lIAe4ub3oL6FDT0O54/view?usp=sharing)
# [7](https://drive.google.com/file/d/1UiXPBh_qxnkYyzGjqtx4cCUbfCgJ0XWQ/view?usp=sharing)
# 
# Here are some of the conclusions which I draw from analysing them:
# 
# 1) The depths are incorrect or biased so much that they carry only the qulitative information.
# On the figures above you can see that images which fit perfectly can have depths different by hundreds of feet.
# On the other hands pictures with only zeros pixels which refers to the area over the sea bottom which is shallow falls in the depths from 0 to 300 ft which seems to be reasonable.
# (explained here: https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/61725#380962)
# 
# 2) The images are most probably not in the same scale.
# Thickness of the reflactions can vary with different frequency spectrum of the seismic data after processing.
# Frequency spectrum depend on the quality of the data and is also decreasing with depth which can lower the resolution of image.
# However this can not explein the differences between 4th and 6th images. The first one is of local scale and the second is of regional scale.
# this mean that some images shows the zoomed( better sampled data) and some shows outzoomed(low sampled data)
# 
# 3) There are some images coming from the neghbouring inlines which can act like doubling some of the pictures.
# I make additional analysis where i checked the correlation between only rightmost columns of the images and i found those (right image is always flipped left-right):
# 
# [a](https://drive.google.com/file/d/16NmT22SFL1HB-TXC_TZfnunO-SELL6SE/view?usp=sharing)
# [b](https://drive.google.com/file/d/1FAHD57BMLEKpouYCDnQdAqUdIKy52JqB/view?usp=sharing)
# 
# As you can see images are very similiar, but not the same. They are probably from neighbouring inlines (if that need to be explained let me know in the comments).
# this results in effect close to augmentation or doubling the images.
# 
# 4) If you can find neighbouring train images then you can get different augmentation, by getting middle part of connected images like in the one below
# 

# In[ ]:


m=[3535]
for x in m:

    corr_r=[]
    corr_l=[]

    for i in range(0,len(l)):
        if i==x:
            corr_r.append(0)
        else:
            corr_r.append(np.corrcoef(r[x],l[i])[0,1])

    # Finding index of max correlation
    ri=corr_r.index(max(corr_r))

    # Normalization of images
    a=imag[x]-np.mean(imag[x])
    b=imag[ri]-np.mean(imag[ri])

    mina=min([np.amin(a),np.amin(b)])
    maxa=min([np.amax(a),np.amax(b)])

    plt.figure(figsize=(10,7))
    plt.subplot(121)
    plt.imshow(a,cmap='gray',vmin=mina,vmax=maxa)
    plt.imshow(mask[x],cmap='Greens',vmin=0,vmax=255,alpha=0.2)
    plt.axis('off')
    plt.text(50, 100, dept[x], fontsize=12)
    plt.subplot(122)
    plt.imshow(b,cmap='gray',vmin=mina,vmax=maxa)
    plt.imshow(mask[ri],cmap='Greens',vmin=0,vmax=255,alpha=0.2)
    plt.axis('off')
    plt.text(0, 10, str(max(corr_r))[:5], fontsize=12)
    plt.text(50, 100, dept[ri], fontsize=12)

    # Tight layout of images
    plt.suptitle(str(x))
    plt.subplots_adjust(wspace=0, hspace=0)


# 
# I hope that you find this kernel intersting and useful. If there will be interest I could also public a kernel about depth analysis.
