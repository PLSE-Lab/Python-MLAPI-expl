#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import skimage.io
import imageio
import os
import scipy.ndimage as ndi
import random
from skimage import color
from collections import Counter 
import scipy.ndimage.measurements
from skimage.feature import greycomatrix,greycoprops
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


get_ipython().system('pip install imagecodecs')


# In[ ]:


def file_names(path):
    file_names_pictures=[]
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            #print(os.path.join(dirname, filename))
            file_names_pictures.append(os.path.join(dirname, filename))
    return np.sort(file_names_pictures)
def show_image_in_path(path):
    """Show image in first dimension .
    
    Args:
    str: image path
    """
    image_path=path
    im=imageio.imread(image_path)
    show_image(im[:,:,0])
def read_image(image_path,part_to_read):
    im=imageio.imread(image_path)
    if part_to_read==0:
        return im
    elif part_to_read==1:
        return im[:,:,0]
    elif part_to_read==2:
        return im[:,:,1]
    elif part_to_read==3:
        return im[:,:,2]
    else:
        raise ValueError("argument number 2 must be 0,1,2 or 3")
def show_image(image):
    """Show images.
    
    Args:
    numpy.Array: image to show
    """
    plt.imshow(image)
def equalizer(image):
    """Equalize images.
    
    Args:
    numpy.Array: image to equalize
    
    Returns:
    numpy.Array: equalized image
    """
    hist=ndi.histogram(image,min=0,max=255,bins=256)
    cdf=hist.cumsum()/hist.sum()
    im_equalized=cdf[image]*255
    return im_equalized
def histogram_1D(image):
    return ndi.histogram(image,min=0,max=1,bins=1000)
def CDF_1D(image):
    hist=ndi.histogram(image,min=0,max=1,bins=1000)
    return hist.cumsum()/hist.sum()
def histogram(image,i):
    #im=imageio.imread(image_path)
    if i==0:
        return ndi.histogram(image,min=0,max=255,bins=256)
    elif i==1:
        return ndi.histogram(image[:,:,0],min=0,max=255,bins=256)
    elif i==2:
        return ndi.histogram(image[:,:,1],min=0,max=255,bins=256)
    elif i==3:
        return ndi.histogram(image[:,:,2],min=0,max=255,bins=256)
    else:
        raise ValueError("argument number 2 must be 0,1,2 or 3")
def CDF(image,i):
    #im=imageio.imread(image_path)
    if i==0:
        hist=ndi.histogram(image,min=0,max=255,bins=256)
        return hist.cumsum()/hist.sum()
    elif i==1:
        hist=ndi.histogram(image[:,:,0],min=0,max=255,bins=256)
        return hist.cumsum()/hist.sum()
    elif i==2:
        hist=ndi.histogram(image[:,:,1],min=0,max=255,bins=256)
        return hist.cumsum()/hist.sum()
    elif i==3:
        hist=ndi.histogram(image[:,:,2],min=0,max=255,bins=256)
        return hist.cumsum()/hist.sum()
    else:
        raise ValueError("argument number 2 must be 0,1,2 or 3")
    
def plot_histogram(image):
    hist=ndi.histogram(image,min=0,max=255,bins=256)
    plt.plot(hist)
def mask_apply_1(image):
    B=image<255
    im_selected=np.where(B,image,0)
    #ndi.binary_closing(B,iterations=3)
    return im_selected,ndi.binary_closing(B,iterations=3)
def mask_apply(image,mask):
    B=mask>0
    im_selected=np.where(B,image,0)
    return im_selected
def plot_histogram_in_image_parts(image,mask):
    im_selected=mask_apply(image,mask)
    hist=ndi.histogram(im_selected,min=1,max=255,bins=255)
    plt.plot(hist)
def dimensions_after_mask_application(image,mask):
    try:
        im_selected=mask_apply(image,mask)
    except ValueError:
        im_selected,_=mask_apply_1(image)
    dimensions=[]
    for i in range(im_selected.shape[0]):
        if im_selected[i,:].any():
            dimensions.append(i)
            break
    for j in range(dimensions[0],im_selected.shape[0]):
        if not im_selected[j,:].any():
            dimensions.append(j)
            break
    for k in range(im_selected.shape[1]):
        if im_selected[:,k].any():
            dimensions.append(k)
            break
    for l in range(dimensions[2],im_selected.shape[1]):
        if not im_selected[:,l].any():
            dimensions.append(l)
            break
    return dimensions
def extract_patchs(im_mask,im,dimensions,patch_number,patch_length,patch_width):
    selected_patchs=[]
    selected_dimentions=[]
    for i in range(100000):
        if len(selected_patchs)>=patch_number:
            break;
        pixel_coordinates=[random.randint(dimensions[0], dimensions[1]),random.randint(dimensions[2], dimensions[3])]
        if  im_mask[pixel_coordinates[0]:pixel_coordinates[0]+patch_width,pixel_coordinates[1]:pixel_coordinates[1]+patch_length].all():
            selected_patchs.append(im[pixel_coordinates[0]:pixel_coordinates[0]+patch_width,pixel_coordinates[1]:pixel_coordinates[1]+patch_length,:])
            selected_dimentions.append((pixel_coordinates[0],pixel_coordinates[1]))
    return selected_patchs,selected_dimentions
def extract_patch_layers(selected_patchs,layer_number):
    selected_patchs_created=[]
    if layer_number==0:
        for i in range(len(selected_patchs)):
            selected_patchs_created.append(selected_patchs[i][:,:,0])
        return selected_patchs_created
    elif layer_number==1:
        for i in range(len(selected_patchs)):
            selected_patchs_created.append(selected_patchs[i][:,:,1])
        return selected_patchs_created
    elif layer_number==2:
        for i in range(len(selected_patchs)):
            selected_patchs_created.append(selected_patchs[i][:,:,2])
        return selected_patchs_created
    else:
        raise ValueError("argument number 2 must be 0,1 or 2")
def show_patchs(figSize,columns_number,rows_number,selected_patchs):
    fig=plt.figure(figsize=figSize)
    for i in range(1, columns_number*rows_number +1):
        #img = np.random.randint(10, size=(h,w))
        fig.add_subplot(rows_number, columns_number, i)
        #print(i)
        plt.imshow(selected_patchs[i-1])
    plt.show()
def show_patch_location(selected_patch,selected_image,selected_dimention,image):
    mask1=np.ones(selected_image.shape,dtype=bool)
    mask1[selected_dimention[0]:selected_dimention[0]+400,selected_dimention[1]:selected_dimention[1]+400]=False
    im_position=np.where(mask1,image[:,:,0],0)
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(im_position[dimensions[0]:dimensions[1],dimensions[2]:dimensions[3]])
    axarr[1].imshow(selected_patch)
def show_histograms(selected_patchs):
    n=len(selected_patchs)
    l=[]
    gs = plt.GridSpec(n,4) 
    fig = plt.figure(figsize=(20,50))
    for i in range(n):
        l.append(fig.add_subplot(gs[i,0]))
        l.append(fig.add_subplot(gs[i,1]))
        l.append(fig.add_subplot(gs[i,2]))
        l.append(fig.add_subplot(gs[i,3]))
    k=0
    for i in range(0,n*4,4):
        l[i].imshow(selected_patchs[k])
        l[i+1].plot(histogram(selected_patchs[k],1))
        l[i+2].plot(histogram(selected_patchs[k],2))
        l[i+3].plot(histogram(selected_patchs[k],3))
        k=k+1
def show_CDF(selected_patchs):
    n=len(selected_patchs)
    l=[]
    gs = plt.GridSpec(n,4) 
    fig = plt.figure(figsize=(20,50))
    for i in range(n):
        l.append(fig.add_subplot(gs[i,0]))
        l.append(fig.add_subplot(gs[i,1]))
        l.append(fig.add_subplot(gs[i,2]))
        l.append(fig.add_subplot(gs[i,3]))
    k=0
    for i in range(0,n*4,4):
        l[i].imshow(selected_patchs[k])
        l[i+1].plot(CDF(selected_patchs[k],1))
        l[i+2].plot(CDF(selected_patchs[k],2))
        l[i+3].plot(CDF(selected_patchs[k],3))
        k=k+1
def show_histograms_1D(selected_patchs):
    n=len(selected_patchs)
    l=[]
    gs = plt.GridSpec(n,2) 
    fig = plt.figure(figsize=(20,50))
    for i in range(n):
        l.append(fig.add_subplot(gs[i,0]))
        l.append(fig.add_subplot(gs[i,1]))
        
    k=0
    for i in range(0,n*2,2):
        l[i].imshow(selected_patchs[k])
        l[i+1].plot(histogram_1D(selected_patchs[k]))
        
        k=k+1
def show_CDF_1D(selected_patchs):
    n=len(selected_patchs)
    l=[]
    gs = plt.GridSpec(n,2) 
    fig = plt.figure(figsize=(20,50))
    for i in range(n):
        l.append(fig.add_subplot(gs[i,0]))
        l.append(fig.add_subplot(gs[i,1]))
        
    k=0
    for i in range(0,n*2,2):
        l[i].imshow(selected_patchs[k])
        l[i+1].plot(CDF_1D(selected_patchs[k]))
        
        k=k+1
def position_construction(im_selected,selected_patch,selected_dimention,image):
    #selected_patch=selected_patchs[k]
    #selected_image=im_selected
    #selected_dimention=selected_dimentions[k]
    #image=im
    mask1=np.ones(im_selected.shape,dtype=bool)
    mask1[selected_dimention[0]:selected_dimention[0]+400,selected_dimention[1]:selected_dimention[1]+400]=False
    im_position=np.where(mask1,image[:,:,0],0)
    return im_position
def features(image):
    a=ndi.mean(image)
    b=ndi.median(image)
    c=ndi.sum(image)
    d=ndi.maximum(image)
    e=ndi.minimum(image)
    f=ndi.standard_deviation(image)
    #g=ndi.variance(image)
    return [a,b,c,d,e,f]
def more_features(image):
    a=ndi.mean(image)
    b=ndi.median(image)
    c=ndi.sum(image)
    d=ndi.maximum(image)
    e=ndi.minimum(image)
    f=ndi.standard_deviation(image)
    glcm=greycomatrix(image,distances=[5],angles=[0],levels=256,symmetric=True,normed=True)
    g=greycoprops(glcm,'dissimilarity')[0,0]
    h=greycoprops(glcm,'correlation')[0,0]
    #g=ndi.variance(image)
    return [a,b,c,d,e,f,g,h]
def all_selected_features(image,image_grey_scale):
    a1=ndi.mean(image[:,:,0])
    b1=ndi.median(image[:,:,0])
    c1=ndi.sum(image[:,:,0])
    d1=ndi.standard_deviation(image[:,:,0])
    glcm=greycomatrix(image[:,:,0],distances=[5],angles=[0],levels=256,symmetric=True,normed=True)
    e1=greycoprops(glcm,'dissimilarity')[0,0]
    f1=greycoprops(glcm,'correlation')[0,0]
    a2=ndi.mean(image[:,:,1])
    b2=ndi.median(image[:,:,1])
    c2=ndi.sum(image[:,:,1])
    d2=ndi.standard_deviation(image[:,:,1])
    glcm=greycomatrix(image[:,:,1],distances=[5],angles=[0],levels=256,symmetric=True,normed=True)
    e2=greycoprops(glcm,'dissimilarity')[0,0]
    f2=greycoprops(glcm,'correlation')[0,0]
    a3=ndi.mean(image[:,:,2])
    b3=ndi.median(image[:,:,2])
    c3=ndi.sum(image[:,:,2])
    d3=ndi.standard_deviation(image[:,:,2])
    glcm=greycomatrix(image[:,:,2],distances=[5],angles=[0],levels=256,symmetric=True,normed=True)
    e3=greycoprops(glcm,'dissimilarity')[0,0]
    f3=greycoprops(glcm,'correlation')[0,0]
    a4=ndi.mean(image_grey_scale)
    b4=ndi.median(image_grey_scale)
    c4=ndi.sum(image_grey_scale)
    d4=ndi.standard_deviation(image_grey_scale)
    return [a1,b1,c1,d1,e1,f1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4]
def show_features(im):
    n=round(len(im)/2)
    l=[]
    gs = plt.GridSpec(n,2) 
    fig = plt.figure(figsize=(10,110))
    for i in range(n):
        l.append(fig.add_subplot(gs[i,0]))
        l.append(fig.add_subplot(gs[i,1]))
    k=0
    for i in range(0,n*2,2):
    
        #im_position=position_construction(im_selected,selected_patchs[k],selected_dimentions[k],im)

        L=features(im[k])
        textstr = '\n'.join(( r'$\mu=%.3f$' % (L[0], ),r'$\mathrm{median}=%.3f$' % (L[1], ),r'$\mathrm{sum}=%.3f$' % (L[2], ),r'$\mathrm{maximum}=%.3f$' % (L[3], ),r'$\mathrm{minimum}=%.3f$' % (L[4], ),r'$\mathrm{std}=%.3f$' % (L[5], )))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)


        l[i].imshow(im[k])
        l[i].text(0.05, 0.95, textstr, transform=l[i].transAxes, fontsize=14,verticalalignment='top', bbox=props)


        k=k+1

        #im_position=position_construction(im_selected,selected_patchs[k],selected_dimentions[k],im)


        L=features(im[k])
        textstr = '\n'.join(( r'$\mu=%.3f$' % (L[0], ),r'$\mathrm{median}=%.3f$' % (L[1], ),r'$\mathrm{sum}=%.3f$' % (L[2], ),r'$\mathrm{maximum}=%.3f$' % (L[3], ),r'$\mathrm{minimum}=%.3f$' % (L[4], ),r'$\mathrm{std}=%.3f$' % (L[5], )))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)


        l[i+1].imshow(im[k])
        l[i+1].text(0.05, 0.95, textstr, transform=l[i+1].transAxes, fontsize=14,verticalalignment='top', bbox=props)


        k=k+1
def show_more_features(im):
    n=round(len(im)/2)
    l=[]
    gs = plt.GridSpec(n,2) 
    fig = plt.figure(figsize=(10,110))
    for i in range(n):
        l.append(fig.add_subplot(gs[i,0]))
        l.append(fig.add_subplot(gs[i,1]))
    k=0
    for i in range(0,n*2,2):
    
        #im_position=position_construction(im_selected,selected_patchs[k],selected_dimentions[k],im)

        L=more_features(im[k])
        textstr = '\n'.join(( r'$\mu=%.3f$' % (L[0], ),r'$\mathrm{median}=%.3f$' % (L[1], ),r'$\mathrm{sum}=%.3f$' % (L[2], ),r'$\mathrm{maximum}=%.3f$' % (L[3], ),r'$\mathrm{minimum}=%.3f$' % (L[4], ),r'$\mathrm{std}=%.3f$' % (L[5], ),r'$\mathrm{dissimilarity}=%.3f$' % (L[6], ),r'$\mathrm{correlation}=%.3f$' % (L[7], )))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)


        l[i].imshow(im[k])
        l[i].text(0.05, 0.95, textstr, transform=l[i].transAxes, fontsize=14,verticalalignment='top', bbox=props)


        k=k+1

        #im_position=position_construction(im_selected,selected_patchs[k],selected_dimentions[k],im)


        L=more_features(im[k])
        textstr = '\n'.join(( r'$\mu=%.3f$' % (L[0], ),r'$\mathrm{median}=%.3f$' % (L[1], ),r'$\mathrm{sum}=%.3f$' % (L[2], ),r'$\mathrm{maximum}=%.3f$' % (L[3], ),r'$\mathrm{minimum}=%.3f$' % (L[4], ),r'$\mathrm{std}=%.3f$' % (L[5], ),r'$\mathrm{dissimilarity}=%.3f$' % (L[6], ),r'$\mathrm{correlation}=%.3f$' % (L[7], )))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)


        l[i+1].imshow(im[k])
        l[i+1].text(0.05, 0.95, textstr, transform=l[i+1].transAxes, fontsize=14,verticalalignment='top', bbox=props)


        k=k+1
def show_all_features(im,im_grey):
    n=round(len(im)/2)
    l=[]
    gs = plt.GridSpec(n,2) 
    fig = plt.figure(figsize=(10,150))
    for i in range(n):
        l.append(fig.add_subplot(gs[i,0]))
        l.append(fig.add_subplot(gs[i,1]))
    k=0
    for i in range(0,n*2,2):
    
        #im_position=position_construction(im_selected,selected_patchs[k],selected_dimentions[k],im)

        L=all_selected_features(im[k],im_grey[k])
        textstr = '\n'.join(( r'$\mu_{R}=%.3f$' % (L[0], ),r'$\mathrm{median_{R}}=%.3f$' % (L[1], ),r'$\mathrm{sum_{R}}=%.3f$' % (L[2], ),r'$\mathrm{std_{R}}=%.3f$' % (L[3], ),r'$\mathrm{dissimilarity_{R}}=%.3f$' % (L[4], ),r'$\mathrm{correlation_{R}}=%.3f$' % (L[5], ),r'$\mu_{G}=%.3f$' % (L[6], ),r'$\mathrm{median_{G}}=%.3f$' % (L[7], ),r'$\mathrm{sum_{G}}=%.3f$' % (L[8], ),r'$\mathrm{std_{G}}=%.3f$' % (L[9], ),r'$\mathrm{dissimilarity_{G}}=%.3f$' % (L[10], ),r'$\mathrm{correlation_{G}}=%.3f$' % (L[11], ),r'$\mu_{B}=%.3f$' % (L[12], ),r'$\mathrm{median_{B}}=%.3f$' % (L[13], ),r'$\mathrm{sum_{B}}=%.3f$' % (L[14], ),r'$\mathrm{std_{B}}=%.3f$' % (L[15], ),r'$\mathrm{dissimilarity_{B}}=%.3f$' % (L[16], ),r'$\mathrm{correlation_{B}}=%.3f$' % (L[17], ),r'$\mu_{grey}=%.3f$' % (L[18], ),r'$\mathrm{median_{grey}}=%.3f$' % (L[19], ),r'$\mathrm{sum_{grey}}=%.3f$' % (L[20], ),r'$\mathrm{std_{grey}}=%.3f$' % (L[21], )))
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)


        l[i].imshow(im[k])
        l[i].text(0.05, 0.95, textstr, transform=l[i].transAxes, fontsize=8,verticalalignment='top', bbox=props)


        k=k+1

        #im_position=position_construction(im_selected,selected_patchs[k],selected_dimentions[k],im)


        L=all_selected_features(im[k],im_grey[k])
        textstr = '\n'.join(( r'$\mu_{R}=%.3f$' % (L[0], ),r'$\mathrm{median_{R}}=%.3f$' % (L[1], ),r'$\mathrm{sum_{R}}=%.3f$' % (L[2], ),r'$\mathrm{std_{R}}=%.3f$' % (L[3], ),r'$\mathrm{dissimilarity_{R}}=%.3f$' % (L[4], ),r'$\mathrm{correlation_{R}}=%.3f$' % (L[5], ),r'$\mu_{G}=%.3f$' % (L[6], ),r'$\mathrm{median_{G}}=%.3f$' % (L[7], ),r'$\mathrm{sum_{G}}=%.3f$' % (L[8], ),r'$\mathrm{std_{G}}=%.3f$' % (L[9], ),r'$\mathrm{dissimilarity_{G}}=%.3f$' % (L[10], ),r'$\mathrm{correlation_{G}}=%.3f$' % (L[11], ),r'$\mu_{B}=%.3f$' % (L[12], ),r'$\mathrm{median_{B}}=%.3f$' % (L[13], ),r'$\mathrm{sum_{B}}=%.3f$' % (L[14], ),r'$\mathrm{std_{B}}=%.3f$' % (L[15], ),r'$\mathrm{dissimilarity_{B}}=%.3f$' % (L[16], ),r'$\mathrm{correlation_{B}}=%.3f$' % (L[17], ),r'$\mu_{grey}=%.3f$' % (L[18], ),r'$\mathrm{median_{grey}}=%.3f$' % (L[19], ),r'$\mathrm{sum_{grey}}=%.3f$' % (L[20], ),r'$\mathrm{std_{grey}}=%.3f$' % (L[21], )))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)


        l[i+1].imshow(im[k])
        l[i+1].text(0.05, 0.95, textstr, transform=l[i+1].transAxes, fontsize=8,verticalalignment='top', bbox=props)


        k=k+1
def selected_patchs_in_file_name(file_name,file_mask):
    try:
        #print("noValueError")
        im=read_image(file_name,1)
        im_mask=read_image(file_mask,1)
        dimensions=dimensions_after_mask_application(im,im_mask)
        im=read_image(file_name,0)
        selected_patchs,selected_dimentions=extract_patchs(im_mask,im,dimensions,50,patch_length=400,patch_width=400)
    except ValueError:
        print("valueError")
        selected_patchs,selected_dimentions=extract_patchs_1(file_name,patch_width=400,patch_length=400,patch_number=50)
    return selected_patchs,selected_dimentions
def extract_patchs_1(image,patch_width,patch_length,patch_number):
    im=read_image(image,1)
    _,im_mask=mask_apply_1(im)
    dimensions=dimensions_after_mask_application(im,im_mask)
    im=read_image(image,0)
    pixel_coordinates=[random.randint(dimensions[0], dimensions[1]),random.randint(dimensions[2], dimensions[3])]
    selected_patchs=[]
    selected_dimentions=[]
    for i in range(100000):
        if len(selected_patchs)>=patch_number:
            break;
        pixel_coordinates=[random.randint(dimensions[0], dimensions[1]),random.randint(dimensions[2], dimensions[3])]
        if  im_mask[pixel_coordinates[0]:pixel_coordinates[0]+patch_width,pixel_coordinates[1]:pixel_coordinates[1]+patch_length].all():
            selected_patchs.append(im[pixel_coordinates[0]:pixel_coordinates[0]+patch_width,pixel_coordinates[1]:pixel_coordinates[1]+patch_length,:])
            selected_dimentions.append((pixel_coordinates[0],pixel_coordinates[1]))
    return selected_patchs,selected_dimentions


# In[ ]:


file_names_pictures=file_names('/kaggle/input/prostate-cancer-grade-assessment/train_images')


# In[ ]:


file_names_pictures


# In[ ]:


file_names_mask=file_names('/kaggle/input/prostate-cancer-grade-assessment/train_label_masks')


# In[ ]:


file_names_mask


# In[ ]:


file_names_test=file_names('../input/prostate-cancer-grade-assessment/test_images')


# In[ ]:


#file_names_test


# In[ ]:





# In[ ]:





# In[ ]:


class Patch():
    def __init__(self, filepath,patch_width,patch_length):
        self.filepath=filepath
        self.data=imageio.imread(filepath)
        self.patch_width=patch_width
        self.patch_length=patch_length
    def show_image(self):
        plt.imshow(self.data)
    def mask_apply(self):
        im_selected=np.where(self.data[:,:,0]<255,self.data[:,:,0],0)
        return im_selected,ndi.binary_closing(self.data[:,:,0]<255,iterations=3)
    def dimensions_after_mask_application(self):
        #image=self.data[:,:,0]
        im_selected,_=self.mask_apply()
        dimensions=[]
        for i in range(0,im_selected.shape[0],200):
            if im_selected[i,:].any():
                dimensions.append(i)
                break
        for j in range(dimensions[0],im_selected.shape[0],100):
            if not im_selected[j,:].any():
                dimensions.append(j)
                break
        for k in range(0,im_selected.shape[1],200):
            if im_selected[:,k].any():
                dimensions.append(k)
                break
        for l in range(dimensions[2],im_selected.shape[1],100):
            if not im_selected[:,l].any():
                dimensions.append(l)
                break
        return dimensions
    def extract_patch(self):
        _,im_mask=self.mask_apply()
        dimensions=self.dimensions_after_mask_application()
        for i in range(10000):
            pixel_coordinates=[random.randint(dimensions[0], dimensions[1]),random.randint(dimensions[2], dimensions[3])]
            if  im_mask[pixel_coordinates[0]:pixel_coordinates[0]+self.patch_width,pixel_coordinates[1]:pixel_coordinates[1]+self.patch_length].all():
                return self.data[pixel_coordinates[0]:pixel_coordinates[0]+self.patch_width,pixel_coordinates[1]:pixel_coordinates[1]+self.patch_length,:]
            
        
        


# In[ ]:


class Features():
    def __init__(self, image):
        self.image=image
        self.mean_1=ndi.mean(image[:,:,0])
        self.median_1=ndi.median(image[:,:,0])
        self.sum_1=ndi.sum(image[:,:,0])
        self.standard_deviation_1=ndi.standard_deviation(image[:,:,0])
        self.glcm_1=greycomatrix(image[:,:,0],distances=[5],angles=[0],levels=256,symmetric=True,normed=True)
        self.dissimilarity_1=greycoprops(self.glcm_1,'dissimilarity')[0,0]
        self.correlation_1=greycoprops(self.glcm_1,'correlation')[0,0]
        self.mean_2=ndi.mean(image[:,:,1])
        self.median_2=ndi.median(image[:,:,1])
        self.sum_2=ndi.sum(image[:,:,1])
        self.standard_deviation_2=ndi.standard_deviation(image[:,:,1])
        self.glcm_2=greycomatrix(image[:,:,1],distances=[5],angles=[0],levels=256,symmetric=True,normed=True)
        self.dissimilarity_2=greycoprops(self.glcm_2,'dissimilarity')[0,0]
        self.correlation_2=greycoprops(self.glcm_2,'correlation')[0,0]
        self.mean_3=ndi.mean(image[:,:,2])
        self.median_3=ndi.median(image[:,:,2])
        self.sum_3=ndi.sum(image[:,:,2])
        self.standard_deviation_3=ndi.standard_deviation(image[:,:,2])
        self.glcm_3=greycomatrix(image[:,:,2],distances=[5],angles=[0],levels=256,symmetric=True,normed=True)
        self.dissimilarity_3=greycoprops(self.glcm_3,'dissimilarity')[0,0]
        self.correlation_3=greycoprops(self.glcm_3,'correlation')[0,0]
        self.mean_4=ndi.mean(color.rgb2gray(image))
        self.median_4=ndi.median(color.rgb2gray(image))
        self.sum_4=ndi.sum(color.rgb2gray(image))
        self.standard_deviation_4=ndi.standard_deviation(color.rgb2gray(image))
    def extract_features(self):
        return [self.mean_1,self.median_1,self.sum_1,self.standard_deviation_1,self.dissimilarity_1,self.correlation_1,self.mean_2,self.median_2,self.sum_2,self.standard_deviation_2,self.dissimilarity_2,self.correlation_2,self.mean_3,self.median_3,self.sum_3,self.standard_deviation_3,self.dissimilarity_3,self.correlation_3,self.mean_4,self.median_4,self.sum_4,self.standard_deviation_4]
     
        


# In[ ]:


class ImageFeatures():
    def __init__(self,PatchNumber,filepath,patch_width,patch_length,idx):
        self.PatchNumber=PatchNumber
        self.filepath=filepath
        self.patch_width=patch_width
        self.patch_length=patch_length
        self.idx=idx
        
    def extract_image_features(self):
        features=[]
        P=Patch(self.filepath,self.patch_width,self.patch_length)
        for i in range(self.PatchNumber):
            #P=Patch(self.filepath,self.patch_width,self.patch_length)
            F=Features(P.extract_patch())
            features.append(F.extract_features())
        return features
    
    def extract_image_features_dataframe(self):
        df = pd.DataFrame(self.extract_image_features(), columns = ['mean_1','median_1','sum_1','standard_deviation_1','dissimilarity_1','correlation_1','mean_2','median_2','sum_2','standard_deviation_2','dissimilarity_2','correlation_2','mean_3','median_3','sum_3','standard_deviation_3','dissimilarity_3','correlation_3','mean_4','median_4','sum_4','standard_deviation_4'])
        df["picture_number"]=list(np.round(np.ones(self.PatchNumber)*self.idx))
        return df,self.idx
    def extract_mean_image_features(self):
        df=self.extract_image_features_dataframe()
        
        return [np.mean(df['mean_1']),np.mean(df['median_1']),np.mean(df['sum_1']),np.mean(df['standard_deviation_1']),np.mean(df['correlation_1']),np.mean(df['mean_2']),np.mean(df['median_2']),np.mean(df['sum_2']),np.mean(df['standard_deviation_2']),np.mean(df['dissimilarity_2']),np.mean(df['correlation_2']),np.mean(df['mean_3']),np.mean(df['median_3']),np.mean(df['sum_3']),np.mean(df['standard_deviation_3']),np.mean(df['dissimilarity_3']),np.mean(df['correlation_3']),np.mean(df['mean_4']),np.mean(df['median_4']),np.mean(df['sum_4']),np.mean(df['standard_deviation_4']),np.mean(df['mean_1'])]
    


# In[ ]:


class DatasetFeatures():
    def __init__(self,fileNames,firstIndex,lastIndex,patchNumber,patch_width,patch_length,isup_grade):
        self.fileNames=fileNames
        self.firstIndex=firstIndex
        self.lastIndex=lastIndex
        self.patchNumber=patchNumber
        self.patch_width=patch_width
        self.patch_length=patch_length
        self.Y=isup_grade
        self.index=0
    def extract_dataset_features(self):
        dataset_features=[]
        for i in range(self.firstIndex,self.lastIndex):
            p=Patch(self.fileNames[i],self.patch_width,self.patch_length)
            im=p.extract_patch()
            if im is not None:
                ImageF=ImageFeatures(self.patchNumber,self.fileNames[i],self.patch_width,self.patch_length)
                dataset_features.append(ImageF.extract_mean_image_features())
                self.index=self.index+1
                print("image number "+str(self.index)+" is processed")
                #dataset_features.append(self.Y)
            else:
                dataset_features.append([None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None])
                print("image number "+str(self.index)+" is nontype")
        
        return dataset_features
    def extract_images_dataset_features_dataframe(self):
        
        df = pd.DataFrame(self.extract_dataset_features(), columns = ['mean_1','median_1','sum_1','standard_deviation_1','dissimilarity_1','correlation_1','mean_2','median_2','sum_2','standard_deviation_2','dissimilarity_2','correlation_2','mean_3','median_3','sum_3','standard_deviation_3','dissimilarity_3','correlation_3','mean_4','median_4','sum_4','standard_deviation_4'])
        df["isup_grade"]=self.Y[self.firstIndex:self.lastIndex]
        return df
        


# In[ ]:





# In[ ]:





# In[ ]:


df1 = pd.read_csv(r'../input/data-prostatcancer/data.csv')


# In[ ]:


df1


# In[ ]:


df1["isup_grade"]


# In[ ]:





# In[ ]:


DATA = '../input/prostate-cancer-grade-assessment/test_images'
data=[]
if os.path.exists(DATA):
    file_names_picture=file_names(DATA)
    isup_grade=np.zeros(len(file_names_pictures),dtype=np.int8)
    imDF=pd.DataFrame()
    for i in range(0,len(file_names_pictures)):
        try:
            imF=ImageFeatures(5,file_names_pictures[i],400,400,i)
            p,idx=imF.extract_image_features_dataframe()
            imDF = pd.concat([imDF,p],ignore_index=True)
            print("image number "+str(idx)+" is processed")
        except TypeError:
            continue
        except IndexError:
            continue
    arr=df1.to_numpy()
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(arr[:,0:21])
    knn = KNeighborsClassifier()
    knn.fit(x_scaled, arr[:,22])
    imDF=imDF.dropna(axis=0)
    arr=imDF.to_numpy()
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(arr[:,0:21])
    v=knn.predict(x_scaled)
    result_dict_test={}
    i=0
    for num in set(imDF["picture_number"]):
        # Add each name to the names_by_rank dictionary using rank as the key
        result_dict_test[num] = list(v[i:i+5])
        i=i+5
    L=[]
    for idx in result_dict_test.keys(): 
        L.append((idx,Counter(result_dict_test[idx]).most_common(1)[0][0]))
    for x in L:
        isup_grade[round(x[0])]=round(x[1])
    L1=[]
    for file in file_names_pictures: 
        L1.append(file.split("/")[-1].split(".")[-2])
    
    for n1,n2 in zip(L1,list(isup_grade)):
        data.append([n1,n2])
df = pd.DataFrame(data, columns = ['image_id', 'isup_grade']) 
df.to_csv('./submission.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




