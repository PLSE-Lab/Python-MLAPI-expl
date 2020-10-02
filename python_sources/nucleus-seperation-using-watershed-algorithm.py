#!/usr/bin/env python
# coding: utf-8

# **Initially applying Image segmentation for one Image**

# In[6]:


import numpy as np 
import pandas as pd
import os
import pathlib
import imageio
import cv2
import skimage
print(os.listdir("../input"))
#GLobalising Training data and creating a single image path
train_paths = pathlib.Path('../input/stage1_train').glob('*/images/*.png')
train_sorted = sorted([x for x in train_paths])
im_path = train_sorted[45]
im = cv2.imread(str(im_path))#Read Images
#im


# In[7]:


#First we will analyse one image 
#Image ID
im_id = im_path.parts[-3]
im_id


# In[10]:


#Converting image to grayscale for better analysis
im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
im1=im


# In[11]:


#Plotting data for comparision of grayscale versus original image
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
plt.subplot(121)
plt.imshow(im)
plt.axis('off')
plt.title('Original Image')
plt.subplot(122)
plt.imshow(im_gray, cmap='gray')
plt.axis('off')
plt.title('Grayscale Image')
plt.tight_layout()
plt.show()


# In[13]:


#In image processing gaussioan blur is applied to reduce the noise in the image 
#Gaussian blur is neccesary to be applied before watershedding
im_blur=cv2.GaussianBlur(im_gray,(5,5),0)
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
plt.subplot(121)
plt.imshow(im)
plt.axis('off')
plt.title('Original Image')
plt.subplot(122)
plt.imshow(im_blur, cmap='gray')
plt.axis('off')
plt.title('Blurred Grayscale Image')
plt.tight_layout()
plt.show()


# 
# 

# **APPLYING IMAGE SEGMENTATION USING WATERSHED**

# In[14]:


#Using Watershed
ret,th = cv2.threshold(im_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(th,cmap='gray')
plt.axis("off")
plt.show()


# In[15]:


# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel, iterations = 2)
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(opening,cmap='gray')
plt.axis("off")
plt.show()


# In[16]:


# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.005*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(markers,cmap='jet')
plt.axis("off")
plt.show()


# In[17]:


markers = cv2.watershed(im1,markers)
im1[markers == -1] = [255,0,0]
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(im1,cmap='gray')
plt.axis("off")
plt.show()


# In[18]:


fig=plt.figure(figsize=(20, 20), dpi= 80, facecolor='w', edgecolor='k')
plt.axis("off")
plt.subplot(121)
plt.imshow(im1)
plt.axis("off")
plt.subplot(122)
plt.imshow(markers,cmap='gray')
plt.axis("off")
plt.show()



# **Calculating Total Number of Masks**

# In[19]:


mask = np.where(markers > sure_fg, 1, 0)
# Make sure the larger portion of the mask is considered background
if np.sum(mask==0) < np.sum(mask==1):
    mask = np.where(mask, 0, 1)


# In[20]:


from scipy import ndimage
labels, nlabels = ndimage.label(mask)
# Regenerate the labels
label_arrays = []
for label_num in range(1, nlabels+1):
    label_mask = np.where(labels == label_num, 1, 0)
    label_arrays.append(label_mask)
    
print('There are {} separate components / objects detected.'.format(nlabels))


# In[21]:


for label_ind, label_coords in enumerate(ndimage.find_objects(labels)):
    cell = markers[label_coords]
    
    # Check if the label size is too small
    if np.product(cell.shape) < 10: 
        #print('Label {} is too small! Setting to 0.'.format(label_ind))
        mask = np.where(labels==label_ind+1, 0, mask)

# Regenerate the labels
labels, nlabels = ndimage.label(mask)

label_arrays = []
for label_num in range(1, nlabels+1):
    label_mask = np.where(labels == label_num, 1, 0)
    label_arrays.append(label_mask)
    
print('There are now {} separate components / objects detected.'.format(nlabels))


# In[22]:


#RLE Encoding Function
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
    return " ".join([str(i) for i in run_lengths])

print('RLE Encoding for the current mask is: {}'.format(rle_encoding(label_mask)))


# **Now for Multiple Images **

# In[ ]:


#Writing a function to carry out all processing given above for the whole test dataset Images
import pandas as pd


def analyze_image(im_path):
    '''
    Take an image_path (pathlib.Path object), preprocess and label it, extract the RLE strings 
    and dump it into a Pandas DataFrame.
    '''
    # Read in data and convert to grayscale
    im_id = im_path.parts[-3]
    im = cv2.imread(str(im_path))
    #COnverting to grayscale
    im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #im1 = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im1=im
    
    im_blur=cv2.GaussianBlur(im_gray,(5,5),0)
    import matplotlib.pyplot as plt

    #plt.figure(figsize=(8,8))

    #plt.subplot(121)
    #plt.imshow(im)
    #plt.axis('off')
    #plt.title('Original Image')

    #plt.subplot(122)
    #plt.imshow(im_blur, cmap='gray')
    #plt.axis('off')
    #plt.title('Blurred Grayscale Image')

    #plt.tight_layout()
    #plt.show()
    
    ret,th = cv2.threshold(im_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #plt.figure(figsize=(10,10))
    #plt.subplot(121)
    #plt.imshow(th,cmap='gray')
    #plt.axis("off")
    #plt.show()
    
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel, iterations = 2)
    #plt.figure(figsize=(10,10))
    #plt.subplot(121)
    #plt.imshow(opening,cmap='gray')
    #plt.axis("off")
    #plt.show()
    
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.005*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    #plt.figure(figsize=(10,10))
    #plt.subplot(121)
    #plt.imshow(markers,cmap='jet')
    #plt.axis("off")
    #plt.show()
    markers = cv2.watershed(im1,markers)
    im1[markers == -1] = [255,0,0]
    #fig=plt.figure(figsize=(20, 20), dpi= 80, facecolor='w', edgecolor='k')
    #plt.axis("off")
    #plt.subplot(131)
    #plt.imshow(im1)
    #plt.axis("off")
    #plt.subplot(132)
    #plt.imshow(markers,cmap='gray')
    #plt.axis("off")
    #plt.show()
    #plt.subplot(133)
    #plt.imshow(im_gray,cmap='gray')
    #plt.axis("off")
    #plt.show()
    mask = np.where(markers > sure_fg, 1, 0)

    # Make sure the larger portion of the mask is considered background
    if np.sum(mask==0) < np.sum(mask==1):
        mask = np.where(mask, 0, 1)
        
    from scipy import ndimage
    labels, nlabels = ndimage.label(mask)
    #print('There are {} separate components / objects detected.'.format(nlabels))
    
    for label_ind, label_coords in enumerate(ndimage.find_objects(labels)):
         cell = markers[label_coords]
    
         # Check if the label size is too small
         if np.product(cell.shape) < 10: 
             #print('Label {} is too small! Setting to 0.'.format(label_ind))
             mask = np.where(labels==label_ind+1, 0, mask)

    # Regenerate the labels
    labels, nlabels = ndimage.label(mask)

    label_arrays = []
    for label_num in range(1, nlabels+1):
        label_mask = np.where(labels == label_num, 1, 0)
        label_arrays.append(label_mask)
    
    #print('There are now {} separate components / objects detected.'.format(nlabels))
    
    def rle_encoding(x):
        dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
        run_lengths = []
        prev = -2
        for b in dots:
            if (b>prev+1): run_lengths.extend((b+1, 0))
            run_lengths[-1] += 1
            prev = b
        return " ".join([str(i) for i in run_lengths])

    #print('RLE Encoding for the current mask is: {}'.format(rle_encoding(label_mask)))
    
    # Loop through labels and add each to a DataFrame
    im_df = pd.DataFrame()
    for label_num in range(1, nlabels+1):
        label_mask = np.where(labels == label_num, 1, 0)
        if label_mask.flatten().sum() > 10:
            rle = rle_encoding(label_mask)
            s = pd.Series({'ImageId': im_id, 'EncodedPixels': rle})
            im_df = im_df.append(s, ignore_index=True)
    
    return im_df


def analyze_list_of_images(im_path_list):
    '''
    Takes a list of image paths (pathlib.Path objects), analyzes each,
    and returns a submission-ready DataFrame.'''
    all_df = pd.DataFrame()
    for im_path in im_path_list:
        im_df = analyze_image(im_path)
        all_df = all_df.append(im_df, ignore_index=True)
    
    return all_df


# In[ ]:


testing = pathlib.Path('../input/stage2_test_final/').glob('*/images/*.png')
df = analyze_list_of_images(list(testing))
df.to_csv('submission.csv', index=None)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




