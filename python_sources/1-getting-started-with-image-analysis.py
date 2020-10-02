#!/usr/bin/env python
# coding: utf-8

# Author: ashukr
# Date: (6 September 2018)
# >Still working..in progress
# 
# >[Reference notebook](https://www.kaggle.com/stkbailey/teaching-notebook-for-total-imaging-newbies)
# 
# > suggestions to learn and improve always welcome

# 

# In[ ]:


import pathlib
# this module has class for file-system with semantics appropiate for different operating system
import imageio
# this module provides an easy interface to read volumes of image data
import numpy as np


# In[ ]:


train_path = pathlib.Path('../input/train/images').glob('*.png')
# train_path is a generator which can be iterated only once, the result will 
# be empty if we wish to iterate twice over the generator train_path


# In[ ]:


# for x in train_path:
#     print(x)


# In[ ]:


# for x in train_path:
#     print(x)


# In[ ]:


train_path_sorted = sorted([x for x in train_path])
# the piece of code sorts the concerned path of images


# In[ ]:


im_path = train_path_sorted[41]
# we access an image with the help of its index
# we may get out of bound error if we iterate over the train_path second time


# In[ ]:


#print(im_path.parts)


# In[ ]:


#print(im_path.parts[-1][0:-4])


# In[ ]:


type(im_path)


# In[ ]:


im = imageio.imread(str(im_path))
# the im_path which is a posixPath is first converted to string and then the corresponding image is read


# In[ ]:


im.shape
# the presence of three dimensions suggests that the colour scale is RGB


# In[ ]:


print("image original shape :{}".format(im.shape))
# helps to print the shape of the image


# In[ ]:





# In[ ]:


from skimage.color import rgb2gray
# the rgb2gray compute the illuminance of an RGB image


# In[ ]:


im_gray = rgb2gray(im)
# calculating the illuminance apparantly gives us the gray scale of the image


# In[ ]:


print("the shape of the image in the grayScale :{}".format(im_gray.shape))
# here we get two dimensional image matrix, significant for grayScale images


# In[ ]:


#import matplotlib.pyplot as plt
# plot a line, implicitly creating a subplot(111)
#plt.plot([1,2,3])
# now create a subplot which represents the top plot of a grid
# with 2 rows and 1 column. Since this subplot will overlap the
# first, the plot (and its axes) previously created, will be removed
# plt.subplot(2,1,1)
# plt.plot(range(12))
# plt.subplot(212, facecolor='y') # creates 2nd subplot with yellow background


# In[ ]:


import matplotlib.pyplot as plt
# the module to visualize the image
plt.figure(figsize = (10,4))
# we  decide the display size of the visualisation
plt.subplot(1,2,1)
# In the current figure, create and return an Axes, at position index of a (virtual) grid of nrows by ncols axes. Indexes go from 1 to nrows * ncols, incrementing in row-major order.
# If nrows, ncols and index are all less than 10, they can also be given as a single, concatenated, three-digit number.
plt.imshow(im)
#displays the image on the axis
plt.axis('off')
#turns off the axis and the labels
plt.title("the original image")

plt.subplot(122)
#if the value of parameters in the arguements are less than 10 then they can be written in the sequence
plt.imshow(im_gray,cmap = 'gray')
#for gray scale image a corresponding parameter for Cmap is passed
plt.axis('off')
plt.title("grayScaleImage")
# the display of the grayScale image
plt.tight_layout()
# tight_layout automatically adjusts subplot params so that the subplot(s) fits in to the figure area.
plt.show()


# In[ ]:


# to remove the image and the background, there is method of seperation (i am not very sure if this approach of seperation will help in any way..
# ..i am doing it cause the tutorial does it :-) ).

# the eaisest way is to find the simple descriptive statistics like mean median and mode, the other approach is to "otsu" method, which finds the bimodal 
# distribution and finds the optimal seperation value


#otsu method named after Noboyuki Otsu is a method in which the gray scale image are converted to binary image or clustering based image thresholding
#to find the optimal seperation between the image and its background


# In[ ]:


from skimage.filters import threshold_otsu
thres_val = threshold_otsu(im_gray)
# thres_val contains the threshold value for the separation of image and its background as per the otsu's algorithm
# for that particular image


# In[ ]:


mask = np.where(im_gray > thres_val,1,0)
#the np.where(,[1,0]) returns the value 1 for true and the value 0 for false


# In[ ]:


# we are considering the larger portion of he mask as the background
if np.sum(mask==0) < np.sum(mask==1):
    mask = np.where(mask,0,1)


# In[ ]:


plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
im_pixels = im_gray.flatten()
#numpy flatten returns the array collapsed in one dimension
plt.hist(im_pixels,bins=50)
plt.vlines(thres_val, 0, 100000, linestyle='--')
#the plt.vlines draws a vertical line at every X from y_min to y_max
plt.ylim([0,50000])
plt.title('Grayscale Histogram')


# In[ ]:


plt.subplot(1,2,2)
mask_for_display = np.where(mask, mask, np.nan)
# the np.nan substitutes the nan in place of all the zeros
plt.imshow(im_gray, cmap='gray')
plt.imshow(mask_for_display, cmap='rainbow', alpha=0.5)
plt.axis('off')
plt.title('Image w/ Mask')

plt.show()


# In[ ]:


#we assign a label to each component in the mask and add each label to 
#an iterable such as list

from scipy import ndimage
#the package for multi dimensional image processing
labels, nlabels = ndimage.label(mask)
#the ndimage label returns the number of features and every feature marked 
#with the labels 


# In[ ]:


label_arrays = []
for label_num in range(1, nlabels+1):
    label_mask = np.where(labels == label_num, 1, 0)
    label_arrays.append(label_mask)
print('There are {} seperate components/objects/features detected.'.format(nlabels))


# In[ ]:


#listedColourMap maps a color from a list of color, after creating an object of colourMap
from matplotlib.colors import ListedColormap
rand_cmap = ListedColormap(np.random.rand(256,3))


# In[ ]:


type(rand_cmap)


# In[ ]:


labels_for_display = np.where(labels > 0, labels, np.nan)
plt.imshow(im_gray, cmap='gray')
#plt.imshow(labels_for_display, cmap=rand_cmap)
plt.axis('off')
plt.title('Labeled Objects ({} Objects)'.format(nlabels))
plt.show()


# In[ ]:


labels_for_display = np.where(labels > 0, labels, np.nan)
plt.imshow(im_gray, cmap='gray')
#we have added the colourMap to visualize the different components
plt.imshow(labels_for_display, cmap=rand_cmap)
plt.axis('off')
plt.title('Labeled Objects ({} Objects)'.format(nlabels))
plt.show()


# In[ ]:


#now we can use the ndimage.find_objects to iterate over 
#the different objects in the image and process them individually
for label_ind, label_coords in enumerate(ndimage.find_objects(labels)):
   # print(label_ind)
    #print(label_coords)
    cell = im_gray[label_coords]
    
    # Check if the label size is too small
    #prefer np.prod over np.product as the later uses the former and may be depricated in future release
    #np.prod multiplies the elements of an array, either all of them together or the multiplication along the specefied axis
    if np.product(cell.shape) < 10: 
        print('Label {} is too small! Setting to 0.'.format(label_ind))
        mask = np.where(labels==label_ind+1, 0, mask)

# Regenerate the labels
labels, nlabels = ndimage.label(mask)
print('There are now {} separate components / objects detected.'.format(nlabels))


# In[ ]:


ig, axes = plt.subplots(1,6, figsize=(10,6))
#this block of code is getting out every component of the given image and is visualizing them 
for ii, obj_indices in enumerate(ndimage.find_objects(labels)[0:6]):
    cell = im_gray[obj_indices]
    axes[ii].imshow(cell, cmap='gray')
    axes[ii].axis('off')
    axes[ii].set_title('Label #{}\nSize: {}'.format(ii+1, cell.shape))

plt.tight_layout()
plt.show() 


# In[ ]:


#shrinking the mask so as to get the components more confidently is called mask erosion
#we can re-dilate the mask so as to retrieve the original components
#now we get an image and perform the binary opening procedure
two_cell_indices = ndimage.find_objects(labels)[1]
cell_mask = mask[two_cell_indices]
cell_mask_opened = ndimage.binary_opening(cell_mask, iterations=8)


# In[ ]:


dots = np.where(label_mask.T.flatten()==1)[0]


# In[ ]:


dots


# In[ ]:


# run_lengths = []
# prev = -2
# for b in dots:
#     print(b)
#     if (b>prev+1): 
#         print(b)
#         run_lengths.extend((b+1, 0))
#         print(run_lengths)
#     run_lengths[-1] += 1
#     prev = b


# In[ ]:


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


# In[ ]:


from skimage.color import rgb2gray
im_gray = rgb2gray(im)
thres_val = threshold_otsu(im_gray)


# In[ ]:


# now we combine together all of the function that we have created above and then process all of the images for a submission
import pandas as pd
i=0
def analyze_image(im_path):
    '''
    Take an image_path (pathlib.Path object), preprocess and label it, extract the RLE strings 
    and dump it into a Pandas DataFrame.
    '''
    print()
    # Read in data and convert to grayscale
    im_id = im_path.parts[-1][0:-4]
    im = imageio.imread(str(im_path))
    im_gray = rgb2gray(im)
    
    # Mask out background and extract connected objects
    thresh_val = threshold_otsu(im_gray)
    mask = np.where(im_gray > thresh_val, 1, 0)
    if np.sum(mask==0) < np.sum(mask==1):
        mask = np.where(mask, 0, 1)    
        labels, nlabels = ndimage.label(mask)
    labels, nlabels = ndimage.label(mask)
    
    # Loop through labels and add each to a DataFrame
    im_df = pd.DataFrame()
    for label_num in range(1, nlabels+1):
        label_mask = np.where(labels == label_num, 1, 0)
        if label_mask.flatten().sum() > 10:
            rle = rle_encoding(label_mask)
            s = pd.Series({'id': im_id, 'rle_mask': rle})
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


testing = pathlib.Path('../input/train/images').glob('*.png')
df = analyze_list_of_images(list(testing))
df.to_csv('submission.csv', index=None)


# ## the method of separating the background image with the rel image components and then extracting their encodings did not work for this task, as there are some images which have only one colour, and for suc images the otsu_separation do not work.

# In[ ]:




