#!/usr/bin/env python
# coding: utf-8

# ## <center> <span style="color:#aaa"> Prostate cANcer graDe Assessment (PANDA) Challenge </span></center>
# 
# 
# # <center> Looking into the Effect of  </center>
# # <center>  <font size="6"> <span style="color:#ffa7b6"> IMAGE ENHANCEMENT </span> </font></center>   

# ##  <span style="color:pink">  In this notebook: </span>
# 
# ### 1. First, applied a contrast enhancement filter to remove gray areas in Radboud samples.
# 
# ### 2. Experimented with other filters:
#  
#    a. Histogram equalization: did not work!
# 
#    b. Unsharp masking: results are good!
# 
# ### <center> I will add more filters if Kagglers like this notebook. Please upvote if it is useful!

# ##  <span style="color:pink">  Import Packages </span>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
from tqdm.notebook import tqdm
import random
import cv2
plt.rcParams['figure.figsize'] = [15,8]


# ## <span style="color:pink">  Data I/O

# In[ ]:


DATA_DIR = '/kaggle/input/prostate-cancer-grade-assessment/'
data = pd.read_csv(DATA_DIR + 'train.csv')
data_karolinska = data[data.data_provider=="karolinska"].reset_index().drop(columns=['index'])
data_radboud = data[data.data_provider=="radboud"].reset_index().drop(columns=['index'])


# ##  <span style="color:pink">  Utilities

# In[ ]:


def show_grid(dataframe):
    N = len(dataframe)
    for i in range(3):
        for j in range(3):
            plt.subplot(3,3,j+3*i+1)
            img = skimage.io.MultiImage(DATA_DIR + 'train_images/' + dataframe.image_id[int(N*random.random())] + '.tiff')
            plt.imshow(img[2])
            plt.axis('off')


# ##  <span style="color:pink">  Let's Look at Radboud and Karolinska Samples </span>
# 
# We randomly sample ```Radboud``` and ```Karolinska``` data for visualization.
# 
# - If you look carefully, you will notice that in the ```Radboud``` samples, there is a distinct gray thick boundary around the tissue border due to transparent slides used in the experiment. 
# 
# - This is not visible as predominently in the ```Karolinska``` data.
# 
# ### <span style="color:orange"> This region can cause issue in segmenting forground from the background! </span>

# In[ ]:


print('Radboud Samples')
show_grid(data_radboud)


# In[ ]:


print('Karolinska Samples')
show_grid(data_karolinska)


# ---
# ##  Filter 1: <span style="color:pink">  Removing Gray Borders in Radboud Samples using </span> <span style="color:orange">Contrast Enhancement </span>
# ---
# 
# For contrast enhancement we will use OpenCV function ```addWeighted``` for this purpose.
# 
# ```
# img_enhanced = cv2.addWeighted( img, contrast, img, 0, brightness)
# ```
# 
# The same enhancement can also be safely applied to ```Karolinska``` samples, as I observed from my limited experiments. See the following figures for comparison.
# 
# #### Reference: https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
# 

# In[ ]:


def enhance_image(image, contrast=1, brightness=15):
    """
    Enhance constrast and brightness of images
    """
    img_enhanced = cv2.addWeighted(image, contrast, image, 0, brightness)
    return img_enhanced

factor = 4 
channel = 1# using 2nd channel
img = skimage.io.MultiImage(DATA_DIR + 'train_images/' + data_radboud.image_id[50] + '.tiff')[channel][factor*0:factor*300,factor*0:factor*300]
enhnaced_img = enhance_image(img)
img_concat = np.concatenate((img, enhnaced_img), axis=1)
plt.imshow(img_concat);plt.axis('off');
plt.title('A RADBOUD SAMPLE \n\n [Left] Before Image Enhancement, [Right] After Image Enhancement')
plt.show();

img = skimage.io.MultiImage(DATA_DIR + 'train_images/' + data_karolinska.image_id[50] + '.tiff')[channel][factor*0:factor*1300,factor*0:factor*1300]
enhnaced_img = enhance_image(img)
img_concat = np.concatenate((img, enhnaced_img), axis=1)
plt.imshow(img_concat);plt.axis('off');
plt.title('A KAROLINSKA SAMPLE \n\n [Left] Before Image Enhancement, [Right] After Image Enhancement')
plt.show();


# #  <span style="color:#aaa">  Further Enhancements</span>

# ---
# ## Filter 2: <span style="color:pink"> Histogram Equalization </span>
# ---
# 
# Note: We are applying histogram equalization to already contrast enhanced images.
# 
# <span style="color:orange"> The result is not good. So we cannot use it for this competitions. Please let me know if you made it work, I will be happy to update here! </span>
# 
# We can use opencv ```equalizeHist``` function.

# In[ ]:


def RGB_histogram_equalization(img):
    """ Histogram Equalization of 3-Channel images"""
    equalized_image1 = cv2.equalizeHist(img[:,:,0])
    equalized_image2 = cv2.equalizeHist(img[:,:,1])
    equalized_image3 = cv2.equalizeHist(img[:,:,2])
    return cv2.merge((equalized_image1,equalized_image2,equalized_image3))

factor = 4 
channel = 1# using 2nd channel
img = skimage.io.MultiImage(DATA_DIR + 'train_images/' + data_radboud.image_id[50] + '.tiff')[channel][factor*0:factor*300,factor*0:factor*300]
img = enhance_image(img)
equalized_image = RGB_histogram_equalization(img)
img_concat = np.concatenate((img, equalized_image),axis=1)
plt.imshow(img_concat);plt.axis('off');
plt.title('A RADBOUD SAMPLE \n\n [Left] Before Histogram Equalization, [Right] After Histogram Equalization')
plt.show();

img = skimage.io.MultiImage(DATA_DIR + 'train_images/' + data_karolinska.image_id[50] + '.tiff')[1][factor*0:factor*1300,factor*0:factor*1300]
img = enhance_image(img)
equalized_image = RGB_histogram_equalization(img)
img_concat = np.concatenate((img, equalized_image),axis=1)
plt.imshow(img_concat);plt.axis('off');
plt.title('A KAROLINSKA SAMPLE \n\n [Left] Before Histogram Equalization, [Right] After Histogram Equalization')
plt.show();


# ---
# ## Filter 3: <span style="color:pink"> Unsharp Masking </span>
# ---
# 
# Unsharp masking enhances the high frequency details. This can be implemented using a gaussian filter and subtracting it from the original image.
# 
# ### <span style="color:orange"> You can see that this filter enhance the details! </span>

# In[ ]:


def unsharp_masking(img):
    """ Unsharp masking of an RGB image"""
    img_gaussian = cv2.GaussianBlur(img, (21,21), 10.0)
    return cv2.addWeighted(img, 1.8, img_gaussian, -0.8, 0, img)

factor = 4
img = skimage.io.MultiImage(DATA_DIR + 'train_images/' + data_radboud.image_id[500] + '.tiff')[1][factor*150:factor*250,factor*150:factor*250]
img = enhance_image(img)
unsharp_image = unsharp_masking(img.copy())
img_concat = np.concatenate((img, unsharp_image),axis=1)
plt.imshow(img_concat);plt.axis('off');
plt.title('A RADBOUD SAMPLE \n\n [Left] Before Unsharp Masking, [Right] After Unsharp Masking')
plt.show();

img = skimage.io.MultiImage(DATA_DIR + 'train_images/' + data_karolinska.image_id[500] + '.tiff')[1][factor*600:factor*1000,factor*600:factor*1000]
img = enhance_image(img)
unsharp_image = unsharp_masking(img.copy())
img_concat = np.concatenate((img, unsharp_image),axis=1)
plt.imshow(img_concat);plt.axis('off');
plt.title('A KAROLINSKA SAMPLE \n\n [Left] Before Unsharp Masking, [Right] Unsharp Masking')
plt.show();


# # <span style="color:orange"> More filters coming if Kagglers show interest!
#   
