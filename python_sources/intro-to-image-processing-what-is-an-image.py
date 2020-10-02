#!/usr/bin/env python
# coding: utf-8

# # About the tutorials
# ___
# Hi everyone! 
# 
# There are several tutorials out there teaching about Computer Vision, but most of them starts right from the Deep Learning part, applying Convolutional Neural Networks (CNNs) and other algorithms to classification, segmentation and other use cases. However, I've seen little to no image processing background tutorials. What an image actually is? What is equalization? What are filters and how do we use them? 
# 
# This series of tutorials is meant as a foundation for further readings in Computer Vision. Throughout it we'll make use of the popular OpenCV library.
# 
# 
# # Tutorial Index
# ___
# - [What is an image?](https://www.kaggle.com/hrmello/intro-to-image-processing-what-is-an-image/)
# - [Colorspaces](https://www.kaggle.com/hrmello/intro-to-image-processing-colorspaces)
# - [Image Enhancement Part 1](https://www.kaggle.com/hrmello/intro-to-image-processing-image-enhancement-pt-1)
# 
# # Introduction
# ___
# 
# 
# First let's talk about what images are. At their core, images are nothing but numbers, each one representing the intensity of a given color or other numeric value -- depending on the colorspace you are in. I'll talk more about it in the next tutorial. 
# 
# ## Gray images
# 
# In the image below, we see several pixels in grayscale. 
# ![Image](http://www.whydomath.org/node/wavlets/images/imagebasicslowerleft.gif) 
# 
# This is how the image looks like for our computers. Just a bunch of numbers representing the intensity of the gray color in each pixel:
# ![Pixels](https://i.ibb.co/Hp7t80B/Screenshot-from-2019-04-25-22-00-06.png)
# 
# The intensities vary from 0 to 255 -- so there are 256 possible values for each pixel to take.
# 
# ## Colored images
# Colored images are similar, except they are represented by three of those pixel intensity matrices, called channels. When an image is in RGB color system, each of those matrices represent the intensities of Red, Green and Blue colors respectively, as can be seen on the image below.
# 
# ![rgb_matrix](https://www.researchgate.net/profile/Jane_Courtney2/publication/267210444/figure/fig6/AS:295732335661069@1447519491773/A-three-dimensional-RGB-matrix-Each-layer-of-the-matrix-is-a-two-dimensional-matrix.png)
# 
# Here, too, the intensities can vary from 0 to 255 in each channel, giving us the overwhelming number of 256x256x256 = 16777216 different colors each pixel can be.
# 
# Now that we are familiar with the very basics from images, let's see this with actual code. 
# 
# 

# In[ ]:


import numpy as np # linear algebra
import cv2 # OpenCV module that handles computer vision methods
import matplotlib.pyplot as plt # make plots
import os

ANNOTATION_DIR = '../input/annotations/Annotation/'
IMAGES_DIR = '../input/images/Images/'

# list of breeds of dogs in the dataset
breed_list = os.listdir(ANNOTATION_DIR)

## set the seed for the np.random module, so we always get the same image when run this code cell
np.random.seed(35)

# since we just want one image, I'll ramdomly choose a breed and a dog from that breed
breed = np.random.choice(breed_list)
dog = np.random.choice(os.listdir(ANNOTATION_DIR + breed))

# opening one image
img = cv2.imread(IMAGES_DIR + breed + '/' + dog + '.jpg') 

# this line is necessary because cv2 reads an image in BGR format (Blue, Green, Red) by default. 
# So we will convert it to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

plt.figure(figsize = (10,10))
plt.imshow(img);


# That's our testing image. Now, this image is colored, which means that it has three color channels. Let's check them one by one and see how they differ.

# In[ ]:


f, axes = plt.subplots(1,3, figsize = (15,15))
i = 0
colors = {'0':'red', '1': 'green', '2':'blue'}
for ax in axes:
    ax.imshow(img[:,:,i], cmap = "gray")
    i+=1


# The channels show different intensities as we can see by the gray shades in each one of them. The image has lots of red(ish) colors, so the red channel has brighter spots in those areas, while green and blue channels are darker, meaning a lack of those colors.
# 
# RGB isn't the only colorspace that can be used. On the next part we will see the other usual colorspaces that can be used and the math behind the conversion to them starting from and RGB image.
# 
# ## Try it yourself!
# ___
# Use another image and try to spot the difference in intensities in each channel. Given a picture, is the channel-splitted images what you expected? For instance, orange tones will be much brighter in red channel than in blue channel. 
# 
# ## Next tutorial
# - [Colorspaces](https://www.kaggle.com/hrmello/intro-to-image-processing-colorspaces)
