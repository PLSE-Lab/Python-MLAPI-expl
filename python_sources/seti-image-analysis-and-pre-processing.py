#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## SETI Explanatory Analysis and Preprocessing Images
# 
# Let's start by examining images in each class, below snippet will randomly select two images from the training data and displays them.

# In[ ]:


# all classes 
classes = ["brightpixel",
            "narrowband",
            "narrowbanddrd",
            "noise",
            "squarepulsednarrowband",
            "squiggle",
            "squigglesquarepulsednarrowband"]
num_images = 2
for _class in classes:
    # start off by observing images
    path = os.path.join("../input/primary_small/train", _class)
    image_files = os.listdir(path)
    random_images = random.sample(range(0, len(image_files)-1), num_images)
    fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(12, 14), squeeze=False)
    fig.tight_layout()
    for l in range(1):
        for m in range(num_images):
            axes[l][m].imshow(cv2.imread(os.path.join(path, image_files[random_images[m]]), 0), cmap="gray")
            axes[l][m].axis("off")
            axes[l][m].set_title(_class)
# done displaying


# One can observe that despite many signals being strong with less noice, we still have some images with high level of noice and even some signals hiding along noice. So we'll do some preprocessing to the images to see if that can extract the features we're looking for.
# 
# Let's pick a random image from "narrowband" class and perform some pre-processing on them and see how it turns out. Before that let's construct a helper method that takes a numpy array and displays it as a image.

# In[ ]:


def display(image):
    fig = plt.figure(figsize=(9, 11))
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()


# In[ ]:


_random = random.choice(os.listdir(os.path.join("../input/primary_small/train/narrowband")))
# lets read the image and display it
narrowband = cv2.imread(os.path.join("../input/primary_small/train/narrowband", _random))
display(narrowband)


# Notice how the image is displayed in weird colorspace? This is because openCV reads an image in "BGR" format when compared to matplotlib expectation of "RGB". Let's make the image grayscale and start applying preprocessing.

# In[ ]:


# convert from BGR to Grayscale
narrowband = cv2.cvtColor(narrowband, cv2.COLOR_BGR2GRAY)
display(narrowband)


# After inspecting from the below features, we can clearly see that there are outliers in our images which are far away from the mean

# In[ ]:


# now let's extract some features from the image
low = np.min(narrowband)
high = np.max(narrowband)
mean = np.mean(narrowband)
std = np.std(narrowband)
variance = np.var(narrowband)
# print
print("Min: {}".format(low))
print("Max: {}".format(high))
print("Mean: {}".format(mean))
print("Standard Deviation: {}".format(std))
print("Variance: {}".format(variance))


# Let's make an assumption that every pixel which is 3.5 times standard deviations away from mean value is an outlier and clip it's value to the 3.5 * std

# In[ ]:


clipped = np.clip(narrowband, mean-3.5*std, mean+3.5*std)
# print
print("Min: {}".format(np.min(clipped)))
print("Max: {}".format(np.max(clipped)))
display(clipped)


# We can see that the maximum pixel intensity reduced by clipping the image which makes much smoother image. Now let's perform some image arithmetic methods like applying Gaussian Blurr, Morphing and Gradient selection

# In[ ]:


# Gaussian blurr
gaussian = cv2.GaussianBlur(narrowband, (3, 3), 1)
print("Min: {}".format(np.min(gaussian)))
print("Max: {}".format(np.max(gaussian)))
print("Mean: {}".format(np.mean(gaussian)))
display(gaussian)


# The Gaussian Blurr increased the min and reduced the max pixel intensities further for the image packing the pixels closer to each other that smoothens the image. Unofrtunately for images that have weak signals Gaussian blurr actually blurrs part of the signal itself which we may not help, so it's upto us on if we want to use the Gaussian blurr or not(I decided to use it for further downstream transformations). 
# 
# Let's move on to morphological operations on the narrowband image.

# In[ ]:


# lets do a morphological closing on the clipped image which is dilation + erosion
morphed = cv2.morphologyEx(gaussian, cv2.MORPH_CLOSE, kernel=np.ones((3, 3), dtype=np.float32))
display(morphed)


# This morphological closing clearly helped some of the low intensity signal pixels to expose more and brighten the signal compared to background noice. 
# 
# Now, lets observe all the images from all classes (scroll above to 3rd cell) and we can notice that each signal has some gradient along the horizontal axis and we'll make use of that gradient by applying Sobel operations to the morphed image

# In[ ]:


# we'll start by applying sobel edge detection along x-axis
sobelx = cv2.Sobel(morphed, cv2.CV_64F, 1, 0, 2)
display(sobelx)


# In[ ]:


# let's apply sobel ege detection along y-axis
sobely = cv2.Sobel(morphed, cv2.CV_64F, 0, 1, 2)
display(sobely)


# Now, let's blend both images with more horizontal weight as we know all signals have a gradient along horizontal axis

# In[ ]:


blended = cv2.addWeighted(src1=sobelx, alpha=0.7, src2=sobely, beta=0.3, gamma=0)
display(blended)


# The above image is so far the smoothest image and it clearly distinguishes the signal from background which will save a lot of time later when classifying them via a Convolutional Neural Network.
# 
# Finally let's apply same set of transformations to couple of images in each class and see if the results are uniform. Before that let's copy all the above transformations into a method.

# In[ ]:


def process_image(image):
    # grayscale conversion
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # clip intensities
    mean = np.mean(image)
    std = np.std(image)
    image = np.clip(image, mean-3.5*std, mean+3.5*std)
    # morph close 
    morphed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel=np.ones((3, 3), dtype=np.float32))
    # gradient in both directions
    sobelx = cv2.Sobel(morphed, cv2.CV_64F, 1, 0, 2)
    sobely = cv2.Sobel(morphed, cv2.CV_64F, 0, 1, 2)
    # blend 
    blended = cv2.addWeighted(src1=sobelx, alpha=0.7, src2=sobely, beta=0.3, gamma=0)
    return blended


# In[ ]:


for _class in classes:
    # start off by observing images
    path = os.path.join("../input/primary_small/train", _class)
    image_files = os.listdir(path)
    random_images = random.sample(range(0, len(image_files)-1), num_images)
    fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(11, 12), squeeze=False)
    fig.tight_layout()
    for l in range(1):
        for m in range(num_images):
            axes[l][m].imshow(process_image(cv2.imread(os.path.join(path, image_files[random_images[m]]))), cmap="gray")
            axes[l][m].axis("off")
            axes[l][m].set_title(_class)
# done displaying


# As you see that the above pre-processign algorithm did really good in highlighting the signal in each class,even for the classes that have some vertical gradient in them such as Squiggle and Squigglesquarepulsednarrowband by preserving their shape.
# 
# We can pre-process the images further by applying Image PCA, Dialtion and other image arithmetic techniques, but these should be sufficient to feed the images to any Machine Learning/Deep Learning algorithms for successful classification.
