#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Analysis
# 
# Firstly we will display the original images and from there, we will develop a strategy to segment the letters from the background, without using machine learning / deep learning techniques.

# In[ ]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# For reproducibility purposes
np.random.seed(42)

def get_random_images(images_path, n=5):
    image_collection = []
    
    for i in np.random.permutation(len(images_path))[:n]:
        image = cv2.imread(images_path[i])
        image_collection.append(image)
        
    return image_collection


# We obtain the path of all the original images
base_path = "../input/Captcha Dataset/original"
images_path = list(map(lambda x: os.path.join(base_path, x), os.listdir(base_path)))

random_images = get_random_images(images_path, n=9)

plt.figure(figsize=(9, 3))

for i, image in enumerate(random_images):
    plt.subplot(3, 3, i + 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

plt.suptitle('Random Captchas')
plt.tight_layout()
plt.show()


# ### Histograms
# 
# If we want to extract the letters from the background, it is necesary to find a threshold level that will be applied to the image to binarize it. The main problem that we face is that in not all the images the letters are of the same color. So in order to find that level, we will need to explore the images using the histograms.

# In[ ]:


def plot_histogram(image, colorspace):
    chans = cv2.split(image)
    
    plt.title(f"{colorspace} Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    
    for (chan_data, channel_name) in zip(chans, list(colorspace)):
        if colorspace == "BGR":
            hist = cv2.calcHist([chan_data], [0], None, [256], [0, 256])
            plt.plot(hist, label=channel_name, color=channel_name.lower())
        elif colorspace == "HSV":
            if channel_name == "H":
                hist = cv2.calcHist([chan_data], [0], None, [180], [0, 256])
            else:
                hist = cv2.calcHist([chan_data], [0], None, [256], [0, 256])
                
            plt.plot(hist, label=channel_name)
        
    
    plt.xlim([0, 256])
    plt.legend(loc="upper right")


# In[ ]:


plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.imshow(cv2.cvtColor(random_images[0], cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(3, 1, 2)
plot_histogram(image, "BGR")
plt.subplot(3, 1, 3)
plot_histogram(cv2.cvtColor(random_images[0], cv2.COLOR_BGR2HSV), "HSV")
plt.tight_layout()
plt.show()


# Usually while working with colors in image processing, is better to use the HSV because it gives the colors a more "logical" way of grouping similar colors. After looking at a couple of images, we arrive to the conclusion that the letters are usually darker than its background. So the channel that will be of most use, in the HSV space, will be the value channel; the lower the value, the darker the color is.
# 
# ### Channels plotting

# In[ ]:


random_images = get_random_images(images_path)

for image in random_images:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    channels = cv2.split(image)
    
    plt.figure(figsize=(20, 10))
    
    for index, channel in enumerate(channels):
        plt.subplot(1, 3, 1 + index)
        plt.imshow(channel, cmap="gray")
        
    plt.show()


# The first column represents the Hue channel, the second represents the Saturation channel and the third one represents the Value channel. Out of the three channels, the value channel seems to be the most useful to separete between letters and background; because as it was said before, the letters are usually darker than the rest of the image.

# ## Image Processing
# 
# We will be using the value channel to process the images. The general idea behind the processing is to get a binary image in order to find the contours. Hopefuly the contours will be the ones of the letters.
# 
# ### Thresholding
# 
# To find an adecuate threshold level, we will be showing the image thresholded at different levels and seek for the one that yields better results.

# In[ ]:


def threshold_image(image, level):
    # image should be in BGR space
    
    # Conversion and extraction of `value` channel
    image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)[:, :, 2]
    
    # Thresholding at specified level
    ret, thresh_img = cv2.threshold(image, level, 255, cv2.THRESH_BINARY_INV)
    
    return thresh_img


# Plotting at different threshold levels
levels = np.linspace(0, 255, 10).astype('uint8')

plt.figure(figsize=(10, 5))

for i, level in enumerate(levels):
    plt.subplot(5, 2, i + 1)
    
    plt.title(f'Threshold = {level}')    
    plt.imshow(threshold_image(random_images[0], level), cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()


# It seems that the best results lie between a threshold level of 85 and 113. Let's plot the random images thresholded at a level of 99, which is the intermediate value between 113 and 85.

# In[ ]:


plt.figure(figsize=(5, 5))

for i, image in enumerate(random_images):
    thresholded = threshold_image(image, 99)
    
    plt.subplot(5, 1, i + 1)
    plt.imshow(thresholded, cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()


# Bingo! We can clearly separate the letters from the background. Now we need to find their contours to be able to extract them from the image.
# 
# ## Contours Finding
# 
# So far the pipepline to segmentate the letters from the background has been:
# 
# 1. Load the image
# 2. Convert from BGR to HSV space and extract the Value channel
# 3. Apply thresholding to the value channel with a threshold level of 99
# 
# The last step before classifying the letters is to detect the contours of the letters to be able to extract them.
# 
# OpenCV comes with a handy function that detect the contours of a grayscale / binarized image: `cv2.findContours()`. Let's see it into action:

# In[ ]:


threshold_level = 99

# 1: Load the images
for image in random_images:
    # 2, 3: Convert to HSV, obtain the value channel and threshold it
    binarized = threshold_image(image, threshold_level)
    
    # 4: Obtain the contours over the binarized image
    cnts, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4.1: Create a copy of the binary image to draw over it
    gray = cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)
    
    # 4.2: Contour drawing
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 1)
        
    # 5: Visualization
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(gray)
    plt.axis('off')
    
    plt.show()


# Almost perfect! But there is still some room for improvement.
# 
# First, we need to filter the contours to avoid enclosing noise as a letter. This problem can be easily solved using the area of the contour. If the area of the contour is below a certain threshold level, ignore it.
# 
# Second, sometimes there is an opening in the letter, splitting it into two and at the moment of detecting the contours. After applying the area threshold level, we still have it and this will impact the classification of the letter. This can be solved by applying a closing morphological operation over the binarized image.
# 
# The following plots try to address these problems.

# In[ ]:


threshold_level = 99
area_threshold = 100

# NEW: Kernel for the closing operation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# 1: Load the images
for image in random_images:
    # 2, 3: Convert to HSV, obtain the value channel and threshold it
    binarized = threshold_image(image, threshold_level)
    
    # NEW: Apply a closing operation
    binarized = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)
    
    # 4: Obtain the contours over the binarized image
    cnts, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4.1: Create a copy of the binary image to draw over it
    gray = cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)
    
    # 4.2: Contour drawing
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        
        # NEW: Apply an area threshold
        if w * h > area_threshold:
            cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 1)
        
    # 5: Visualization
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(gray)
    plt.axis('off')
    
    plt.show()


# ## Letter Extraction
# 
# Finally we need to extract the letters from the image. In this scenario the best idea would be to create a function which input is the original image and its output an array containing the binarized letters for further processing.

# In[ ]:


def get_letters(image, threshold_level=99, area_threshold=100):
    letters_bin = []
    rects = []
    
    # 2, 3: Convert to HSV, obtain the value channel and threshold it
    binarized = threshold_image(image, threshold_level)
    
    # 3.1: Apply a closing operation
    binarized = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)
    
    # 4: Obtain the contours over the binarized image
    cnts, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4.2: Create a copy of the binary image to draw over it
    gray = cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)
    
    # Creation of bounding rects
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        
        # 4.4: Apply an area threshold
        if w * h > area_threshold:
            rects.append((x, y, w, h))
    
    # Sort by position along the x axis. This is necessary because the order does matter when
    # typing the answer to a captcha
    rects = sorted(rects, key=lambda x: x[0])
    
    for (x, y, w, h) in rects:
        # 4.5: Extract the letter and append it into an array
        letters_bin.append(binarized[y:(y + h), x:(x + w)])
            
            
    return letters_bin


# In[ ]:


image = get_random_images(images_path, n=1)[0]

plt.figure()
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

letters = get_letters(image)

plt.figure()
for i, letter in enumerate(letters):
    plt.subplot(len(letters), 1, i + 1)
    plt.imshow(letter, cmap='gray')
    plt.axis('off')
    
plt.tight_layout()
plt.show()

