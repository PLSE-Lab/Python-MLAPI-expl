#!/usr/bin/env python
# coding: utf-8

# # 1. Loading and visualizing the images
# ### Airbus Ship Detection Challenge - A quick overview for computer vision noobs
# 
# &nbsp;
# 
# 
# Hi, and welcome! This is the first kernel of the series `Airbus Ship Detection Challenge - A quick overview for computer vision noobs.` 
# In this short kernel (~50 lines) we will find, load, resize and display the jpg satellital images using [PIL](https://pillow.readthedocs.io/en/latest/), we will map them to a rgb matrix representation using numpy [array](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html) and we will display those rgb matrices on a nice grid using matplotlib's [imshow](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html) and [subplots](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html)).
# 
# 
# 
# The full series consist of the following notebooks:
# 1. *[Loading and visualizing the images](https://www.kaggle.com/julian3833/1-loading-and-visualizing-the-images)*
# 2. [Understanding and plotting rle bounding boxes](https://www.kaggle.com/julian3833/2-understanding-and-plotting-rle-bounding-boxes) 
# 3. [Basic exploratory analysis](https://www.kaggle.com/julian3833/3-basic-exploratory-analysis)
# 4. [Exploring public models](https://www.kaggle.com/julian3833/4-exploring-models-shared-by-the-community)
# 5. [1.0 submission: submitting the test file](https://www.kaggle.com/julian3833/5-1-0-submission-submitting-the-test-file)
# 
# This project aims to get some good understanding about the specific topic (image segmentation), including going over the dataset, learning common approaches and understanding the best models proposed by the community from a technical and theoretical point of view. The ideal reader is a data scientist noob with some general knowledge about deep learning.
# 
# This is an ongoing project, so expect more notebooks to be added to the series soon. Actually, we are currently working on the following ones:
# * Understanding and exploiting the data leak
# * A quick overview of image segmentation domain
# * Jumping into Pytorch
# * Understanding U-net
# * Proposing a simple improvement to U-net model

# <a id='one'></a>
# ## 1. Finding the images

# Let's start with a quick glance of the `input` directory (remember to refer to the challenge's [Data tab](https://www.kaggle.com/c/airbus-ship-detection/data)  for more information about the dataset). 
# There are 3 csvs and 2 directories:

# In[ ]:


ls ../input/


# The directories contain roughly 100,000 and 85,000 images each. In this kernel, we will focus only on them, leaving the analysis of the csvs content for [another](https://https://www.kaggle.com/julian3833/2-understanding-and-plotting-rle-bounding-boxes) kernel:

# In[ ]:


import os
train = os.listdir("../input/train")
test = os.listdir("../input/test")
print(f"Train files: {len(train)}. ---> {train[:3]}")
print(f"Test files :  {len(test)}. ---> {test[:3]}")


# <a id='two'></a>
# ## 2. Quick image display with [PIL](https://pillow.readthedocs.io/en/latest/)

# The first thing we can do is to open and display some of the images from whitin Jupyter notebook itself. Using [PIL](https://pillow.readthedocs.io/en/latest/) is a good idea: it will allow us to load, resize and display a jpg image in one readable, pythonic line of code. 
# 
# Let's start opening an example jpg with [PIL.Image.open()](https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#open-rotate-and-display-an-image-using-the-default-viewer) function. The image is automatically displayed by Jupyter:

# In[ ]:


import PIL # We will import the packages at "use-time (just for this kernel)

PIL.Image.open("../input/train/000c34352.jpg")


# In[ ]:


PIL.Image.open("../input/train/000c34352.jpg").size


# <a id='three'></a>
# ## 3. Resizing images

# In read-only mode it's not noticiable, but running the previous cell takes quite some time (>5 seconds): most of that time is spended by PIL and Jupyter trying to render a $(768 \times 768)$ jpg.  Luckily,  we can overcome this by just resizing the image with the PIL Image's [resize()](https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.resize) method. 
# 
# Check the cell below: it's much faster, isn't it?

# In[ ]:


PIL.Image.open('../input/train/000c34352.jpg').resize((200, 200))


# <a id='four'></a>
# ## 4. JPG2<span style='color:red'>R</span><span style='color:green'>G</span><span style='color:blue'>B</span> with [np.array()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html)

# Casting these PIL Image objects into a rgb matrix representation is as easy as resizing them. We just need to pass the Image to [np.array()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html) to obtain a (768, 768, 3) matrix. This matrix contains 3 integers for each of the 768$\times$768 pixels, representing the red, green and blue saturation of each point in the picture:

# In[ ]:


import numpy as np

# Taking a shrinked version of the image to avoid unnecessary computation
img = PIL.Image.open('../input/train/000c34352.jpg').resize((200, 200))

rgb_pixels = np.array(img)
rgb_pixels.shape


# In[ ]:


# Red saturation of the top-left most 2x2 square pixels
rgb_pixels[0:2, 0:2, 0]


# <a id='five'></a>
# ## 5. Visualizing rgb matrices with [plt.imshow()](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html)

# We can easily plot this three-channel rgb representation with matplotlib's [imshow()](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html):

# In[ ]:


import matplotlib.pyplot as plt
plt.imshow(rgb_pixels);


# In[ ]:


# Note that imshow can plot a 1-channel image on monochrome
plt.imshow(rgb_pixels[:, :, 1], cmap='Greys');


# In[ ]:


# And also:
plt.imshow(np.random.random(size=(10, 10)));


# <a id='six'></a>
# ## 6. Displaying more than one image in the same cell with [plt.show()](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.show.html)

# To display more than one image per cell  the easiest - but somehow insatisfactory - path we can take is to ask matplotlib kindly to flush the current figure calling [plt.show()](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.show.html) and start a new one:

# In[ ]:


plt.imshow(rgb_pixels)
plt.title("Full image"); # We can add a title with plt.title()
plt.show()

plt.imshow(rgb_pixels[0:200,0:100]) # And with can crop the image using standard python slicing
plt.title("Left half");
plt.show()

plt.imshow(rgb_pixels[0:200,100:200])
plt.title("Right half");
plt.show()


# <a id='seven'></a>
# ## 7. Displaying images compactly using [plt.subplots()](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html)

# As we mentioned, the previous display mechanism has a drawback: it doesn't exploit the screen width very well... unfortunately - and as far as we know - there is no easy solution for this with Jupyter, but using [subplots](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html) is not that complicated:

# In[ ]:


# these two variables are "the parameters" of this cell
w = 6
h = 6

# this function uses the open, resize and array functions we have seen before
load_img = lambda filename: np.array(PIL.Image.open(f"../input/train/{filename}").resize((200, 200)))

_, axes_list = plt.subplots(h, w, figsize=(2*w, 2*h)) # define a grid of (w, h)

for axes in axes_list:
    for ax in axes:
        ax.axis('off')
        img = np.random.choice(train) # take a random train filename (like 000c34352.jpg)
        ax.imshow(load_img(img)) # load and show
        ax.set_title(img)
        


# ### References
# * [Airbus ship data vizualization](https://www.kaggle.com/meaninglesslives/airbus-ship-detection-data-visualization) - a nice data visualization and exploratory data analysis kernel. We didn't copy code from there, but it helped us to quick start the project.
# 
# ### What's next?
# You can check the [next kernel](https://www.kaggle.com/julian3833/2-understanding-and-plotting-rle-bounding-boxes) of the series, where we understand the rle-encoding and plot the bounding boxes over the images.

# In[ ]:




