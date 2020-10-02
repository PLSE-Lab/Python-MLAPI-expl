#!/usr/bin/env python
# coding: utf-8

# ## Import Python libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Used to change filepaths
from pathlib import Path
import matplotlib.pyplot as plt
import IPython
from IPython.display import display
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input"))


# In[ ]:


IPython.display.Image(filename='../input/flowers/flowers/rose/12202373204_34fb07205b.jpg') 


# <em>A rose.</em>

# The question at hand is: can a machine identify a flower as a rose or a sunflower? These flowers have different appearances,but given the variety of backgrounds, positions, and image resolutions it can be a challenge for machines to tell them apart.
# 
# Being able to identify flowers types from images is a task that ultimately would allow researchers to more quickly and effectively collect field data.

# In[ ]:


IPython.display.Image(filename='../input/flowers/flowers/sunflower/1008566138_6927679c8a.jpg') 


# 
# <em>A sunflower.</em>

# This notebook walks through loading and processing images. After loading and processing these images, they will be ready for building models that can automatically detect rose and sunflowers.

# In[ ]:


# generate test_data.
test_data=np.random.beta(1, 1, size=(120, 120, 3))

# display the test_data
plt.imshow(test_data)


# ## Opening images with PIL
# Now that we have all of our imports ready, it is time to work with some real images.
# 
# Pillow is a very flexible image loading and manipulation library. It works with many different image formats, for example, <code>.png</code>, <code>.jpg</code>, <code>.gif</code> and more. For most image data, one can work with images using the Pillow library (which is imported as <code>PIL</code>).</p>
# <p>Now we want to load an image, display it in the notebook, and print out the dimensions of the image. By dimensions, we mean the width of the image and the height of the image. These are measured in pixels. The documentation for <a href="https://pillow.readthedocs.io/en/5.1.x/reference/Image.html">Image</a> in Pillow gives a comprehensive view of what this object can do.</p>

# In[ ]:


# open the image
img = Image.open('../input/flowers/flowers/rose/14510185271_b5d75dd98e_n.jpg')
# Get the image size
img_size = img.size

print("The image size is: {}".format(img_size))

# Just having the image as the last line in the cell will display it in the notebook
img


# ## Image manipulation with PIL
# Pillow has a number of common image manipulation tasks built into the library. For example, one may want to resize an image so that the file size is smaller. Or, perhaps, convert an image to black-and-white instead of color. Operations that Pillow provides include:
# <ul>
# <li>resizing</li>
# <li>cropping</li>
# <li>rotating</li>
# <li>flipping</li>
# <li>converting to greyscale (or other <a href="https://pillow.readthedocs.io/en/5.1.x/handbook/concepts.html#concept-modes">color modes</a>)</li>
# </ul>
# 
# Often, these kinds of manipulations are part of the pipeline for turning a small number of images into more images to create training data for machine learning algorithms. This technique is called <a href="http://cs231n.stanford.edu/reports/2017/pdfs/300.pdf">data augmentation</a>, and it is a common technique for image classification.
# <p>We'll try a couple of these operations and look at the results.</p>

# In[ ]:


# Crop the image to 25, 25, 75, 75
img_cropped = img.crop([25,25,75,75])
display(img_cropped)

# rotate the image by 45 degrees
img_rotated = img.rotate(45,expand=25)
display(img_rotated)

# flip the image left to right
img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT) 
display(img_flipped)


# ## Images as arrays of data
# What is an image? So far, PIL has handled loading images and displaying them. However, if we're going to use images as data, we need to understand what that data looks like.
# 
# Most image formats have three color <a href="https://en.wikipedia.org/wiki/RGB_color_model">"channels": red, green, and blue</a> (some images also have a fourth channel called "alpha" that controls transparency). For each pixel in an image, there is a value for every channel.</p>
# ![RGB Colors](https://lh3.googleusercontent.com/JYhIGVIvnCU_txibv3XBIpI1PMcLZuijPn-_o65uzL8IyfZsl5jKI6-VeLpwHXcGxcQayPmMOhpueslHtwvJQvcdu8V_gQLal-VfAQ4SKXtDddZ9nhwnrZDYUsaGbe2Lzqngq24qGuKQMObRTgllmRTHePYhS7_qb5Xa4Kw73pWu2SeiOWkuDw0ZVlOohbWzkRlbTjHUCPHpYKEZfg7OAIPX3RdUhBl4DmjdMipr1-Kf9esqX8zwNTHSXjD3mI0uJ4XvtsK6GzBcBBMh93lO90J7uzSesXhSVN3H11wdl2thbxUm_YYvEGphHfDZyyBANQ9hmrHLb3P6fePxaOLFkroqs7qMmsYMPaaqSOHVeF1WnncH2KUY6XuNYf4fbCvywmeUP2FfOJH1UpF1zipTm699bxhCV1inG-yPdabQIrczF50O3BRDLMBE5uWM3nAcAdGxBy0Ea0XUV9-yq-UhJFz70hhl8XbH_ZSr1Y5pnsmUEdwlzu5GVIC9J4HVDMwHQuNA7SnNzmPZuE0O_x8X6c0yvy582Q5yaq5Wwf8XkGVGCE3nwosK_yaslgSegB2Qw_ISJEkpZM6-VRRpjHwZOFrhVArGf9yHKbP9KcLPoz1vXwhZ8UKKQP7aTqNpZvmh-Lis8UY3r8FxuI8Myuvo3EawZbEV5Oo=s400-no)
# The way this is represented as data is as a three-dimensional matrix. The width of the matrix is the width of the image, the height of the matrix is the height of the image, and the depth of the matrix is the number of channels. So, as we saw, the height and width of our image are both 100 pixels. This means that the underlying data is a matrix with the dimensions <code>100x100x3</code>.</p>

# In[ ]:


# Turn our image object into a NumPy array
img_data = np.array(img)

# get the shape of the resulting array
img_data_shape = img_data.shape

print("Our NumPy array has the shape: {}".format(img_data_shape))

# plot the data with `imshow` 
plt.imshow(img_data)
plt.show()

# plot the red channel
plt.imshow(img_data[:,:,0], cmap=plt.cm.Reds_r)
plt.show()

# plot the green channel
plt.imshow(img_data[:,:,1], cmap=plt.cm.Greens_r)
plt.show()

# plot the blue channel
plt.imshow(img_data[:,:,2], cmap=plt.cm.Blues_r)
plt.show()


# ## Explore the color channels
# Color channels can help provide more information about an image. A picture of the ocean will be more blue, whereas a picture of a field will be more green. This kind of information can be useful when building models or examining the differences between images.
# 
# We'll look at the <a href="https://en.wikipedia.org/wiki/Kernel_density_estimation">kernel density estimate</a> for each of the color channels on the same plot so that we can understand how they differ.
# 
# When we make this plot, we'll see that a shape that appears further to the right means more of that color, whereas further to the left means less of that color.

# In[ ]:


def plot_kde(channel, color):
    """ Plots a kernel density estimate for the given data.
        
        `channel` must be a 2d array
        `color` must be a color string, e.g. 'r', 'g', or 'b'
    """
    data = channel.flatten()
    return pd.Series(data).plot.density(c=color)

# create the list of channels
channels = ['r','g','b']
    
def plot_rgb(image_data):
    # use enumerate to loop over colors and indexes
    for ix, color in enumerate(channels):
        plt.imshow(image_data[:,:,ix])
        plt.show()
    
plot_rgb(img_data)


# ## Rose and Sunflower(i)
# Now we'll look at two different images and some of the differences between them. The first image is of a rose, and the second image is of a sunflower.
# <p>First, let's look at the rose.</p>

# In[ ]:


# load rose
rose =Image.open('../input/flowers/flowers/rose/16001846141_393fdb887e_n.jpg')
# display the rose image
display(rose)

# NumPy array of the rose image data
rose_data=np.array(rose)
# plot the rgb densities for the rose image
plot_rgb(rose_data)


# ## Rose and Sunflower(ii)
# Now let's look at the sunflower.

# In[ ]:


# load sunflower
sunflower =Image.open('../input/flowers/flowers/sunflower/1044296388_912143e1d4.jpg')
# display the sunflower image
display(sunflower)
# NumPy array of the sunflower image data
sunflower_data=np.array(sunflower)
# plot the rgb densities for the sunflower image
plot_rgb(sunflower_data)


# ## Simplify, simplify, simplify
# While sometimes color information is useful, other times it can be distracting. In this examples where we are looking at flowers, the flowers are very similar colors.so let's convert these images to <a href="https://en.wikipedia.org/wiki/Grayscale">black-and-white, or "grayscale."</a>
# 
# Grayscale is just one of the <a href="https://pillow.readthedocs.io/en/5.0.0/handbook/concepts.html#modes">modes that Pillow supports</a>. Switching between modes is done with the <code>.convert()</code> method, which is passed a string for the new mode.
# 
# Because we change the number of color "channels," the shape of our array changes with this change. It also will be interesting to look at how the KDE of the grayscale version compares to the RGB version above.

# In[ ]:


# convert rose to grayscale
rose_bw = rose.convert("L")
display(rose_bw)

# convert the image to a NumPy array
rose_bw_arr = np.array(rose_bw)

# get the shape of the resulting array
rose_bw_arr_shape = rose_bw_arr.shape
print("Our NumPy array has the shape: {}".format(rose_bw_arr_shape))

# plot the array using matplotlib
plt.imshow(rose_bw_arr, cmap=plt.cm.gray)
plt.show()

# plot the kde of the new black and white array
plot_kde(rose_bw_arr, 'k')


# ## Save your work!
# We've been talking this whole time about making changes to images and the manipulations that might be useful as part of a machine learning pipeline. To use these images in the future, we'll have to save our work after we've made changes.
# 
# Now, we'll make a couple changes to the <code>Image</code> object from Pillow and save that. We'll flip the image left-to-right, just as we did with the color version. Then, we'll change the NumPy version of the data by clipping it. Using the <code>np.maximum</code> function, we can take any number in the array smaller than <code>100</code> and replace it with <code>100</code>. Because this reduces the range of values, it will increase the <a href="https://en.wikipedia.org/wiki/Contrast_(vision)">contrast of the image</a>. We'll then convert that back to an <code>Image</code> and save the result.

# In[ ]:


# flip the image left-right with transpose
rose_bw_flip = rose.transpose(Image.FLIP_LEFT_RIGHT)

# show the flipped image
display(rose_bw_flip)

# save the flipped image
rose_bw_flip.save("bw_flipped.jpg")

# create higher contrast by reducing range
rose_hc_arr = np.maximum(rose_bw_arr, 100)

# show the higher contrast version
plt.imshow(rose_hc_arr, cmap=plt.cm.gray)

# convert the NumPy array of high contrast to an Image
rose_bw_hc = Image.fromarray(rose_hc_arr,"L")

# save the high contrast version
rose_bw_hc.save("bw_hc.jpg")


# ## Make a pipeline
# Now it's time to create an image processing pipeline. We have all the tools in our toolbox to load images, transform them, and save the results.
# 
# In this pipeline we will do the following:
# <ul>
# <li>Load the image with <code>Image.open</code> and create paths to save our images to</li>
# <li>Convert the image to grayscale</li>
# <li>Save the grayscale image</li>
# <li>Rotate, crop, and zoom in on the image and save the new image</li>
# </ul>

# In[ ]:


# take only four image from sunflower
image_paths = ['../input/flowers/flowers/sunflower/1022552036_67d33d5bd8_n.jpg',
               '../input/flowers/flowers/sunflower/14121915990_4b76718077_m.jpg',
               '../input/flowers/flowers/sunflower/1043442695_4556c4c13d_n.jpg',
               '../input/flowers/flowers/sunflower/14472246629_72373111e6_m.jpg']

def process_image(path):
    img = Image.open(path)

    # create paths to save files to
    bw_path = "bw_{}.jpg".format(path.stem)
    rcz_path = "rcz_{}.jpg".format(path.stem)

    print("Creating grayscale version of {} and saving to {}.".format(path, bw_path))
    bw = img.convert("L").save(bw_path)
    print("Creating rotated, cropped, and zoomed version of {} and saving to {}.".format(path, bw_path))
    rcz = img.rotate(45).crop([25, 25, 75, 75]).resize((100, 100)).save(rcz_path)

# for loop over image paths
for img_path in image_paths:
    process_image(Path(img_path))

