#!/usr/bin/env python
# coding: utf-8

# # Part 1.
# 

# First of all, we are going to talk about the mathematical definition of convolution and connect it with convolutuion in Neural Networks. It will allow you to deeper understand how convolution is calculated and why it's helpful in image processing.

# Some terms: matrices (2D or 3D) used in convolution are called **filters** or **kernel**.

# ## Mathematical Definition

# Convolutional Layers are called this way because operation carried out by them is called a convolution. It's defined for two functions $f$ and $g$ as follows:
# 
# $$(f*g)(x) = \int \limits^{+\infty}_{-\infty} f(\tau)g(x - \tau) d\tau$$
# 
# If we were convolving 2D image with a kernel then $f$ would be a discrete function of two variables (x, y) returning the value of a pixel at position (x, y) and $g$ would be a similar discrete function for the kernel.
#     (_To be precise the analogy is not completely right and some differences exist but they are irrelevant for us right now_)

# There are many explanations for what convolution does. We will examine the most interesting of them to get intuition. 

# ## Convolution as signal processing tool
# 

# ### One-dimensional signals (advanced)
# 

# If you ever worked on processing signals then you have definitely met convolution before. It's one of the most basic tools which is used everywhere. As an example of a task for convolution, we can take denoising of a signal. The result is shown in the picture below. We can see the dirty signal at the top and the denoised result on the bottom right image. To remove noise convolution uses a sliding window which takes a weighted average of signal values in some area.
# 
# <img src="https://cdn-images-1.medium.com/max/1600/1*V2j4icieU9aAi3gW2ESqOw.png" width=400>
# <img src="https://cdn-images-1.medium.com/max/1600/1*OiCIOkDIockKUIw6snQIxw.png" width=700>
# (Image source: [medium article](https://towardsdatascience.com/convolution-a-journey-through-a-familiar-operators-deeper-roots-2e3311f23379))

# ### Images

# &mdash; Well, then what about images? Are they signal too?
# 
# &mdash; YEP!
# 
# If we encode an image with a discrete function as it was explained earlier than the image becomes a signal and we can process it accordingly.
# 
# Important note: we discussed the way encode gray image with a two-variable function but for a colored image, the function would have three coordinates (x, y, c) where c is a channel.

# &mdash; In the example of signal processing, we removed noise from a signal by effectively blurring it. How can we do the same for an image? 
# 
# &mdash; We simply have to make 
# 
# You can see those kernels at http://setosa.io/ev/image-kernels. For me, this is the most useful illustration you can have. 
# Try to gather all you have learned so far, play on the site and write your own convolution.

# # [Task 1]

# In[ ]:


get_ipython().system('pip install imageio')

import scipy.misc
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from imageio import imread
from ipywidgets import interact, FloatSlider

get_ipython().run_line_magic('matplotlib', 'inline')

def grayscale(img):
    """
    To transform an image to grey we will simply take its red channel.
    """
    return img[:, :, 0]

def show_img(img):
    """
    Nicely show an image with matplotlib.
    """
    plt.figure(figsize=(10, 6))
    if len(img.shape) == 2:
        plt.imshow(img, cmap=cm.Greys_r)
    else:
        plt.imshow(img)
    plt.axis('off')
    plt.show()
    
def seed_random(size, seed):
    """
    Function to generate seeded random matrices.
    """
    np.random.seed(seed)
    return np.random.normal(size=size)

def load_img(img_path):
    """
    Read an image from path and convert it to [0, 1] pixel format.
    
    !!!For different computers the result of a function may differ. Pleease use Google Colab for reproducability!!!
    """
    return imread(img_path) / 255


# In[ ]:


# Load an image
gray_img = grayscale(load_img('./img.jpeg'))
show_img(gray_img)


# In[ ]:


def convolve(img, kernel):
    """
    This function should take 2d numpy array representing an image and a kernel and carry out convolution.
    img.shape = (y, x), and kernel.shape = (y, x). 
    NOTICE: the convolution works with 2D images right now. We will get to 3D convolution a bit later.
    COnvolution should have stride 1 and no padding.
    The function has to return the result with one output channel.
    Do not add Bias or apply activation function. Only convolution.
    """
    kernel_y, kernel_x = kernel.shape[:2]
    img_y, img_x = img.shape[:2]

    result = np.zeros((img_y - kernel_y + 1, img_x - kernel_x + 1))
    # iterate over x and y, take slices of numpy arrays and perform element-wise product and so on.
    <Your code here>
    
    return result


# Lets test our convolution with different filters. Each run will take approxiamately 10 seconds.

# In[ ]:


# apply blur filter
blur_filter = np.array([[0.0625, 0.125, 0.0625],
                           [0.125, 0.25, 0.125],
                           [0.0625, 0.125, 0.0625]])

sharpened_img = convolve(gray_img, blur_filter).clip(0, 1)
show_img(sharpened_img)


# In[ ]:


# apply sharpening filter
sharpen_filter = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

sharpened_img = convolve(gray_img, sharpen_filter).clip(0, 1)
show_img(sharpened_img)


# In[ ]:


# apply filter, which detects borders
outline_filter = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])

outline_img = convolve(gray_img, outline_filter).clip(0, 1)
show_img(outline_img)


# Hopefully, you have got everything right and the function is working. In particular, in convolution, we take the sum of pair-wise products and we want you to change this.
# 
# __!!!
# Replace sum of pairwise products with the multiplication, apply convolution to gray_img with outline_filter. .!!!__

# In[ ]:


def my_operation(img, kernel):
    """
    Copy the implementation of convolution from above and replace sum with multiplication.
    """
    kernel_y, kernel_x = kernel.shape[:2]
    img_y, img_x = img.shape[:2]

    <Your code here>
    
    return result


# In[ ]:


# apply convolution and take sum
<Your code here>


# ## Convolution as local feature extractor

# **[Advanced]**
# 
# If we take a closer look at the definition of convolution then we can see that its value at some point as simply a weighted sum of values of $f$ with one interesting feature: the weights are defined by function $g$ and its center is always located at the point at which we want to know the value of convolution.
# $$(f*g)(x) = \int \limits^{+\infty}_{-\infty} f(\tau)g(x - \tau) d\tau$$
# 
# 
# **[Not advanced]**
# 
# Now, let's return to convolutions in neural networks. At the lecture, we found out why fully-connected layers are worse at processing images. The reason is that ConvLayers exploit some external knowledge about image structure:
# 
# * Pixels which are close together are much more connected than those which are separate.
# * If we move an object around an image it is still the same object.
# 
# This assumption can be formulated as "In images, the local structure is the most important". Using the assumptions we create restrictions for convolutional layers which allow them to operate using a lot fewer weights.

# ## Convolution as a pattern-matcher

# Lets take a look at how convolution is calculated in 1D case. (The illustration won't work in Colab and you need to run it on your computer).

# In[ ]:


def f(x):
    """
    Some nice function
    """
    return 1/(2 + x**2 * (0.1 + np.sin(x)**2))


def g(x):
    """
    Normal distribution. It's also nice
    """
    return np.exp(-x**2/2) / np.sqrt(2 * np.pi)


x = np.linspace(-10, 10, 100)
@interact(g_offset=FloatSlider(min=-10, max=10, step=0.5))
def plot_and_calc(g_offset):
    plt.figure(figsize=(10, 7))
    f_val = f(x)
    g_val = g(g_offset - x)
    mul_vals = f_val * g_val

    plt.plot(x, f_val, label='f(x)')
    plt.plot(x, g_val, label='g(x)')
    plt.plot(x, mul_vals, label='f(x)*g(x)')
    plt.gca().fill_between(x, 0, mul_vals)
    plt.legend()

    plt.text(-10.5, 0.55,
             "Approximate conv value at {} = {:.2f}".format(g_offset, mul_vals.sum()))


# The value of a slider is the point at which we want to calculate the value of convolution. If you move the slider you can see that the function $g(x)$, which defines weights for aggregation, is also moving. We multiply the function $f(x)$ by $g(x)$ and then calculate area under the curve. This area is the value of convolution we wanted to find. (NOTE: we didn't plot convolution itself on the plot, only $f(x)$ and $g(x)$ to illustrate the process of calculating 1D convolution)
# 
# It's easy to notice, that the value of convolution is bigger when functions have similar shapes. The maximum is reached when g_offset (the point at which we calculate convolution) is equal to 0 and the biggest peak from $f(x)$ matches the peak from $g(x)$. However, the value is also big when $g(x)$  matches any of the smaller peaks. 
# 
# **Apparently, looking only at values of convolution we can identify points at which the peaks from $f(x)$ are located.**

# Using convolution to find some special patterns in an image is the most important interpretations of convolution. This case the image is taken as $f(x)$ and the kernel, which we slide over the image to find matches, becomes $g(x)$. However, the kernels are not predefined as in the examples. They are learned by the neural network itself.
# 
# For CNN's, which are trained on big datasets, the filters of first Convolutional Layer are some basic patterns which network tries to find in an image: 
# <img src="http://cs231n.github.io/assets/nn3/cnnweights.jpg" width=600>
# (Source http://cs231n.github.io/neural-networks-3/#vis)

# # [Task 2]

# <img src="https://i.imgflip.com/31jdp4.jpg">

# In[ ]:


# It's time to play a role of a detective
# Load file noisy_data.txt and using convolution you have written earlier find three crosses 
# which are shaped like + with height of 5 and width of 5. 
# The crosses consist of some relatively big positive integer number while everything around is just random numbers
# drawn from normal distribution with mean=0 and std=1. (Which means that in general numbers are small)
noise = np.loadtxt('./noisy_data.txt')
# Create 5x5 pattern which would give a high value if it overlaps with the cross and a small one otherwise.
pattern = <Your code here>

# It's better to use your own convolution from earlier task
convolution_activation = <Your code here>


# In[ ]:


# There are three crosses so you should find coordinates of three biggest values from convolution_activation
# I used numpy.where and hardcoded threshold to find them but there are many other ways.
<Your code here>


# Print the regions of an image with the cross to confirm that you have one. Also, it will help you understand where the center of a cross is located (the coordinates you have found above are most likely the coordinates of the left-top corner while the task requires finding their centers).
# **!!!To get the answer find the __centers__ of all three crosses and find the sum of their coordinates. !!!**

# ## Convolutional Layer

# I hope you have already understood how 2D convolution works. Now we will move to convolutional layers. The input of a Convolutional Layer is a batch of colored 3D images. The batch has shape **[batch_size, x, y, channel]**. 
# 
# Since our images have 3 channels the kernels will also have 3 channels. Also, in the convolutional layer we have several kernels, we apply convolution to each image with every kernel. The outputs of convolutions with different 3D kernels are then stacked into a new image with several channels where each channel corresponds to a kernel.
# 
# You are going to write class ConvLayer(in_channels, out_channels, kernel_size). Link to a nice illustration from the lecture http://cs231n.github.io/convolutional-networks.

# In[ ]:


class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # initialize kernels
        self.kernels =seed_random((out_channels, kernel_size, kernel_size, in_channels), 42)
        # To understand what's going on lets recap some info:
        # 1) In convolutional layer we apply several filters and each of them tries to find some special
        # feature in an image. Each filter produces 2D output which is then stacked to form a new output 
        # image. It all means that the number of kernels is equal to the number of kernels.
        # 2) Each filter has several pattern maps of size kernel_size*kernel_size to gather information 
        # from every channel. The number of such 2D pattern maps is equal to a number of channels in input images.
        
        # Also we want to add biases now
        self.biases = seed_random((out_channels), 13)
        
    def forward(self, X):       
        # Initialize array with the result of convolution.
        res = np.zeros((X.shape[0], X.shape[1] - self.kernel_size + 1,
                        X.shape[2] - self.kernel_size + 1, self.out_channels))
        # Use 4 nested for-loops to iterate over images in a batch, over filters and over coordinate x and y. 
        # !!!Don't forget to add bias.
        # Do not use activation function
        for i, img in enumerate(X):
            <Your code here>
        
        return res


# To test convolution we will return to an image of a cat and try to make resulting image colored. To do this
# we will use three filters each will blur its own channel. The first outputs blurred red, the second outputs blurred green and so on.

# In[ ]:


# Create Layer
conv = ConvLayer(3, 3, 3)
r_filter = np.zeros((3, 3, 3))
r_filter[:, :, 0] = blur_filter

g_filter = np.zeros((3, 3, 3))
g_filter[:, :, 1] = blur_filter

b_filter = np.zeros((3, 3, 3))
b_filter[:, :, 2] = blur_filter
# We eill override default random filters with our predefined kernels
conv.kernels = np.array([r_filter, g_filter, b_filter])
conv.biases = np.zeros((3))


# In[ ]:


img = load_img('./img.jpeg')
res = conv.forward(img[np.newaxis, :, :, :])
# Show blurred image
show_img((res[0]).clip(0, 1))


# # [Task 3]

# Let's test our Conv Layer.
# 
# **!!!Just like in the first task copy code of ConvLayer and replace sum with multiplication. (Replace sum with multiplication only when you take sum of pair-wise products. You should still add bias to the product of pair-wise products)!!!**
# 
# **Create a ModifiedCOnvLayer and apply it to the cat image we used above. To get the answer sum all pixels of a ModifiedConvLayer output. Do not clip the result**

# In[ ]:


class ModifiedConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Initialize array of kernels
        self.kernels =seed_random((out_channels, kernel_size, kernel_size, in_channels), 42)
        self.biases = seed_random((out_channels), 13)
        
    def forward(self, X):       
        # Initialize array with the result
        res = np.zeros((X.shape[0], X.shape[1] - self.kernel_size + 1,
                        X.shape[2] - self.kernel_size + 1, self.out_channels))
        <Your code here>
        
        return res


# In[ ]:


# Create ModifiedCOnvLayer and apply it to the cat image. Use insqueeze to add dimensoin to an image
# so that it looks like a batch from one image.
# Calculate the required statistic. Do not change the kernels created in __init__
img = load_img('./img.jpeg')

<Your code here>

