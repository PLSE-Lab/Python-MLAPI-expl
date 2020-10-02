#!/usr/bin/env python
# coding: utf-8

# Convolution is an important concept in image processing and Convolutional Neural Networks. Here I will focus on 2d Convolution.
# In 2d convolution the basic idea is to:
# * take a predefined matrix know as a kernel
# * slide it over a matrix representing an image
# * multiple the elements in the image matrix by the corresponding elements in the kernel matrix
# * sum the result to create a new element
# * move to the next element and repeat for every element in the image matrix to create a new matrix
# 
# ![Convolution](https://developer.apple.com/library/content/documentation/Performance/Conceptual/vImage/Art/kernel_convolution.jpg)
# 
# Checkout this link for an excellent interactive example of convolution
# http://setosa.io/ev/image-kernels/
# 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
from scipy import signal


# **Load and format data**

# In[ ]:



train_df = pd.read_json("../input/train.json")

# Train data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_2"]])


# **Creating and applying convolution filters to an image**

# In[ ]:


# Edge Detection kernel
edge = np.array([[-100,-100,-100],[-100,100,-100],[-100,-100,-100]])
arr = signal.convolve2d(np.reshape(np.array(x_band1[5]),(75,75)),edge,mode='valid')
plt.imshow(arr)
plt.show()

# Sharpen kernel
sharpen = np.array([[0,-1,0],[-1,-5,-1],[0,-1,0]])
arr = signal.convolve2d(np.reshape(np.array(x_band1[5]),(75,75)),sharpen,mode='valid')
plt.imshow(arr)
plt.show()


# **Further Reading**
# * https://en.wikipedia.org/wiki/Kernel_(image_processing)#Convolution
# * https://www.cs.cornell.edu/courses/cs1114/2013sp/sections/S06_convolution.pdf
# * https://en.wikipedia.org/wiki/Convolution
