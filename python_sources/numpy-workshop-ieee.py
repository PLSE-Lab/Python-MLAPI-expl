#!/usr/bin/env python
# coding: utf-8

# # Numpy

# Numpy is the core library for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays.

# # Numpy Arrays

# In[2]:


import numpy as np # Standard way to import numpy


# In[ ]:


a = np.array([1, 2, 3]) # create a "rank 1" array


# In[ ]:


print(a)
print(a.dtype)
print(a.shape)


# In[ ]:


a[0] = 100
print(a)


# In[ ]:


b = np.array([[1, 2, 3], 
              [4, 5, 6]])


# In[ ]:


print(b)
print(b.shape)


# In[ ]:


print(b[0]) # first row
print(b[0, 1]) # first row, second column
print(b[0][1]) # also first row, second column


# # Different ways to create a numpy array

# In[ ]:


a = np.zeros((2, 3)) # array of all zeros with given shape
print(a)


# In[ ]:


b = np.ones((3, 3)) # array of all ones with given shape
print(b)


# In[ ]:


b = np.ones((3, 3), dtype=np.int32) # array of all ones with given shape
print(b)


# In[ ]:


c = np.full((3, 2), 10) # Constant array
print(c)


# In[ ]:


d = np.eye(3) # 3x3 identity matrix
print(d)


# In[1]:


e = np.random.random((2, 4)) # Random array of given shape with values [0, 1)
print(e)


# In[ ]:


f = np.random.randint(1, 10, size=(3, 3)) # Random array of given shape/size with values [1, 10)
print(f)


# In[ ]:


g = np.arange(10)
print(g)

h = np.arange(1, 11)
print(h)


# In[ ]:


i = np.linspace(-1, 1, 100) # 100 equally spaced values between -1 and 1
print(i)


# In[ ]:





# # Array Indexing

# **1) SLICING**

# In[ ]:


a = np.random.randint(10, size=(3, 4))
print(a)


# In[ ]:


print(a[0]) # First row or a[0, :]
print(a[:, 0]) # First column
print(a[:2, 1: 3]) # Rows 0, 1 and columns 1, 2


# In[ ]:


a[:, 1] = 100 # assign 100 to complete second column
print(a)


# In[ ]:


b = np.array([0, 1, 0, 2])
print(a[:, b])


# **2) Boolean array indexing**

# In[ ]:


a = np.random.randint(100, size=(4, 4))
print(a)


# In[ ]:


a>50


# In[ ]:


a[a > 50] = 0
print(a)


# In[ ]:


a[a < 50] # Get rank 1 array with elements for which a[i] < 50


# # Array Math

# In[ ]:


a = np.arange(9).reshape((3, 3)) # reshape [0, 1, ... 8] array to 3x3
print(a)


# In[ ]:


b = np.random.randint(20, 40, size=(3, 3)) # Random array with values [20, 40)
print(b)


# In[ ]:


print(a + b)
print(np.add(a, b))


# In[ ]:


print(a - b, end='\n\n')
print(a * b, end='\n\n') # NOTE: elementwise-multiplication, not matrix multiplication
print(a / b)


# In[ ]:


# Elementwise operations
print(np.sqrt(a), end='\n\n')
print(np.square(b), end='\n\n')
print(np.sin(a), end='\n\n')
print(a * 10.0, end='\n\n')


# In[ ]:


x = np.random.randint(10, size=(3, 2))
y = np.random.randint(10, size=(2, 4))
matmul = np.matmul(x, y) # or np.dot(x, y)


# In[ ]:


matmul.shape


# In[ ]:


z = np.random.randint(10, size=(4, 2))
print(z.shape)
print(z.T.shape) # Transpose operation


# In[ ]:


np.matmul(x, z.T) # matmul for (3, 2) and (2, 4)


# In[ ]:





# In[ ]:


print(x)


# In[ ]:


print(np.sum(x))         # Sum of all elements
print(np.sum(x, axis=0)) # Sum of each column
print(np.sum(x, axis=1)) # Sum of each row


# # Broadcasting

# In[ ]:


a = np.arange(9).reshape((3, 3))
print(a)


# In[ ]:


b = np.array([10, 20, 30])
print(b)
print(b.shape)


# In[ ]:


a + b


# In[ ]:


a + b.reshape((3, 1)) # Add `b` as a column vector


# # Save and load large numpy arrays

# In[ ]:


a = np.random.random(size=(400, 400))


# In[ ]:


np.save("abc.npy", a)


# In[ ]:


b = np.load("abc.npy")


# In[ ]:


np.all(b == a) # Check if a and b are elementwise equal


# In[ ]:





# # Visualizations

# In[3]:


import matplotlib.pyplot as plt


# In[ ]:


x = np.linspace(-np.pi, np.pi, 100)
sin = np.sin(x)
cos = np.cos(x)

# Draw a scatter plot
plt.plot(x, sin, c='r')
plt.plot(x, cos, c='b')


# In[5]:


img = plt.imread("../input/lanes.jpg")


# In[6]:


img.shape


# In[7]:


plt.imshow(img)


# An RGB image has three channels - **Red**, **Green** and **Blue**. This corresponds to 0, 1 and 2 channels of the 3D image.

# In[8]:


red = img.copy() # Now changing `red` won't change `img`
red[:, :, 1:] = 0 # Turn channels G(1) and B(2) to 0
plt.imshow(red)


# In[9]:


green = img.copy() # Now changing `green` won't change `img`
green[:, :, [0, 2]] = 0
plt.imshow(green)


# In[10]:


blue = img.copy() # Now changing `blue` won't change `img`
blue[:, :, :2] = 0
plt.imshow(blue)


# In[11]:


cropped = img[100: 400, 100: 400, :] # Crop image
plt.imshow(cropped)


# In[13]:


plt.imsave("lanes_cropped.jpg", cropped)


# In[14]:


plt.imshow(plt.imread("lanes_cropped.jpg"))


# In[ ]:





# In[ ]:




