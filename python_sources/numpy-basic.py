#!/usr/bin/env python
# coding: utf-8

# We should know about multidimensional array concept if you work in image processing. So am learning and also share my code for all. This code we will learn how to create image through numpy array and aslo apply basic operation. I have not explain, because my english not very good.  If it useful please upvote me.<br/>
# **Thanks**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Create image in numpy

# In[ ]:


arr = np.array([[[3, 4, 5], [8, 2, 5], [1, 9, 5],[3, 4, 5], [8, 2, 5], [1, 9, 5]],
               [[3, 0, 1], [0, 0, 0], [1, 10, 3],[3, 4, 5], [8, 20, 5], [1, 9, 5]],
               [[35, 50, 51], [8, 52, 57], [20, 20, 20],[3, 4, 11], [12, 2, 5], [1, 9, 5]],
               [[3, 0, 1], [8, 2, 7], [1, 10, 3],[3, 4, 5], [14, 2, 20], [1, 9, 5]],
               [[3, 0, 1], [8, 2, 7], [20, 20, 21],[3, 4, 5], [16, 2, 18], [1, 9, 5]],
               [[3, 0, 1], [13, 19, 16], [1, 10, 3],[5, 4, 6], [9, 2, 10], [1, 9, 5]]])
arr.shape


# In[ ]:


plt.imshow(arr)
plt.show()


# ### Multply 10 in Red channel value

# In[ ]:


arr[:,:,0] *= 10 


# In[ ]:


plt.imshow(arr)
plt.show()


# ### Multply 7 in Red channel value

# In[ ]:


arr[:,:,1] *= 7


# In[ ]:


plt.imshow(arr)
plt.show()


# In[ ]:


arr1 = np.copy(arr)


# In[ ]:


np.random.seed(5)
np.random.shuffle(arr1)
plt.imshow(arr1)
plt.show()


# In[ ]:


_,(ax) = plt.subplots(ncols=3, figsize=(16,5)) 
ax[0].hist(arr1[:,:,0],bins=10)
ax[1].hist(arr1[:,:,1],bins=10)
ax[2].hist(arr1[:,:,2],bins=10)
plt.show()


# In[ ]:


arr2 = arr1[:,:,0]+arr1[:,:,0]
plt.imshow(arr2)
plt.show()


# In[ ]:


red1 = np.concatenate((arr1[:,:,0], arr1[:,:,0]), axis=0)
green1 = np.concatenate((arr1[:,:,1], arr1[:,:,1]), axis=0)
blue1 = np.concatenate((arr1[:,:,2], arr1[:,:,2]), axis=0)


# In[ ]:


red1


# In[ ]:


_,(ax) = plt.subplots(ncols=3, figsize=(16,5)) 
ax[0].imshow(red1)
ax[1].imshow(green1)
ax[2].imshow(blue1)
plt.show()


# In[ ]:


image = cv2.merge((red1,green1, blue1))
plt.imshow(image)
plt.show()


# ### Image rotation using numpy

# In[ ]:


image_rotation = np.rot90(image)
plt.imshow(image_rotation)
plt.show()


# ### Image flip

# In[ ]:


image_flip = np.fliplr(image)
plt.imshow(image_flip)
plt.show()


# In[ ]:


img_rec = np.copy(image)


# In[ ]:


img_rec[1,:,0] = 0
img_rec[1,:,1] = 50
img_rec[1,:,2] = 150


# In[ ]:


plt.imshow(img_rec)
plt.show()


# In[ ]:


img_rec[:,3,0] = 0
img_rec[:,3,1] = 50
img_rec[:,3,2] = 150
plt.imshow(img_rec)
plt.show()


# ### Open grid in numpy

# In[ ]:


ogrid_x, ogrid_y = np.ogrid[0:10, 0:5]


# In[ ]:


ogrid_x


# In[ ]:


ogrid_img = ogrid_x+ogrid_y
print(ogrid_img)
plt.imshow(ogrid_img)
plt.show()


# ### Transpose 

# In[ ]:


ogrid_img = ogrid_img.T
plt.imshow(ogrid_img)
plt.show()


# ### Meshgrid

# In[ ]:


YY, XX = np.mgrid[10:40:10, 1:4]
ZZ = XX + YY 
ZZ


# In[ ]:


circle_mask = 5**2 + 7**2 <= 100**2
circle_mask


# In[ ]:


demo_image = plt.imread('/kaggle/input/sample-images-for-kaggle-demos/1928768_1035869614877_9398_n.jpg')
plt.imshow(demo_image)
plt.show()


# In[ ]:


# Get the dimensions
n,m,d = demo_image.shape
print(n,m,d)


# We are get x and y axes index values through <code> ogrid </code> method. It is most importent task to get x and y index values in array. We are used this in add geometric shapes and also use rotation.
# Note: Predefined functions have available in **numpy** and **opencv** for image processing.

# In[ ]:


# Create an open grid for our image
x,y = np.ogrid[0:n,0:m]


# ### Add geometric shapes

# In[ ]:


#copy image
copyImg = demo_image.copy()

#get the x and y center points of our image
center_x = n/2
center_y = m/2
print("Center x and Center y", center_x, center_y)

#create a circle mask which is centered in the middle of the image
circle_mask = (x-center_x)**2+(y-center_y)**2 <= 8000

copyImg[circle_mask] = [0, 0,0]

plt.imshow(copyImg)
plt.show()


# ### Create a square mask
# 
# 

# In[ ]:


square_mask = (x<200)&(x>100)&(y<500)&(y>400)

copyImg[square_mask] = [255, 0,0]

plt.imshow(copyImg)
plt.show()


# ### Rotation
# 

# In[ ]:


copyImg = demo_image.copy()

copyImg = demo_image[x, -y]

plt.imshow(copyImg)
plt.show()


# In[ ]:


copyImg = demo_image.copy()

copyImg = demo_image[-x, y]

plt.imshow(copyImg)
plt.show()


# Thank you for visting. Happy to receive any comments :) <br/><br/>
# **References**
# 
# https://towardsdatascience.com/the-little-known-ogrid-function-in-numpy-19ead3bdae40
