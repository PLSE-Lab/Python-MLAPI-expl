#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <h2>1.Loading Libraries</h2>

# In[ ]:


import cv2
# Open source computer vision (Opencv)is a library help to work with images 
import numpy as np 

import pandas as pd 

import sklearn as sk 

import matplotlib.pyplot as plt 

import seaborn as sns


# <h2>2.Loading Image</h2>

# In[ ]:


# Loading  a image as a grayscale 
image = cv2.imread('/kaggle/input/plane.jpg',cv2.IMREAD_GRAYSCALE)


# > Above command Reads the image and converts it into grayscale image . We can use matplot lib to view the image

# In[ ]:


# Show Image 
plt.imshow(image,cmap='gray')
# in the place of image we can also give 2d or 3d arrays as input then matplot lib will plot the corresponding output
plt.axis('off')
plt.show()


# > Basically images are mase up of numbers of different dimensions i.e if gray scale(2D) ,if it is Color (3d) . So openCv library converts every image into a numpy array.And we can plot the numpy array using the plt.imshow command

# In[ ]:


type(image)


# > As we can see above the image is converted into numpy array

# In[ ]:


print(image.ndim)
image.shape


# > The converted image is of the size 2270 X 3600

# >  Above we are dealing with gray scale images so our picture dimension are 2d . But when we are dealing with colored images,the images will be stored as 3d numpy array in blue,green,red(BGR) format.

# In[ ]:


# Loading image in color 
image_clr = cv2.imread('/kaggle/input/plane.jpg',cv2.IMREAD_COLOR)
print(image_clr.shape)
print('I am a colored image and my dimensions are ',image_clr.ndim,'and I am in BGR format not RGB')


# > This is the small issue with the openCV library it stores data in B,G,R format and we need to convert to R,G,B format before plotting it with matplotlib

# In[ ]:


# Converting the bgr image to rgb format to plot in matplotlib 
image_rgb = cv2.cvtColor(image_clr,cv2.COLOR_BGR2RGB)
# above command converts bgr image to rgb
plt.imshow(image_clr)
plt.title('Without converting in to RGB format ')
plt.show()
plt.imshow(image_rgb)
plt.title('after converting in to RGB format ')
plt.show()


# <h3>3.Resizing the image </h3>

# > Sometimes we need to resize the image so that the size of the  image is reduced and we can do machine learning on to of it . However,if we resize an image the lot of information in the image will be lost so we need to be okay with the cost it's coming with

# In[ ]:


image_100X100 = cv2.resize(image,(100,100))
# resizing the image in to 100X100 pixels
plt.imshow(image_100X100,cmap ='gray')
plt.show()
print('I am rescaled and now my size is ',image_100X100.shape)


# > Now we have rescaled image to 100 X 100 though lots of detailed information is lost we can still identify it as a plane so it will be fine

# <h2>4.Cropping the Image</h2>

# > As mentioned above images are nothing but a 2d or 3d array so we can achieve image cropping by slicing the array

# In[ ]:


image_crop = image[:,:2000]
plt.imshow(image_crop,cmap='gray')
plt.show()


# > Cropping will be useful when we want to remove unwanted portions and use only important content of the image

# <h2>5.Blurring Image</h2>

# >We can blur an image by placing kernel over the image this kernel is also called filter .To achieve the effect of blurring each pixel in the image is tranformed to be average value of its neighbours . The effect of blurring depends on the size of kernel we choose . This concept of filtering with kernels is used in convolution neural networks to extract important features of the image please refer to this link to get to know more about CNN https://medium.com/@nadeemhqazi/a-brief-introduction-to-convolution-neural-network-4821215aa591

# In[ ]:


# Blurring image 
image_blur = cv2.blur(image,(50,50))
# above I have used 50X50 kernel 
plt.imshow(image_blur,cmap='gray')
plt.show()


# > Kernels are widely used in image propcessing we can do anything like filtering ,sharpening ,edgedetection , blurring etc.. by defining appropriate kernels or filters

# <h2>6.Sharpening Image</h2>

# In[ ]:


# Defining a kernel to sharpening the image 
kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])

# sharpen image 
image_sharp = cv2.filter2D(image,-1,kernel)

plt.imshow(image_sharp,cmap='gray')
plt.show()


# > Sharpening works similar to the blurring but here we used a kernel that highlights every pixel in the image by adding a weight of 5 to the current pixel and removing the effect of neighbours by giving 0 or -1 weights".Similar to this we can perform more operations like isolating the colors in the image , enhancing the contrast etc..

# * <h2>7.Edge Detecting</h2>

# > Edge Detection is simply a case of trying to find the regions in an image where we have a sharp change in the intensity or a sharp change in the color a high value indicates steep change a low value indicates shallow change

# In[ ]:


image_gray2 = cv2.imread('/kaggle/input/plane_256x256.jpg',cv2.IMREAD_GRAYSCALE)
# calculate the median intensity 
median_intensity = np.median(image_gray2)
# seting thershold to be one of the standard deviation above and below the median
lower_thershold = int((max(0,(1-.33)*median_intensity)))
upper_thershold = int((min(255,(1+.33)*median_intensity)))
#applying canny edge detector 
image_canny = cv2.Canny(image_gray2,lower_thershold,upper_thershold)

#show Image
plt.imshow(image_canny ,cmap='gray')


# > To Know what is canny and how edge detection works please refer to this link :-https://www.youtube.com/watch?v=uihBwtPIBxM

# > This are some of the image processing techiniques to get started with . I have leart this methods from a book called MACHINE LEARNING WITH PYTHON COOKBOOK written by CHRIS ALBON 

# In[ ]:




