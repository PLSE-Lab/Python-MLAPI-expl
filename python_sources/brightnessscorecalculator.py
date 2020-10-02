#!/usr/bin/env python
# coding: utf-8

# # Brightness Score Calculator
# Returns the score of the brightness your image

# ## Importing Libraries
# <p>
# Importing all the required libraries.<br>
# <br>
#     <b>Cv2</b> : Open-CV Library for image-processing.<br>
#     <b>Numpy</b> : Used for Faster calculations of arrays and numbers.<br>
#     <b>Matplotlib</b> : Used mainly for Plotting and also used to show images.<br>
#     <b>Os</b> : It interacts with System enviornment and used to work with directories, files etc.<br>
#     <b>Glob</b> : Optimized library to work with directories and files.<br>
# </p>

# In[ ]:


import cv2

import numpy as np
import matplotlib.pyplot as plt

import glob,os


# ## Setting the data path
# Here I used flowers dataset which consists of 50 flower images with dimensions 128x128.

# In[ ]:


path = '../input/'
print(os.listdir(path))


# ## Loading the Data
# **load_data** function will be given a path which contains our data, and this load_data function recursively iterate through all the image files with given extension. All these will be read with cv2.imread function and the matrix data will be returned. Here I've shown you a sample picture.

# In[ ]:


def load_data(path,ext='.png'):
  files = glob.glob(path+'*'+ext)
  data = []
  for file in files:
    img = cv2.imread(file)
    data.append(img)
  return data

data = load_data(path,ext='.png')
data = np.array(data)
plt.imshow(data[0])
plt.axis('off')
plt.show()


# ## Data Augmentation
# - I made a dataset using augmentation which converts all our 50 images with different brightness values.
# - This is done using the adjust_gamma function. It takes a image and apply the Gamma transformation and will return the required image with either low brightness or high brightness.

# In[ ]:


def adjust_gamma(image, gamma=1.0):
  table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)])
  return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))


# In[ ]:


k = './augmented_data/'
flag=0
if(os.path.isdir(k)):
  print('path '+k+' is already existed.')
  aug_path = k
else:
  print('path '+k+' is created.')
  os.mkdir(k)
  aug_path = k
  flag=1
  
if flag:
  print('Data Augmentation is being processed...')
  for i in range(len(data)):
    r = np.round(np.random.uniform(low=0.01, high=5),2)
    cv2.imwrite(aug_path+str(i)+'.png',adjust_gamma(data[i],r))


# In[ ]:


aug_data = load_data(aug_path,ext='.png')  
aug_data = np.array(aug_data)
plt.imshow(aug_data[4])
plt.axis('off')
plt.show()


# ## Calculating the Brightness Score
# 
# ### Method-1 :
# Input image will be converted to the gray scale image and we'll take the mean value of all the pixels. 
# This is done because, the gray scale image is nothing but the image with pixels having the **brightness** range in 0 to 255. So I simply used this concept to calculate the score and I normalized the image brightness by taking the mean and multiplying with 10/255..
# ### Method-2 :
#  In this method, we'll convert our image from **RGB** model to **HSV** model. The **HSV** model is the most closest model which will interprete the image as the human perceive. And **V** value here indicates the Brightness. Using this **V**, Here also I normalized the image brightness by taking the mean and multiplying with 10/255.

# In[ ]:


def brightness_score(img,method=1):
  if(method==1):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.round((np.mean(gray)/255)*10,2)
  else:    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    return np.round((np.mean(v)/255)*10,2)

idx = 3
sample_img = aug_data[idx]
plt.subplot(121)
plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
plt.title('Method1 - score :'+str(brightness_score(sample_img,method=1)))
plt.axis('off')
plt.subplot(122)
plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
plt.title('Method2 - score :'+str(brightness_score(sample_img,method=2)))
plt.axis('off')
plt.show()


# ## Verifying our Brightness score function
# 
# Here I've taken pure black and pure white images which must have brightness score 0 and 10 respectively.

# In[ ]:


white = cv2.imread(path+'white.jpg')
white[white<255]=255
black = cv2.imread(path+'black.jpg')
black[black>0]=0

plt.subplot(221)
plt.imshow(cv2.cvtColor(white, cv2.COLOR_BGR2RGB))
plt.title('Method1 - score :'+str(brightness_score(white,method=1)))
plt.axis('off')
plt.subplot(222)
plt.imshow(cv2.cvtColor(black, cv2.COLOR_BGR2RGB))
plt.title('Method1 - score :'+str(brightness_score(black,method=1)))
plt.axis('off')
plt.subplot(223)
plt.imshow(cv2.cvtColor(white, cv2.COLOR_BGR2RGB))
plt.title('Method2 - score :'+str(brightness_score(white,method=2)))
plt.axis('off')
plt.subplot(224)
plt.imshow(cv2.cvtColor(black, cv2.COLOR_BGR2RGB))
plt.title('Method2 - score :'+str(brightness_score(black,method=2)))
plt.axis('off')
plt.show()


# ## Looking through the Dataset (Method-1)
# 
# I've taken some random images to observe the working of our brightness score function.

# In[ ]:


c=1
plt.figure(figsize=(10,10))
for i in range(3):
  for j in range(3):
    plt.subplot(3,3,c)
    idx = np.random.choice(range(50), replace=False)
    img = aug_data[idx]
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('score :'+str(brightness_score(img,method=1)))
    plt.axis('off')
    c+=1


# ## Looking through the Dataset (Method-2)
# 
# I've taken some random images to observe t

# In[ ]:


c=1
plt.figure(figsize=(10,10))
for i in range(3):
  for j in range(3):
    plt.subplot(3,3,c)
    idx = np.random.choice(range(50), replace=False)
    img = aug_data[idx]
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('score :'+str(brightness_score(img,method=2)))
    plt.axis('off')
    c+=1


# ### Thank You
# ### Feel free to share your views and comments.
# @V.V.S.S. Anil Kumar, vvssak1023@gmail.com

# In[ ]:




