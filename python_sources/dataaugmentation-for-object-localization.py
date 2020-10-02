#!/usr/bin/env python
# coding: utf-8

# ## Data Augmentation for Object Localization 
# In this kernel i'm going to augment data for the task of object localization that is Bouding Box prediction(Only).
# Input is in the following form
# 
# > Image_name, x1, y1, x2, y2
# 
# **Note** ; Same augmentation techniques can be used if the data is in the form of bx,by,bh,bw
# 
# Where x1, y1 are the top left cornor coordinates of bouding box and x2,y2 are bottom right cornor coordinate of bounding box.
# 
# ### Data Augmentation Techniques
# 1. Horizontal Flip
# 2. Vertical Flip
# 3. Horizontal and Vertical Flip (Combined)
# 
# As we are specifically doing object localization so other Data Augmentation techniques like Random Crop, transforms, Angle change cannot be applied here.

# ## Reading the Input

# In[ ]:


import numpy as np 
import pandas as pd 
import os

print(os.listdir("/kaggle/input/repository/savannahar68-DataAugmentation-dce458d/"))
PATH = "/kaggle/input/repository/savannahar68-DataAugmentation-dce458d/"


# In[ ]:


df = pd.read_csv(PATH + 'dataAug.csv')
df.head(5)


# In[ ]:


df = df.drop(df.columns[0], axis=1)


# In[ ]:


df.shape


# ## Importing libraries for visualizing images

# In[ ]:


import numpy as np, os, cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Displaying orignal image
# 

# In[ ]:


df.values


# In[ ]:


images = df.values
img_name = images[0][0]
x1 = images[0][1]
x2 = images[0][2]
y1 = images[0][3]
y2 = images[0][4]


# Here the Rectangle function expect input in the form of x1,y1,width,height
# 
# So width can be removed using x2-x1 
# 
# and Height by y2-y1

# In[ ]:


plt.imshow(Image.open(PATH + 'images/' + img_name))
plt.gca().add_patch(Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,edgecolor='r',facecolor='none'))


# ## Horizontal Flip
# If you are given x1,y1,x2,y2 as coordinates and you have to h_flip the image then the coordinates of the new image changes to 
# 
# Change the **x1** and **x2** to following 
# - x1 = image_width - x1
# - x2 = image_width - x2
# 
# Keep **y1**,**y2** same
# 
# To **h_flip** the image pass argument **1** to cv2.flip function of Opencv
# 
# Here we have image shape as (640,480,3) so image_width = 640

# In[ ]:


x11 = 640 - x1
x22 = 640 - x2
y11 = y1
y22 = y2
plt.imshow(cv2.flip(cv2.imread(PATH + "images/"+ img_name),1))
plt.gca().add_patch(Rectangle((x11,y11),x22-x11,y22-y11,linewidth=1,edgecolor='r',facecolor='none'))


# ## Vertical Flip
# If you are given x1,y1,x2,y2 as coordinates and you have to v_flip the image then the coordinates of the new image changes to 
# 
# Change the **y1** and **y2** to following 
# - y1 = image_height - y1
# - y2 = image_height - y2
# 
# Keep **x1**,**x2** same
# 
# To **v_flip** the image pass argument **0** to cv2.flip function of Opencv
# 
# Here we have image shape as (640,480,3) so image_height = 480

# In[ ]:


x11 = x1
x22 = x2
y11 = 480 - y1
y22 = 480 - y2
plt.imshow(cv2.flip(cv2.imread(PATH + "images/"+ img_name),0))
plt.gca().add_patch(Rectangle((x11,y11),x22-x11,y22-y11,linewidth=1,edgecolor='r',facecolor='none'))


# ## Horizontal Vertical Flip
# If you are given x1,y1,x2,y2 as coordinates and you have to h_flip_v_flip the image then the coordinates of the new image changes to 
# 
# Change the **x1** ,**x2** , **y1** , **y2** to following 
# - x1 = image_width - x1
# - x2 = image_width - x2
# - y1 = image_height - y1
# - y2 = image_height - y2
# 
# To **h_flip_v_flip** the image pass argument **-1** to cv2.flip function of Opencv
# 
# Here we have image shape as (640,480,3) so image_width = 640 and  image_height = 480

# In[ ]:


x11 = 640 - x1
x22 = 640 - x2
y11 = 480 - y1
y22 = 480 - y2
plt.imshow(cv2.flip(cv2.imread(PATH + "images/"+ img_name),-1))
plt.gca().add_patch(Rectangle((x11,y11),x22-x11,y22-y11,linewidth=1,edgecolor='r',facecolor='none'))


# **Now let's combine the above techniques and cerate the images and also append the generated images to the dataframe.

# In[ ]:


for i,img in enumerate(df.values):
    x1 = img[1]
    x2 = img[2]
    y1 = img[3]
    y2 = img[4]
    
    df = df.append({'image_name':img[0].split('.')[0]+'H_FLIP.png', 'x1':640-x1, 'x2':640-x2, 'y1':y1, 'y2':y2}, ignore_index=True)
    cv2.imwrite(PATH + "images/" + img[0].split('.')[0]+'H_FLIP.png', cv2.flip(cv2.imread(PATH+"images/"+img[0]),1))
    
    df = df.append({'image_name':img[0].split('.')[0]+'V_FLIP.png', 'x1':x1, 'x2':x2, 'y1':480-y1, 'y2':480-y2}, ignore_index=True)
    cv2.imwrite(PATH + "images/"+img[0].split('.')[0]+'V_FLIP.png', cv2.flip(cv2.imread(PATH+"images/"+img[0]),0)) 
    
    df = df.append({'image_name':img[0].split('.')[0]+'H_FLIPV_FLIP.png', 'x1':640-x1, 'x2':640-x2, 'y1':480-y1, 'y2':480-y2}, ignore_index=True)
    cv2.imwrite(PATH + "images/"+img[0].split('.')[0]+'H_FLIPV_FLIP.png', cv2.flip(cv2.imread(PATH+"images/"+img[0].split('.')[0]+'H_FLIP.png'),0))
   


# Earlier we had 5 images in our dataset, now we have 5 + 5*3 types of augmentation images in out dataset
# 
# 

# In[ ]:


df.shape


# In[ ]:


df.head(20) #This we have successfully augmented the data for bouding box prediction


# ## If the dataset has bx,by,bh,bw as coordinates then
# 
# ### For H_flip change bx to 
#    bx = image_width - bx
#    
#    rest all the coordinated remain same
#    
# ### For V_flip change by to 
#    by = image_height - by
#    
#    rest all the coordinated remain same
# 
# ### For H_flip_V_flip change bx,by to 
#    bx = image_width - bx
#    
#    by = image_height - by
#    
#    rest all the coordinated remain same(bw,bh)
#   
#   The below code satisfies the purpose

# In[ ]:


for i,img in enumerate(df.values):
    bx = img[1]
    by = img[2]
    bh = img[3]
    bw = img[4]
    
    df = df.append({'image_name':img[0].split('.')[0]+'H_FLIP.png', 'bx':640 - bx, 'by':by, 'bh':bh, 'bw':bw}, ignore_index=True)
    cv2.imwrite(PATH + "images/" + img[0].split('.')[0]+'H_FLIP.png', cv2.flip(cv2.imread(PATH+"images/"+img[0]),1))
    
    df = df.append({'image_name':img[0].split('.')[0]+'V_FLIP.png', 'bx':bx, 'by':480 - by, 'bh':bh, 'bw':bw}, ignore_index=True)
    cv2.imwrite(PATH + "images/"+img[0].split('.')[0]+'V_FLIP.png', cv2.flip(cv2.imread(PATH+"images/"+img[0]),0)) 
    
    df = df.append({'image_name':img[0].split('.')[0]+'H_FLIPV_FLIP.png', 'bx':640 - bx, 'by':640-by, 'bh':bh, 'bw':bw}, ignore_index=True)
    cv2.imwrite(PATH + "images/"+img[0].split('.')[0]+'H_FLIPV_FLIP.png', cv2.flip(cv2.imread(PATH+"images/"+img[0].split('.')[0]+'H_FLIP.png'),0))
   


# ### Thank you
# Connect with me on linkedin - https://www.linkedin.com/in/savannahar/
