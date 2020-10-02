#!/usr/bin/env python
# coding: utf-8

# #### My work on the chapters 1, 2 and 3. 
# ### Content:
# * Uploading and viewing an image
# * Getting image resolution
# * Looking at pixel values
# * Converting color spaces
# * Saving an image
# * Creating basic drawings
# * Doing gamma correction
# * Rotating, shifting, and scalling images
# * Determining structural similarity

# # Chapters #1 and #2 were just warm-ups
# ### #1: Setup Environment.
# ### #2: Introduction to Image Processing.

# In[ ]:


#import cv2


# In[ ]:


#import keras


# In[ ]:


#import sklearn
#import skimage


# # Chapter #3: Basics of Python and Scikit Image

# ### Uploading and viewing an image:

# In[ ]:


from skimage import io, data
## I tryied to load the BOVESPA image but it didnt work. It says that it doesnt recognize the file path.
## I used io.imread('../input/imgtesting/BOVESPA-prego-trading-post-inicio-70-1.jpg')
cell_img = data.cell()
io.imshow(cell_img)


# In[ ]:


astronaut_img = data.astronaut()
io.imshow(astronaut_img)


# ### Getting Image Resolution:

# In[ ]:


#Getting image resolution, and number of channels
astronaut_img.shape


# ### Looking at Pixel Values:

# In[ ]:


#Getting pixel values
import pandas as pd
df = pd.DataFrame(astronaut_img.flatten())
#file_path = 'pixel_values_1.xlsx'
#df.to_excel(file_path, index=False)


# In[ ]:


astronaut_img.flatten()


# In[ ]:


df


# In[ ]:


astronaut_img


# ### Converting Color Space:

# In[ ]:


from skimage import color
from pylab import *


# In[ ]:


#Converting from RGB to HSV:
img_hsv = color.rgb2hsv(astronaut_img)

#Converting back from HSV to RGB:
img_hsv_rgb = color.hsv2rgb(img_hsv)


# In[ ]:


img_hsv.flatten()


# In[ ]:


img_hsv_rgb.flatten()


# In[ ]:


#Showing both images:
figure(0) #function figure in module matplotlib.pyplot; Create a new figure.
io.imshow(img_hsv)
figure(1)
io.imshow(img_hsv_rgb)


# In[ ]:


#Converting from RGB to XYZ
img_xyz = color.rgb2xyz(astronaut_img)
#and back to rgb
img_xyz_rgb = color.xyz2rgb(img_xyz)

figure(0)
io.imshow(img_xyz)
figure(1)
io.imshow(img_xyz_rgb)


# In[ ]:


#RGB to LAB:
img_lab = color.rgb2lab(astronaut_img)
#and back to rgb:
img_lab_rgb = color.lab2rgb(img_lab)

figure(0)
io.imshow(img_lab)
figure(1)
io.imshow(img_lab_rgb)


# In[ ]:


#rgb to YUV
img_yuv = color.rgb2yuv(astronaut_img)
#and back to RGB:
img_yuv_rgb = color.yuv2rgb(img_yuv)

figure(0)
io.imshow(img_yuv)
figure(1)
io.imshow(img_yuv_rgb)


# In[ ]:


#RGB to YIQ:
img_yiq = color.rgb2yiq(astronaut_img)
#back to RGB:
img_yiq_rgb = color.yiq2rgb(img_yiq)

figure(0)
io.imshow(img_yiq)
figure(1)
io.imshow(img_yiq_rgb)


# In[ ]:


#RGB to YPbPr
img_ypbpr = color.rgb2ypbpr(astronaut_img)
#back to rgb
img_ypbpr_rgb = color.ypbpr2rgb(img_ypbpr)

figure(0)
io.imshow(img_ypbpr)
figure(1)
io.imshow(img_ypbpr_rgb)


# For saving:

# In[ ]:


io.imsave("this.jpg", img_ypbpr)


# ### Creating basic Drawings

# In[ ]:


from skimage import draw


# Lines:

# In[ ]:


x, y = draw.line(0, 0, 511, 511)
astronaut_img[x, y] = 0 ## changing the color of the line
# (x, y) are the coordinates of the line
io.imshow(astronaut_img)


# Polygons:

# In[ ]:


#Rectangle example:
def rectangle(x, y, w, h):
    rr, cc = [x, x+w, x+w, x], [y, y, y+h, y+h]
    return (draw.polygon(rr, cc))

rr, cc = rectangle(30, 30, 100, 100)
astronaut_img[rr, cc] = 1
io.imshow(astronaut_img)


# Cyrcle:

# In[ ]:


#Defining circle coordinates and radius:
x, y = draw.circle(300, 300, 100)
#Draw circle:
astronaut_img[x, y] = 1
#show image:
io.imshow(astronaut_img)


# Bezier Curve:

# In[ ]:


#Defining Bezier Curve coordinates:
x, y = draw.bezier_curve(0,0, 100, 100, 300, 450, 200)
#Drawing Bezier Curve:
astronaut_img[x, y] = 250

io.imshow(astronaut_img)


# ### Doing Gamma Correction

# In[ ]:


from skimage import exposure
from pylab import *

img = data.astronaut()

gamma_corrected0 = exposure.adjust_gamma(img, 0.2)
gamma_corrected1 = exposure.adjust_gamma(img, 0.5)
gamma_corrected2 = exposure.adjust_gamma(img, 2)
gamma_corrected3 = exposure.adjust_gamma(img, 5)

figure(0)
io.imshow(gamma_corrected0)
figure(1)
io.imshow(gamma_corrected1)
figure(2)
io.imshow(gamma_corrected2)
figure(3)
io.imshow(gamma_corrected3)


# ### Rotating, Shifting and Scalling Images

# In[ ]:


from skimage.transform import rotate
img_rot_180 = rotate(img, 180) #how many degrees to rotate? 180
io.imshow(img_rot_180)


# In[ ]:


from skimage.transform import resize

img_resized10 = resize(img, (10, 10))
img_resized20 = resize(img, (20, 20))
img_resized40 = resize(img, (40, 40))
img_resized80 = resize(img, (80, 80))
img_resized160 = resize(img, (160, 160))

figure(0)
io.imshow(img_resized10)
figure(1)
io.imshow(img_resized20)
figure(2)
io.imshow(img_resized40)
figure(3)
io.imshow(img_resized80)
figure(4)
io.imshow(img_resized160)


# ### Determining Strutural Similarity

# In[ ]:


from skimage.measure import compare_ssim as ssim

ssim_original = ssim(img, img, data_range=img.max()-img.min(), multichannel=True)
ssim_different = ssim(img, img_xyz, data_range=(img_xyz.max() - img_xyz.min()), multichannel=True)

print(ssim_original, "/", ssim_different)


# In[ ]:




