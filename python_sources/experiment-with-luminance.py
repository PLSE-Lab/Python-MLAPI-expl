#!/usr/bin/env python
# coding: utf-8

# **Read files**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os
from glob import glob
TRAIN_DATA = "../input/train"
type_1_files = glob(os.path.join(TRAIN_DATA, "Type_1", "*.jpg"))
type_1_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_1"))+1:-4] for s in type_1_files])
type_2_files = glob(os.path.join(TRAIN_DATA, "Type_2", "*.jpg"))
type_2_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_2"))+1:-4] for s in type_2_files])
type_3_files = glob(os.path.join(TRAIN_DATA, "Type_3", "*.jpg"))
type_3_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_3"))+1:-4] for s in type_3_files])

print(len(type_1_files), len(type_2_files), len(type_3_files))
print("Type 1", type_1_ids[:10])
print("Type 2", type_2_ids[:10])
print("Type 3", type_3_ids[:10])


# In[ ]:


def get_filename(image_id, image_type):
    """
    Method to get image file path from its id and type   
    """
    if image_type == "Type_1" or         image_type == "Type_2" or         image_type == "Type_3":
        data_path = os.path.join(TRAIN_DATA, image_type)
    elif image_type == "Test":
        data_path = TEST_DATA
    elif image_type == "AType_1" or           image_type == "AType_2" or           image_type == "AType_3":
        data_path = os.path.join(ADDITIONAL_DATA, image_type[1:])
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    ext = 'jpg'
    return os.path.join(data_path, "{}.{}".format(image_id, ext))


def get_image_data(image_id, image_type):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# In[ ]:


import matplotlib.pylab as plt

def plt_st(l1,l2):
    plt.figure(figsize=(l1,l2))


# In[ ]:


img_1 = get_image_data('208', 'Type_1')
file_1 = get_filename('208','Type_1')


# In[ ]:


plt_st(20, 20)
plt.imshow(img_1)
plt.title("Training dataset of type %i" % (1))


# In[ ]:


from PIL import Image, ImageDraw
from PIL import ImageFont

imag = Image.open(file_1)
#Convert the image te RGB if it is a .gif for example
imag = imag.convert ('RGB')
#coordinates of the pixel
X,Y = 0,0
#Get RGB
pixelRGB = imag.getpixel((X,Y))
R,G,B = pixelRGB 
#Luminance photometric/digital
Y = 0.2126*R + 0.7152*G + 0.0722*B


# In[ ]:


def luminosity(rgb, rcoeff=0.2126, gcoeff=0.7152, bcoeff=0.0722):
   return rcoeff*rgb[0] + gcoeff*rgb[1] + bcoeff*rgb[2]

def gen_pix_factory(im):
   num_cols, num_rows=im.size
   r, c = 0,0
   while r!=num_rows:
       c = c%num_cols
       print (c)
       print (r)
       yield ((c,r), im.getpixel((c,r)) 
       if c == (num_cols-1): 
              r+=1
       c+=1
def rgb_to_gray_level(rgb_img, conversion=luminosity):        
       gl_img = Image.new('L',rgb_img.size)
       gen_pix = gen_pix_factory(gl_img)
       lum_pix = ((gp[0],conversion(gp[1])) for gp in gen_pix)
       for lp in lum_pix:
           gl_img.putpixel(lp[0],int(lp[1]))
       return gl_img
   
def binarize(gl_img, thresh=70):
   gen_pix = gen_pix_factory(gl_img)
   for pix in gen_pix:
       if pix[1] <= thesh:
           gl_img.putpixel(pix[0], 0)
       else:
           gl_img.putpixel(pix[0],255)
       
       


# In[ ]:


from PIL import Image, ImageDraw
from PIL import ImageFont
im = Image.open(file_1)
im2 = rgb_to_gray_level(im, conversion=luminosity)
plt_st(20, 20)
plt.imshow(im2)
plt.title("Training dataset of type %i" % (1))

