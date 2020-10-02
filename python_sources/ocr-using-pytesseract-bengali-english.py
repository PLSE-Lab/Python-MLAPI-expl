#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pytesseract
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# First let's see the name of the images and explore the directory
filenames = os.listdir('../input/images/')
print(filenames)


# In[ ]:


# Let's start with a simple image
img = cv2.imread("../input/images/img_en_1.jpg") # image in BGR format
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig = plt.figure(figsize = [10,10])
height,width,channel = img.shape
plt.imshow(img)

print(type(img))
print(height,width,channel)


# In[ ]:


# as the image is simple enough, image_to_string method reads all characters almost perfectly!
text = pytesseract.image_to_string(img)
print(text)


# In[ ]:


# the output of OCR can be saved in a file in necessary
file = open('output.txt','a') # file opened in append mode
file.write(text)
file.close()


# In[ ]:


## lets start with a bit more complex image
# the illumination isn't good as previous one. So the accuracy of OCR is deterioorating

img2 = cv2.imread('../input/images/img_en_2.jpg')
img2 = cv2.cvtColor(img2 , cv2.COLOR_BGR2RGB) # we want the image in RGB mode

text2= pytesseract.image_to_string(img2)
fig = plt.figure(figsize = [10,10])
plt.imshow(img2)

print(img2.shape)
print(text2)


#we can see that the OCR for this image isn't that great. 
#Also it accidently read some of the texts form the previous page!


# In[ ]:


## Let's do some image processing for better OCR

img2= cv2.resize(img2,None, fx=.5, fy=0.5) #resizing the image
print(img2.shape)
fig= plt.figure(figsize= [10,10])
plt.imshow(img2)


# In[ ]:


gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)   # converting image to grayscale
fig = plt.figure(figsize= [10,10])
plt.imshow(gray,cmap='gray', vmin=0, vmax=255)  # while plotting grayscale image with matplotlib, cmap should be defined


# In[ ]:


text2= pytesseract.image_to_string(gray)
print(text2)
## we can already see the improvement in result


# In[ ]:


#some letters are coming from previous page.lets try adaptive thresholding
adaptive_threshold = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY ,85, 11 )
fig = plt.figure(figsize= [10,10])
plt.imshow(adaptive_threshold,cmap='gray', vmin=0, vmax=255)


# In[ ]:


text2= pytesseract.image_to_string(adaptive_threshold)
print(text2)
## the accuracy has improved even more!
# however, still pytesseract couldn't recognize a lot of characters, further processing can improve the scenatio.


# ## working with Bengali OCR
# * now the language sould be defined. But pytesseract doesn't have Bengali language by default. So a file called Bengali.trineddata needed to be added in the directory in which pytecessaract is installed in this virtual machine!

# In[ ]:


img3 = cv2.imread('../input/images/img_bn_3.png')
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
plt.imshow(gray3,cmap='gray', vmin=0, vmax=255)


# In[ ]:


## the following line will throw error as bengali is not added as language by default
text3 = pytesseract.image_to_string(gray3, lang='../input/book-pages/Bengali.traineddata' )
print(text3)


# In[ ]:


## From the error messaage, we can see that pytesessaract is installed in the directory called '/usr/share/tesseract-ocr'
# it is trying to search for the language file in the following directory
filenames = os.listdir('/usr/share/tesseract-ocr/4.00/tessdata/')
print(filenames)

## we need to add the bengali.tranerdata in this directory


# In[ ]:


## this block of code is taken from (https://realpython.com/working-with-files-in-python/) tutorial

import shutil
src = '../input/Bengali.traineddata'
dest = '/usr/share/tesseract-ocr/4.00/tessdata/'
shutil.copy(src, dest)

filenames = os.listdir('/usr/share/tesseract-ocr/4.00/tessdata/')
print(filenames) # check that the file is added in this directory which was not present before.


# In[ ]:


## Now our language is set !
## let's process the image and detect the characters!

img3 = cv2.imread('../input/images/img_bn_3.png')
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
ret,thresh3 = cv2.threshold(img3,127,255,cv2.THRESH_BINARY)  # binary thresholding


# In[ ]:


## this function shows two images side-by-side
# this function will plot two images side by side
def plot_two_images(img1, img2, title1="", title2=""):
    fig = plt.figure(figsize=[15,15])
    ax1= fig.add_subplot(121)
    ax1.imshow(img1, cmap="gray")
    ax1.set(xticks=[], yticks=[], title=title1)
    
    ax2= fig.add_subplot(122)
    ax2.imshow(img2, cmap="gray")
    ax2.set(xticks=[], yticks=[], title=title2)


# In[ ]:


plot_two_images(img3, thresh3, 'Original image', 'Processed image')


# In[ ]:


text3 = pytesseract.image_to_string(thresh3, lang='Bengali' )
print(text3)


# # References:
# 1. [Text recognition(OCR) with Tesseract and Python](https://www.youtube.com/watch?v=JkzFjj2hjtw)
# 2. [Pytesseract for Bengali language](https://stackoverflow.com/questions/43034112/detecting-bangla-character-using-pytesseract)
# 3. [Bengali traineddata for pytesseract](https://github.com/tesseract-ocr/tessdata/blob/master/script/Bengali.traineddata)
# 4. [File mapulation with python OS directory](https://realpython.com/working-with-files-in-python/)
