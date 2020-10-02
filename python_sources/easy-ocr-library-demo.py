#!/usr/bin/env python
# coding: utf-8

# Special Thanks : Bhavesh Bhatt video
# 
# link : [How to extract text from images using EasyOCR Python Library (Deep Learning)](https://www.youtube.com/watch?v=ic4chj-iMaI)
# 
# Github Repo : [Easy OCR](https://github.com/JaidedAI/EasyOCR)

# In[ ]:


get_ipython().system('pip install easyocr')


# In[ ]:


import matplotlib.pyplot as plt
import cv2
import easyocr
from pylab import rcParams
from IPython.display import Image
rcParams['figure.figsize'] = 8, 16


# In[ ]:


reader = easyocr.Reader(['en'])


# In[ ]:


get_ipython().system('ls ../input/')


# In[ ]:


path = "../input/"


# In[ ]:


import PIL
from PIL import ImageDraw
img = PIL.Image.open(path+"1.jpg")
img


# In[ ]:


output = reader.readtext(path+'1.jpg')


# In[ ]:


output[0][-2],output[1][-2],output[2][-2]


# In[ ]:


def draw_boxes(image, bounds, color='yellow', width=2):
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image

draw_boxes(img,output)


# ## This is pretty cool!! 
# Lets try little tougher

# In[ ]:


img = PIL.Image.open(path+"3.png")


# In[ ]:


img


# In[ ]:


output = reader.readtext(path+"3.png")
draw_boxes(img,output)


# In[ ]:


for i in range(len(output)):
    print(output[i][-2])


# ## OOPS! Didn't read character "I".

# In[ ]:


img = PIL.Image.open(path+"4.jpg")
img


# In[ ]:


output = reader.readtext(path+"4.jpg")
print(output)


# In[ ]:


for i in range(len(output)):
    print(output[i][-2])


# In[ ]:


draw_boxes(img,output)


# In[ ]:


img = PIL.Image.open(path+"1.png")
img


# In[ ]:


output = reader.readtext(path+"1.png")
print(output)


# In[ ]:


for i in range(len(output)):
    print(output[i][-2])


# In[ ]:


draw_boxes(img,output)


# In[ ]:




