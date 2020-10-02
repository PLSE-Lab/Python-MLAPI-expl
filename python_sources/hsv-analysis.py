#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
import cv2
import os


# In[46]:


def hsv_analysis(img_path,show_pic=False):    
    image=cv2.imread(img_path)
    if show_pic:
        image_rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         try:
#             plt.subplot(3, 3, i)
#         except:pass
        plt.imshow(image_rgb)
        plt.show()
        
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,s,v=np.average(hsv_image,axis=(0,1))
    return h,s,v
# print(os.listdir('../input'))
train = pd.read_csv('../input/imet-2019-fgvc6/train.csv')
read_len=train.id.__len__()
read_len=5
hsv_list=[]
for i in range(read_len):    
    img_path='../input/imet-2019-fgvc6/train/'+train.id[i]+".png"    
    hsv_list.append(hsv_analysis(img_path,show_pic=True))

hsv_list_sum=np.average(np.array(hsv_list),axis=0)
print(hsv_list_sum)


# What is hsv:
# ![](http://zh.wikipedia.org/wiki/HSL%E5%92%8CHSV%E8%89%B2%E5%BD%A9%E7%A9%BA%E9%97%B4)
# https://en.wikipedia.org/wiki/HSL_and_HSV

# In[48]:


# the first 1000 pics
# read_len=train.id.__len__()
read_len=1000
hsv_list=[]
for i in range(read_len):    
    img_path='../input/imet-2019-fgvc6/train/'+train.id[i]+".png"    
    hsv_list.append(hsv_analysis(img_path,show_pic=False))

hsv_list_sum=np.average(np.array(hsv_list),axis=0)
print(hsv_list_sum)
np.save('hsv_list.h5',np.array(hsv_list),)


# In[49]:


import seaborn as sns
df = pd.DataFrame(hsv_list, columns=["Hue", "y",'Brightness(Values)'])
sns.jointplot(x="Hue", y="Brightness(Values)", data=df)


# From the distribution , the brightness value is not low, largely because the background is white.
# 
# Most of the pictures are concentrated in the 0-40 range, indicating that the tue is warmer, probably because of the antiques.

# In[50]:


assert os.path.exists("../input/hsv-list-h5/hsv_list.h5.npy")
hsv_list=np.load("../input/hsv-list-h5/hsv_list.h5.npy")
import seaborn as sns
df = pd.DataFrame(hsv_list, columns=["Hue", "y",'Brightness(Values)'])
sns.jointplot(x="Hue", y="Brightness(Values)", data=df)


# The above picture is the distribution of all the pictures, and the distribution of the first 1000 is basically the same.

# In[ ]:





# In[ ]:




