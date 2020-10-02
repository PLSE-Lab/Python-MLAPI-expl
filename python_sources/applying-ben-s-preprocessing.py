#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 
import matplotlib.pyplot as plt

np.random.seed(55)
import os

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/train.csv')
#data.head()


# <h1>Applying Ben's Preprocessing</h1>
# 
# So in this kernel I will try to apply the Ben's preprocessing that is popular in APTOS competition. This preprocessing method dated back to previous diabetic retinopathy competitions (Ben Graham is the competition winner). 
# 
# According to the previous diabetic retinopathy winner the reasoning behind doing this preprocessing (points on the bracket is why I believe this might work for this competition) :
# * Enhance finer details (Enhance thin strokes)
# * Tackle different illumination problem. (Background color differences especially between B/W and Color scans)

# In[ ]:


view_count =  15
i_chk = np.random.randint(0,len(data), size = view_count)
sample_imgs = []
ben_sample_imgs = []
file_list = ['../input/train_images/{}.jpg'.format(data['image_id'].values[i_chk[i]]) for i in range(view_count)]

for i in range(view_count) :
    sample_img = cv2.imread(file_list[i])
    sample_img = cv2.cvtColor(sample_img,cv2.COLOR_BGR2RGB)
    sample_imgs.append(sample_img)
    ben_sample_imgs.append(cv2.addWeighted (sample_img,4, cv2.GaussianBlur(sample_img, (0,0) , 10) ,-4 ,128))


# In[ ]:


for i in range(15) :
    fig , ax = plt.subplots(1,2,figsize = (12,15))
    ax[0].imshow(sample_imgs[i])
    ax[1].imshow(ben_sample_imgs[i])
    #plt.autoscale(tight = 'True' , axis = 'y')
    ax[0].set_title(data['image_id'].values[i_chk[i]], y = 1)
    ax[1].set_title(str(data['image_id'].values[i_chk[i]]) + ' with preprocess', y = 1)
    


# <h2>Observations</h2>
# There are few things that I observed if I do this Image processing 
# * The difference between Colored scan and B/W scan are less apparent
# * Applies more contrasts to the text and might also enhance thinner strokes at the end of each stroke.
# 
# However I feel this also introduce minor problem as it also enhance the character from the back page.
# 
# Keep in mind that these differences are only from my human eyes, and might not applies directly to the model.
# 
# P.S. : This kernel is only a "what-if" kernel that is made out of my curiousity. 
# Please do not judge harshly I am still a noob and correct me if there are any mistakes on the comment section.
