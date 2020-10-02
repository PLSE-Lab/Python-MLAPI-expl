#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Hello guys! Hope everything is ok.
# 
# From my point of view, one of the most difficult parts on this challenge is related with Satellite images. Which kind of information we can extract from the images and how it will affect to our models for predicting oil prices.
# 
# So, I have one theory. Maybe is not the best but i think we can link both worlds in different ways. This is just a draft about my thoughts.
# 
# First of all, what is supposed to happen when a big pandemic like this impact the whole world? And a countrys government announce an alarm state?
# Yes, the people run outside cities. So if we are right, we will find that the lights of the small cities on the outsides should be brighter, while the lights of the big cities will remain the same (the quality and resolution of the photos dont allow us to identify differences at these levels)
# 
# Then, as a first step, we have to identify the light differences over time, giving a variation data for each 10 days period of time. Lets go!!

# In[ ]:


import cv2
import matplotlib.pyplot as plt


# We are going to use Italy as first reference(Coronavirus FIFO :P) , but it will be simple to switch between countries. The idea is to compare image differences between 2 mean images. these differences will show how people have moved around the country from one period to another, and hopefully, we'll see spikes when the virus starts to spread and states set the national alarm.

# In[ ]:


#Lets load the data
train = pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/COVID-19_train.csv')
test = pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/COVID-19_test.csv')

#Load images ( I took it from another kaggler). If you want to try with another dataset just replace "Italy" (put "France" as an Example)
TIF = "/kaggle/input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/tif/"
tif_list = os.listdir("/kaggle/input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/tif/")
italy_list = [i for i in tif_list if i.startswith("Italy")]
italy_list.sort()

#List where we are going to save the differences between images.
res_list = []


# This is important. If we want to measure the differences, we have to "delete" the cloud effect on the photos. Thats one of the challenges of analyzing the images, and my idea is to keep the brightest spots in the image. So lets define a mask. If you find a better way to do this, would be nice if you share it!

# In[ ]:



#define spots. The lower_white variable could be considered as a thresshold
lower_white = np.array([0,0,220], dtype=np.uint8)    
upper_white = np.array([0,0,255], dtype=np.uint8)


# Now, we have to apply the mask to every image, compare them and get the differences.

# In[ ]:


for i in range(len(italy_list)-1):
    Img = cv2.imread(TIF+italy_list[i])
    Img2 = cv2.imread(TIF+italy_list[i+1])
    
    #change colors
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
    Img2 = cv2.cvtColor(Img2, cv2.COLOR_BGR2HSV)
    
    #define masks
    mask_white_1 = cv2.inRange(Img, lower_white, upper_white)
    mask_white_2 = cv2.inRange(Img2, lower_white, upper_white)

    #IMG1
    result_white_1 = cv2.bitwise_and(Img, Img, mask=mask_white_1)
    result_white_1 = cv2.cvtColor(result_white_1, cv2.COLOR_HSV2BGR)

    #IMG2
    result_white_2 = cv2.bitwise_and(Img2, Img2, mask=mask_white_2)
    result_white_2 = cv2.cvtColor(result_white_2, cv2.COLOR_HSV2BGR)

    #save the differences (Im sure there is a better way to perform this operation)                                        
    difference = result_white_2 - result_white_1  
    res_list.append(difference)
    
    

#This is an output example
plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
plt.imshow(res_list[0])
plt.subplot(2, 2, 2)
plt.imshow(res_list[1])
plt.subplot(2, 2, 3)
plt.imshow(res_list[2])
plt.subplot(2, 2, 4)
plt.imshow(res_list[3])




# Now we have to count the new pixels on the image for each period, create an array and normalize it.

# In[ ]:


#white pixels on the image
movements = []
for i in range(len(res_list)):
    n_white_pix = np.sum(res_list[i] >= 150)
    movements.append(n_white_pix)    
print(movements)

#Normalizing
norm = np.linalg.norm(movements)
normal_array = movements/norm
print(normal_array)

#the dates will be the average of each pair of periods
date = ["2/3 jan", "3/3 jan", "1/3 Feb", "2/3 Feb" ,"3/3 Feb", "1/3 Mar", "2/3 Mar" ,"3/3 Mar", "1/3 Apr", "2/3 Apr" ,"3/3 Apr", "1/3 May", "2/3 May" ,"3/3 May"]


plt.figure(figsize=(15,10))
plt.plot(date, normal_array)
plt.show()





# So.. here it is! the biggest point seems to match with the moment when italy put its state of alarm, and also with the 2 weeks when all kind of trade was closed (there is at least no movement between 2/3 March and 3/3 March. Once those two weeks were over, the main industries were back (although the citizens were still at home), so we could explain the subsequent movements. 
# 
# Or maybe it is a coincidence! (remember that we dont have quality enough satellite images) but it seems to be more or less related right?

# Let me put the Italy new cases statistics to guess when the people started freaking out and moving to his little towns.

# In[ ]:



fig, ax = plt.subplots(figsize=(15,10))
ax.plot(train['Date'],train['Italy_new_cases'])
plt.grid()
plt.xticks(rotation='vertical')
ax.set(xlabel="Date",
       ylabel="cases",
       title="Italy cases over time") 


# As a final conclusion, i think light differences could be the key on this question, I will let you to test with another country as France (It seems to match well also), to see the results and guess what happened on each peak.
# 
# Thank you very much and every commentary will be well accepted. Also Forgive me for my level of programming, I'm a little out of practice.
