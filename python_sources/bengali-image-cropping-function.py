#!/usr/bin/env python
# coding: utf-8

# ## Purpose
# This image cropping function is developped for the competition "Bengali.AI Handwritten Grapheme Classification". Original Image data has some problems to be solved:
# * image is not located in the center
# * some of images have noise, which disturbs Machine Learning

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import resize
import time
import gc
from sklearn.metrics import confusion_matrix


# In[ ]:


IM_SIZE = 64
n_file = 4
n_each_train = 50210


# ## 1. Functions
# Cropping function and image plot function are defined. Cropping function has two features: cropping and centering.

# In[ ]:


#croping image
def crop_image(array, image_size, noise):
    threshold = 10
    threshold2 = 150
    # to avoid noise on the boundary
    if np.mean(array, axis = 1)[0] > noise + 10:
        array = array[2:,:]
    if np.mean(array, axis = 1)[-1] > noise + 10:
        array = array[0:-3,:]
    if np.mean(array, axis = 0)[0] > noise + 10:
        array = array[:,2:]
    if np.mean(array, axis = 0)[-1] > noise + 10:
        array = array[:,0:-3]
    
    x_max = np.quantile(array, axis = 1, q = 0.95)
    x_max2 = np.where(x_max >= threshold)[0]     
    
    T = 0.02
    while x_max2.shape[0] < 2:
        x_max = np.quantile(array, axis = 1, q = 0.95 + T)
        x_max2 = np.where(x_max >= threshold)[0] 
        T += 0.02
        T = min(T, 0.05)
    
    y_max = np.quantile(array, axis = 0, q = 0.95)    
    y_max2 = np.where(y_max >= threshold)[0] 
    
    T = 0.02
    while y_max2.shape[0] < 2:
        y_max = np.quantile(array, axis = 0, q = 0.95 + T)    
        y_max2 = np.where(y_max >= threshold)[0] 
        T += 0.02
        T = min(T, 0.05)
    
    x_max_m = np.max(array, axis = 1)
    y_max_m = np.max(array, axis = 0)    
    x_max2_m = np.where(x_max_m >= threshold2)[0]   
    y_max2_m = np.where(y_max_m >= threshold2)[0]         
    
    T = 10
    while x_max2_m.shape[0] < 2:
        x_max2_m = np.where(x_max_m >= threshold2 - T)[0] 
        T += 25
    T = 10
    while y_max2_m.shape[0] < 2:
        y_max2_m = np.where(y_max_m >= threshold2 - T)[0] 
        T += 25    
    
    x1 = int(np.mean(np.array([x_max2[0], x_max2_m[0]])))
    x2 = int(np.mean(np.array([x_max2[-1], x_max2_m[-1]]))) 
    
    y1 = int(np.mean(np.array([y_max2[0], y_max2_m[0]])))
    y2 = int(np.mean(np.array([y_max2[-1], y_max2_m[-1]])))
    
    margin = 3
    LX = x2-x1
    LY = y2-y1    
    DL = LX - LY
    ML = np.max([x2-x1, y2-y1])
    array2 = np.zeros((ML + 2*margin, ML + 2*margin))
    
    # centering
    if DL > 0:
        array2[margin:-margin,int(abs(DL)/2)+margin:int(abs(DL)/2)+LY+margin] = array[x1:x2,y1:y2]
    
    else:
        array2[int(abs(DL)/2)+margin:int(abs(DL)/2)+LX+margin,margin:-margin] = array[x1:x2,y1:y2]

    return resize(array2, (image_size, image_size), preserve_range=True).astype("uint8")


# In[ ]:


def image_plot(array1, array2):
    n = array1.shape[0]
    m = int(n/10)
    fig, ax = plt.subplots(10, 10, figsize = (19, 20))
    cmap = "gray"
    
    for i in range(n):
        x = int(i/10)
        y = i % 10
        ax[2*x, y].imshow(array1[i,:,:], cmap)
        ax[2*x, y].set_xticks([], [])
        ax[2*x, y].set_yticks([], [])
        ax[2*x+1, y].imshow(array2[i,:,:], cmap)
        ax[2*x+1, y].set_xticks([], [])
        ax[2*x+1, y].set_yticks([], [])
    
    plt.show()
    


# ## 2. Read Train files

# In[ ]:


list_col = []
for i in range(32332):
    list_col.append(str(i))


# In[ ]:


time1 = time.time()
train_m = np.zeros((8000, IM_SIZE, IM_SIZE), dtype = "uint8")
train_before = np.zeros((8000, 137, 236), dtype = "uint8")
train_m0 = np.zeros((137, 236), dtype = "int16")
k = 0
for i in range(n_file):
    time3 = time.time()
    print("reading train file",i)
    directory = "/kaggle/input/bengaliai-cv19/train_image_data_"+str(i)+".parquet"
    train_df0 = pd.read_parquet(directory, engine = "pyarrow", columns = list_col)

    gc.collect()

    print("image cropping start",i)
    time3 = time.time()
    samples = np.random.randint(0,n_each_train,2000)
    for j in samples:
        train_m0[:,:] = np.reshape(np.array(train_df0.iloc[j,:], dtype = "int16"), (137, 236))
        train_m0[:,:] = - train_m0[:,:] + 255
        add = 255 - np.max(train_m0)
        train_m0[:,:] = train_m0[:,:] + add
        noise = 65 + add
        train_m0[train_m0 < noise] = 0 # Noise Reduction
        train_m[k,:,:] = crop_image(train_m0, image_size = IM_SIZE, noise = noise).astype("uint8")
        train_before[k,:,:] = train_m0
        k += 1   

    del train_df0
    gc.collect()

time2 = time.time()
print(int(time2-time1), "sec")

del train_m0
gc.collect()


# ## 3. Cropped Images
# Here, both original and cropped images are plotted. Total number is 500 respectively.

# In[ ]:


samples = np.random.randint(0,8000, size = 50)
image_plot(train_before[samples,:,:], train_m[samples,:,:])


# In[ ]:


samples = np.random.randint(0,8000, size = 50)
image_plot(train_before[samples,:,:], train_m[samples,:,:])


# In[ ]:


samples = np.random.randint(0,8000, size = 50)
image_plot(train_before[samples,:,:], train_m[samples,:,:])


# In[ ]:


samples = np.random.randint(0,8000, size = 50)
image_plot(train_before[samples,:,:], train_m[samples,:,:])


# In[ ]:


samples = np.random.randint(0,8000, size = 50)
image_plot(train_before[samples,:,:], train_m[samples,:,:])


# In[ ]:


samples = np.random.randint(0,8000, size = 50)
image_plot(train_before[samples,:,:], train_m[samples,:,:])


# In[ ]:


samples = np.random.randint(0,8000, size = 50)
image_plot(train_before[samples,:,:], train_m[samples,:,:])


# In[ ]:


samples = np.random.randint(0,8000, size = 50)
image_plot(train_before[samples,:,:], train_m[samples,:,:])


# In[ ]:


samples = np.random.randint(0,8000, size = 50)
image_plot(train_before[samples,:,:], train_m[samples,:,:])


# In[ ]:


samples = np.random.randint(0,8000, size = 50)
image_plot(train_before[samples,:,:], train_m[samples,:,:])

