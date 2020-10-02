#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import os
import time
import shutil


# In[ ]:


ls


# In[ ]:


data=pd.read_csv("../input/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv")
data


# In[ ]:


unique=[]
for i in data['Label']:
    if i not in unique:
        unique.append(i)
unique


# In[ ]:


Path_train="../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/"
Total_images=len(os.listdir(Path_train))
normal=0
infected =0
name=data['X_ray_image_name']
label=data['Label']
image_type=data['Dataset_type']
all_dir=os.listdir(Path_train)

os.mkdir("train")
os.mkdir("train/Infected")
os.mkdir("train/Normal")

wrong=0


# In[ ]:


for i in range(len(image_type)):
    if image_type[i]=='TRAIN':
        if name[i] in all_dir:
            if label[i]=='Normal':
                normal=normal+1
                shutil.copy(Path_train+'/'+name[i],'train/Normal/'+name[i])
            else:
                infected=infected+1
                shutil.copy(Path_train+'/'+name[i],'train/Infected//'+name[i])
        else:
            wrong=wrong+1


# In[ ]:


print(normal,infected,wrong)


# In[ ]:


Path_test="../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/"
Total_images=len(os.listdir(Path_test))
normal=0
infected =0
name=data['X_ray_image_name']
label=data['Label']
image_type=data['Dataset_type']
all_dir=os.listdir(Path_test)

os.mkdir("test")
os.mkdir("test/Infected")
os.mkdir("test/Normal")


wrong=0


# In[ ]:


for i in range(len(image_type)):
    if image_type[i]=='TEST':
        if name[i] in all_dir:
            if label[i]=='Normal':
                normal=normal+1
                shutil.copy(Path_test+'/'+name[i],'test/Normal/'+name[i])
            else:
                infected=infected+1
                shutil.copy(Path_test+'/'+name[i],'test/Infected//'+name[i])
        else:
            wrong=wrong+1


# In[ ]:


print(normal,infected,wrong)


# In[ ]:


import cv2
import matplotlib.pyplot as plt

normal_sample = cv2.imread("train/Normal/"+os.listdir("train/Normal")[3])
infected_sample = cv2.imread("train/Infected/"+os.listdir("train/Infected")[2])

plt.imshow(normal_sample)
plt.title("NORMAL")
plt.show()

plt.imshow(infected_sample)
plt.title("INFECTED")
plt.show()


# In[ ]:


from fastai.vision import *
from fastai.utils.mem import *
from fastai.callbacks.hooks import *


# In[ ]:


classes = ['covid','normal']
data_path='train'


# In[ ]:


batch = 8
data = ImageDataBunch.from_folder(data_path,
       train = Path_train, valid_pct = 0.4,
       ds_tfms = get_transforms(),size = 224,seed = 42,bs=batch).normalize(imagenet_stats)


# In[ ]:


data.classes


# In[ ]:


len(data.train_ds)


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=[accuracy])


# In[ ]:


epochs=10


# In[ ]:


learn.fit_one_cycle(epochs)


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find(start_lr = 1e-6, end_lr = 1e-1)


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(epochs,max_lr=slice(1e-5,1e-4))


# In[ ]:


learn.save('model')


# In[ ]:


learn.export()


# In[ ]:




