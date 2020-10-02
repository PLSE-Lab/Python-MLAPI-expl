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
count = 0
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        count+=1
        #print(os.path.join(dirname, filename))
print(count)
# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('/kaggle/input/jantahackcomputervision/train_SOaYf6m/train.csv')
test = pd.read_csv('/kaggle/input/jantahackcomputervision/test_vc2kHdQ.csv')
train.head()


# In[ ]:


train.emergency_or_not.value_counts()


# In[ ]:


from PIL import Image

im = Image.open(r'/kaggle/input/jantahackcomputervision/train_SOaYf6m/images/1503.jpg')
width, height = im.size
print(width, height)


# In[ ]:


from IPython.display import Image 
pil_img = Image(filename='/kaggle/input/jantahackcomputervision/train_SOaYf6m/images/1503.jpg')
display(pil_img)


# In[ ]:


from fastai.imports import *
from fastai import *
from fastai.vision import *
from torchvision.models import *


# In[ ]:


path = '/kaggle/input/jantahackcomputervision/'
test_path = path+'Test Images/'
def generateSubmission(learn, string):
    id_list=list(test.image_names)
    label=[]
    for iname in id_list:
        img=open_image(path+"train_SOaYf6m/images/"+iname)
        label.append(learn.predict(img)[0])
        if (len(label)%350 == 0):
            print(f'{len(label)} images done!')
    #submissions = pd.DataFrame({'image_names':id_list,'emergency_or_not':label_vgg})#, 'res': label_res, 'dense': label_dense})
    #submissions.to_csv(string+"_submit.csv",index = False)
    return label


# In[ ]:


tfms = get_transforms(do_flip = True, max_lighting = 0.2, max_zoom= 1.1, max_warp = 0.15, max_rotate = 45)
data = ImageDataBunch.from_csv(path, folder= 'train_SOaYf6m/images', 
                              valid_pct = 0.0,
                              csv_labels = 'train_SOaYf6m/train.csv',
                              ds_tfms = tfms, 
                              fn_col = 'image_names',
                              #test = 'train_SOaYf6m/images', 
                              label_col = 'emergency_or_not',
                              bs = 16,
                              size = 300).normalize(imagenet_stats)


# In[ ]:


fbeta = FBeta(average='weighted', beta = 1)
learn = cnn_learner(data, models.resnet101, metrics=[accuracy, fbeta])


# In[ ]:


learn.fit(epochs = 30, lr = 1.5e-4)


# In[ ]:


sub = pd.read_csv('/kaggle/input/jantahackcomputervision/ss.csv')
sub['res_101'] = generateSubmission(learn, 'res_101')
sub['res_101'] = sub['res_101'].astype('int')
sub.head()


# In[ ]:


del learn
learn = cnn_learner(data, models.resnet50, metrics=[accuracy, fbeta])
learn.fit(epochs = 30, lr = 1.5e-4)


# In[ ]:


sub['res_50'] = generateSubmission(learn, 'res_50')
sub['res_50'] = sub['res_50'].astype('int')
sub.head()


# In[ ]:


del learn
learn = cnn_learner(data, models.densenet121, metrics=[accuracy, fbeta])
learn.fit(epochs = 50, lr = 6e-5)


# In[ ]:


sub['dense_121'] = generateSubmission(learn, 'res_50')
sub['dense_121'] = sub['dense_121'].astype('int')
sub.head()


# In[ ]:


del learn
learn = cnn_learner(data, models.densenet161, metrics=[accuracy, fbeta])
learn.fit(epochs = 25, lr = 1e-4)


# In[ ]:


sub['dense_161'] = generateSubmission(learn, 'res_50')
sub['dense_161'] = sub['dense_161'].astype('int')
sub.head()


# In[ ]:


del learn
learn = cnn_learner(data, models.resnet152, metrics=[accuracy, fbeta])
learn.fit(epochs = 25, lr = 5e-5)


# In[ ]:


sub['res_152'] = generateSubmission(learn, 'res_50')
sub['res_152'] = sub['res_152'].astype('int')
sub.head()


# In[ ]:


sub = sub.drop(['emergency_or_not'], axis = 1)
sub['emergency_or_not'] = sub.mode(axis = 1, numeric_only = True)


# In[ ]:


sub.info()


# In[ ]:


sub.head()


# In[ ]:


sub[['image_names', 'emergency_or_not']].to_csv('ensemble5.csv', index = False)
sub.to_csv('sub5.csv', index = False)


# In[ ]:




