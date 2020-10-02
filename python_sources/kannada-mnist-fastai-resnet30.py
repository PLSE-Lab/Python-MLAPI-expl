#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import shutil
import cv2
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


path = Path('../input/Kannada-MNIST')
train = pd.read_csv('../input/Kannada-MNIST/train.csv')
test  =pd.read_csv('../input/Kannada-MNIST/test.csv')
#sub = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
#dig = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')


# In[ ]:


#check the kaggle directory folder.
outpath = '/kaggle/'
os.listdir(outpath)
#create the output folder for train and test images
shutil.os.mkdir(outpath + "output")
shutil.os.mkdir(outpath + "output/train")
shutil.os.mkdir(outpath + "output/test")

#code to delete a directory and its contents.
#shutil.rmtree(outpath + "output") 
#shutil.rmtree(outpath + "output") 

#Check if the directory is created properly
outpath = '/kaggle/output'
os.listdir(outpath)


# In[ ]:


train.head()
test.head()


# In[ ]:


def save_images(outpath,df):
    for index, row in df.iterrows():
        pixels=np.asarray(row['pixel0':'pixel783'])
        img=pixels.reshape((28,28))
        pathname=os.path.join(outpath,'image_'+str(row['label'])+'_'+str(index)+'.jpg')
        cv2.imwrite(pathname,img)
        print('image saved ias {}'.format(pathname))

outpath = '/kaggle/output/train'
save_images(outpath,train)


# In[ ]:


path = Path('/kaggle/output/train')
path.ls()


# In[ ]:


#Lets us view couple of sample images
img = open_image('/kaggle/output/train/image_7_28127.jpg')
img.show(figsize=(5,5))


# #### Creating the data bunch

# In[ ]:


path_img = Path('/kaggle/output/train')

fnames = get_image_files(path_img)
fnames[:5]

# tempsrt = '/kaggle/output/train/image_5_39645.jpg'
# pat = r'/([^/]+)_\d+_\d+.jpg$'
# pat = r'/image_(\d+)_'
# m = re.search(pat,tempsrt)
# print(m.group(1))


# In[ ]:


np.random.seed(42)

#define the batch size
bs = 64

#define the regex pattern to get the label from the file name.
#pat = r'/([^/]+)_\d+_\d+.jpg$'
pat = r'/image_(\d+)_'

#lets do some data augmentation
#tfms = get_transforms(do_flip=False)

data = ImageDataBunch.from_name_re(path_img, fnames, pat, valid_pct=0.2,
        ds_tfms=get_transforms(), size=64, bs=bs, num_workers=4).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(7,8))


# In[ ]:


data.classes


# ### Building the model

# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir='/output/model/')


# In[ ]:


learn.fit_one_cycle(8)


# In[ ]:


learn.model
learn.save('stage-1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(8, max_lr=slice(1e-6,1e-4))


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(8, max_lr=slice(1e-6,1e-4))


# In[ ]:


lr=1e-5


# In[ ]:


learn.fit_one_cycle(8, lr)


# In[ ]:


submit = pd.DataFrame(columns=['id','label'])


# ### Lets get the predictions on the test images

# In[ ]:


#Remove the 
submit = pd.DataFrame(columns=['id','label'])
submit['id'] = test['id']

for index, row in test.iterrows():
#    sub.at[index, 'id'] = row.at[index,'id']
    pixels=np.asarray(row['pixel0':'pixel783'])
    arr=pixels.reshape((28,28))
    arr = np.stack([arr]*3,axis=0)
    img = Image(FloatTensor(arr))
    submit.at[index, 'label'] = int(learn.predict(img)[1])
    
submit.head(30)


# In[ ]:


submit.to_csv('submission.csv',index=False)

