#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


os.listdir("../input")


# In[ ]:


train_imgs=os.listdir("../input/train/train/")


# In[ ]:


train_imgs[:5]


# In[ ]:


import pathlib


# In[ ]:


path=pathlib.Path("../input/train/train/")


# In[ ]:


fnames=get_image_files(path)
fnames[:5]


# In[ ]:


def return_label(path):
    if str(path).find('dog')==-1:
        return "cat"
    else:
        return "dog"


# In[ ]:


return_label(pathlib.Path("../input/train/train/cat.1.jpg"))


# In[ ]:


data = ImageDataBunch.from_name_func(path, fnames, return_label, ds_tfms=get_transforms(), size=224,bs=8)
data.normalize(imagenet_stats)


# In[ ]:


data.train_ds.y.classes


# In[ ]:


TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"

learn=create_cnn(data,models.resnet34,metrics=[accuracy,error_rate],path=TMP_PATH,model_dir=MODEL_PATH)


# In[ ]:


learn.fit_one_cycle(1)


# In[ ]:


learn.save('model_224')


# In[ ]:


os.listdir("../../tmp/model/")


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


interp=ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


#dog=1 and cat=0
test_path=pathlib.Path("../input/test1/test1/")
test_images=get_image_files(test_path)
test_images[:5]


# In[ ]:


id_list=[]
labels=[]
import re
for img_path in test_images:
    #print(img)
    img_id=re.findall('\d+',str(img_path))
    id_list.append(int(img_id[2]))
    img=open_image(img_path)
    label=learn.predict(img)
    if str(label[0])=='cat':
        labels.append(0)
    else:
        labels.append(1)
    print(img_id,"  ",label)


# In[ ]:


len(labels)


# In[ ]:


len(id_list)


# In[ ]:


os.listdir("../input")


# In[ ]:


my_submission=pd.DataFrame({"id":id_list,"label":labels})


# In[ ]:


my_submission.to_csv('submission.csv', index=False)


# In[ ]:


os.listdir()


# In[ ]:


sub=pd.read_csv("submission.csv")


# In[ ]:


sub.head()


# In[ ]:




