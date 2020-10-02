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


# In[ ]:


import torch
from fastai.vision import *
from fastai.metrics import error_rate


# In[ ]:


torch.cuda.set_device(0)


# In[ ]:


torch.cuda.get_device_name()


# In[ ]:


traindf = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')


# In[ ]:


traindf.head()


# In[ ]:


traindf.shape


# In[ ]:


classdata = (traindf.healthy+traindf.multiple_diseases+traindf.rust+traindf.scab)


# In[ ]:


classdata.head()


# In[ ]:


any(classdata >1)


# In[ ]:


traindf['image_id'] = traindf['image_id'].astype('str') + ".jpg"


# In[ ]:


traindf.head()


# In[ ]:


traindf['label'] = (0*traindf.healthy+1*traindf.multiple_diseases+2*traindf.rust+3*traindf.scab)


# In[ ]:


traindf.drop(columns=['healthy','multiple_diseases','rust','scab'],inplace=True)


# In[ ]:


traindf.head()


# In[ ]:


tfms = get_transforms(do_flip =True,
                     flip_vert=True,
                     max_lighting=0.1,
                     max_zoom=1.05,
                     max_warp=0.1,
                     max_rotate=20,
                     p_affine=0.75,
                     p_lighting=0.75)


# In[ ]:


path = '/kaggle/input/plant-pathology-2020-fgvc7/'


# In[ ]:


data = ImageDataBunch.from_df(path=path,
                             df=traindf,
                             folder="images",
                             label_delim=None,
                             valid_pct=0.2,
                             seed=100,
                             fn_col=0,
                             label_col=1,
                             suffix='',
                             ds_tfms=tfms,
                             size=512,
                             bs=64,
                             val_bs=32,)


# In[ ]:


data.show_batch(rows=3,figsize=(8,8))


# In[ ]:


data = data.normalize(imagenet_stats)


# In[ ]:


learner = cnn_learner(data,models.resnet50,pretrained=True,metrics=[error_rate,accuracy]).to_fp16()


# In[ ]:


learner.model_dir = '/kaggle/working/models'


# In[ ]:


learner.lr_find(start_lr=1e-07,end_lr=0.2,num_it=100)


# In[ ]:


learner.recorder.plot(suggestion=True)


# In[ ]:


mingradlr = learner.recorder.min_grad_lr
mingradlr


# In[ ]:


lr=mingradlr


# In[ ]:


learner.fit_one_cycle(2,lr)


# In[ ]:


learner.unfreeze()


# In[ ]:


learner.lr_find(start_lr=1e-07,end_lr=0.2,num_it=100)
learner.recorder.plot(suggestion=True)


# In[ ]:


mingradlr01 = learner.recorder.min_grad_lr
mingradlr01


# In[ ]:


learner.fit_one_cycle(3,slice(mingradlr01,mingradlr01/20))


# In[ ]:


learner.show_results()


# In[ ]:


inter = ClassificationInterpretation.from_learner(learner)


# In[ ]:


inter.plot_confusion_matrix(title='Confusion matrix')


# In[ ]:


test_df = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')


# In[ ]:


test_df.head()


# In[ ]:


sample_sub = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')


# In[ ]:


sample_sub.head()


# In[ ]:


path = '/kaggle/input/plant-pathology-2020-fgvc7/'


# In[ ]:


testdata = ImageList.from_folder(path+"images")


# In[ ]:


testdata.filter_by_func(lambda x: x.name.startswith("Test"))


# In[ ]:


testdata.items[0]


# In[ ]:


img = open_image(testdata.items[0])


# In[ ]:


testdata.items[0]


# In[ ]:


img


# In[ ]:


learner.predict(img)


# In[ ]:


val = learner.predict(img)[2].tolist()


# In[ ]:


val


# In[ ]:


resultlist = []
for item in testdata.items:
    img = open_image(item)
    pred = learner.predict(img)[2].tolist()
    pred.insert(0,item.name[:-4:])
    resultlist.append(pred)


# In[ ]:


resultlist[0:5]


# In[ ]:


resultdf = DataFrame(resultlist)


# In[ ]:


resultdf.head()


# In[ ]:


resultdf.columns = sample_sub.columns


# In[ ]:


resultdf.head()


# In[ ]:


resultdf.set_index("image_id",inplace=True)


# In[ ]:


resultdf.head()


# In[ ]:


resultdf = resultdf.loc[sample_sub.image_id,:]


# In[ ]:


resultdf.head()


# In[ ]:


resultdf.reset_index(inplace=True)


# In[ ]:


resultdf.head()


# In[ ]:


resultdf.to_csv("submission.csv",index=False)


# In[ ]:




