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


from fastai.vision import *
from fastai.metrics import error_rate


# In[ ]:


path=pathlib.Path(os.path.normpath('../input/car_data/car_data'))


# In[ ]:


os.listdir(path)


# In[ ]:


train=path/'train'
test=path/'test'


# In[ ]:


tfms=get_transforms()


# In[ ]:


data=ImageDataBunch.from_folder(path=path,valid='test',bs=32,ds_tfms=tfms,size=(320,320)).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3,figsize=(12,12))


# In[ ]:


learn=cnn_learner(data,models.resnet50,pretrained=True,metrics=error_rate,model_dir='/tmp/model/')


# In[ ]:


data.classes


# In[ ]:


learn.fit_one_cycle(10)


# In[ ]:


learn.save('stage-1')


# In[ ]:


interp=ClassificationInterpretation.from_learner(learn)
losses,idx=interp.top_losses()
interp.plot_top_losses(9,figsize=(12,12))


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10,max_lr=slice(3e-06,4e-06))


# In[ ]:


learn.save('stage-2')


# In[ ]:


data.classes


# In[ ]:


listofSubdirectory=os.listdir(test)
len(listofSubdirectory)


# In[ ]:


import glob
from PIL import Image
root='../input/car_data/car_data/test/'
images=list()
for i in listofSubdirectory:
    images.append(os.listdir(root+i))
    
images


# In[ ]:


learn.load('stage-2')


# In[ ]:


learn.export('/tmp/model/export.pkl')


# In[ ]:


learn=load_learner('/tmp/model/')


# In[ ]:


columns = ['Id','Predicted']
listofprediction=list()
for i in range(len(listofSubdirectory)):
    for j in images[i]:
        pt=root+listofSubdirectory[i]+'/'+j
        img=open_image(pt)
        name=j.replace('.jpg','')
        _,pred_idx,output=learn.predict(img)
        
        listofprediction.append([name,pred_idx.item()])


# In[ ]:


array=np.array(listofprediction)


# In[ ]:


submission=pd.DataFrame(array,columns=columns)
submission


# For training and validation accuracy please refer to the cell No 17 as I don't know how to print it but the final value of it is the accuracy for training and valid and error_rate is the error on valid loss whose accuracy = 1-error_rate

# In[ ]:


submission.to_csv('submission.csv',index=False)


# In[ ]:




