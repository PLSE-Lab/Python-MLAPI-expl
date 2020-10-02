#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../"))

# Any results you write to the current directory are saved as output.


# In[2]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from fastai import *
from fastai.vision import *


# In[4]:


bs=64


# In[5]:


get_ipython().system('mkdir data')


# In[6]:


mycsv=pd.read_csv("../input/train.csv")
mycsv.head()


# In[7]:


get_ipython().system('mkdir ./data/train')
get_ipython().system('mkdir ./data/train/1')
get_ipython().system('mkdir ./data/train/0')
for i in mycsv.values:
    shutil.copy("../input/train/train/"+i[0],"./data/train/"+str(i[1])+"/"+i[0])


# In[8]:


path=Path('./data')


# In[9]:


data= ImageDataBunch.from_folder(path,train="./train",valid_pct=0.2,ds_tfms=get_transforms(),size=bs,num_workers=0).normalize(imagenet_stats)


# In[10]:


data.show_batch(rows=3,figsize=(7,6))


# In[ ]:


#!mkdir ../test
#for file in os.listdir('../input/test/test'):
    #shutil.copy('../input/test/test/'+file,'../test/'+file)


# In[17]:


data.classes


# In[11]:


learn=create_cnn(data,models.resnet34, metrics=error_rate)


# In[13]:


learn.lr_find()


# In[14]:


learn.recorder.plot()


# In[19]:


learn.fit_one_cycle(4)


# In[35]:


interp= ClassificationInterpretation.from_learner(learn)
losses,indxs=interp.top_losses()
len(data.valid_ds)==len(losses)==len(indxs)


# In[36]:


interp.plot_top_losses(9, figsize=(8,8))


# In[37]:


interp.plot_confusion_matrix(figsize=(12,12),dpi=60)


# In[23]:


learn.save('stage-1')


# In[ ]:


get_ipython().system('cd data/models && ls')


# In[ ]:


os.listdir('../input/test/test/')[0]


# In[ ]:


img=open_image("../input/test/test/c662bde123f0f83b3caae0ffda237a93.jpg")


# In[24]:


learn.unfreeze()
learn.fit_one_cycle(2,max_lr=slice(1e-4,1e-2))
learn.save('stage-2')


# In[25]:


data2= ImageDataBunch.single_from_classes(path,["0","1"],ds_tfms=get_transforms(),size=64).normalize(imagenet_stats)


# In[34]:


learn=create_cnn(data2,models.resnet34).load('stage-2')


# In[27]:


mydf={'id':[],'has_cactus':[]}
for i in os.listdir('../input/test/test'):
    img=open_image("../input/test/test/"+i)
    pred_class, pred_idxs, outputs = learn.predict(img)
    mydf['id'].append(i)
    mydf['has_cactus'].append(pred_class)


# In[ ]:


mydf=pd.DataFrame(mydf)
mydf.head()


# In[ ]:


file=mydf.to_csv('test.csv',sep=',',index=False)


# In[ ]:


get_ipython().system('rm -r ./data')


# In[ ]:


from IPython.display import FileLink
FileLink('test.csv')

