#!/usr/bin/env python
# coding: utf-8

# In[35]:


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


# In[36]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[37]:


import matplotlib.pyplot as plt
import os

from fastai.vision import *
import torchvision.models as models


# In[38]:


get_ipython().system('ls ../input')


# In[39]:


all_train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print(all_train_df.shape, test_df.shape)


# In[40]:


all_train_df.head()


# In[41]:


val_df = all_train_df.sample(frac=0.2, random_state=1337)
train_df = all_train_df.drop(val_df.index)
train_df['fn'] = train_df.index
print(train_df.shape, val_df.shape)


# In[42]:


train_df.head()


# In[43]:


class PixelImageItemList(ImageList):
    def open(self,fn):
        regex = re.compile(r'\d+')
        fn = re.findall(regex,fn)
        df = self.inner_df[self.inner_df.fn.values == int(fn[0])]
        df_fn = df[df.fn.values == int(fn[0])]
        img_pixel = df_fn.drop(labels=['label','fn'],axis=1).values
        img_pixel = img_pixel.reshape(28,28)
        img_pixel = np.stack((img_pixel,)*3,axis=-1)
        return vision.Image(pil2tensor(img_pixel,np.float32).div_(255))


# In[44]:


src = (PixelImageItemList.from_df(train_df,'.',cols='fn')
      .split_by_rand_pct()
      .label_from_df(cols='label'))


# In[45]:


data = (src.transform(tfms=(rand_pad(padding=5,size=28,mode='zeros'),[]))
       .databunch(num_workers=2,bs=128))


# In[46]:


data


# In[80]:


arch = vision.models.resnet152
learn = cnn_learner(data, arch, pretrained=True, metrics=accuracy)


# In[81]:


learn.lr_find()


# In[82]:


learn.recorder.plot()


# In[83]:


lr = 5e-2
learn.fit_one_cycle(3, slice(lr))


# In[84]:


learn.save('frozen-resnet50')
learn.unfreeze()


# In[85]:


learn.recorder.plot_losses()


# In[86]:


learn.lr_find()
learn.recorder.plot()


# In[87]:


learn.fit_one_cycle(3, slice(5e-5, lr/10))


# In[88]:


learn.recorder.plot_losses()


# In[89]:


df_test = pd.read_csv('../input/test.csv')
df_test['label'] = 0
df_test['fn'] = df_test.index
df_test.head()


# In[90]:


learn.data.add_test(PixelImageItemList.from_df(df_test, path='.', cols='fn'))
test_pred, test_y, test_loss = learn.get_preds(ds_type=DatasetType.Test, with_loss=True)
test_result = torch.argmax(test_pred,dim=1)
result = test_result.numpy()


# In[91]:


test_pred.shape


# In[92]:


#create a CSV file to submit
final = pd.Series(result,name='Label')
submission = pd.concat([pd.Series(range(1,28001),name='ImageId'),final],axis=1)
submission.to_csv('submission.csv',index=False)


# In[33]:


get_ipython().system('ls')


# In[ ]:




