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


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.widgets import *


# In[ ]:


# these codes are from @Dreamdragon's Kernel: https://www.kaggle.com/heye0507/fastai-1-0-with-customized-itemlist/notebook
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


# # Data Pre-processing

# In[ ]:


df_train = pd.read_csv('/kaggle/input/train.csv')
df_test = pd.read_csv('/kaggle/input/test.csv')


# In[ ]:


df_train['fn'] = df_train.index
df_train.head()


# In[ ]:


df_test['label'] = 0
df_test['fn'] = df_test.index
df_test.head()


# In[ ]:


# add transform (optional), we are only zero pad and random zoom in this case. 
# calling get_transform() with flipping / lighting wont do much good for the 28*28 grey imgs(even though is RGB now)
data = (PixelImageItemList.from_df(df_train, '.', cols='fn')
        .split_by_rand_pct()
        .label_from_df(cols='label')
        .transform(tfms=get_transforms(do_flip=False))
        .databunch(num_workers=2, bs=128)
        .normalize(imagenet_stats))


# In[ ]:


data.classes, data.c


# In[ ]:


print(len(data.train_ds), len(data.valid_ds))


# In[ ]:


data.train_ds[0][0]


# In[ ]:


data.valid_ds[0][0]


# # Training

# ## ResNet-34

# In[ ]:


np.random.seed(42)
learn = cnn_learner(data, models.resnet34, metrics=accuracy, callback_fns=ShowGraph, model_dir='/kaggle/working')


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4, max_lr=1e-02)


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.load('stage-1');
learn.unfreeze()
learn.lr_find(end_lr=10)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(12, max_lr=slice(1e-05, 1e-04))


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(10, 10))


# In[ ]:


learn.save('stage-2')


# ## ResNet-50

# In[ ]:


np.random.seed(42)
learn_50 = cnn_learner(data, models.resnet50, metrics=accuracy, callback_fns=ShowGraph, model_dir='/kaggle/working')


# In[ ]:


learn_50.lr_find(end_lr=100)
learn_50.recorder.plot()


# In[ ]:


learn_50.fit_one_cycle(4, max_lr=1e-02)


# In[ ]:


learn_50.save('stage-1-resnet50')


# In[ ]:


learn_50.load('stage-1-resnet50')
learn_50.unfreeze()
learn_50.lr_find(end_lr=10)
learn_50.recorder.plot()


# In[ ]:


learn_50.fit_one_cycle(12, max_lr=slice(1e-05, 1e-04))


# In[ ]:


learn_50.save('stage-2-resnet50')


# In[ ]:


learn_50.show_results()


# In[ ]:


learn_50.data.add_test(PixelImageItemList.from_df(df_test,path='.',cols='fn'))


# In[ ]:


pred_test = learn.get_preds(ds_type=DatasetType.Test)
test_result = torch.argmax(pred_test[0],dim=1)
result = test_result.numpy()


# In[83]:


final = pd.Series(result,name='Label')
submission = pd.concat([pd.Series(range(1,28001),name='ImageId'),final],axis=1)
submission.to_csv('submission.csv',index=False)
submission.head()


# In[84]:


def create_download_link(title='Download file', filename='submission.csv'):
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title, filename=filename)
    return HTML(html)


# In[85]:


from IPython.display import HTML


# In[86]:


create_download_link(filename='submission.csv')


# In[ ]:




