#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai import *
from fastai.vision import *

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('ls ../input')
get_ipython().system('mkdir -p /tmp/.torch/models/')
get_ipython().system('cp ../input/pretrained-pytorch-models/* /tmp/.torch/models/')
get_ipython().system('cp ../input/resnet-from-fastai/* /tmp/.torch/models')


# In[ ]:


path = Path('../input/digit-recognizer')


# Pull in digit reader

# In[ ]:


class CustomImageItemList(ImageItemList):
    def open(self, fn):
        img = fn.reshape(28, 28)
        img = np.stack((img,)*3, axis=-1) # convert to 3 channels
        return Image(pil2tensor(img, dtype=np.float32))

    @classmethod
    def from_csv_custom(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs) -> 'ItemList':
        df = pd.read_csv(Path(path)/csv_name, header=header)
        res = super().from_df(df, path=path, cols=0, **kwargs)
        # convert pixels to an ndarray
        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 783.0, axis=1).values
        return res


# In[ ]:


test = CustomImageItemList.from_csv_custom(path=path, csv_name='test.csv', imgIdx=0)
data = (CustomImageItemList.from_csv_custom(path=path, csv_name='train.csv')
                       .random_split_by_pct(.2)
                       .label_from_df(cols='label')
                       .add_test(test, label=0)
                       .databunch(bs=64, num_workers=0)
                       .normalize(imagenet_stats))


# In[ ]:


data.show_batch(rows=3, figsize=(12,9))


# In[ ]:


arch = models.resnet34


# In[ ]:


learn = create_cnn(data, arch, metrics=error_rate, model_dir='/tmp/models')


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


lr = 0.02


# In[ ]:


learn.fit_one_cycle(4, slice(lr))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.load('stage-1')


# In[ ]:


interp = learn.interpret()


# In[ ]:


interp.plot_top_losses(9, figsize=(7,7))


# In[ ]:


preds, y, losses = learn.get_preds(ds_type=DatasetType.Test, with_loss=True)


# In[ ]:


# Bug in fastai? Why is this needed?
y = torch.argmax(preds, dim=1)


# In[ ]:


submission_df = pd.DataFrame({'ImageId': range(1, len(y) + 1), 'Label': y}, columns=['ImageId', 'Label'])
submission_df.head()


# In[ ]:


submission_df.to_csv('submission.csv', index=False)


# In[ ]:


get_ipython().system('head submission.csv')

