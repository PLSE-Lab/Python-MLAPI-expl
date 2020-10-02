#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.imports import *
from fastai import *
from fastai.vision import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


path = Path('../input/')
get_ipython().system('ls ../input')


# In[ ]:


class CustomImageItemList(ImageList):
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


data.show_batch(rows=3, figsize=(6,6))


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=accuracy, model_dir='/tmp/models')
learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


# fit learner
get_ipython().run_line_magic('time', 'learn.fit(2,slice(1e-2))')


# In[ ]:


# disable precompute and unfreeze layers
learn.precompute=False
learn.unfreeze()


# In[ ]:


# define differential learning rates
lr = np.array([0.001, 0.0075, 0.01])


# In[ ]:


# retrain full model
learn.fit_one_cycle(9,slice(2e-3,2e-5), wd=.1)


# In[ ]:


test_pred, test_y, test_loss = learn.get_preds(ds_type=DatasetType.Test, with_loss=True)


# In[ ]:


test_result = torch.argmax(test_pred,dim=1)
result = test_result.numpy()
print(result.shape)


# In[ ]:


submission_df = pd.DataFrame({'ImageId': range(1, len(test_y) + 1), 'Label': result}, columns=['ImageId', 'Label'])
submission_df.head()


# In[ ]:


submission_df.to_csv("submission.csv",index=None)


# In[ ]:




