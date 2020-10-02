#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *
import numpy as np # linear algebra
import pandas as pd 


# In[ ]:


import os
path = '../input'
print(os.listdir(path))


# In[ ]:



class CustomImageList(ImageList):
    def open(self, fn):
        img = fn.reshape(28,28)
        img = np.stack((img,)*3, axis=-1)
        return Image(pil2tensor(img, dtype=np.float32))
    
    @classmethod
    def from_csv_custom(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs)->'ItemList': 
        df = pd.read_csv(Path(path)/csv_name, header=header)
        res = super().from_df(df, path=path, cols=0, **kwargs)
        
        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values
        
        return res


# In[ ]:


test = CustomImageList.from_csv_custom(path=path, csv_name='test.csv', imgIdx=0)


# In[ ]:


tfms = get_transforms(do_flip=False)
data = (CustomImageList.from_csv_custom(path=path, csv_name='train.csv', imgIdx=1)
            .split_by_rand_pct(.2)
            .label_from_df(cols='label')
            .add_test(test, label=0)
             #src.transform(tfms, size=256)
            .transform(tfms,size=28)
            .databunch(bs=128, num_workers=0)
            .normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(5,5))


# In[ ]:


#learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir='/kaggle/working/models')
learn = cnn_learner(data, models.resnet50, metrics=accuracy, model_dir='/kaggle/working/models')


# In[ ]:


data.show_batch(rows=3, figsize=(5,5))


# In[ ]:


learn.fit_one_cycle(4)


# # Training with different sizes

# In[ ]:


classes = [10,14,18,22,25,28]


# In[ ]:


for c in classes:
    tfms = get_transforms(do_flip=False)
    data = (CustomImageList.from_csv_custom(path=path, csv_name='train.csv', imgIdx=1)
                .split_by_rand_pct(.2)
                .label_from_df(cols='label')
                .add_test(test, label=0)
                 #src.transform(tfms, size=256)
                .transform(tfms,size=c)
                .databunch(bs=128, num_workers=0)
                .normalize(imagenet_stats))
    data.show_batch(rows=3, figsize=(5,5))
    learn.data = data
    c
    learn.fit_one_cycle(8)


# In[ ]:


#learn.data = data


# In[ ]:


learn.save('stage-1-resnet50')


# In[ ]:


learn.load('stage-1-resnet50')


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(12, max_lr=slice(1e-6,5e-2))


# In[ ]:


learn.save('stage-1-resnet50')


# In[ ]:


learn.load('stage-1-resnet50')
learn.validate()


# In[ ]:


learn.unfreeze()
learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-4))


# In[ ]:


learn.save('stage-1-resnet50')


# In[ ]:


learn.load('stage-1-resnet50')
learn.validate()


# In[ ]:


#learn.lr_find()


# In[ ]:


#learn.recorder.plot()


# In[ ]:


#learn.lr_find()


# In[ ]:


#learn.recorder.plot()


# In[ ]:


#learn.unfreeze()
#learn.fit_one_cycle(5,1e-4)


# In[ ]:


#learn.save('stage-1-resnet50')


# In[ ]:


#learn.load('stage-1-resnet50')
#learn.validate()


# In[ ]:


#learn.lr_find()


# In[ ]:


#learn.recorder.plot()


# In[ ]:


#learn.unfreeze()
#learn.fit_one_cycle(8, max_lr=slice(1e-5, 2.5e-4))


# In[ ]:


#learn.save('stage-1-resnet50')
#learn.load('stage-1-resnet50')
#learn.validate()


# In[ ]:


# get the predictions
predictions, *_ = learn.get_preds(DatasetType.Test)
labels = np.argmax(predictions, 1)
# output to a file
submission_df = pd.DataFrame({'ImageId': list(range(1,len(labels)+1)), 'Label': labels})
submission_df.to_csv(f'submission.csv', index=False)

