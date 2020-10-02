#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import cv2
import os

from fastai.vision import *
from fastai.metrics import error_rate

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('../input/digit-recognizer/train.csv' , encoding='latin1')
test  = pd.read_csv('../input/digit-recognizer/test.csv')
sub =   pd.read_csv('../input/digit-recognizer/sample_submission.csv')


# Take a look at the dataset

# In[ ]:


train.tail()


# Create an image from ds

# In[ ]:


class CustomImageList(ImageList):
    def open(self, fn):
        img = fn.reshape(28,28)
#         img = np.stack((img,)*3, axis=-1)
        return Image(pil2tensor(img, dtype=np.float32))
    
    @classmethod
    def from_csv_custom(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs)->'ItemList': 
        df = pd.read_csv(Path(path)/csv_name, header=header)
        res = super().from_df(df, path=path, cols=0, **kwargs)
        
        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values
        
        return res


# **Dataloader**

# In[ ]:


tfms = get_transforms(do_flip=False )
data = (CustomImageList.from_csv_custom(path='../input/digit-recognizer/', csv_name='train.csv', imgIdx=1 , convert_mode='binary')
                .split_by_rand_pct(.2)
                .label_from_df(cols='label')
                .add_test(test)
                .transform(tfms)
                .databunch(bs=128, num_workers=0)
                .normalize(imagenet_stats))


# **Shows example**

# In[ ]:


data.show_batch(rows=3, figsize=(5,5))


# **Create a learner**

# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=accuracy, model_dir='/kaggle/working/models')
learn.lr_find()
learn.recorder.plot(suggestion=True)


# **Train the last layers**

# In[ ]:


learn.fit_one_cycle(1,2e-2)


# In[ ]:


learn.save('one_epoch')


# Let's try to have a bigger accuracy

# In[ ]:


learn.fit_one_cycle(10,slice(1e-3,1e-2))


# In[ ]:


learn.save('second_epoch')


# **Train all the layers**

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.load('second_epoch')


# In[ ]:


learn.fit_one_cycle(8, max_lr=slice(1e-6,1e-4))


# In[ ]:


learn.save('third_step')


# Show images in top_losses along with their prediction, actual, loss, and probability of actual class.

# In[ ]:


preds,y,losses = learn.get_preds(with_loss=True)
interp = ClassificationInterpretation(learn, preds, y, losses)


# In[ ]:


interp.plot_top_losses(9, figsize=(7,7))


# it's easy to understand why the computer had a dificuly to distinguish between the numbers, 
# to increase the next step would be to erase this images and retrain the model to make it more accurate

# **Submission**

# In[ ]:


# get the predictions
predictions, *_ = learn.get_preds(DatasetType.Test)
labels = np.argmax(predictions, 1)
# output to a file
submission_df = pd.DataFrame({'ImageId': list(range(1,len(labels)+1)), 'Label': labels})
submission_df.to_csv(f'submission.csv', index=False)

