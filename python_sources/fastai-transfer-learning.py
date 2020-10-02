#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
GPU_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)


# In[ ]:


import warnings
warnings.filterwarnings("ignore")

import fastai
print(fastai.__version__)
from fastai.vision import *
from fastai.callbacks import SaveModelCallback
import time


# ### Create a Path instance

# In[ ]:


path = Path('../input/hymenoptera-data/hymenoptera_data')
print(type(path))
path.ls()


# In[ ]:


(path/'train').ls()


# ### Create an ImageList instance

# In[ ]:


il = ImageList.from_folder(path)
il.items[0]


# In[ ]:


il


# In[ ]:


il[0].show()


# ### Create item lists for train and valid

# In[ ]:


sd = il.split_by_folder(train='train', valid='val')
sd


# ### Create a label list

# In[ ]:


ll = sd.label_from_folder()
ll


# ### Show an image with label

# In[ ]:


get_ipython().run_cell_magic('time', '', 'x,y = ll.train[0]\nx.show()\nprint(y,x.shape)')


# ### Apply transformations

# In[ ]:


tfms = get_transforms(max_rotate=25); len(tfms)


# In[ ]:


ll = ll.transform(tfms,size=224)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'x,y = ll.train[0]\nx.show()\nprint(y,x.shape)')


# ### Create a databunch instance

# In[ ]:


get_ipython().run_cell_magic('time', '', 'bs = 32\ndata = ll.databunch(bs=bs).normalize(imagenet_stats)')


# In[ ]:


x,y = data.train_ds[0]
x.show()
print(y)


# ### Show random transformations of the same image

# In[ ]:


def _plot(i,j,ax): data.train_ds[0][0].show(ax)
plot_multi(_plot, 3, 3, figsize=(8,8))


# ### show a batch of images with labels

# In[ ]:


xb,yb = data.one_batch()
print(xb.shape,yb.shape)
data.show_batch(rows=3, figsize=(10,8))


# ### Create a CNN learner

# In[ ]:


get_ipython().run_cell_magic('time', '', "learn = cnn_learner(data, models.resnet18, metrics=accuracy)\nlearn.model_dir = '/kaggle/working/models'")


# ### find a proper learning rate

# In[ ]:


get_ipython().system('pwd')


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# ### training

# In[ ]:


learn.fit_one_cycle(10,max_lr=slice(0.007),callbacks=[
            SaveModelCallback(learn, every='improvement', monitor='accuracy'),
            ])


# In[ ]:


pred, truth = learn.get_preds()


# In[ ]:


pred = pred.numpy()
truth = truth.numpy()
acc = np.mean(np.argmax(pred,axis=1) == truth)
print('Validation Accuracy %.4f'%acc)

