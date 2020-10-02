#!/usr/bin/env python
# coding: utf-8

# # Captchas

# [This](https://medium.com/@oneironaut.oml/solving-captchas-with-deeplearning-part-1-multi-label-classification-b9f745c3a599) article explains what's going in this kernel.

# In[ ]:


from fastai.vision import *
import os

path = Path(r'../input/samples/')
print(os.listdir(path/'samples')[:10])


# In[ ]:


from IPython.display import Image
Image(filename='../input/samples/samples/bny23.png')


# ## Multilabel Classification

# In[ ]:


def label_from_filename(path):
    label = [char for char in path.name[:-4]]
    return label


# In[ ]:


data = (ImageList.from_folder(path)
        .split_by_rand_pct(0.2)
        .label_from_func(label_from_filename)
        .transform(get_transforms(do_flip=False))
        .databunch()
        .normalize()
       )
data.show_batch(3)


# In[ ]:


acc_02 = partial(accuracy_thresh, thresh=0.2)


# In[ ]:


learn = learn = cnn_learner(data, models.resnet18, model_dir='/tmp', metrics=acc_02)
lr_find(learn)
learn.recorder.plot()


# In[ ]:


lr = 5e-2
learn.fit_one_cycle(5, lr)


# In[ ]:


import copy
losses = copy.deepcopy(learn.recorder.losses)


# In[ ]:


learn.unfreeze()
lr_find(learn)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(15, slice(1e-3, lr/5))


# In[ ]:


losses += learn.recorder.losses

fig, ax = plt.subplots(figsize=(14,7))
ax.plot(losses, linewidth=2)
ax.set_ylabel('loss', fontsize=16)
ax.set_xlabel('iteration', fontsize=16)
plt.show()

