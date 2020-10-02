#!/usr/bin/env python
# coding: utf-8

# In[ ]:


path = "../input/chest_xray/chest_xray/"
print(path)


# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


# set the batch size i.e. the number of images to train at a time
# reduce this number if you get an out of memory error 
bs = 64


# In[ ]:


# create a data bunch
np.random.seed(42)
data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(do_flip=False), 
                                  size=224, num_workers=0, 
                                  bs=bs, valid_pct=0.2).normalize(imagenet_stats)


# In[ ]:


# display 3 rows of data
data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


# verify the classes
print(data.classes)
len(data.classes),data.c


# In[ ]:


# create a neural network
learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir="/tmp/model/")


# In[ ]:


learn.to_fp16()


# In[ ]:


# fit 4 layers
learn.fit_one_cycle(4)


# In[ ]:


# interpret the results
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


learn.to_fp32()


# In[ ]:


# plot the top losses
interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


# plot the confusion matrix
interp.plot_confusion_matrix(figsize=(5,5), dpi=60)


# In[ ]:


learn.save('/kaggle/working/stage-1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.to_fp16()


# In[ ]:


learn.fit_one_cycle(1)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))


# In[ ]:


# interpret the results
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()


# In[ ]:


interp.plot_confusion_matrix(figsize=(5,5), dpi=60)


# We have managed to significantly reduce the error rate and predict the Pneumonia cases correctly which is the class we care about!

# In[ ]:


learn.save('/kaggle/working/stage-2')

