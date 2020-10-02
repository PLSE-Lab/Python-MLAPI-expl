#!/usr/bin/env python
# coding: utf-8

# ## Experiments for Intel Scene Classification challenge

# In[ ]:


import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import torch
from fastai.vision import *
from fastai.metrics import *

np.random.seed(7)
torch.cuda.manual_seed_all(7)

import os
print(os.listdir("../input"))


# In[ ]:


pred = get_ipython().getoutput('ls ../input/seg_pred/seg_pred')
len(pred)


# In[ ]:


buildings = get_ipython().getoutput('ls ../input/seg_train/seg_train/buildings')
forest = get_ipython().getoutput('ls ../input/seg_train/seg_train/forest')
glacier = get_ipython().getoutput('ls ../input/seg_train/seg_train/glacier')
mountain = get_ipython().getoutput('ls ../input/seg_train/seg_train/mountain')
sea = get_ipython().getoutput('ls ../input/seg_train/seg_train/sea')
street = get_ipython().getoutput('ls ../input/seg_train/seg_train/street')

len(buildings) + len(forest) + len(glacier) + len(mountain) + len(sea) + len(street)


# In[ ]:


buildings = get_ipython().getoutput('ls ../input/seg_test/seg_test/buildings')
forest = get_ipython().getoutput('ls ../input/seg_test/seg_test/forest')
glacier = get_ipython().getoutput('ls ../input/seg_test/seg_test/glacier')
mountain = get_ipython().getoutput('ls ../input/seg_test/seg_test/mountain')
sea = get_ipython().getoutput('ls ../input/seg_test/seg_test/sea')
street = get_ipython().getoutput('ls ../input/seg_test/seg_test/street')

len(buildings) + len(forest) + len(glacier) + len(mountain) + len(sea) + len(street)


# In[ ]:


# Batches of 256
databunch = ImageDataBunch.from_folder(Path('../input'), ds_tfms=get_transforms(),
                               train='seg_train', valid='seg_test',
                                   size=64, bs=256).normalize(imagenet_stats)


# In[ ]:


# Add the test images to the DataBunch
test_img = ImageList.from_folder(path=Path('../input/seg_pred', folder='swg_pred'))
databunch.add_test(test_img)


# In[ ]:


databunch.show_batch(rows=4)


# In[ ]:


databunch.classes


# In[ ]:


databunch.label_list


# In[ ]:


learner = cnn_learner(databunch, models.resnet50, metrics=accuracy, model_dir='/tmp/models')
learner.lr_find();
learner.recorder.plot()


# In[ ]:


# Crazy story of learning rates you see :P
learner.fit_one_cycle(5, max_lr=(2e-04, 1e-03, 3*1e-02))
learner.recorder.plot_losses()


# In[ ]:


del learner
gc.collect()


# In[ ]:


# Towards a more stable training
learner = cnn_learner(databunch, models.resnet50, metrics=accuracy, model_dir='/tmp/models').to_fp16()
learner.fit_one_cycle(5, max_lr=slice(1e-02, 1e-03))


# Slightly overfits!

# In[ ]:


learner.recorder.plot_losses()


# In[ ]:


learner.show_results(ds_type=DatasetType.Valid)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learner.to_fp32())
interp.plot_top_losses(9, figsize=(12,10))


# In[ ]:


interp.plot_confusion_matrix(figsize=(8,8))


# In[ ]:


del learner
gc.collect()


# In[ ]:


learner = cnn_learner(databunch, models.resnet50, metrics=accuracy, model_dir='/tmp/models').to_fp16()
learner.fit_one_cycle(10, max_lr=slice(1e-02, 1e-03))
learner.recorder.plot_losses()


# In[ ]:


del learner
gc.collect()


# ## A bit more data augmentation

# In[ ]:


new_transforms = get_transforms(do_flip=True, flip_vert=True, 
                      max_lighting=0.3, max_warp=0.3, max_rotate=20., max_zoom=0.05)


# In[ ]:


# Batches of 256
new_databunch = ImageDataBunch.from_folder(Path('../input'), ds_tfms=new_transforms,
                               train='seg_train', valid='seg_test',
                                   size=64, bs=256).normalize(imagenet_stats)

new_databunch.add_test(test_img)


# In[ ]:


new_databunch.show_batch(rows=4)


# In[ ]:


learner = cnn_learner(new_databunch, models.resnet50, metrics=accuracy, model_dir='/tmp/models').to_fp16()
learner.fit_one_cycle(10, max_lr=slice(1e-02, 1e-03))
learner.recorder.plot_losses()


# It overfits after augmentation. The augmented transforms were may be way to much. 

# In[ ]:


del learner
gc.collect()


# In[ ]:


less_transforms = get_transforms(do_flip=True, max_warp=0.3, max_zoom=0.05)


# In[ ]:


# Batches of 256
new_databunch_2 = ImageDataBunch.from_folder(Path('../input'), ds_tfms=less_transforms,
                               train='seg_train', valid='seg_test',
                                   size=64, bs=256).normalize(imagenet_stats)

new_databunch_2.add_test(test_img)


# In[ ]:


learner = cnn_learner(new_databunch_2, models.resnet50, metrics=accuracy, model_dir='/tmp/models').to_fp16()
learner.fit_one_cycle(10, max_lr=slice(1e-02, 1e-03))
learner.recorder.plot_losses()


# Class performance right here.

# In[ ]:


learner.save('stage-1-rn50')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learner.to_fp32())
interp.plot_top_losses(9, figsize=(12,10))


# In[ ]:


interp.plot_confusion_matrix(figsize=(8,8))


# In[ ]:


del learner
gc.collect()


# In[ ]:


learner = cnn_learner(new_databunch_2, models.resnet50, metrics=accuracy, model_dir='/tmp/models').to_fp16()
learner.fit_one_cycle(15, max_lr=slice(1e-02, 1e-03))
learner.recorder.plot_losses()


# Overfits slightly.

# In[ ]:


learner.save('stage-2-rn50')


# In[ ]:


del learner
gc.collect()


# ## Trying deeper architectures

# In[ ]:


learner = cnn_learner(new_databunch_2, models.resnet101, metrics=accuracy, model_dir='/tmp/models').to_fp16()
learner.lr_find()
learner.recorder.plot()


# In[ ]:


learner.fit_one_cycle(5, max_lr=slice(1e-02, 1e-03))
learner.recorder.plot_losses()


# In[ ]:


del learner
gc.collect()


# In[ ]:


learner = cnn_learner(new_databunch_2, models.resnet101, metrics=accuracy, model_dir='/tmp/models').to_fp16()
learner.fit_one_cycle(10, max_lr=slice(1e-02, 1e-03))
learner.recorder.plot_losses()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learner)
interp.plot_confusion_matrix(figsize=(8,8))


# In[ ]:


learner.save('stage-1-rn101')


# In[ ]:


del learner
gc.collect()


# ## Loading data in a larger size

# In[ ]:


# Size of 150*150
new_databunch_3 = ImageDataBunch.from_folder(Path('../input'), ds_tfms=less_transforms,
                               train='seg_train', valid='seg_test',
                                   size=150, bs=256).normalize(imagenet_stats)

new_databunch_3.add_test(test_img)


# In[ ]:


new_databunch_3.show_batch(rows=4)


# In[ ]:


learner = cnn_learner(new_databunch_3, models.resnet101, metrics=accuracy, model_dir='/tmp/models').to_fp16()
learner.lr_find();
learner.recorder.plot()


# In[ ]:


learner.fit_one_cycle(10, max_lr=slice(1e-02, 1e-03))
learner.recorder.plot_losses()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learner)
interp.plot_confusion_matrix(figsize=(8,8))


# In[ ]:


del learner
gc.collect()


# In[ ]:


new_databunch_4 = ImageDataBunch.from_folder(Path('../input'), ds_tfms=get_transforms(),
                               train='seg_train', valid='seg_test',
                                   size=150, bs=256).normalize(imagenet_stats)

new_databunch_4.add_test(test_img)


# In[ ]:


learner = cnn_learner(new_databunch_4, models.resnet101, metrics=accuracy, model_dir='/tmp/models').to_fp16()
learner.fit_one_cycle(10, max_lr=slice(1e-02, 1e-03))
learner.recorder.plot_losses()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learner)
interp.plot_confusion_matrix(figsize=(8,8))


# Ensembling the last two models will definitely help. 
