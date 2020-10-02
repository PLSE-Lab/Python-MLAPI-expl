#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *
import numpy as np; import pandas as pd
import matplotlib; import matplotlib.pyplot as plt
get_ipython().system("pip freeze > '../working/requirements.txt'")
img_dir='../input/jurkat cells (merged only)/jurkat cells (merged only)/'


# In[ ]:


path=Path(img_dir)
data=ImageDataBunch.from_folder(path, train=".",valid_pct=0.3,ds_tfms=get_transforms(do_flip=True,flip_vert=True,max_rotate=90,max_lighting=0.3),size=224,bs=64,num_workers=0).normalize(imagenet_stats)
learn=create_cnn(data, models.resnet34, metrics=accuracy, model_dir="/tmp/model/")
learn.fit_one_cycle(10)
interp=ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix(figsize=(4,4), dpi=60)

