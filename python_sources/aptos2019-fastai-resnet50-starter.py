#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, sys
from fastai import *
from fastai.vision import *


# In[ ]:


import fastai
fastai.__version__


# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(42)


# ## Data

# In[ ]:


PATH = Path('../input/aptos2019-blindness-detection')
df_train = pd.read_csv(PATH/"train.csv")            .assign(filename = lambda df: "train_images/" + df.id_code + ".png")
df_test = pd.read_csv(PATH/"test.csv")           .assign(filename = lambda df: "test_images/" + df.id_code + ".png")


# In[ ]:


_ = df_train.hist()


# In[ ]:


transforms = get_transforms(
    do_flip = True,
    flip_vert = True,
    max_zoom = 1,
    max_rotate = 180, #default 10
    max_lighting = 0.2, #default 0.2
    max_warp = 0.1 #default 0.1
)


# In[ ]:


data = ImageDataBunch.from_df(path = "../input/aptos2019-blindness-detection",
                              df = df_train,
                              fn_col = "filename",
                              label_col = "diagnosis",
                              ds_tfms = transforms,
                             size=224)\
        .normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# ## Model

# In[ ]:


# copy pretrained weights for resnet50 to the folder fastai will search by default
Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)
get_ipython().system("cp '../input/resnet50/resnet50.pth' '/tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth'")


# In[ ]:


kappa = KappaScore()
kappa.weights = "quadratic"
learn = cnn_learner(data, models.resnet50,
                    metrics=[error_rate, kappa],
                    model_dir="/tmp/model/")


# In[ ]:


learn.lr_find(end_lr=0.5)
learn.recorder.plot(suggestion=True)


# In[ ]:


lr = 7e-3
learn.fit_one_cycle(10, lr)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.unfreeze()


# In[ ]:


lrs = slice(lr/400,lr/4)
learn.fit_one_cycle(10,lrs)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(8,8), dpi=60)


# ## Submission

# In[ ]:


sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
sample_df.head()


# In[ ]:


learn.data.add_test(ImageList.from_df(
    sample_df, PATH,
    folder='test_images',
    suffix='.png'
))


# In[ ]:


preds,y = learn.get_preds(DatasetType.Test)


# In[ ]:


sample_df.diagnosis = preds.argmax(1)
sample_df.head()


# In[ ]:


sample_df.to_csv('submission.csv',index=False)
_ = sample_df.hist()

