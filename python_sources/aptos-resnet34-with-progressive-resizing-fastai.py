#!/usr/bin/env python
# coding: utf-8

# # Preparing Data
# 

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
import fastai
from fastai.vision import *
import pandas as pd
import os
from shutil import copyfile

print(os.listdir("../input/aptos2019-blindness-detection"))


# In[ ]:


df_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
df_test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

x_train = df_train['id_code']
y_train = df_train['diagnosis']


# In[ ]:


path = Path("../input/aptos2019-blindness-detection")


# In[ ]:


# Creating data into folder format
os.mkdir("../data")
os.mkdir("../data/train")
for i in range(5):
    os.mkdir("../data/train/"+str(i))
def make_img_folder(x, y):
    for img_name ,diagnosis in zip(x,y):
        if diagnosis == 0:
            copyfile('../input/aptos2019-blindness-detection/train_images/{}.png'.format(img_name), '../data/train/0/{}.png'.format(img_name))
        if diagnosis == 1:
            copyfile('../input/aptos2019-blindness-detection/train_images/{}.png'.format(img_name), '../data/train/1/{}.png'.format(img_name))
        if diagnosis == 2:
            copyfile('../input/aptos2019-blindness-detection/train_images/{}.png'.format(img_name), '../data/train/2/{}.png'.format(img_name))
        if diagnosis == 3:
            copyfile('../input/aptos2019-blindness-detection/train_images/{}.png'.format(img_name), '../data/train/3/{}.png'.format(img_name))
        if diagnosis == 4:
            copyfile('../input/aptos2019-blindness-detection/train_images/{}.png'.format(img_name), '../data/train/4/{}.png'.format(img_name))
            
make_img_folder(x_train, y_train)


# In[ ]:


Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)
get_ipython().system("cp '../input/pytorch-pretrained-image-models/resnet34.pth' '/tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth'")


# In[ ]:


np.random.seed(42)
src = (ImageList.from_folder("../data/train")
                .split_by_rand_pct(0.2)
                .label_from_folder())
# Starting with image size 128
data = (src.transform(tfms=get_transforms(flip_vert=True, max_warp=0), size=128)
           .databunch(bs=64, path='.').normalize(imagenet_stats))


# In[ ]:


print(data.classes, data.c)


# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir=".", callback_fns=ShowGraph)


# In[ ]:


# learn.lr_find()
# learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5,3e-3)


# In[ ]:


learn.save("retino_128_1")
learn.load("retino_128_1")


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-5))


# In[ ]:


learn.save("retino128_2")


# ## Progressive Resizing

# In[ ]:


# Creating new data with image size increased to 256
data = (src.transform(tfms=get_transforms(flip_vert=True, max_warp=0), size=256)
           .databunch(bs=64, path='.').normalize(imagenet_stats))


# In[ ]:


learn.data = data


# In[ ]:


learn.fit_one_cycle(4, 1e-3)


# In[ ]:


learn.save("retino256_1")


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-6))


# In[ ]:


learn.save("retino256_2")


# ### Predictions

# In[ ]:


sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
sample_df.head()


# In[ ]:


learn.data.add_test(ImageList.from_df(sample_df,'../input/aptos2019-blindness-detection/',folder='test_images',suffix='.png'))


# In[ ]:


preds,y = learn.get_preds(DatasetType.Test)


# In[ ]:


sample_df.diagnosis = preds.argmax(1)
sample_df.head()


# In[ ]:


sample_df.to_csv('submission.csv',index=False)

