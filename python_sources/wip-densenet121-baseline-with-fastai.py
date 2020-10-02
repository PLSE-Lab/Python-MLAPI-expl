#!/usr/bin/env python
# coding: utf-8

# ## TL;DR
# 
# This is a simple classifier based on an Imagenet-trained Resnet18. It's inspired by Iafoss[](https://www.kaggle.com/iafoss)'s kernels in other competititions.
# 
# ### Update
# 
# Just realized Densnet121 is way better, so let's switch to that for now!

# ## Training 

# Let's start by importing our libararies.

# In[ ]:


get_ipython().system('pip install fastai==0.7.0 --no-deps')
get_ipython().system('pip install torch==0.4.1 torchvision==0.2.1')


# In[ ]:


from fastai.conv_learner import *
from fastai.dataset import *

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


# In[ ]:


MODEL_PATH = 'Dn121_v1'
TRAIN = '../input/train/'
TEST = '../input/test/'
LABELS = '../input/train_labels.csv'
SAMPLE_SUB = '../input/sample_submission.csv'
ORG_SIZE=96
BATCH_SIZE = 128


# The architecture is flexible, I chose Resnet18 since it can fit quite well into a kernel. You may play with this if you want to. 

# In[ ]:


arch = dn121 
nw = 4


# Next, we prapare out dataset to work with Fastai's pipeline.

# In[ ]:


train_df = pd.read_csv(LABELS).set_index('id')
train_names = train_df.index.values
train_labels = np.asarray(train_df['label'].values)
print("Number of positive samples = {:.4f}%".format(np.count_nonzero(train_labels)*100/len(train_labels)))
test_names = [f.replace(".tif","") for f in os.listdir(TEST)]
tr_n, val_n = train_test_split(train_names, test_size=0.15, random_state=42069)
print(len(tr_n), len(val_n))


# In[ ]:


class HCDDataset(FilesDataset):
    def __init__(self, fnames, path, transform):
        self.train_df = train_df
        super().__init__(fnames, transform, path)

    def get_x(self, i):
        img = open_image(os.path.join(self.path, self.fnames[i]+".tif"))
        # We crop the center of the original image for faster training time
        img = img[(ORG_SIZE-self.sz)//2:(ORG_SIZE+self.sz)//2,(ORG_SIZE-self.sz)//2:(ORG_SIZE+self.sz)//2,:]
        return img

    def get_y(self, i):
        if (self.path == TEST): return 0
        return self.train_df.loc[self.fnames[i]]['label']


    def get_c(self):
        return 2


# In[ ]:


def get_data(sz, bs):
    aug_tfms = [RandomRotate(20, tfm_y=TfmType.NO),
                RandomDihedral(tfm_y=TfmType.NO)]
    tfms = tfms_from_model(arch, sz, crop_type=CropType.NO, tfm_y=TfmType.NO,
                           aug_tfms=aug_tfms)
    ds = ImageData.get_ds(HCDDataset, (tr_n[:-(len(tr_n) % bs)], TRAIN),
                          (val_n, TRAIN), tfms, test=(test_names, TEST))
    md = ImageData("./", ds, bs, num_workers=nw, classes=None)
    return md


# In[ ]:


md = get_data(96, BATCH_SIZE)
learn = ConvLearner.pretrained(arch, md) 
learn.opt_fn = optim.Adam


# Uncomment these lines to run Fastai's automatic learning rate finder. 
# 
# The graphs showed that the loss converged at around learning rate = 1e-2, which means we should set our learning rate a bit higher than that.

# In[ ]:


# learn.lr_find()
# learn.sched.plot()
lr = 2e-2


# We start by training only the newly initialized weights, then unfreeze the model and finetune the pretrained weights with reduced learning rate.

# In[ ]:


learn.fit(lr, 1, cycle_len=2)
learn.unfreeze()
lrs = np.array([1e-4, 5e-4, 1.2e-3])
learn.fit(lrs, 1, cycle_len=5, use_clr=(20, 16))
learn.fit(lrs/4, 1, cycle_len=5, use_clr=(10, 8))
learn.fit(lrs/16, 1, cycle_len=5, use_clr=(10, 8))


# ## Predictions
# Let's add some TTA (Test-Time-Augmentation), this usually increases the accuracy a bit, you may need to do some tests here to compare the result with normal prediction.

# In[ ]:


# preds_t,y_t = learn.predict_with_targs(is_test=True) # Predicting without TTA
preds_t,y_t = learn.TTA(is_test=True, n_aug=8)
preds_t = np.stack(preds_t, axis=-1)
preds_t = np.exp(preds_t)
preds_t = preds_t.mean(axis=-1)[:,1]


# Finally, our submission.

# In[ ]:


sample_df = pd.read_csv(SAMPLE_SUB)
sample_list = list(sample_df.id)
pred_list = [p for p in preds_t]
pred_dic = dict((key, value) for (key, value) in zip(learn.data.test_ds.fnames,pred_list))
pred_list_cor = [pred_dic[id] for id in sample_list]
df = pd.DataFrame({'id':sample_list,'label':pred_list_cor})
df.to_csv('submission.csv'.format(MODEL_PATH), header=True, index=False)

