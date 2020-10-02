#!/usr/bin/env python
# coding: utf-8

# **Credits for the notebook goes to Khoi Nguyen's : https://www.kaggle.com/suicaokhoailang/wip-resnet18-baseline-with-fastai-0-375-lb .
# I just wanted to play with resnet34 and see if we can fit a bigger architecture in kaggle kernels.**  <br>
# **If the notebooks gets published then we certainly can.**

# ## Training 

# Let's start by importing our libararies.

# In[ ]:


from fastai.conv_learner import *
from fastai.dataset import *
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import matplotlib.pyplot as plt
import math


# In[ ]:


MODEL_PATH = 'Resnet34_v1'
TRAIN = '../input/train/'
TEST = '../input/test/'
LABELS = '../input/train.csv'
SAMPLE_SUB = '../input/sample_submission.csv'


# The architecture is flexible, I chose Resnet18 since it can fit quite well into a kernel. You may play with this if you want to. 

# In[ ]:


arch = resnet18
nw = 6


# Next, we prapare out dataset to work with Fastai's pipeline.

# In[ ]:


train_df = pd.read_csv(LABELS).set_index('Image')
unique_labels = np.unique(train_df.Id.values)

labels_dict = dict()
labels_list = []
for i in range(len(unique_labels)):
    labels_dict[unique_labels[i]] = i
    labels_list.append(unique_labels[i])
print("Number of classes: {}".format(len(unique_labels)))
train_names = train_df.index.values
train_df.Id = train_df.Id.apply(lambda x: labels_dict[x])
train_labels = np.asarray(train_df.Id.values)
test_names = [f for f in os.listdir(TEST)]


# Let's draw a simple histogram to see the sample-per-class distribution.

# In[ ]:


labels_count = train_df.Id.value_counts()
_, _,_ = plt.hist(labels_count,bins=100)
labels_count


# Ugh, okay, let's kick the elephant out of the room and try again

# In[ ]:


print("Count for class new_whale: {}".format(labels_count[0]))

plt.figure(figsize=(20,8))

plt.hist(labels_count[1:],bins=100,range=[0,100])
plt.hist(labels_count[1:],bins=100,range=[0,100])


# So most of the classes have only one or two sample(s), making **train_test_split** directly on the data impossible. We'll try a simple fix by duplicating the minor classes so that each class have a minimum of 5 samples.

# In[ ]:


dup = []
for idx,row in train_df.iterrows():
    if labels_count[row['Id']] < 5:
        dup.extend([idx]*math.ceil((5 - labels_count[row['Id']])/labels_count[row['Id']]))
train_names = np.concatenate([train_names, dup])
train_names = train_names[np.random.RandomState(seed=42).permutation(train_names.shape[0])]
len(train_names)


# In[ ]:


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42069)
for train_idx, val_idx in sss.split(train_names, np.zeros(train_names.shape)):
    tr_n, val_n = train_names[train_idx], train_names[val_idx]
print(len(tr_n), len(val_n))


# The image sizes seem to vary, so we'll try to see what the average width and height are:

# In[ ]:


avg_width = 0
avg_height = 0
for fn in os.listdir(TRAIN)[:1000]:
    img = cv2.imread(os.path.join(TRAIN,fn))
    avg_width += img.shape[1]
    avg_height += img.shape[0]
avg_width //= 1000
avg_height //= 1000
print(avg_width, avg_height)


# They turn out to be quite big, especially the width, so below you'll see I resize everything back to **average_width/4**. You may consider continue training on bigger size, but that probably won't fit in a kernel. 

# In[ ]:


class HWIDataset(FilesDataset):
    def __init__(self, fnames, path, transform):
        self.train_df = train_df
        super().__init__(fnames, transform, path)

    def get_x(self, i):
        img = open_image(os.path.join(self.path, self.fnames[i]))
        # We crop the center of the original image for faster training time
        img = cv2.resize(img, (self.sz, self.sz))
        return img

    def get_y(self, i):
        if (self.path == TEST): return 0
        return self.train_df.loc[self.fnames[i]]['Id']


    def get_c(self):
        return len(unique_labels)


# In[ ]:


class RandomLighting(Transform):
    def __init__(self, b, c, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.b, self.c = b, c

    def set_state(self):
        self.store.b_rand = rand0(self.b)
        self.store.c_rand = rand0(self.c)

    def do_transform(self, x, is_y):
        if is_y and self.tfm_y != TfmType.PIXEL: return x  # add this line to fix the bug
        b = self.store.b_rand
        c = self.store.c_rand
        c = -1 / (c - 1) if c < 0 else c + 1
        x = lighting(x, b, c)
        return x
    
def get_data(sz, bs):
    aug_tfms = [RandomRotateZoom(deg=20, zoom=2, stretch=1),
                RandomLighting(0.05, 0.05, tfm_y=TfmType.NO),
                RandomBlur(blur_strengths=3,tfm_y=TfmType.NO),
                RandomFlip(tfm_y=TfmType.NO)]
    tfms = tfms_from_model(arch, sz, crop_type=CropType.NO, tfm_y=TfmType.NO,
                           aug_tfms=aug_tfms)
    ds = ImageData.get_ds(HWIDataset, (tr_n[:-(len(tr_n) % bs)], TRAIN),
                          (val_n, TRAIN), tfms, test=(test_names, TEST))
    md = ImageData("./", ds, bs, num_workers=nw, classes=None)
    return md


# In[ ]:


sz = (avg_width//2, avg_height//2)
batch_size = 128
md = get_data(avg_width//4, batch_size)
learn = ConvLearner.pretrained(arch, md) 
learn.opt_fn = optim.Adam


# Uncomment these lines to run Fastai's automatic learning rate finder. 
# 

# In[ ]:


# learn.lr_find()
# learn.sched.plot()
lr = 1e-3


# We start by training only the newly initialized weights, then unfreeze the model and finetune the pretrained weights with reduced learning rate.

# In[ ]:


learn.fit(lr, 1, cycle_len=1)
learn.unfreeze()
lrs = np.array([lr/10, lr/20, lr/40])
learn.fit(lrs, 4, cycle_len=4, use_clr=(20, 16))
learn.fit(lrs/4, 2, cycle_len=4, use_clr=(10, 16))
learn.fit(lrs/16, 1, cycle_len=4, use_clr=(10, 16))


# May be keep training on bigger image for potential performance boost.

# In[ ]:


# batch_size = 32
# md = get_data(avg_width//2, batch_size)
# learn.set_data(md)
# learn.fit(lrs/4, 3, cycle_len=2, use_clr=(10, 8))


# In[ ]:


# batch_size = 16
# md = get_data(avg_width, batch_size)
# learn.set_data(md)
# learn.fit(lrs/16, 1, cycle_len=4, use_clr=(10, 8))


# ## Predictions

# In[ ]:


# preds_t,y_t = learn.predict_with_targs(is_test=True) # Predicting without TTA
preds_t,y_t = learn.TTA(is_test=True,n_aug=8)
preds_t = np.stack(preds_t, axis=-1)
preds_t = np.exp(preds_t)
preds_t = preds_t.mean(axis=-1)


# Finally, our submission.

# In[ ]:


sample_df = pd.read_csv(SAMPLE_SUB)
sample_list = list(sample_df.Image)
pred_list = [[labels_list[i] for i in p.argsort()[-5:][::-1]] for p in preds_t]
pred_dic = dict((key, value) for (key, value) in zip(learn.data.test_ds.fnames,pred_list))
pred_list_cor = [' '.join(pred_dic[id]) for id in sample_list]
df = pd.DataFrame({'Image':sample_list,'Id': pred_list_cor})
df.to_csv('submission.csv'.format(MODEL_PATH), header=True, index=False)
df.head()

