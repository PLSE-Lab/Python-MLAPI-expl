#!/usr/bin/env python
# coding: utf-8

# **Plese Up-Vote if you find the kernel to be useful !**

# In[ ]:


import numpy as np
import pandas as pd
import os
print(os.listdir("/kaggle/input/"))
#os.getcwd()


# In[ ]:


from fastai.conv_learner import *
from fastai.dataset import *

from pathlib import Path
import json
torch.cuda.set_device(0)


# In[ ]:


PATH = Path('/kaggle/input/train')
list(PATH.iterdir())


# In[ ]:


MASKS_FN = 'train.csv'
TRAIN_DN = 'images'
MASKS_DN = 'masks'

PATH1 = Path('/kaggle/input/')
masks_csv = pd.read_csv(PATH1/MASKS_FN)
masks_csv.head()


# In[ ]:


#https://realpython.com/python-pathlib/
import collections
PATH1 = Path('/kaggle/input/train/masks')
collections.Counter(p.suffix for p in PATH1.iterdir())


# In[ ]:


PATH2 = Path('/kaggle/input/train/images')
collections.Counter(p.suffix for p in PATH2.iterdir())


# In[ ]:


PATH3 = Path('/kaggle/input/test/images')
collections.Counter(p.suffix for p in PATH3.iterdir())


# In[ ]:


masks_csv.isnull().sum()


# In[ ]:


#masks_csv=masks_csv.dropna()
masks_csv['id']=masks_csv['id']+".png"


# In[ ]:


masks_csv.count()


# In[ ]:


def show_img(im, figsize=None, ax=None, alpha=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha)
    ax.set_axis_off()
    return ax


# In[ ]:


#Image.open(PATH/TRAIN_DN/f'4875705fb0.png')#.size


# In[ ]:


get_ipython().system('rm -rf /tmp/mask_128')
get_ipython().system('rm -rf /tmp/train_128 ')
get_ipython().system('rm -rf /tmp/test_128 ')
get_ipython().system('mkdir -p /tmp/mask_128')
get_ipython().system('mkdir -p /tmp/train_128')
get_ipython().system('mkdir -p /tmp/test_128')
PATH4 = Path('/kaggle/input/train/masks')
i=os.listdir(PATH4)
for j in range(4000):
     Image.open(PATH4/i[j]).resize((128,128)).save('/tmp/mask_128/'+i[j])
        
PATH4=Path('/kaggle/input/train/images')
for j in range(4000):
     Image.open(PATH4/i[j]).resize((128,128)).save('/tmp/train_128/'+i[j])

PATH4=Path('/kaggle/input/test/images')
i=os.listdir(PATH4)
for j in range(18000):
     Image.open(PATH4/i[j]).resize((128,128)).save('/tmp/test_128/'+i[j])

TRAIN_DN=Path('/tmp/train_128')
MASKS_DN=Path('/tmp/mask_128')
TEST_DN=Path('/tmp/test_128')


# In[ ]:


ims = [open_image(PATH/TRAIN_DN/f'{i}') for i in masks_csv['id'].head(30)]
im_masks = [open_image(PATH/MASKS_DN/f'{i}') for i in masks_csv['id'].head(30)]
fig, axes = plt.subplots(5, 6, figsize=(18, 12))

for i,ax in enumerate(axes.flat):
    ax = show_img(ims[i], ax=ax)
    show_img(im_masks[i][...,0], ax=ax, alpha=0.5)
plt.tight_layout(pad=0.1)


# In[ ]:


x_names = np.array([Path(TRAIN_DN)/o for o in masks_csv['id']])
y_names = np.array([Path(MASKS_DN)/o for o in masks_csv['id']])
len(x_names)//5


# In[ ]:


a=os.listdir('/tmp/test_128')
test_names = np.array([Path(TEST_DN)/o for o in a])
test_name = (test_names ,test_names)


# In[ ]:


val_idxs = list(range(len(x_names)//5))
((val_x,trn_x),(val_y,trn_y)) = split_by_idx(val_idxs, x_names, y_names)
len(val_x),len(trn_x)


# In[ ]:


sz = 128
bs = 64


# In[ ]:


class MatchedFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y=y
        assert(len(fnames)==len(y))
        super().__init__(fnames, transform, path)
    def get_y(self, i): return open_image(os.path.join(self.path, self.y[i]))
    def get_c(self): return 0


aug_tfms = [RandomRotate(4, tfm_y=TfmType.CLASS),
            RandomFlip(tfm_y=TfmType.CLASS),
            RandomLighting(0.05, 0.05)]

tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)
datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x,trn_y), (val_x,val_y), tfms,test = test_name, path=PATH)
md = ImageData(PATH, datasets, bs, num_workers=8, classes=None)


# In[ ]:


denorm = md.trn_ds.denorm
x,y = next(iter(md.aug_dl))
x = denorm(x)


# In[ ]:


fig, axes = plt.subplots(5, 6, figsize=(12, 10))
for i,ax in enumerate(axes.flat):
    ax=show_img(x[i], ax=ax)
    show_img(y[i], ax=ax, alpha=0.5)
plt.tight_layout(pad=0.1)


# In[ ]:


# Model
class StdUpsample(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.conv = nn.ConvTranspose2d(nin, nout, 2, stride=2)
        self.bn = nn.BatchNorm2d(nout)
        
    def forward(self, x): return self.bn(F.relu(self.conv(x)))

flatten_channel = Lambda(lambda x: x[:,0])
simple_up = nn.Sequential(
    nn.ReLU(),
    StdUpsample(512,256),
    StdUpsample(256,256),
    StdUpsample(256,256),
    StdUpsample(256,256),
    nn.ConvTranspose2d(256, 1, 2, stride=2),
    flatten_channel
)


# In[ ]:


# IOU/Jaccard index - metrics
def jacc(pred, targs):
    pred = (pred>0).float()
    dice = 2. * (pred*targs).sum() / (pred+targs).sum()
    return dice/(2-dice)


# In[ ]:


md.path = Path('/tmp')
models =  ConvnetBuilder (resnet34, 0, 0, 0, custom_head=simple_up)
learn = ConvLearner(md, models)
learn.opt_fn=optim.Adam
learn.crit=nn.BCEWithLogitsLoss()
learn.metrics=[accuracy_thresh(0.5)
              # ,jacc
              ]


# In[ ]:


learn.summary()


# In[ ]:


learn.lr_find()
learn.sched.plot()


# In[ ]:


lr=1e-1
learn.fit(lr,1,cycle_len=5,use_clr=(10,5))


# In[ ]:


learn.save('tmp')
learn.load('tmp')
py,ay = learn.predict_with_targs()
ay.shape


# In[ ]:


show_img(ay[8]);


# In[ ]:


show_img(py[8]>0);


# In[ ]:


learn.unfreeze()
#learn.bn_freeze(True)
lrs = np.array([lr/100,lr/10,lr])/10
learn.fit(lrs,1,cycle_len=10,use_clr=(10,10))


# In[ ]:


learn.save('0')
x,y = next(iter(md.val_dl))
py = to_np(learn.model(V(x)))


# In[ ]:


i=52
ax = show_img(denorm(x)[i])
show_img(py[i]>0, ax=ax, alpha=0.5);


# In[ ]:


ax = show_img(denorm(x)[i])
show_img(y[i], ax=ax, alpha=0.5);


# In[ ]:


# Predict on Test data
out=learn.predict(is_test=True)
out.shape


# In[ ]:


#j=452
j=14550
Image.open(PATH/TEST_DN/a[j])


# In[ ]:


columns = ['id']
b= pd.DataFrame(a,columns=columns)
show_img(out[j]>0)
#a[1]


# In[ ]:


# Resize Predictions to 101x101
result_array = np.zeros((18000,101,101),dtype=float)

import cv2
for i in range(18000):
    img = out[i]
    result_array[i] = cv2.resize(img, dsize=(101, 101), interpolation=cv2.INTER_CUBIC)>0.5


# In[ ]:


# Save Predications to disk
get_ipython().system('rm -rf /tmp/output1/*')
get_ipython().system('mkdir -p /tmp/output1')

for i in range(18000):
    plt.imsave('/tmp/output1/'+a[i], result_array[i])


# In[ ]:


OUT_PATH = Path('/tmp/output1')
ims = [open_image(TEST_DN/f'{i}') for i in b['id'].head(40)]
im_masks = [open_image(OUT_PATH/f'{i}')  for i in b['id'].head(40)]
fig, axes = plt.subplots(5, 8, figsize=(18, 12))

for i,ax in enumerate(axes.flat):
    ax = show_img(ims[i], ax=ax)
    show_img(im_masks[i][...,0], ax=ax, alpha=0.5)
plt.tight_layout(pad=0.1)


# In[ ]:


def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# In[ ]:


#Write encoded value to list
temp_list = []
for i in range(18000):
    temp_list.append(rle_encode(result_array[i]))


# In[ ]:


# Merge mask with Dataframe and write csv to disk
b['rle_mask']=pd.Series(temp_list).values
b['id'] = b['id'].astype(str).str.replace(r".png", '')
b.to_csv('/kaggle/working/submission.csv', index = False, header = True)

