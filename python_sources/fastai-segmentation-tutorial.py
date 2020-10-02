#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
GPU_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)


# In[ ]:


import warnings
warnings.filterwarnings("ignore")

import time
import os

from fastai.vision import *
from fastai.callbacks.hooks import *


# In[ ]:


get_ipython().system('pwd')


# ### Create a Path object and view folders

# In[ ]:


path = Path('../input/pennfudanped')
path.ls()


# In[ ]:


path_lbl = path/'PedMasks'
path_img = path/'PNGImages'


# In[ ]:


fnames = get_image_files(path_img)
fnames[:3]


# In[ ]:


lbl_names = get_image_files(path_lbl)
lbl_names[:3]


# ### show an image and its mask

# In[ ]:


img_f = fnames[0]
img = open_image(img_f)
img.show(figsize=(5, 5))


# In[ ]:


print(img_f.stem,img_f.suffix)
get_y_fn = lambda x: path_lbl/f'{x.stem}_mask{x.suffix}'
get_y_fn(img_f)


# In[ ]:


mask = open_mask(get_y_fn(img_f))
mask.show(figsize=(5, 5), alpha=1)
src_size = np.array(mask.shape[1:])


# Although there is only one class, person, objects are labeled with increasing integers. For example, values of object 1 in the mask are 1. values of object 2 in the mask are 2. so on so forth.

# ### so how many objects could there be in one image?

# In[ ]:


df = pd.read_csv(path/'added-object-list.txt',skiprows=1,sep='\t')
df.columns = ['image','objects']
df.head()


# In[ ]:


df['objects'].max()


# In[ ]:


mask = df['objects']==8
df.loc[mask]


# It seems there could be up to 8 objects in one image. Let's verify it.

# In[ ]:


img_f = Path(path/'PNGImages/FudanPed00058.png')
img = open_image(img_f)
img.show(figsize=(5, 5))


# In[ ]:


mask = open_mask(get_y_fn(img_f))
mask.show(figsize=(5, 5), alpha=1)


# yeah, it seems correct. However, some objects could be very small.

# ### Create databunch for segmentation

# In[ ]:


codes = np.array(['background','person'])


# In[ ]:


size = src_size//2
bs = 32


# The mask can have unique values from 0 to 8, which means up to 8 objects in the image. However, they belong to the same class, person. Hence, we binarize the masks. 

# In[ ]:


class MySegmentationLabelList(SegmentationLabelList):
    def open(self, fn): 
        res = open_mask(fn)
        res.px = (res.px>0).float()
        return res

class MySegmentationItemList(ImageList):
    "`ItemList` suitable for segmentation tasks."
    _label_cls,_square_show_res = MySegmentationLabelList,False


# In[ ]:


src = (MySegmentationItemList.from_folder(path_img) # SegmentationItemList
       .split_by_rand_pct(0.2) # SegmentationItemList
       .label_from_func(get_y_fn, classes=codes)) # LabelLists


# In[ ]:


data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))


# In[ ]:


data.show_batch(2, figsize=(5, 5))


# ### Training

# In[ ]:


name2id = {v:k for k,v in enumerate(codes)}
void_code = -1

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    #print(input.size())
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()


# In[ ]:


wd = 1e-2
metrics = acc_camvid
learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)
learn.model_dir = '/kaggle/working/models'


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


lr = 1e-4
learn.fit_one_cycle(10, slice(lr))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'pred,truths = learn.get_preds()')


# ### Let's visually check the predictions

# In[ ]:


class MyImageList(ImageList):
    def __init__(self, *args, imgs=None, **kwargs):
        self.imgs = imgs
        
    def get(self, i):
        res = self.imgs[i]
        return Image(res)


# In[ ]:


pred_masks = MyImageList(imgs = pred.argmax(dim=1,keepdim=True))
true_masks = MyImageList(imgs = truths)


# ### show true masks

# In[ ]:


def _plot(i,j,ax): true_masks[i*3+j].show(ax)
plot_multi(_plot, 3, 3, figsize=(8,8))


# ### show predicted masks

# In[ ]:


def _plot(i,j,ax): pred_masks[i*3+j].show(ax)
plot_multi(_plot, 3, 3, figsize=(8,8))

