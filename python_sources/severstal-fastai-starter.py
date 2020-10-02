#!/usr/bin/env python
# coding: utf-8

# ** WIP **

# # Introduction
# 
# Welcome to the Severstal: Steel Defect Detection competition. This competition is a two-fold competition: classify the type of steel defect, and also segment the parts of the image that contain the defect.
# 
# In this kernel, I will do a quick fastai starter.
# 

# ## EDA

# In[ ]:


import numpy as np
import pandas as pd
import os
from fastai.vision import *


# In[ ]:


print(os.listdir('../input/severstal-steel-defect-detection'))


# Let's look at the training set csv file:

# In[ ]:


comp_dir = Path('../input/severstal-steel-defect-detection')


# In[ ]:


train_df = pd.read_csv(comp_dir/'train.csv')
train_df.head(20)


# In[ ]:


proc_train_df = train_df[train_df['EncodedPixels'].notnull()].reset_index(drop=True)


# In[ ]:


proc_train_df['ImageId_ClassId'] = pd.DataFrame(proc_train_df['ImageId_ClassId'].str.split('_').tolist())[0]
#proc_train_df['ClassId'] = pd.DataFrame(proc_train_df['ImageId_ClassId'].str.split('_').tolist())[1]


# In[ ]:


proc_train_df.head(10)


# In[ ]:


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle2mask(rle, input_shape):
    width, height = input_shape[:2]
    
    mask= np.zeros( width*height ).astype(np.uint8)
    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
        
    return mask.reshape(height, width).T


# In[ ]:


# picking up 10 examples with at least two faults for visualization
examples = proc_train_df['ImageId_ClassId'][5618:5619]
print(examples)
grouped_EncodedPixels = proc_train_df.groupby('ImageId_ClassId')['EncodedPixels'].apply(list)

import cv2        
def viz_steel_from_path(img_path, img_id, mask, convert_to_float=False):
    img = cv2.imread(os.path.join(img_path, img_id))
    mask_decoded = rle2mask(mask, (256, 1600))
    ax = plt.axes()
    print(ax)
    ax.imshow(img)
    ax.imshow(mask_decoded, alpha=0.3, cmap="Reds")
# visualize the image we picked up earlier with mask
for example in examples:
    #print(example)
    #print(grouped_EncodedPixels)
    img_path = '../input/severstal-steel-defect-detection/train_images/'
    img_id = examples
    mask = grouped_EncodedPixels[example]
    masks = mask[0]
    viz_steel_from_path(img_path, example, masks)


# In[ ]:


def open_mask_rle(mask_rle:str, shape:Tuple[int, int])->ImageSegment:
    "Return `ImageSegment` object create from run-length encoded string in `mask_lre` with size in `shape`."
    x = FloatTensor(rle2mask(str(mask_rle), shape).astype(np.uint8))
    x = x.view(shape[0], shape[1], -1)
    return ImageSegment(x.permute(2,0,1))

def open_mk(fn,df):
    indices = df.ImageId_ClassId[df.ImageId_ClassId == fn.split('/')[-1]].index.tolist()
    blah = open_mask_rle(df.iloc[indices[0]].EncodedPixels, shape=(256,1600))
    return blah
    
    


class _SegmentationLabelList(SegmentationLabelList):
    def open(self,fn): return open_mk(fn,proc_train_df)
    def transform(self, tfms:Optional[Tuple[TfmList,TfmList]]=(None,None), **kwargs):
        print(tfms)
        "Set `tfms` to be applied to the xs of the train and validation set."
        if not tfms: tfms=(None,None)
        assert is_listy(tfms) and len(tfms) == 2, "Please pass a list of two lists of transforms (train and valid)."
        self.train.transform(tfms[0], **kwargs)
        self.valid.transform(tfms[1], **kwargs)
        if self.test: self.test.transform(tfms[1], **kwargs)
        return self
class _SegmentationItemList(ImageList):
    _label_cls,_square_show_res = _SegmentationLabelList,False
    def transform(self, tfms:Optional[Tuple[TfmList,TfmList]]=(None,None), **kwargs):
        print(tfms)
        "Set `tfms` to be applied to the xs of the train and validation set."
        if not tfms: tfms=(None,None)
        assert is_listy(tfms) and len(tfms) == 2, "Please pass a list of two lists of transforms (train and valid)."
        self.train.transform(tfms[0], **kwargs)
        self.valid.transform(tfms[1], **kwargs)
        if self.test: self.test.transform(tfms[1], **kwargs)
        return self

src=(_SegmentationItemList.from_df(df=proc_train_df,path=Path('../input/severstal-steel-defect-detection/train_images'))
     .split_by_rand_pct(valid_pct=0.20)
     .label_from_func(func=(lambda x: x),classes=[0,1])
     .databunch(bs=2,num_workers=0)
     .normalize(imagenet_stats)
    )


# In[ ]:


src.show_batch()


# In[ ]:


# Making pretrained weights work without needing to find the default filename
if not os.path.exists('/tmp/.cache/torch/checkpoints/'):
        os.makedirs('/tmp/.cache/torch/checkpoints/')
get_ipython().system("cp '../input/resnet18/resnet18.pth' '/tmp/.cache/torch/checkpoints/resnet18-5c106cde.pth'")


# In[ ]:


from fastai.metrics import dice
learn = unet_learner(src,models.resnet18,metrics=[dice])
learn.path = Path('../')
learn.model_dir = 'models'


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(1,5e-5)


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(5,slice(1e-6,1e-5))


# In[ ]:


sample_df = pd.read_csv('../input/severstal-steel-defect-detection/sample_submission.csv')


# In[ ]:


sample_df.head(10)


# In[ ]:


proc_test_df = sample_df
proc_test_df['ImageId_ClassId'] = pd.DataFrame(sample_df['ImageId_ClassId'].str.split('_').tolist())[0]


# In[ ]:


learn.data.batch_size = 1


# In[ ]:


learn.data.train_dl.num_workers


# In[ ]:


learn.data.train_dl.num_workers = 0


# In[ ]:


learn.path


# In[ ]:


learn.export(destroy=True)


# In[ ]:


import gc;
gc.collect()


# In[ ]:


torch.cuda.empty_cache()


# In[ ]:


learn = load_learner('..', test=_SegmentationItemList.from_df(proc_test_df,'../input/severstal-steel-defect-detection',folder='test_images'))


# In[ ]:


learn.data.add_test(_SegmentationItemList.from_df(proc_test_df,'../input/severstal-steel-defect-detection',folder='test_images'))


# In[ ]:


learn.data.test_dl.batch_size=1
learn.data.test_dl.num_workers=0


# In[ ]:


import gc
gc.collect()
preds, _ = learn.get_preds(ds_type=DatasetType.Test)

