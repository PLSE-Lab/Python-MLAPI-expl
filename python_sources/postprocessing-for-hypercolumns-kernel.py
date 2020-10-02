#!/usr/bin/env python
# coding: utf-8

# ### Overview
# This kernel shows how to do postprocessing for data generated in [Hypercolumns pneumothorax fastai kernel](https://www.kaggle.com/iafoss/hypercolumns-pneumothorax-fastai-0-818-lb) including loading model predictions, threshold selection, generation of the test submission, mask split, and exploiting the leak in sample_submission.csv

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import sys
sys.path.insert(0, '../input/siim-acr-pneumothorax-segmentation')

from mask_functions import *
import gc
import pandas as pd
import numpy as np
import torch
import cv2
from collections import Counter
from scipy import ndimage
import os
from fastai.vision import *


# ### Load validation predictions and masks

# In[ ]:


sz = 256
#to save space, the predictions are multiplied by 255 and saved as byte
#therefore, in this kernel noise_th should be muliplied by 255 as well
noise_th = 75.0*255*(sz/128.0)**2 #threshold for the number of predicted pixels
nfolds = 4
MASKS = '../input/siimacr-pneumothorax-segmentation-data-256/masks'
DATA = '../input/hypercolumns-pneumothorax-fastai-0-819-lb'

preds0, items = [], []
for fold in range(nfolds):
    preds0.append(torch.load(os.path.join(DATA,'preds_fold'+str(fold)+'.pt')))
    items.append(np.load(os.path.join(DATA,'items_fold'+str(fold)+'.npy'), 
                         allow_pickle=True))
preds0 = torch.cat(preds0)
items = np.concatenate(items)

ys = []
for item in items:
    img = cv2.imread(str(item).replace('train', 'masks'),0)
    img = torch.Tensor(img).byte().unsqueeze(0)
    img = (img > 127)
    ys.append(img)
ys = torch.cat(ys)


# ### Threshold selection

# In[ ]:


#dice for threshold selection
def dice_overall(preds, targs):
    n = preds.shape[0]
    preds = preds.view(n, -1)
    targs = targs.view(n, -1)
    intersect = (preds * targs).float().sum(-1)
    union = (preds+targs).float().sum(-1)
    u0 = union==0
    intersect[u0] = 1
    union[u0] = 2
    return (2. * intersect / union)


# In contrast to the main kernel, here RAM is not the limiting factor. Therefore, the threshold is selected in the proper way: based on all oof predictions rather than average of thresholds for each fold. You can run the cell below for several values of noise_th to identify the best value.

# In[ ]:


noise_th = 75.0*255*(sz/128.0)**2 #threshold for the number of predicted pixels

gc.collect();
torch.cuda.empty_cache()
preds = preds0.clone()
preds[preds.view(preds.shape[0],-1).float().sum(-1) < noise_th,...] = 0.0

dices = []
#here evrything is multiplied by 255, so 0.2 threshold corresponds to 51
thrs = np.arange(40, 60, 1)
for th in progress_bar(thrs):
    preds_m = (preds>th).byte()
    dices.append(dice_overall(preds_m, ys).mean())
dices = np.array(dices)    

best_dice = dices.max()
best_thr = thrs[dices.argmax()]

plt.figure(figsize=(8,4))
plt.plot(thrs, dices)
plt.vlines(x=best_thr, ymin=dices.min(), ymax=dices.max())
plt.text(best_thr, best_dice, f'DICE = {best_dice:.3f}', fontsize=14);
plt.show()


# ### Load test predictions

# In[ ]:


TEST = '../input/siimacr-pneumothorax-segmentation-data-256/test/'
SAMPLE = '../input/siim-acr-pneumothorax-segmentation/sample_submission.csv'
items_test = Path(TEST).ls()
ids_test = [p.stem for p in items_test]
preds_test = torch.load(os.path.join(DATA,'preds_test.pt'))
preds_test0 = preds_test.clone()
preds_test[preds_test.view(preds_test.shape[0],-1).float().sum(-1) < noise_th,...] = 0


# ### General submission

# In[ ]:


# Generate rle encodings (images are first converted to the original size)
preds_t = (preds_test>best_thr).long().numpy()
rles = []
for p in progress_bar(preds_t):
    if(p.sum() > 0):
        im = PIL.Image.fromarray((p.T*255).astype(np.uint8)).resize((1024,1024))
        im = np.asarray(im)
        rles.append(mask2rle(im, 1024, 1024))
    else: rles.append('-1')
    
sub_df = pd.DataFrame({'ImageId': ids_test, 'EncodedPixels': rles})
sub_df.to_csv('submission.csv', index=False)
print(len(sub_df))
sub_df.head()


# ### Split masks

# In[ ]:


def split_mask(mask):
    threshold_obj = 30 #ignor predictions composed of 30 pixels or less
    #split disconnected masks with ndimage.label function
    labled,n_objs = ndimage.label(mask > best_thr)
    result = []
    n_pixels = []
    for i in range(n_objs):
        obj = (labled == i + 1).astype(int)
        if(obj.sum() > threshold_obj): 
            result.append(obj)
            n_pixels.append(obj.sum())
    #sort masks based on the number of pixels
    result = [x for _,x in sorted(zip(n_pixels,result),reverse=True,key=lambda x:x[0])]
    return result


# In[ ]:


rles = []
ids_c = []
for p,idx in progress_bar(zip(preds_test,ids_test), total=len(preds_test)):
    if(p.sum() > 0):
        masks = split_mask(to_np(p))
        for mask in masks:
            ids_c.append(idx)
            im = PIL.Image.fromarray((mask.T*255).astype(np.uint8)).resize((1024,1024))
            im = np.asarray(im)
            rles.append(mask2rle(im, 1024, 1024))
        if len(masks) == 0:
            rles.append('-1')
            ids_c.append(idx)
    else: 
        rles.append('-1')
        ids_c.append(idx)
    
sub_df = pd.DataFrame({'ImageId': ids_c, 'EncodedPixels': rles})
sub_df.to_csv('submission_split.csv', index=False)
print(len(sub_df))
sub_df.head()


# ### Use of a leak in sample_submission.csv

# In[ ]:


rles = []
ids_c = []
df_sample = pd.read_csv(SAMPLE)
c = Counter(list(df_sample.ImageId))

for p,p0,idx in progress_bar(zip(preds_test,preds_test0,ids_test),
                             total=len(preds_test)):
    n = c[idx] #number of masks in the submission file
    if n == 1:
        if(p.sum() > 0):
            #no maks split since only one mask is expected
            im = PIL.Image.fromarray((to_np(p > best_thr).T*255).astype(np.uint8))              .resize((1024,1024))
            im = np.asarray(im)
            rles.append(mask2rle(im, 1024, 1024))
        else: rles.append('-1')
        ids_c.append(idx)
    else:
        masks = split_mask(to_np(p0))
        if len(masks) == 0:
            rles.append('-1')
            ids_c.append(idx)
        else:
            for i,mask in enumerate(masks):
                im = PIL.Image.fromarray((mask.T*255).astype(np.uint8))                    .resize((1024,1024))
                im = np.asarray(im)
                rles.append(mask2rle(im, 1024, 1024))
                ids_c.append(idx)
                if i >= n: break 
                #masks are ordered based on the number of pixels
                #therefore, select only first n masks
    
sub_df = pd.DataFrame({'ImageId': ids_c, 'EncodedPixels': rles})
sub_df.to_csv('submission_leak.csv', index=False)
print(len(sub_df))
sub_df.head()


# In[ ]:




