#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# First attempt to build the prediction model.
# 
# This kernel is based on the following:
# * Base foundation: https://www.kaggle.com/mnpinto/pneumothorax-fastai-u-net
# * Adding hypercolumns etc: https://www.kaggle.com/iafoss/hypercolumns-pneumothorax-fastai-0-819-lb

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import sys
print(os.listdir("../input"))
sys.path.insert(0, '../input/siim-acr-pneumothorax-segmentation')

import fastai
from fastai.vision import *
from mask_functions import * 
from fastai.callbacks import SaveModelCallback
import gc
from sklearn.model_selection import KFold
from PIL import Image

fastai.__version__


# In[ ]:


print(os.listdir("../input/pneumotorax128"))
print(os.listdir("../input/pneumotorax128/data128"))
print(os.listdir("../input/pneumotorax128/data128/data128"))


# In[ ]:


sz = 128
bs = 32
n_acc = 64//bs
nfolds = 4


noise_th = 75.0 * (sz/128.0)**2
best_thr0 = 0.2

if sz == 256:
    stats = ([0.540,0.540,0.540],[0.264,0.264,0.264])
    TRAIN = '../input/pneumotorax256/data256/data256/train'
    TEST = '../input/pneumotorax256/data256/data256/test'
    MASKS = '../input/pneumotorax256/data256/data256/masks'
elif sz == 128:
    stats = ([0.615,0.615,0.615],[0.291,0.291,0.291])
    TRAIN = '../input/pneumotorax128/data128/data128/train'
    TEST = '../input/pneumotorax128/data128/data128/test'
    MASKS = '../input/pneumotorax128/data128/data128/masks'


# In[ ]:


Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True,parents=True)
get_ipython().system("cp '../input/resnet34/resnet34.pth' '/tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth'")


# In[ ]:


print(os.listdir("/tmp/"))
print(os.listdir("/tmp/.cache/"))
print(os.listdir("/tmp/.cache/torch"))
print(os.listdir("/tmp/.cache/torch/checkpoints"))


# In[ ]:


SEED = 2019
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #tf.set_random_seed(seed)
seed_everything(SEED)


# In[ ]:


# Setting div=True in open_mask
class SegmentationLabelList(SegmentationLabelList):
    def open(self, fn): return open_mask(fn, div=True)
    
class SegmentationItemList(SegmentationItemList):
    _label_cls = SegmentationLabelList

# Setting transformations on masks to False on test set
def transform(self, tfms:Optional[Tuple[TfmList,TfmList]]=(None,None), **kwargs):
    if not tfms: tfms=(None,None)
    assert is_listy(tfms) and len(tfms) == 2
    self.train.transform(tfms[0], **kwargs)
    self.valid.transform(tfms[1], **kwargs)
    kwargs['tfm_y'] = False # Test data has no labels
    if self.test: self.test.transform(tfms[1], **kwargs)
    return self
fastai.data_block.ItemLists.transform = transform


# In[ ]:


# Create databunch
data = (SegmentationItemList.from_folder(path=Path(TRAIN)).split_by_rand_pct(0.2)
            .label_from_func(lambda x: str(x).replace('train','masks'), classes=[0,1])
            .add_test(Path(TEST).ls(),label=None)
            .transform(get_transforms(), size=sz, tfm_y=True)
            .databunch(path=Path('.'), bs=32)
            .normalize(imagenet_stats)
       )


# In[ ]:


data.show_batch()


# In[ ]:


learn = unet_learner(data, models.resnet34, metrics=[dice])


# In[ ]:


epoch=5
learn.fit_one_cycle(epoch),slice(1e-3)


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(5,slice(3e-5,3e-4))


# In[ ]:


# predictions for the validation set
preds, ys = learn.get_preds()


# In[ ]:


print(preds.shape)
print(ys.shape)


# In[ ]:


preds = preds[:,1,...]
ys=ys.squeeze()


# In[ ]:


n = preds.shape[0]
print(preds.view(n, -1).shape)


# In[ ]:


def dice_overall(preds, targs):
    n = preds.shape[0]
    preds = preds.view(n, -1)
    targs = targs.view(n, -1)
    intersect = (preds * targs).sum(-1).float()
    union = (preds+targs).sum(-1).float()
    u0 = union==0
    intersect[u0] = 1
    union[u0] = 2
    return (2. * intersect / union)


# In[ ]:


# find optimal threshold for DICE. similar in principle with ROC Curve.
dices = []
thrs = np.arange(0.01,1,0.05)
for i in progress_bar(thrs):
    preds_m = (preds>i).long()
    dices.append(dice_overall(preds_m,ys).mean())
dices = np.array(dices)


# In[ ]:


best_dice = dices.max()
best_thr = thrs[dices.argmax()]

plt.figure(figsize=(8,4))
plt.plot(thrs, dices)
plt.vlines(x=best_thr, ymin=dices.min(), ymax=dices.max())
plt.text(best_thr+0.03, best_dice-0.15, f'DICE = {best_dice:.2f} \nThreshold = {best_thr:.2f}', fontsize=14);
plt.show()


# # Plot some samples

# In[ ]:


rows = 20
plot_idx = ys.sum((1,2)).sort(descending=True).indices[:rows]
print(plot_idx)


# In[ ]:


for idx in plot_idx:
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))
    ax0.imshow(data.valid_ds[idx][0].data.numpy().transpose(1,2,0))
    ax1.imshow(ys[idx], vmin=0, vmax=1)
    ax2.imshow(preds[idx], vmin=0, vmax=1)
    ax1.set_title('Targets')
    ax2.set_title('Predictions')


# In[ ]:


# Predictions for test set
preds, _ = learn.get_preds(ds_type=DatasetType.Test)
preds = (preds[:,1,...]>best_thr).long().numpy()
print(preds.sum())


# In[ ]:


# Generate rle encodings (images are first converted to the original size)
rles = []
for p in progress_bar(preds):
    im = PIL.Image.fromarray((p.T*255).astype(np.uint8)).resize((1024,1024))
    im = np.asarray(im)
    rles.append(mask2rle(im, 1024, 1024))


# In[ ]:


ids = [o.stem for o in data.test_ds.items]
sub_df = pd.DataFrame({'ImageId': ids, 'EncodedPixels': rles})
sub_df.loc[sub_df.EncodedPixels=='', 'EncodedPixels'] = '-1'
sub_df.head()


# In[ ]:


sub_df.to_csv('submission.csv', index=False)


# # LIME explainer for pneumotorax prediction

# In[ ]:


def predict_fn(img):
    x = img.convert('RGB')
    test_img = Image(pil2tensor(x, np.float32).div_(255))
    pred_class, pred_idx, outputs = learn.predict(test_img)
    print (pred_class)
    return outputs.numpy()#my_softmax(preds)

def softmax(x):
    tmp = np.zeros_like(x)
    """Compute softmax values for each sets of scores in x."""
    for i in range(x.shape[0]):
        s = np.exp(x[i, :])/np.sum(np.exp(x[i, :]))
        tmp[i, :] = s
    return tmp # only difference

def batch_predict(images):
    for i, img in enumerate(images):
        output = predict_fn(img)
        print (output)
        if i == 0:
            preds = output
        else:
            preds = np.concatenate((preds, output), axis=0 )
    return softmax(preds.reshape(-1, len(pet_classes)))


# In[ ]:


explainer = lime_image.LimeImageExplainer()

