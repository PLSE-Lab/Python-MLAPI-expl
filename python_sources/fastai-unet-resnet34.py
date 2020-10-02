#!/usr/bin/env python
# coding: utf-8

# ## References
# https://www.kaggle.com/mnpinto/pneumothorax-fastai-u-net
# 
# Normalize with imagenet caused weird loss values so removed it
# added TTA 
# 

# In[ ]:


import sys
sys.path.insert(0, '../input/siim-acr-pneumothorax-segmentation')

import fastai
from fastai.vision import *
from mask_functions import *


# In[ ]:


from fastai.callbacks import *
fastai.__version__


# In[ ]:


SZ = 128
path = Path(f'../input/pneumotorax{SZ}/data{SZ}/data{SZ}')


# In[ ]:


# copy pretrained weights for resnet34 to the folder fastai will search by default
Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)
get_ipython().system("cp '../input/resnet34/resnet34.pth' '/tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth'")


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



def dice(input:Tensor, targs:Tensor, eps:float=1e-8)->Rank0Tensor:
    input = input.clone()
    targs = targs.clone()
    n = targs.shape[0]
    input = torch.softmax(input, dim=1).argmax(dim=1)
    input = input.view(n, -1)
    targs = targs.view(n, -1)
    input[input == 0] = -999
    intersect = (input == targs).sum().float()
    union = input[input > 0].sum().float() + targs[targs > 0].sum().float()
    del input, targs
    gc.collect()
    return ((2.0 * intersect + eps) / (union + eps)).mean()

def visualize_one(a, b, c, title):
    fig, ax = plt.subplots(3, 1, figsize=(15, 7))
    ax[0].set_title(title)
    ax[0].imshow(a.permute(1, 2, 0))
    ax[1].imshow(b.squeeze(), vmin=0, vmax=4)
    ax[2].imshow(c.squeeze(), vmin=0, vmax=4)
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()
    plt.show()
    
def visualize_some():
    n_batch = 0
    for batch in learn.data.train_dl:
        x, y = batch
        n_batch += 1
        if n_batch > 8:
            break
        for idx in range(bs):
            predimg, pred, _ = learn.predict(Image(x[idx].cpu()))
            visualize_one(x[idx], y[idx], pred, f"Index: {idx}")
    plt.tight_layout()
    
def print_stats(learn):
    print("Plotting Losses")
    learn.recorder.plot_losses()
    print("Plotting metrics")
    learn.recorder.plot_metrics()
    print("Plotting LR")
    learn.recorder.plot_lr()
    print("Validation losses")
    print(learn.recorder.val_losses)
    print("Metrics")
    print(learn.recorder.metrics)


# In[ ]:


# Create databunch
data = (SegmentationItemList.from_folder(path=path/'train')
        .split_by_rand_pct(0.2)
        .label_from_func(lambda x : str(x).replace('train', 'masks'), classes=[0, 1])
        .add_test((path/'test').ls(), label=None)
        .transform(get_transforms(), size=SZ, tfm_y=True)
        .databunch(path=Path('.'), bs=32)
       #        .normalize(imagenet_stats)
       )


# In[ ]:


# Display some images with masks
data.show_batch(rows=3, figsize=(10,10))


# In[ ]:


# Create U-Net with a pretrained resnet34 as encoder
learn = unet_learner(data, models.resnet34, metrics=[dice],model_dir="/kaggle/working",
                    callback_fns=[partial(EarlyStoppingCallback, monitor='dice',
                                          min_delta=0.01, patience=3)])


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


# Fit one cycle of 6 epochs with max lr of 1e-3
learn.fit_one_cycle(4,max_lr=1e3)


# In[ ]:


learn.recorder.plot_losses()
learn.recorder.plot_metrics()


# In[ ]:


# Unfreeze the encoder (resnet34)
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


# Fit one cycle of 12 epochs
#lr = 1e-3
learn.fit_one_cycle(12, max_lr=slice(1e-5,1e-3))


# ## TTA and Submission

# In[ ]:


from fastai.core import *
from fastai.basic_data import *
from fastai.basic_train import *
from fastai.torch_core import *
def _tta_only(learn:Learner, ds_type:DatasetType=DatasetType.Valid, num_pred:int=10) -> Iterator[List[Tensor]]:
    "Computes the outputs for several augmented inputs for TTA"
    dl = learn.dl(ds_type)
    ds = dl.dataset
    old = ds.tfms
    aug_tfms = [o for o in learn.data.train_ds.tfms]
    try:
        pbar = master_bar(range(num_pred))
        for i in pbar:
            ds.tfms = aug_tfms
            yield get_preds(learn.model, dl, pbar=pbar)[0]
    finally: ds.tfms = old

Learner.tta_only = _tta_only

def _TTA(learn:Learner, beta:float=0, ds_type:DatasetType=DatasetType.Valid, num_pred:int=10, with_loss:bool=False) -> Tensors:
    "Applies TTA to predict on `ds_type` dataset."
    preds,y = learn.get_preds(ds_type)
    all_preds = list(learn.tta_only(ds_type=ds_type, num_pred=num_pred))
    avg_preds = torch.stack(all_preds).mean(0)
    if beta is None: return preds,avg_preds,y
    else:            
        final_preds = preds*beta + avg_preds*(1-beta)
        if with_loss: 
            with NoneReduceOnCPU(learn.loss_func) as lf: loss = lf(final_preds, y)
            return final_preds, y, loss
        return final_preds, y

Learner.TTA = _TTA


# In[ ]:


# Predictions for the validation set
preds, ys = learn.get_preds()
preds = preds[:,1,...]
ys = ys.squeeze()


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


# Find optimal threshold
dices = []
thrs = np.arange(0.01, 1, 0.01)
for i in progress_bar(thrs):
    preds_m = (preds>i).long()
    dices.append(dice_overall(preds_m, ys).mean())
dices = np.array(dices)


# In[ ]:


best_dice = dices.max()
best_thr = thrs[dices.argmax()]

plt.figure(figsize=(8,4))
plt.plot(thrs, dices)
plt.vlines(x=best_thr, ymin=dices.min(), ymax=dices.max())
plt.text(best_thr+0.03, best_dice-0.01, f'DICE = {best_dice:.3f}', fontsize=14);
plt.show()


# In[ ]:


# Plot some samples
rows = 10
plot_idx = ys.sum((1,2)).sort(descending=True).indices[:rows]
for idx in plot_idx:
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))
    ax0.imshow(data.valid_ds[idx][0].data.numpy().transpose(1,2,0))
    ax1.imshow(ys[idx], vmin=0, vmax=1)
    ax2.imshow(preds[idx], vmin=0, vmax=1)
    ax1.set_title('Targets')
    ax2.set_title('Predictions')


# In[ ]:


# Predictions for test set
preds, _ = learn.TTA(ds_type=DatasetType.Test)
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

