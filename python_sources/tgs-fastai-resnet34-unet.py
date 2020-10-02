#!/usr/bin/env python
# coding: utf-8

# Checking FastAI

# In[ ]:


from __future__ import print_function, division

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / union
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []    
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / union)
        ious.append(iou)
    ious = map(mean, zip(*ious)) # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.elu(errors_sorted) +1, Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, only_present=False, per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), only_present=only_present)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), only_present=only_present)
    return loss


def lovasz_softmax_flat(probas, labels, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float() # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (Variable(fg) - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


# In[ ]:


from fastai.conv_learner import *
from fastai.dataset import *
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import os

from pathlib import Path
import json
torch.cuda.set_device(0)


# In[ ]:


MASKS_FN = 'train.csv'
TRAIN_DN = Path('train/images/')
MASKS_DN = Path('train/masks/')
TEST = Path('test/images/')

PATH = Path('/kaggle/input/tgs-salt-identification-challenge/')
PATH128 = Path('/tmp/128/')
TMP = Path('/tmp/')
MODEL = Path('/tmp/model/')
PRETRAINED = Path('/kaggle/input/fork-of-is-there-salt-resnet34/model/resnet34_issalt.h5')
seg = pd.read_csv(PATH/MASKS_FN).set_index('id')
seg.head()

sz = 128
bs = 64
nw = 4


# In[ ]:


ls /kaggle/input/is-there-salt-resnet34/models/res


# In[ ]:


train_names_png = [TRAIN_DN/f for f in os.listdir(PATH/TRAIN_DN)]
train_names = list(seg.index.values)
masks_names_png = [MASKS_DN/f for f in os.listdir(PATH/MASKS_DN)]
test_names_png = [TEST/f for f in os.listdir(PATH/TEST)]


# In[ ]:


train_names_png[0], masks_names_png[0], test_names_png[0]


# In[ ]:


TMP.mkdir(exist_ok=True)
PATH128.mkdir(exist_ok=True)
(PATH128/'train').mkdir(exist_ok=True)
(PATH128/'test').mkdir(exist_ok=True)
(PATH128/MASKS_DN).mkdir(exist_ok=True)
(PATH128/TRAIN_DN).mkdir(exist_ok=True)
(PATH128/TEST).mkdir(exist_ok=True)


# In[ ]:


def resize_mask(fn, sz=128):
    Image.open(PATH/fn).resize((sz,sz)).save(PATH128/fn)


# In[ ]:


with ThreadPoolExecutor(4) as e: e.map(resize_mask, train_names_png)


# In[ ]:


with ThreadPoolExecutor(4) as e: e.map(resize_mask, masks_names_png)


# In[ ]:


with ThreadPoolExecutor(4) as e: e.map(resize_mask, test_names_png)


# In[ ]:


def show_img(im, figsize=None, ax=None, alpha=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha)
    ax.set_axis_off()
    return ax


# In[ ]:


show_img(open_image(str(PATH128/train_names_png[0])))


# In[ ]:


class CustomDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y=y
        assert(len(fnames)==len(y))
        super().__init__(fnames, transform, path)
        
    def get_x(self, i): 
        return open_image(os.path.join(self.path,self.fnames[i]))
    def get_y(self, i): 
        return open_image(os.path.join(self.path,self.y[i]))
    def get_c(self): return 0


# In[ ]:


class TestFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y=y
        self.th = 1
        super().__init__(fnames, transform, path)        
    def get_x(self, i): 
        return np.fliplr(open_image(os.path.join(self.path, self.fnames[i])))
    def get_y(self, i): 
        return np.fliplr(open_image(os.path.join(self.path,self.y[i])))
    def get_c(self): return 0


# In[ ]:


def IoU_np(pred, targs, thres=0):
    pred = (pred>thres)
    intersection = (pred*targs).sum()
    return intersection / ((pred+targs).sum() - intersection + 1.0)

def IoU(pred, targs, thres=0):
    pred = (pred>thres).float()
    intersection = (pred*targs).sum()
    return intersection / ((pred+targs).sum() - intersection + 1.0)


# In[ ]:


def get_base(f, cut):
    layers = cut_model(f(True), cut)
    return nn.Sequential(*layers)

def load_pretrained(model, path):
    weights = torch.load(PRETRAINED, map_location=lambda storage, loc: storage)
    model.load_state_dict(weights, strict=False)
            
    return model


# In[ ]:


class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()


# In[ ]:


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out//2
        self.x_conv  = nn.Conv2d(x_in,  x_out,  1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        
    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p,x_p], dim=1)
        return self.bn(F.relu(cat_p))


# In[ ]:


class Unet34(nn.Module):
    def __init__(self, rn):
        super().__init__()
        self.rn = rn
        self.sfs = [SaveFeatures(rn[i]) for i in [2,4,5,6]]
        self.up1 = UnetBlock(512,256,256)
        self.up2 = UnetBlock(256,128,256)
        self.up3 = UnetBlock(256,64,256)
        self.up4 = UnetBlock(256,64,256)
#         self.conv1 = nn.Conv2d(512, 256, kernel_size=3, bias=False, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(256)
#         self.conv2 = nn.Conv2d(256, 512, kernel_size=1, bias=False, stride=1, padding=0)
#         self.bn2 = nn.BatchNorm2d(512)
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                             nn.Conv2d(512,256,1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(256, 512,1),
                             nn.Sigmoid()
                               )
        self.sSE = nn.Sequential(nn.Conv2d(512,512,1),
                             nn.Sigmoid())
        
        self.up5 = nn.ConvTranspose2d(256, 1, 2, stride=2)
        
    def forward(self,x):
        x = F.dropout(F.relu(self.rn(x)),0.3)
        x = torch.addcmul(x * self.cSE(x), 1, x, self.sSE(x))

#         x = F.leaky_relu(self.bn2(self.conv2(x)))
        
        x1 = self.up1(x, self.sfs[3].features)
        x2 = self.up2(x1, self.sfs[2].features)
        x = self.up3(x2, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x)
        x = torch.cat((x,
                       F.interpolate((x2),scale_factor=8,mode='bilinear',align_corners=True),
                       F.interpolate((x1),scale_factor=16,mode='bilinear', align_corners=True)
                      ),1)
        return F.dropout(x[:,0],0.3)
    
    def close(self):
        for sf in self.sfs: sf.remove()


# In[ ]:


class UnetModel():
    def __init__(self,model,name='unet'):
        self.model,self.name = model,name

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.rn), [lr_cut]))
        return lgs + [children(self.model)[1:]]


# In[ ]:


x_names = [f'{x}.png' for x in train_names]
x_names_path = np.array([str(TRAIN_DN/x) for x in x_names])
y_names = [x for x in x_names]
y_names_path = np.array([str(MASKS_DN/x) for x in x_names])


# In[ ]:


class RandomRotate2(CoordTransform):
    """ Rotates images and (optionally) target y.

    Rotating coordinates is treated differently for x and y on this
    transform.
     Arguments:
        deg (float): degree to rotate.
        p (float): probability of rotation
        mode: type of border
        tfm_y (TfmType): type of y transform
    """
    def __init__(self, deg, p=0.75, mode=cv2.BORDER_REFLECT, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.deg,self.p = deg,p
        if tfm_y == TfmType.COORD or tfm_y == TfmType.CLASS:
            self.modes = (mode,cv2.BORDER_CONSTANT)
        else:
            self.modes = (mode,mode)

    def set_state(self):
        self.store.rdeg = rand0(self.deg)
        self.store.rp = random.random()<self.p

    def do_transform(self, x, is_y):
        if self.store.rp: x = rotate_cv(x, self.store.rdeg, 
                mode = self.modes[0],
                interpolation=cv2.INTER_NEAREST if is_y else cv2.INTER_AREA)
        return x


# In[ ]:


aug_tfms = [RandomRotate2(4, tfm_y=TfmType.CLASS),
            RandomFlip(tfm_y=TfmType.CLASS),
            RandomStretch(0.2,tfm_y=TfmType.CLASS),
            RandomLighting(0.05, 0.05, tfm_y=TfmType.CLASS)]
tfms = tfms_from_model(resnet34, sz=sz, pad=0, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)

def get_data(val_idxs):
    ((val_x,trn_x),(val_y,trn_y)) = split_by_idx(val_idxs, x_names_path, y_names_path)
    datasets = ImageData.get_ds(CustomDataset, (trn_x,trn_y), (val_x,val_y), tfms, (test_x, test_x), path=PATH128)
    return ImageData(PATH128, datasets, bs=32, num_workers=nw, classes=None)
    


# In[ ]:


f = resnet34
cut,lr_cut = model_meta[f]

def get_learner(md):
    m_base = load_pretrained(get_base(f, cut),PRETRAINED)
    m = to_gpu(Unet34(m_base))
    models = UnetModel(m)
    learn = ConvLearner(md, models, tmp_name=TMP, models_name=MODEL)
    learn.opt_fn=optim.Adam
    learn.crit = nn.BCEWithLogitsLoss()
    learn.metrics=[accuracy_thresh(0.5), IoU]
    learn.clip = 0.25
    return learn


# In[ ]:


n_folds = 5
val_size = 4000//n_folds
actual_its = []
lr=3e-3
wd=1e-7
lrs = np.array([lr/100,lr/10,lr/3])
test_x = np.array(test_names_png)

out=np.zeros((18000,sz,sz))
alpha = 0


# # UNET

# In[ ]:


for i in [0,1,2]:
    #fold i
    print(f'Fold{i}------------------------------------')
    val_idxs=list(range(i*val_size, (i+1)*val_size))
    md = get_data(val_idxs)
    
    #learner
    learn = get_learner(md)
    learn.freeze_to(1)
    print(f'fit (lrs, wd): {lrs, wd}')
    learn.fit(lrs, 1, wds=wd, cycle_len=20,use_clr_beta=(10,10, 0.85, 0.9),best_save_name='best1')
    print('unfreezing')
    learn.unfreeze()
#     learn.fit(lrs/10, 1, wds=wd, cycle_len=20,use_clr_beta=(10,10, 0.85, 0.9))
    learn.load('best1')
    learn.crit = lovasz_hinge
    learn.fit(lrs/10, 1, wds=wd, cycle_len=20,use_clr_beta=(10,10, 0.85, 0.9),best_save_name='best2')
    learn.load('best2')
    print(f'computing test set: {i}')
    p = learn.predict(is_test=True)
    learn.save(f'{i}-model')
    print('Computing optimal threshold')
    preds, targs = learn.predict_with_targs()
    IoUs=[]
    for a in np.arange(0, 1, 0.1):
        IoUs.append(IoU_np(preds, targs, a))
    IoU_max = np.array(IoUs).argmax()
    print(f'optimal Threshold: {IoU_max/10.0}')
    alpha+=IoU_max/10.0
    print('TTA')
    md.test_dl.dataset = TestFilesDataset(test_x,test_x,tfms[1],PATH128)
    p_f = learn.predict(is_test=True)
    p_f = p_f[:,:,::-1]
    out = (p+p_f)/2
    actual_its.append(i)


# In[ ]:


print(f'Last Iteration group [0,1,2]: {i} ')


# In[ ]:


for i in [3,4]:
    #fold i
    print(f'Fold{i}------------------------------------')
    val_idxs=list(range(i*val_size, (i+1)*val_size))
    md = get_data(val_idxs)
    
    #learner
    learn = get_learner(md)
    learn.freeze_to(1)
    print(f'fit (lrs, wd): {lrs, wd}')
    learn.fit(lrs, 1, wds=wd, cycle_len=20,use_clr_beta=(10,10, 0.85, 0.9),best_save_name='best1')
    print('unfreezing')
    learn.unfreeze()
#     learn.fit(lrs/10, 1, wds=wd, cycle_len=20,use_clr_beta=(10,10, 0.85, 0.9))
    learn.load('best1')
    learn.crit = lovasz_hinge
    learn.fit(lrs/10, 1, wds=wd, cycle_len=20,use_clr_beta=(10,10, 0.85, 0.9),best_save_name='best2')
    learn.load('best2')
    print(f'computing test set: {i}')
    p = learn.predict(is_test=True)
    learn.save(f'{i}-model')
    print('Computing optimal threshold')
    preds, targs = learn.predict_with_targs()
    IoUs=[]
    for a in np.arange(0, 1, 0.1):
        IoUs.append(IoU_np(preds, targs, a))
    IoU_max = np.array(IoUs).argmax()
    print(f'optimal Threshold: {IoU_max/10.0}')
    alpha+=IoU_max/10.0
    print('TTA')
    md.test_dl.dataset = TestFilesDataset(test_x,test_x,tfms[1],PATH128)
    p_f = learn.predict(is_test=True)
    p_f = p_f[:,:,::-1]
    out = (p+p_f)/2
    actual_its.append(i)


# # Predict

# In[ ]:


print(f'Last Iteration group [3,4]: {i} ')


# In[ ]:


print(actual_its)
out = out/n_folds
alpha = alpha/n_folds


# In[ ]:


fig, axes = plt.subplots(12, 6, figsize=(12, 40))
for i,ax in enumerate(axes.flat):
    ax = show_img(Image.open(PATH128/test_names_png[i+30]), ax=ax)
    show_img(out[i+30]>alpha, ax=ax, alpha=0.2)
plt.tight_layout(pad=0.1)


# In[ ]:


def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# In[ ]:


tmp_list = []
name_list = []
for i in range(18000):
    img = cv2.resize(out[i,:,:], dsize=(101,101), interpolation = cv2.INTER_CUBIC)
    tmp_list.append(rle_encode(img>alpha))
    name_list.append(test_names_png[i].name[0:-4])


# In[ ]:


sub = pd.DataFrame(list(zip(name_list, tmp_list)), columns = ['id', 'rle_mask'])


# In[ ]:


sub.to_csv('submission.csv', index=False)

