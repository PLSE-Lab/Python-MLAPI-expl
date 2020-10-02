#!/usr/bin/env python
# coding: utf-8

# Checking FastAI

# In[ ]:


import numpy as np
import pandas as pd
import os
print(os.listdir("/kaggle/input/tgs-salt-identification-challenge"))
#os.getcwd()


# In[ ]:


from fastai.conv_learner import *
from fastai.dataset import *

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
# PRETRAINED = Path('/kaggle/input/is-there-salt-resnet34/model/resnet34_issalt.h5')
seg = pd.read_csv(PATH/MASKS_FN).set_index('id')
seg.head()

sz = 128
bs = 64
nw = 4


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


def resize_mask(fn):
    Image.open(PATH/fn).resize((128,128)).save(PATH128/fn)


# In[ ]:


with ThreadPoolExecutor(4) as e: e.map(resize_mask, train_names_png)


# In[ ]:


with ThreadPoolExecutor(4) as e: e.map(resize_mask, masks_names_png)


# In[ ]:


with ThreadPoolExecutor(4) as e: e.map(resize_mask, test_names_png)


# In[ ]:


PATH = PATH128 #just for sanity


# In[ ]:


def show_img(im, figsize=None, ax=None, alpha=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha)
    ax.set_axis_off()
    return ax


# In[ ]:


im, mk = Image.open(PATH/train_names_png[0]), Image.open(PATH/masks_names_png[0])


# In[ ]:


ax = show_img(im);
show_img(mk, alpha=0.3);


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


def dice(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def IoU(pred, targs):
    pred = (pred>0).float()
    intersection = (pred*targs).sum()
    return intersection / ((pred+targs).sum() - intersection + 1.0)


# In[ ]:


x_names = [f'{x}.png' for x in train_names]
x_names_path = np.array([str(TRAIN_DN/x) for x in x_names])
y_names = [x for x in x_names]
y_names_path = np.array([str(MASKS_DN/x) for x in x_names])


# In[ ]:


im = open_image(str(PATH/y_names_path[1]))


# In[ ]:


im[:,:,0][:,:,None].shape


# In[ ]:


# val_idxs = list(range(200))
val_idxs=list(range(20))
((val_x,trn_x),(val_y,trn_y)) = split_by_idx(val_idxs, x_names_path, y_names_path)
test_x = np.array(test_names_png)


# In[ ]:


trn_x, trn_y, test_x


# In[ ]:


aug_tfms = [RandomRotate(4, tfm_y=TfmType.CLASS),
            RandomFlip(tfm_y=TfmType.CLASS),
            RandomLighting(0.05, 0.05, tfm_y=TfmType.CLASS)]
# aug_tfms = []


# In[ ]:


tfms = tfms_from_model(resnet34, sz=128, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)
datasets = ImageData.get_ds(CustomDataset, (trn_x,trn_y), (val_x,val_y), tfms, (test_x, test_x), path=PATH)
md = ImageData(PATH, datasets, bs=16, num_workers=nw, classes=None)
denorm = md.trn_ds.denorm


# In[ ]:


x,y = next(iter(md.trn_dl))
x.shape, y.shape


# # VGG - UNET

# In[ ]:


from fastai.models.unet import *


# In[ ]:


def get_encoder(f, cut):
    base_model = (cut_model(f(True), cut))
    return nn.Sequential(*base_model)


# In[ ]:


# Wrap everything nicely
class UpsampleModel():
    def __init__(self, model, cut_lr, name='upsample'):
        self.model,self.name, self.cut_lr = model, name, cut_lr

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.encoder), [self.cut_lr]))
        return lgs + [children(self.model)[1:]]


# In[ ]:


class DynamicUnet2(nn.Module):
    """
    A dynamic implementation of Unet architecture, because calculating connections
    and channels suck!. When an encoder is passed, this network will
    automatically construct a decoder after the first single forward pass for any
    given encoder architecture.

    Decoder part is heavily based on the original Unet paper:
    https://arxiv.org/abs/1505.04597.

    Inputs:
        encoder(nn.Module): Preferably a pretrained model, such as VGG or ResNet
        last (bool): Whether to concat only last activation just before a size change
        n_classes (int): Number of classes to output in final step of decoder

    Important Note: If architecture directly reduces the dimension of an image as soon as the
    first forward pass then output size will not be same as the input size, e.g. ResNet.
    In order to resolve this problem architecture will add an additional extra conv transpose
    layer. Also, currently Dynamic Unet expects size change to be H,W -> H/2, W/2. This is
    not a problem for state-of-the-art architectures as they follow this pattern but it should
    be changed for custom encoders that might have a different size decay.
    """

    def __init__(self, encoder, last=True, n_classes=3):
        super().__init__()
        self.encoder = encoder
        self.n_children = len(list(encoder.children()))
        self.sfs = [SaveFeatures(encoder[i]) for i in range(self.n_children)]
        self.last = last
        self.n_classes = n_classes

    def forward(self, x):
        dtype = x.data.type()

        # get imsize
        imsize = x.size()[-2:]

        # encoder output
        x = F.relu(self.encoder(x))

        # initialize sfs_idxs, sfs_szs, middle_in_c and middle_conv only once
        if not hasattr(self, 'middle_conv'):
            self.sfs_szs = [sfs_feats.features.size() for sfs_feats in self.sfs]
            self.sfs_idxs = get_sfs_idxs(self.sfs, self.last)
            middle_in_c = self.sfs_szs[-1][1]
            middle_conv = nn.Sequential(*conv_bn_relu(middle_in_c, middle_in_c * 2, 3, 1, 1),
                                        *conv_bn_relu(middle_in_c * 2, middle_in_c, 3, 1, 1))
            self.middle_conv = middle_conv.type(dtype)

        # middle conv
        x = self.middle_conv(x)

        # initialize upmodel, extra_block and 1x1 final conv
        if not hasattr(self, 'upmodel'):
            x_copy = Variable(x.data, requires_grad=False)
            upmodel = []
            for idx in self.sfs_idxs[::-1]:
                up_in_c, x_in_c = int(x_copy.size()[1]), int(self.sfs_szs[idx][1])
                unet_block = UnetBlock(up_in_c, x_in_c).type(dtype)
                upmodel.append(unet_block)
                x_copy = unet_block(x_copy, self.sfs[idx].features)
                self.upmodel = nn.Sequential(*upmodel)

            if imsize != self.sfs_szs[0][-2:]:
                extra_in_c = self.upmodel[-1].conv2.out_channels
                self.extra_block = nn.ConvTranspose2d(extra_in_c, extra_in_c, 2, 2).type(dtype)

            final_in_c = self.upmodel[-1].conv2.out_channels
            self.final_conv = nn.Conv2d(final_in_c, self.n_classes, 1).type(dtype)

        # run upsample
        for block, idx in zip(self.upmodel, self.sfs_idxs[::-1]):
            x = block(x, self.sfs[idx].features)
        if hasattr(self, 'extra_block'):
            x = self.extra_block(x)

        out = self.final_conv(x)
        if self.n_classes == 1:
            out = out.squeeze(1)
        return out


# In[ ]:


f = vgg16
cut,lr_cut = model_meta[f]
cut,lr_cut


# In[ ]:


encoder= get_encoder(f, 30)
# m_base = get_base()
m = DynamicUnet2(encoder, n_classes=1).cuda()
# m = to_gpu(Unet34(m_base))
# models = UnetModel(m)


# In[ ]:


inp = torch.ones(1, 3, 128, 128)
out = m(V(inp))


# In[ ]:


out.shape


# In[ ]:


models = UpsampleModel(m, cut_lr=9)


# In[ ]:


learn = ConvLearner(md, models, tmp_name=TMP, models_name=MODEL)
learn.opt_fn=optim.Adam
learn.crit=nn.BCEWithLogitsLoss()
learn.metrics=[accuracy_thresh(0.5),dice, IoU]


# In[ ]:


learn.freeze_to(1)


# In[ ]:


learn.lr_find()
learn.sched.plot()


# In[ ]:


lr=1e-2
wd=1e-4

lrs = np.array([lr/100,lr/100,lr/10])


# In[ ]:


learn.fit(lr,1,wds=wd,cycle_len=20,use_clr=(5,8))


# In[ ]:


learn.save('128unet1')


# In[ ]:


learn.load('128unet1')


# In[ ]:


learn.unfreeze()
learn.bn_freeze(True)


# In[ ]:


learn.fit(lrs/10, 1, wds=wd, cycle_len=10,use_clr=(20,10))


# In[ ]:


# learn.fit(lrs/3, 1, wds=wd, cycle_len=10,use_clr=(8,5))


# In[ ]:


py,ay = learn.predict_with_targs()


# In[ ]:


fig, axes = plt.subplots(5, 4, figsize=(12, 12))
for i,ax in enumerate(axes.flat):
    ax = show_img((ay[i]), ax=ax)
plt.tight_layout(pad=0.1)


# In[ ]:


fig, axes = plt.subplots(5, 4, figsize=(12, 12))
for i,ax in enumerate(axes.flat):
    ax = show_img((py[i]>0), ax=ax)
plt.tight_layout(pad=0.1)


# # Predict

# In[ ]:


# Predict on Test data
out = learn.predict(is_test=True)
out.shape


# In[ ]:


fig, axes = plt.subplots(6, 6, figsize=(12, 12))
for i,ax in enumerate(axes.flat):
#     ax = show_img(Image.open(PATH/test_names_png[i]), ax=ax)
    show_img(out[i]>0, ax=ax, alpha=0.8)
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


tmp_list = []
name_list = []
for i in range(18000):
    img = cv2.resize(out[i,:,:], dsize=(101,101), interpolation = cv2.INTER_CUBIC)
    tmp_list.append(rle_encode(img>0))
    name_list.append(test_names_png[i].name[0:-4])


# In[ ]:


sub = pd.DataFrame(list(zip(name_list, tmp_list)), columns = ['id', 'rle_mask'])


# In[ ]:


sub.head()


# In[ ]:


sub.to_csv('submission.csv', index=False)


# In[ ]:


#last version 1PM - 03/10/

