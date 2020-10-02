#!/usr/bin/env python
# coding: utf-8

# https://www.kaggle.com/imrandude/tgs-salt-identification-fastai-unet-resnet

# In[ ]:


get_ipython().system('pip3 install pycocotools')


# In[ ]:


get_ipython().system('git clone https://github.com/dromosys/TGS-SaltIdentification-Open-Solution-fastai')
import sys
sys.path.insert(0, '/kaggle/working/TGS-SaltIdentification-Open-Solution-fastai')


# In[ ]:


import numpy as np
import pandas as pd
import os
print(os.listdir("/kaggle/input/tgs-salt-identification-challenge"))
#os.getcwd()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from fastai.conv_learner import *
from fastai.dataset import *
from fastai.models.resnet import vgg_resnet50
from fastai.models.senet import *
from skimage.transform import resize
import json
from sklearn.model_selection import train_test_split, StratifiedKFold , KFold
from sklearn.metrics import jaccard_similarity_score
from pycocotools import mask as cocomask
from utils import my_eval,intersection_over_union_thresholds,RLenc
from lovasz_losses import lovasz_hinge
print(torch.__version__)
torch.cuda.is_available()
torch.backends.cudnn.benchmark=True


# In[ ]:


MASKS_FN = 'train.csv'
TRAIN_DN = Path('train/images/')
MASKS_DN = Path('train/masks/')
TEST = Path('test/images/')

PATH = Path('/kaggle/input/tgs-salt-identification-challenge/')
PATH128 = Path('/tmp/128/')
TMP = Path('/tmp/')
MODEL = Path('/tmp/model/')
PRETRAINED = Path('/kaggle/input/is-there-salt-resnet34/model/resnet34_issalt.h5')
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


def resize_mask(fn, sz=128):
    Image.open(PATH/fn).resize((sz,sz)).save(PATH128/fn)


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


from datasets import CustomDataset


# In[ ]:


def dice(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def IoU_np(pred, targs, thres=0):
    pred = (pred>thres)
    intersection = (pred*targs).sum()
    return intersection / ((pred+targs).sum() - intersection + 1.0)

def IoU(pred, targs, thres=0):
    pred = (pred>thres).float()
    intersection = (pred*targs).sum()
    return intersection / ((pred+targs).sum() - intersection + 1.0)


# In[ ]:


def get_base():
    layers = cut_model(f(True), cut)
    return nn.Sequential(*layers)

def load_pretrained(model, path): #load a model pretrained on ship/no-ship classification
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
        self.up5 = nn.ConvTranspose2d(256, 1, 2, stride=2)
        
    def forward(self,x):
        x = F.dropout(F.relu(self.rn(x)),0.2)
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x)
        return x[:,0]
    
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


aug_tfms = [RandomRotate(4, tfm_y=TfmType.CLASS),
            RandomFlip(tfm_y=TfmType.CLASS),
            RandomLighting(0.05, 0.05, tfm_y=TfmType.CLASS)]
# aug_tfms = []


# In[ ]:


sz


# # UNET

# In[ ]:


lr=3e-3
wd=1e-7
lrs = np.array([lr/100,lr/10,lr])

n_folds = 10
out=np.zeros((18000,sz,sz))
alpha = 0
for i in range(n_folds):
    val_size = 4000//n_folds
    val_idxs=list(range(i*val_size, (i+1)*val_size))
    ((val_x,trn_x),(val_y,trn_y)) = split_by_idx(val_idxs, x_names_path, y_names_path)
    test_x = np.array(test_names_png)
    
    tfms = tfms_from_model(resnet34, sz=sz, pad=0, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)
    datasets = ImageData.get_ds(CustomDataset, (trn_x,trn_y), (val_x,val_y), tfms, (test_x, test_x), path=PATH)
    md = ImageData(PATH, datasets, bs=64, num_workers=nw, classes=None)
    denorm = md.trn_ds.denorm
    
    f = resnet34
    cut,lr_cut = model_meta[f]
    m_base = load_pretrained(get_base(),PRETRAINED)
    m = to_gpu(Unet34(m_base))
    models = UnetModel(m)
    learn = ConvLearner(md, models, tmp_name=TMP, models_name=MODEL)
    learn.opt_fn=optim.Adam
#     learn.crit=nn.BCEWithLogitsLoss()
    learn.crit = lovasz_hinge
    learn.metrics=[accuracy_thresh(0.5),dice, IoU]
    
    learn.freeze_to(2)
#     learn.fit(lr,1)
    learn.fit(lr,2,wds=wd,cycle_len=10,use_clr_beta=(10,10, 0.85, 0.9))
    learn.unfreeze()
    learn.fit(lrs, 3, wds=wd, cycle_len=10,use_clr_beta=(10,10, 0.85, 0.9))
    print(f'computing test set: {i}')
    out+=learn.predict(is_test=True)
    print('Computing optimal threshold')
    preds, targs = learn.predict_with_targs()
    IoUs=[]
    for a in np.arange(0, 1, 0.1):
        IoUs.append(IoU_np(preds, targs, a))
    IoU_max = np.array(IoUs).argmax()
    print(f'optimal Threshold: {IoU_max/10.0}')
    alpha+=IoU_max/10.0


# # Predict

# In[ ]:


out = out/n_folds
alpha = alpha/n_folds


# In[ ]:


fig, axes = plt.subplots(6, 6, figsize=(12, 12))
for i,ax in enumerate(axes.flat):
    ax = show_img(Image.open(PATH/test_names_png[i+30]), ax=ax)
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


test_names_png[0], test_x[0]


# In[ ]:


sub = pd.DataFrame(list(zip(name_list, tmp_list)), columns = ['id', 'rle_mask'])


# Filter with classification probabilities previously computed

# In[ ]:


sub.to_csv('submission.csv', index=False)


# # Optimal threshold finder

# In[ ]:


out


# In[ ]:


ls


# In[ ]:


rm -rf /tmp/model/


# In[ ]:


rm -rf /tmp/128


# In[ ]:


rm -rf /kaggle/working/TGS-SaltIdentification-Open-Solution-fastai


# In[ ]:




