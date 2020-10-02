#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
#sys.path = [ ###ResNext Models
#	'/gpfs/research/chaohuang/panda/models/semi-supervised-ImageNet1K-models-master',
#] + sys.path
sys.path = [ ###Utility scripts
	'../input/utility-script',
] + sys.path
import fastai
from fastai.vision import *
from fastai.callbacks import SaveModelCallback
import os
import numpy as np
import pandas as pd

#from sklearn.model_selection import KFold
from radam import *
from csvlogger import *
from mish_activation import *
#from hubconf import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score,confusion_matrix
from skimage.transform import resize
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.optim.optimizer import Optimizer, required
#import spams

fastai.__version__


# In[ ]:


# remove this cell if run locally
get_ipython().system("mkdir 'cache'")
get_ipython().system("mkdir 'cache/torch'")
get_ipython().system("mkdir 'cache/torch/checkpoints'")
get_ipython().system("cp '../input/pytorch-pretrained-models/semi_supervised_resnext50_32x4-ddb3e555.pth' 'cache/torch/checkpoints/'")
torch.hub.DEFAULT_CACHE_DIR = 'cache'


# In[ ]:


sz = 128
bs = 8
nfolds = 4
SEED = 2020
N = 36 #number of tiles per image
#TRAIN = '/gpfs/research/chaohuang/panda/data/concat_image/tiles_36x256x256_stain_norm/'
TRAIN = '../input/panda-tiles-36x256x256-stain-norm-downsampling128/tiles_36x256x256_stain_norm_downsampling128/tiles_36x256x256_stain_norm_downsampling128/'
LABELS = '../input/prostate-cancer-grade-assessment/train.csv'


# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)


# In[ ]:


########################## Data ###########################
### Use stratified KFold split.

df = pd.read_csv(LABELS).set_index('image_id')
files = sorted(set([p[:32] for p in os.listdir(TRAIN)]))
df = df.loc[files]
df = df.reset_index()
splits = StratifiedKFold(n_splits=nfolds, random_state=SEED, shuffle=True)
splits = list(splits.split(df,df.isup_grade))
folds_splits = np.zeros(len(df)).astype(np.int)
for i in range(nfolds): folds_splits[splits[i][1]] = i
df['split'] = folds_splits
print(df.head())
#df.to_csv('/gpfs/research/chaohuang/panda/data/train_folds.csv',index = False)

df_folds = pd.read_csv('../input/panda-tiles-36x256x256-stain-norm-downsampling128/train_folds_4.csv')
print(df_folds.head())


# In[ ]:


mean = torch.tensor([1.0-0.8182546, 1.0-0.65889897, 1.0-0.84993991])  
std = torch.tensor([0.33994367, 0.48929796, 0.3426114])


# In[ ]:


def open_image(fn:PathOrStr, div:bool=True, convert_mode:str='RGB', cls:type=Image,
        after_open:Callable=None)->Image:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) # EXIF warning from TiffPlugin
        x = PIL.Image.open(fn).convert(convert_mode)
    if after_open: x = after_open(x)
    x = pil2tensor(x,np.float32)
    if div: x.div_(255)
    return cls(1.0-x) #invert image for zero padding

class MImage(ItemBase):
    def __init__(self, imgs):
        self.obj, self.data =           (imgs), [(imgs.data - mean[...,None,None])/std[...,None,None]]
        #(imgs), [(imgs[i].data - mean[...,None,None])/std[...,None,None] for i in range(len(imgs))]

    def apply_tfms(self, tfms,*args, **kwargs):
        self.obj = self.obj.apply_tfms(tfms, *args, **kwargs)
        self.data = (self.obj.data - mean[...,None,None])/std[...,None,None]
        return self
        ##for i in range(len(self.obj)):        
            #self.obj[i] = self.obj[i].apply_tfms(tfms, *args, **kwargs)
            #self.data[i] = (self.obj[i].data - mean[...,None,None])/std[...,None,None]
        #return self


    def __repr__(self): return f'{self.__class__.__name__} {self.obj.shape}'
    #def __repr__(self): return f'{self.__class__.__name__} {img.shape for img in self.obj}'
    def to_one(self):
        img = self.data
        #img = torch.stack(self.data,1)
        #img = img.view(3,-1,N,sz,sz).permute(0,1,3,2,4).contiguous().view(3,-1,sz*N)
        img = img.view(3,-1,1,3*sz,4*sz).permute(0,1,3,2,4).contiguous().view(3,-1,4*sz)
        return Image(1.0 - (mean[...,None,None]+img*std[...,None,None]))

class MImageItemList(ImageList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __len__(self)->int: return len(self.items) or 1 
    
    def get(self, i):
        fn = Path(self.items[i])
        fnames = Path(str(fn)+'.png')
        imgs = open_image(fnames, convert_mode=self.convert_mode, after_open=self.after_open)
        return MImage(imgs)

    #def reconstruct(self, t):
    #    return MImage([mean[...,None,None]+_t*std[...,None,None] for _t in t])
    
    def show_xys(self, xs, ys, figsize:Tuple[int,int]=(300,50), **kwargs):
        rows = min(len(xs),8)
        fig, axs = plt.subplots(rows,1,figsize=figsize)
        xs.show(ax=axs, y=ys, **kwargs)
        plt.tight_layout()        
        #for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
        #    xs[i].to_one().show(ax=ax, y=ys[i], **kwargs)
        #plt.tight_layout()

#collate function to combine multiple images into one tensor
def MImage_collate(batch:ItemsList)->Tensor:
    result = torch.utils.data.dataloader.default_collate(to_data(batch))
    if isinstance(result[0],list):
        result = [torch.stack(result[0],1),result[1]]
    return result


# In[ ]:


fn = TRAIN

fnames = [Path(str(fn) + files[i]+'.png') for i in range(12)]
imgs = [open_image(fname) for fname in fnames]
imgs =[(imgs[i].data - mean[...,None,None])/std[...,None,None] for i in range(len(imgs))]
print(imgs[0].shape)
IMG = imgs[0].view(3,-1,1,6*sz,6*sz).permute(0,1,3,2,4).contiguous().view(3,-1,6*sz)
print(IMG.shape)

def get_data(fold=0):
    return (MImageItemList.from_df(df_folds, path='.', folder=TRAIN, cols='image_id')
      .split_by_idx(df_folds.index[df_folds.split == fold].tolist())
      .label_from_df(cols=['isup_grade'])
      .transform(get_transforms(flip_vert=True,max_rotate=15),size=6*sz,padding_mode='zeros')
      .databunch(bs=bs,num_workers=4))


# In[ ]:


### Model ###
# The code below implements Concat Tile pooling idea. 
# As a backbone I use Semi-Weakly Supervised ImageNet pretrained ResNeXt50 model, 
# which worked for me quite well in a number of previous competitions.

def _resnext(url, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    #state_dict = load_state_dict_from_url(url, progress=progress)
    #model.load_state_dict(state_dict)
    return model

class Model(nn.Module):
    def __init__(self, arch='resnext50_32x4d_ssl', n=6, pre=True):
        super().__init__()
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        #m = _resnext(semi_supervised_model_urls[arch], Bottleneck, [3, 4, 6, 3], False, 
        #        progress=False,groups=32,width_per_group=4)
        self.enc = nn.Sequential(*list(m.children())[:-2])       
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(AdaptiveConcatPool2d(),Flatten(),nn.Linear(2*nc,512),
                            Mish(),nn.BatchNorm1d(512), nn.Dropout(0.5),nn.Linear(512,n))
        
    def forward(self, x):
        N = 1
        sz1 = 6*sz
        sz2 = 6*sz
        x = x.view(-1,N,3,sz1,sz2)
        
        shape = x.shape
        n = shape[1]
        x = x.view(-1,shape[2],shape[3],shape[4])
        x = self.enc(x)
        shape = x.shape
        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous()          .view(-1,shape[1],shape[2]*n,shape[3])
        x = self.head(x)
        x = F.softmax(x,dim=1)
        return x


# In[ ]:


fold = 0


# In[ ]:


### Training ###
fname = 'RNXT50_36x256x256_stain_norm_downsampling128'
pred,prob,target = [],[],[]
#for fold in range(nfolds):
#for fold in range(1):
data = get_data(fold)
model = Model()
learn = Learner(data, model, loss_func=nn.CrossEntropyLoss(), opt_func=Over9000, 
            metrics=[KappaScore(weights='quadratic')]).to_fp16()
#logger = tf.keras.callbacks.CSVLogger(learn, f'log_{fname}_{fold}')
logger = CSVLogger(learn, f'log_{fname}_{fold}')
learn.clip_grad = 1.0
learn.split([model.head])
learn.unfreeze()

cyc_len = 16
learn.fit_one_cycle(cyc_len, max_lr=1e-4, div_factor=100, pct_start=0.0, 
  callbacks = [SaveModelCallback(learn,name=f'model',monitor='kappa_score')])
torch.save(learn.model.state_dict(), f'{fname}_{fold}.pth')

learn.model.eval()
with torch.no_grad():
    for step, (x, y) in progress_bar(enumerate(data.dl(DatasetType.Valid)),
                                 total=len(data.dl(DatasetType.Valid))):
        p = learn.model(x)
        prob.append(p.float().cpu())
        target.append(y.cpu())


# In[ ]:


### Display probabilities, prediction labels and target labels of train data ###           
probs = torch.cat(prob,0).numpy()      
preds = torch.argmax(torch.cat(prob,0),1).numpy()
t = torch.cat(target)
#print(probs,preds,t)
print(f'probs = ',probs,'\npreds = ',preds,'\ntargets = ',t)
print('quadratic kappa score for valid dataset = ', cohen_kappa_score(t,preds,weights='quadratic'))
print('confusion matrix = ', confusion_matrix(t,preds))

