#!/usr/bin/env python
# coding: utf-8

# # Fastai conform dataset with resnet18

# I created this kernel to get to know the fastai dataformats a bit better.
# I currently get 0.87 score, which is not too high but Ithink it should be easy to improve my approach.
# 
# I had to fix a tiny bug in the lr_find2 so I'll import my hotfix of fastai. The problem was that some augmentations changed the shape of an image if it only has a single channel.
# For now the runtime of 9hrs on kernels is the main bottleneck and im not sure how to fix this,
# I can currently train 12 cycles with 5k images per category. If anyone has Ideas for improvements go ahead.
# 
# My next Ideas would be trying some different augmentations and image sizes.
# Also an Idea would be tuning the learning hyperparameters for the 1 cycle learning policy as described here:
#     
# 1. [blog post by Sylvain Gugger summarizing the following papers](https://sgugger.github.io/the-1cycle-policy.html#the-1cycle-policy)
# 2. [original papers by leslie smith on hyperparameter tuning](https://arxiv.org/pdf/1803.09820.pdf) 
# 3. [ and Superconvergence, the 1 cycle policy learning](https://arxiv.org/pdf/1708.07120.pdf)
# 

# # Imports and definition of necessary functions

# In[ ]:


get_ipython().system('mkdir fastai_lib;cd fastai_lib;git clone https://github.com/rpauli/fastai.git;ln -s fastai_lib/fastai/old/fastai/ ../')


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import ast
import json

import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from fastai.imports import *

from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '')


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# ### Thanks to  [Belugas Kernel](https://www.kaggle.com/gaborfodor/greyscale-mobilenet-lb-0-892/notebook)

# In[ ]:



def f2cat(filename: str) -> str:
    return filename.split('.')[0]

def list_all_categories():
    files = os.listdir(os.path.join(path_base, path_train))
    return sorted([f2cat(f) for f in files], key=str.lower)


# ## Define apk Metric and FocalLoss

# In[ ]:


def calc_apk(pred,actual,k=3):
    """
    pred 2D array, 
    axis 0 is predicted items
    axis 1 is probabilities per label
    """
    pred=to_np(pred)
    actual=to_np(actual)
    top_idxs=top_k_preds(pred,k)
    idx_0=(np.where(actual==1)[1][:,None]-top_idxs)
    idx_correct=idx_0==0
    weighted_scores=1/np.arange(1,k+1)
    score_per_data=weighted_scores[np.argmax(idx_correct,axis=1)]*np.max(idx_correct,axis=1)
    mean_score=np.mean(score_per_data)
    return mean_score

def top_k_preds(pred,k):
    top_idxs=np.argsort(-to_np(pred),axis=1)[:,:k]
    return top_idxs

def acc_metric(preds,actual):
    return accuracy_score(np.where(actual==1)[1],np.argmax(to_np(preds),axis=1))


# In[ ]:


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=1, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


# ## Read in image from raw stroke [(again thanks @Beluga)](https://www.kaggle.com/gaborfodor/greyscale-mobilenet-lb-0-892/notebook) 
# This version color codes the strokes from red to green in order, and the blue channel is the stroke velocity.

# In[ ]:


def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):
    img = np.zeros((BASE_SIZE, BASE_SIZE,3), np.float32)
    t_max=max(1,len(raw_strokes))
    
    velos=[np.sqrt(np.sum(np.square(np.diff(stroke)),axis=0)).astype(int) for stroke in raw_strokes]
    velo_min=np.min([np.min(velo) for velo in velos])
    velo_max=max(1,np.max([np.max(velo) for velo in velos]))
    for t, stroke in enumerate(raw_strokes):
        velo=velos[t]
        velo=(velo)*0.9/(velo_max)+0.1
        velo*=255
        for i in range(len(stroke[0]) - 1):
            try:
                color = (255 - int(t*255./t_max), int(t*255./t_max), int(velo[i])) if time_color else 255    
            except:
                print(t,t_max,t*255./t_max,int(t*255./t_max),int(velo[i]))
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    img/=255
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img


# ## Define Dataset and Dataloader using the fastai library data format

# In[ ]:


class DoodleDataset(Dataset):
    """Adapted the fastai Dataset class for my needs, mainly the get_x function"""
    def __init__(self, x, y, transform=None,sz=64):
        self.x = x
        self.y = y
        self.transform = transform
        self.n = self.get_n()
        self.c = self.get_c()
        self.sz = sz
    @abstractmethod
    def get_x(self, i):
        return draw_cv2(json.loads(self.x[i]),self.sz)
    @abstractmethod
    def get_y(self, i):
        y= np.zeros(self.c,dtype=np.float)
        y[int(self.y[i])]=1
        return y
    @property
    def is_multi(self): return True
    @property
    def is_reg(self):return False
    #this flag is set to remove the output sigmoid that allows log(sigmoid) optimization
    #of the numerical stability of the loss function
    
    def get1item(self, idx):
        x,y = self.get_x(idx),self.get_y(idx)
        return self.get(self.transform, x, y)


    def __getitem__(self, idx):
        if isinstance(idx,slice):
            xs,ys = zip(*[self.get1item(i) for i in range(*idx.indices(self.n))])
            return np.stack(xs),ys
        return self.get1item(idx)

    def __len__(self): return self.n

    def get(self, tfm, x, y):     
        return (x,y) if tfm is None else tfm(x,y)    
    
    @abstractmethod
    def get_n(self):
        return len(self.y)

    @abstractmethod
    def get_c(self):
        return len(np.unique(self.y))

    @abstractmethod
    def get_sz(self):
        return self.sz


    @property
    def is_multi(self):
        """Returns true if this data set contains multiple labels per sample."""
        return True



class DoodleImageData(ImageData):

    def get_dl(self, ds, shuffle):
        if ds is None: return None
        return DataLoader(ds, batch_size=self.bs, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=False)


# In[ ]:


def show_img(ims, idx, figsize=(5,5), normed=True, ax=None):
    if ax is None: fig,ax = plt.subplots(figsize=figsize)
    if normed: ims = np.rollaxis(to_np(ims),1,4);ims*=0.07367;ims+=0.09587;
    else:      ims = np.rollaxis(to_np(ims),1,4)
    ax.imshow(np.clip(ims,0,1)[idx,:,:,:])
    ax.axis('off')


# ## Defining the learn object based on the CNN objects from the fastai library

# In[ ]:


class DoodleConvnetBuilder(ConvnetBuilder):
    def __init__(self, f, c, is_multi, is_reg, ps=None, xtra_fc=None, xtra_cut=0, custom_head=None, pretrained=True):
        self.f,self.c,self.is_multi,self.is_reg,self.xtra_cut = f,c,is_multi,is_reg,xtra_cut
        if xtra_fc is None: xtra_fc = [512]
        if ps is None: ps = [0.25]*len(xtra_fc) + [0.5]
        self.ps,self.xtra_fc = ps,xtra_fc

        if f in model_meta: cut,self.lr_cut = model_meta[f]
        else: cut,self.lr_cut = 0,0
        cut-=xtra_cut
        layers = cut_model(f(pretrained), cut)
        
        layers[0] = nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3, 3), bias=False)
        
        self.nf = model_features[f] if f in model_features else (num_features(layers)*2)
        if not custom_head: layers += [AdaptiveConcatPool2d(), Flatten()]
        self.top_model = nn.Sequential(*layers)

        n_fc = len(self.xtra_fc)+1
        if not isinstance(self.ps, list): self.ps = [self.ps]*n_fc

        if custom_head: fc_layers = [custom_head]
        else: fc_layers = self.get_fc_layers()
        self.n_fc = len(fc_layers)
        self.fc_model = to_gpu(nn.Sequential(*fc_layers))
        if not custom_head: apply_init(self.fc_model, kaiming_normal)
        self.model = to_gpu(nn.Sequential(*(layers+fc_layers)))
        

        
class DoodleConvLearner(ConvLearner):
    @classmethod
    def pretrained(cls, f, data, ps=None, xtra_fc=None, xtra_cut=0, custom_head=None, precompute=False,
                   pretrained=True, **kwargs):
        models = DoodleConvnetBuilder(f, data.c, data.is_multi, data.is_reg,
            ps=ps, xtra_fc=xtra_fc, xtra_cut=xtra_cut, custom_head=custom_head, pretrained=pretrained)
        return cls(data, models, precompute, **kwargs)

    @classmethod
    def lsuv_learner(cls, f, data, ps=None, xtra_fc=None, xtra_cut=0, custom_head=None, precompute=False,
                  needed_std=1.0, std_tol=0.1, max_attempts=10, do_orthonorm=False, **kwargs):
        models = DoodleConvnetBuilder(f, data.c, data.is_multi, data.is_reg,
            ps=ps, xtra_fc=xtra_fc, xtra_cut=xtra_cut, custom_head=custom_head, pretrained=False)
        convlearn=cls(data, models, precompute, **kwargs)
        convlearn.lsuv_init()
        return convlearn


# In[ ]:


def get_stats(X_trn,X_val,y_trn,y_val,N=100):
    bs=340
    num_workers=4 # Apprently 2 cpus per kaggle node
    sz=64
    aug_tfms = None
    #dummy transformations, i subtract mean 0 and devide by 1 so it shouldnt change the stats
    tfms = tfms_from_stats((np.array([0.,0.,0.]), np.array([1.,1.,1.])), sz=sz, aug_tfms=aug_tfms,tfm_y=TfmType.NO)
    datasets=DoodleImageData.get_ds(DoodleDataset,(X_trn,y_trn),(X_val, y_val),tfms=tfms,sz=sz)
    doodleimageloader=DoodleImageData('.',datasets,bs,num_workers,enc.classes_)
    mn,var=[],[]
    for i,(x,y) in tqdm(enumerate(iter(doodleimageloader.trn_dl))):
        batch=to_np(x).swapaxes(1,3).reshape([-1,3])
        mn.append(np.mean(batch,axis=0))
        var.append(np.var(batch,axis=0))
        if i>N:
            break
    channel_mean,channel_var=np.mean(mn,axis=0),np.mean(var,axis=0)
    return channel_mean,channel_var


# In[ ]:


def get_set(N_cat,N_rows,N_skip):
    if N_cats < len(list_all_categories()):
        df = pd.concat([pd.read_csv(f'{path_base}/{path_train}/{cat}.csv',usecols=['drawing','word','key_id','recognized'],nrows=N_rows,skiprows=np.arange(1,N_skip+1)) for cat in  tqdm(np.random.choice(list_all_categories(),N_cats,replace=False))])
    elif  N_cats == len(list_all_categories()):
        df = pd.concat([pd.read_csv(f'{path_base}/{path_train}/{cat}.csv',usecols=['drawing','word','key_id','recognized'],nrows=N_rows,skiprows=np.arange(1,N_skip+1)) for cat in  tqdm(list_all_categories())])
    else:
        print('Somethings wrong')
    df_cleaned=df[df.recognized==True].drop(['recognized'],axis=1)
    #df_cleaned=df.drop(['recognized'],axis=1)
    df_cleaned.word=df_cleaned.word.str.replace(' ','_')
    return df


# # Preprocessing

# I can currently only run about 8k samples with the 9h time limit on kaggle, a good size for the validation set seems to be 100 samples per category from what a read of other kernels and the discussion. I will also disregard any unregognized and later see if it makes a difference. In some cases there were class impurities where the doodle was missclassified, I hope those would drop out for regognized=True

# In[ ]:


path_base='../input'
path_train='train_simplified'
# path_base='.'
# path_train='train'


# In[ ]:


np.random.seed(0)
#subsample for now
N_cats=len(list_all_categories())
N_rows_train=10000 #I can currently do about 8k samples which is not nearly enough
N_rows_val=100 # This is a common value among kagglers 100 items per category to validate
N_cats


# ### Read in subset of row and categories

# In[ ]:


df_val=get_set(N_cats,N_rows_val,0)
df_trn=get_set(N_cats,N_rows_train,N_rows_val)
df_val=df_val.set_index('key_id')
df_trn=df_trn.set_index('key_id')
df_val.index.intersection(df_trn.index)


# ### load in test data

# In[ ]:


test_file='test_simplified.csv'
df_test=pd.read_csv(f'{path_base}/{test_file}')
df_test.info()


# 
# ## Ordinal encoding of labels and splitting into train and validation set
# Also shuffle samples, apparently sklearn.utils.shuffle is the fastest method [ apparently sklearn.utils.shuffle is the fastest method](https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows)

# In[ ]:


enc=LabelEncoder()
enc.fit(df_trn.word)
df_trn=shuffle(df_trn)
df_val=shuffle(df_val)
y_trn=enc.transform(df_trn.word)
y_val=enc.transform(df_val.word)
X_trn=df_trn.drawing.values
X_val=df_val.drawing.values


# In[ ]:


bs=256
num_workers=4 # Apprently 2 cpus per kaggle node
BASE_SIZE=256
sz=128
arch=resnet18


# ## Fill datasets and corresponding dataloader and model architecture (Resnet18)
# The Image statistics of the doodle will be wildly different from imagenet, so I load in N batches of doodles and calculate channel mean and variance for proper normalization

# In[ ]:


channel_mean,channel_var=get_stats(X_trn,X_val,y_trn,y_val,50)


# In[ ]:


channel_mean,channel_var


# In[ ]:


aug_tfms = transforms_side_on+[RandomRotate(10)]
tfms = tfms_from_stats((channel_mean, channel_var), sz=sz, aug_tfms=aug_tfms)


# # Load Datasets in fastai format

# In[ ]:


datasets=DoodleImageData.get_ds(DoodleDataset,(X_trn,y_trn),(X_val, y_val),tfms=tfms,sz=sz,test=df_test.drawing.values)
datasets_warmup=DoodleImageData.get_ds(DoodleDataset,(X_trn[:200*bs],y_trn[:200*bs]),(X_val, y_val),tfms=tfms,sz=sz,test=df_test.drawing.values)


# In[ ]:


doodleplot=DoodleImageData('.',datasets_warmup,8,num_workers,enc.classes_)
doodle_warmup=DoodleImageData('.',datasets_warmup,bs,num_workers,enc.classes_)
doodleimageloader=DoodleImageData('.',datasets,bs,num_workers,enc.classes_)


# #### Some images,time information is encoded in the brightness
# Some of these drawings are absolutely hilarious

# In[ ]:


idx=0
batches = [next(iter(doodleplot.trn_dl)) for i in range(8)]
fig, axes = plt.subplots(1,8, figsize=(18,9))
for i,(x,y) in enumerate(batches):
    show_img(x,idx, ax=axes.flat[i])
    axes.flat[i].set_title(enc.inverse_transform(np.where(y==1)[1][idx]))


# #### An image and some augmentations,

# In[ ]:


idx=2
batches = [next(iter(doodleplot.aug_dl)) for i in range(8)]
fig, axes = plt.subplots(1,8, figsize=(18,9))
for i,(x,y) in enumerate(batches):
    show_img(x,idx, ax=axes.flat[i])
    axes.flat[i].set_title(enc.inverse_transform(np.where(y==1)[1][idx]))


# # Start learning stuff!

# This is the learner object, it has all the fancy learning rate finding and cyclicle learning rate with momentum methods.

# In[ ]:


learn=ConvLearner.pretrained(arch,doodle_warmup,metrics=[acc_metric,calc_apk])
learn.crit=FocalLoss()


# ## Logarithmic sweep to check for a good learning rate

# An Appropriate learning rate should be on a slope but before the minimum.
# [(Great Explanation)](https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10)
# An important thing to know is, in the original paper, the minimum is mentioned as the best value.
# To smooth the curve a running average is used in the fastai library, so we have to take an earlier value. Something like minimum/10 works well for me.

# In[ ]:


learn.lr_find(end_lr=1000,start_lr=0.0001)


# In[ ]:


learn.sched.plot(n_skip_end=1)


# ## For now the layers are frozen except the last one to not unlearn all the tuned weights
# 

# In[ ]:


lr=10
learn.fit(lr,1,cycle_len=1,use_clr_beta = (10,10,0.95,0.85))


# In[ ]:


learn.sched.plot_loss()
learn.sched.plot_lr()


# ## Unfreeze network and check again if lr is appropriate

# In[ ]:


learn.save('freeze')


# In[ ]:


learn.load('freeze')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find(end_lr=10000,start_lr=0.0001)
learn.sched.plot(n_skip_end=1)


# ## Learning rate for earlier layers is supposed to be lower

# The learning rate from the layers is the optimal learning rate from the find_lr method. We also decrease the learning rate for earlier layers so all the learned features from the resnet18 are not unlearned.

# In[ ]:


lr=100
lrs=[lr/9,lr/3,lr]
learn.set_data(doodleimageloader)
learn.load('freeze')
learn.unfreeze()
learn.fit(lrs,1,cycle_len=1,use_clr_beta = (10,10,0.95,0.85))


# In[ ]:


learn.sched.plot_loss()

learn.sched.plot_lr()


# In[ ]:


learn.save('unfreeze')


# In[ ]:


learn.load('unfreeze')


# ### Predit the validation data using TTA
# Here for every image we want to predict on, n_augs images are augmented form the original image.
# We can then compare the predictions on for example the image and the image flipped / roated / slightly different crop/ lighting/stretched etc. 
# For now only the diherdral and rotations are used. THis gives a nice extra percent or two when compared to the auc above after training where not TTA is used. 
# I also test if mean or max is better to use on the image and its augments but it can't conclude anything yet.
# 

# In[ ]:


log_preds_test,y_test=learn.TTA(is_test=True,n_aug=16)


# In[ ]:


mean_test_preds=np.mean(np.exp(log_preds_test),0)
max_test_preds=np.max(np.exp(log_preds_test),0)


# In[ ]:


def prepare_submission(preds,fname):
    preds_transformed=enc.inverse_transform(top_k_preds(preds,3))
    preds_joined=[' '.join(words) for words in preds_transformed]
    df_test['word']=preds_joined
    df_test.loc[:,['key_id','word']].to_csv(fname,index=None)


# ## I add the score to the name of the file so I can later plot the leaderboard score versus my validation score
# In the fastai course Jeremy mentions that if you have a monotonic relation between validation and LB score the way you set up your validation set matches what the test set consists of.

# In[ ]:


prepare_submission(mean_test_preds,f'submission_mean.csv')
prepare_submission(max_test_preds,f'submission_max.csv')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('rm fastai')


# In[ ]:


get_ipython().system('rm -r fastai_lib/')

