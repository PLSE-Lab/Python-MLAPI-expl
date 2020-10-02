#!/usr/bin/env python
# coding: utf-8

# # TReNDS - Exploring tabular augmentation

# Augmentation is broadly used for sample generation and regularization in image classification. This is an experiment to use the techniques on tabular data. I'm starting here with mixup and may add more in the future, if I feel motivated.
# 
# ## Credits
# Many ideas of my notebook are derived from this [notebook](https://www.kaggle.com/iafoss/grapheme-fast-ai-starter-lb-0-964#MixUp) from the Bengaliai competition earlier this year. Please go there and upvote if you find this or other references usefull.
# Here are the references in detail:
# - https://www.kaggle.com/iafoss/grapheme-fast-ai-starter-lb-0-964#MixUp: Multi head - metrics, - loss function, - mixup.
# - https://github.com/fastai/fastai/tree/master/fastai: mixup and many more
# - https://forums.fast.ai/t/tabulardata-mixup/52011/6: tabular mixup

# In[ ]:


import numpy as np
import pandas as pd

from pathlib import Path

from fastai.tabular import * 
from fastai import *

import os, shutil
import sys


# In[ ]:


# Notebook Settings

# np.set_printoptions(threshold=sys.maxsize)


# In[ ]:


kaggle_path = Path('.')
kaggle_input_path = Path('/kaggle/input/trends-assessment-prediction')

#for dirname, _, filenames in os.walk(kaggle_input_path):
#    print(dirname, filenames)


# # Parameters

# In[ ]:


INCLUDE_FNC_DATA = False
IMPUTATION_STRAT = 'IGNORE_ON_TRAIN' # 'IGNORE_ON_TRAIN', 'MEAN' 
LOSS_BASE = 'MSE' # 'L1'
LOSS_WEIGHTS = [0.4, 0.15, 0.15, 0.15, 0.15]
BS = 128


# # Prepare data

# In[ ]:


l_data = pd.read_csv(kaggle_input_path/'loading.csv')

if INCLUDE_FNC_DATA:
    f_data = pd.read_csv(kaggle_input_path/'fnc.csv')
    l_data = l_data.merge(f_data, on='Id', how = 'inner')

y_data = pd.read_csv(kaggle_input_path/'train_scores.csv')

idx_site2 = pd.read_csv(kaggle_input_path/'reveal_ID_site2.csv')
#submission = pd.read_csv(kaggle_input_path/'sample_submission.csv')


# In[ ]:


display(y_data.head())
display(y_data.describe())
y_data.shape


# In[ ]:


display(l_data.tail())
display(l_data.describe()),
l_data.shape


# ## Impute missing data

# In[ ]:


if IMPUTATION_STRAT == 'IGNORE_ON_TRAIN':
    ## will later ignore the value when executing the loss function
    y_data = y_data.fillna(0)
else: #'MEAN'
    y_data = y_data.fillna(mean())
    
y_data


# ## Combine Xs and Ys

# In[ ]:


train = l_data.merge(y_data, on='Id', how='inner').sort_values(by='Id').reset_index(drop = True)
idx_train = train.pop('Id') 
train


# In[ ]:


test = l_data.merge(y_data, on='Id', how='outer', indicator = True)
test = test[test['_merge'] == 'left_only'].drop(['age',
                                                 'domain1_var1', 
                                                 'domain1_var2',
                                                 'domain2_var1',
                                                 'domain2_var2',
                                                 '_merge'], axis = 1).sort_values(by='Id').reset_index(drop = True)
idx_test = test.pop('Id') 
test


# # Model

# ## Metrics

# In[ ]:


# variation of https://github.com/fastai/fastai/blob/master/fastai/metrics.py#L85

def norm_absolute_error(preds, targs):
    "Normalized absolute error between `pred` and `targ`."
    sg=targs.sign()
    y=targs*sg
        
    pred, targ = flatten_check(preds*sg, y)
    return torch.abs(targ - pred).sum() / targ.sum()

def weighted_nae(preds, targs):
    return 0.3 * norm_absolute_error(preds[:,0],targs[:,0]) +            0.175 * norm_absolute_error(preds[:,1],targs[:,1]) +            0.175 * norm_absolute_error(preds[:,2],targs[:,2]) +            0.175 * norm_absolute_error(preds[:,3],targs[:,3]) +            0.175 * norm_absolute_error(preds[:,4],targs[:,4])


# The customized metric callback is a variation of https://www.kaggle.com/iafoss/grapheme-fast-ai-starter-lb-0-964. 
# It is used to keep track of the five single targets and the combined metric. 

# In[ ]:


# variation of https://www.kaggle.com/iafoss/grapheme-fast-ai-starter-lb-0-964

class Metric_idx(Callback):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx
        
    def on_epoch_begin(self, **kwargs):
        self.targs, self.preds = Tensor([]), Tensor([])
    
    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
        last_output = last_output[self.idx]
        last_target = last_target[self.idx]
        
        self.preds = torch.cat((self.preds, last_output.float().cpu()))
        self.targs = torch.cat((self.targs, last_target.float().cpu()))
        
    def _norm_absolute_error(self):
        return norm_absolute_error(self.preds, self.targs)
    
    def on_epoch_end(self, last_metrics, **kwargs): 
        return add_metrics(last_metrics, self._norm_absolute_error())

    
Metric_age = partial(Metric_idx,0)
Metric_domain1_var1 = partial(Metric_idx,1)
Metric_domain1_var2 = partial(Metric_idx,2)
Metric_domain2_var1 = partial(Metric_idx,3)
Metric_domain2_var2 = partial(Metric_idx,4)

class Metric_total(Callback):
    def __init__(self):
        super().__init__()
        self.age = Metric_idx(0)
        self.domain1_var1 = Metric_idx(1)
        self.domain1_var2 = Metric_idx(2)
        self.domain2_var1 = Metric_idx(3)
        self.domain2_var2 = Metric_idx(4)
        
    def on_epoch_begin(self, **kwargs):
        self.age.on_epoch_begin(**kwargs)
        self.domain1_var1.on_epoch_begin(**kwargs)
        self.domain1_var2.on_epoch_begin(**kwargs)
        self.domain2_var1.on_epoch_begin(**kwargs)
        self.domain2_var2.on_epoch_begin(**kwargs)
        
    
    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
        self.age.on_batch_end(last_output, last_target, **kwargs)
        self.domain1_var1.on_batch_end(last_output, last_target, **kwargs)
        self.domain1_var2.on_batch_end(last_output, last_target, **kwargs)
        self.domain2_var1.on_batch_end(last_output, last_target, **kwargs)
        self.domain2_var2.on_batch_end(last_output, last_target, **kwargs)
 
        
    def on_epoch_end(self, last_metrics, **kwargs): 
        return add_metrics(last_metrics, 
                           0.3 * self.age._norm_absolute_error() +
                           0.175*self.domain1_var1._norm_absolute_error()  +
                           0.175*self.domain1_var2._norm_absolute_error()  +
                           0.175*self.domain2_var1._norm_absolute_error()  +
                           0.175*self.domain2_var2._norm_absolute_error()
                          )


# ## Loss function
# The customized metric callback is a variation of https://www.kaggle.com/iafoss/grapheme-fast-ai-starter-lb-0-964. It weights and combines the five single losses.

# In[ ]:


# variation of https://www.kaggle.com/iafoss/grapheme-fast-ai-starter-lb-0-964

class Loss_combine(nn.Module):
    def __init__(self, loss_weights = [0.4,0.15,0.15,0.15,0.15], loss_base = 'MSE'):
        super().__init__()
        
        self.loss_base = loss_base
        
        self.loss_weights = loss_weights
        self.fw = Tensor(LOSS_WEIGHTS).cuda()
          
            
    def forward(self, input, target,reduction='mean'): #mean
        
        x0,x1,x2,x3,x4 = input.T
        
        if IMPUTATION_STRAT == 'IGNORE_ON_TRAIN':
            sg = target.float().sign()
            x0,x1,x2,x3,x4 = x0.float()*sg[:,0],x1.float()*sg[:,1],x2.float()*sg[:,2],x3.float()*sg[:,3],x4.float()*sg[:,4]
        else: # 'MEAN'
            sg = 1
            x0,x1,x2,x3,x4 = x0.float(),x1.float(),x2.float(),x3.float(),x4.float()
            
        y = target.float()*sg
        
        if self.loss_base == 'MSE':
            loss_func = F.mse_loss 
            reduction = 'sum'
            return self.fw[0]*loss_func(x0,y[:,0],reduction=reduction)/sum(y[:,0]**2) +                self.fw[1]*loss_func(x1,y[:,1],reduction=reduction)/sum(y[:,1]**2) +                self.fw[2]*loss_func(x2,y[:,2],reduction=reduction)/sum(y[:,2]**2) +                self.fw[3]*loss_func(x3,y[:,3],reduction=reduction)/sum(y[:,3]**2) +                self.fw[4]*loss_func(x4,y[:,4],reduction=reduction)/sum(y[:,4]**2)
        else: # 'L1'
            loss_func = F.l1_loss 
            reduction = 'mean'
            return self.fw[0]*loss_func(x0,y[:,0],reduction=reduction) +                self.fw[1]*loss_func(x1,y[:,1],reduction=reduction) +                self.fw[2]*loss_func(x2,y[:,2],reduction=reduction) +                self.fw[3]*loss_func(x3,y[:,3],reduction=reduction) +                self.fw[4]*loss_func(x4,y[:,4],reduction=reduction)


# ### Augmentation
# Augmentation is broadly used for sample generation and regularization in image classification. This is a try to use the techniques on tabular data.
# 
# The Mixup implementation is a variation of https://github.com/fastai/fastai/blob/master/fastai/callbacks/mixup.py following this [example](https://www.kaggle.com/iafoss/grapheme-fast-ai-starter-lb-0-964#MixUp) and adaptied for tabular data following this [thread](https://forums.fast.ai/t/tabulardata-mixup/52011/6).

# In[ ]:


# Vartiation https://github.com/fastai/fastai/blob/master/fastai/callbacks/mixup.py#L8
# and https://www.kaggle.com/iafoss/grapheme-fast-ai-starter-lb-0-964#MixUp
# and https://forums.fast.ai/t/tabulardata-mixup/52011/6

class MixUpLoss(Module):
    "Adapt the loss function `crit` to go with mixup."
    
    def __init__(self, crit, reduction='mean'): #mean
        super().__init__()
        if hasattr(crit, 'reduction'): 
            self.crit = crit
            self.old_red = crit.reduction
            setattr(self.crit, 'reduction', 'none')
        else: 
            self.crit = partial(crit, reduction='none')
            self.old_crit = crit
        self.reduction = reduction
        
    def forward(self, output, target):
        if len(target.shape) == 2 and target.shape[1] == 11:
            loss1, loss2 = self.crit(output,target[:,0:5].long()), self.crit(output,target[:,5:10].long())
            d = loss1 * target[:,-1] + loss2 * (1-target[:,-1])
        else:  d = self.crit(output, target)
        
        if self.reduction == 'mean':    return d.mean()
        elif self.reduction == 'sum':   return d.sum()
        return d
    
    def get_old(self):
        if hasattr(self, 'old_crit'):  return self.old_crit
        elif hasattr(self, 'old_red'): 
            setattr(self.crit, 'reduction', self.old_red)
            return self.crit

class MixUpCallback(LearnerCallback):
    "Callback that creates the mixed-up input and target."
    def __init__(self, learn:Learner, alpha:float=0.4, stack_x:bool=False, stack_y:bool=True):
        super().__init__(learn)
        self.alpha,self.stack_x,self.stack_y = alpha,stack_x,stack_y
    
    def on_train_begin(self, **kwargs):
        if self.stack_y: self.learn.loss_func = MixUpLoss(self.learn.loss_func)
        
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        "Applies mixup to `last_input` and `last_target` if `train`."
        if not train: return
        
        # last_input[0] ==> embedded categorical data
        # last_input[1] ==> continous data
        rnd = 1
        #if .5<np.random.uniform():
        #    rnd=np.random.uniform()/20
        
        l_org = last_input[1] * rnd
        last_input = last_input[1] *rnd #0
        
        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))
        lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
        
        lambd = last_input.new(lambd)
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        x1, y1 = last_input[shuffle], last_target[shuffle]
        if self.stack_x:
            new_input = [last_input, last_input[shuffle], lambd]
        else: 
            out_shape = [lambd.size(0)] + [1 for _ in range(len(x1.shape) - 1)]
            new_input = (last_input * lambd.view(out_shape) + x1 * (1-lambd).view(out_shape))
        if self.stack_y:
            new_target = torch.cat([last_target.float(), y1.float(), lambd[:,None].float()], 1)
        else:
            if len(last_target.shape) == 2:
                lambd = lambd.unsqueeze(1).float()
            new_target = last_target.float() * lambd + y1.float() * (1-lambd)
        
        return {'last_input': [l_org, new_input], 'last_target': new_target}  
    
    def on_train_end(self, **kwargs):
        if self.stack_y: self.learn.loss_func = self.learn.loss_func.get_old()
            


# ## Build the model

# In[ ]:


bs = 128
valid_idx = range(200, 400)
dep_var = ['age','domain1_var1','domain1_var2', 'domain2_var1', 'domain2_var2']
    
def prep_data(bs, valid_idx):
    procs = [FillMissing, Categorify, Normalize]
    cont_names = list(set(train.columns) - set(['age','domain1_var1','domain1_var2', 'domain2_var1', 'domain2_var2'])-set(dep_var))
    cat_names = []

    tlist = (TabularList.from_df(train, 
                                path=kaggle_path, 
                                cat_names=cat_names, 
                                cont_names=cont_names, 
                                procs=procs))

    if valid_idx == None:
        tlist = tlist.split_none()
    else:
        tlist = tlist.split_by_idx(valid_idx)

    data = (tlist.label_from_df(cols=dep_var)
                 .add_test(TabularList.from_df(test, 
                                               cat_names=cat_names,
                                               cont_names=cont_names, 
                                               procs = procs))
                 .databunch(path = kaggle_path, bs = bs))
    
    return data


def prep_learn(data):
    
    learn = tabular_learner(data, 
                        layers = [256,128,256,128,64], #[1024,1024,128,1024,128,1024,1024],#
                        ps = 0.3,
                        loss_func = Loss_combine(loss_weights = LOSS_WEIGHTS,  loss_base= LOSS_BASE),
                        metrics=[Metric_age(),
                                 Metric_domain1_var1(),
                                 Metric_domain1_var2(),
                                 Metric_domain2_var1(),
                                 Metric_domain2_var2(),
                                 Metric_total()],
                       y_range=(Tensor([12,12,0,0,0]).cuda(),Tensor([90,90,100,100,100]).cuda())
                       )#.to_fp16()

    learn.clip_grad = 1.0
    
    return learn
    


# ## Training experiment without augmentation

# In[ ]:


data = prep_data(bs, valid_idx)
learn = prep_learn(data)


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


lr = 2e-2
reduceLR = callbacks.ReduceLROnPlateauCallback(learn=learn, monitor = 'valid_loss', mode = 'auto', patience = 2, factor = 0.2, min_delta = 0)
learn.fit_one_cycle(10, lr, callbacks=[reduceLR])


# In[ ]:


yv, yv_truth= learn.get_preds(ds_type=DatasetType.Valid)
yt, yt_truth= learn.get_preds(ds_type=DatasetType.Train)

print(f'Without augmentation:')
print(f'Weighted normalized absolute error (Valid): {weighted_nae(yv,yv_truth)}')
print(f'Weighted normalized absolute error (Train): {weighted_nae(yt,yt_truth)}')


# ## Training experiment with augmentation

# In[ ]:


data = prep_data(bs, valid_idx)
learn = prep_learn(data)

lr = 2e-2
reduceLR = callbacks.ReduceLROnPlateauCallback(learn=learn, monitor = 'valid_loss', mode = 'auto', patience = 2, factor = 0.2, min_delta = 0)
learn.fit_one_cycle(10, lr, callbacks=[MixUpCallback(learn, alpha=0.4),reduceLR])


# In[ ]:


yv, yv_truth= learn.get_preds(ds_type=DatasetType.Valid)
yt, yt_truth= learn.get_preds(ds_type=DatasetType.Train)

print(f'With augmentation:')
print(f'Weighted normalized absolute error (Valid): {weighted_nae(yv,yv_truth)}')
print(f'Weighted normalized absolute error (Train): {weighted_nae(yt,yt_truth)}')


# ## Training conclution
# 
# The choosen augmentation technique in this configuration doesn't improve the validation score. 
# 
# Let's see if it generalizes better on the LB.

# # Prediction

# In[ ]:


def make_submission(learn, postfix = ''):
    preds = learn.get_preds(ds_type=DatasetType.Test)[0]
    
    rec = pd.DataFrame(idx_test)
    rec['Id'] = rec['Id'].astype(str)+'_'
    rec['Predicted'] = preds[:,1]
    
    submission=None

    for t, tcol in enumerate(dep_var):
        rec = pd.DataFrame(idx_test)
        rec['Id'] = rec['Id'].astype(str)+'_'+tcol
        rec['Predicted'] = preds[:,t]
        if isinstance(submission, pd.DataFrame):
            submission = submission.append(rec)
        else:
            submission = rec

    submission = submission.sort_values('Id').reset_index(drop=True)
    
    display(submission.head(10))
    submission.to_csv('submission'+postfix+'.csv', index=False)


# ## Predict without augmentation

# In[ ]:


data = prep_data(bs, valid_idx = None)
learn = prep_learn(data)

lr = 2e-2
learn.fit_one_cycle(10, lr)


# In[ ]:


make_submission(learn, postfix = '_wo_aug')


# ## Predict with augmentation

# In[ ]:


data = prep_data(bs, valid_idx = None)
learn = prep_learn(data)

lr = 2e-2
learn.fit_one_cycle(10, lr, callbacks=[MixUpCallback(learn, alpha=0.4)])


# In[ ]:


make_submission(learn, postfix = '_with_aug')


# # Conclusion
# LB without augmentation: 0.164
# 
# LB with augmentation: 0.165
# 
# ==> No advantage from augmentation is this particular setting. Got to try something else ... ;)

# In[ ]:




