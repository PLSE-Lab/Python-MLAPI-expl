#!/usr/bin/env python
# coding: utf-8

# # TReNDS - Tabular NN 0.159

# LB: 0.159
# 
# ## Credits
# Many ideas of my notebook are derived from this [notebook](https://www.kaggle.com/iafoss/grapheme-fast-ai-starter-lb-0-964#MixUp) from the Bengaliai competition earlier this year. Please go there and upvote if you find this or other references usefull.
# Here are the references in detail:
# - https://www.kaggle.com/iafoss/grapheme-fast-ai-starter-lb-0-964#MixUp: Multi head - metrics, - loss function
# 
# - Schaefer 2018 -features: https://www.kaggle.com/kpriyanshu256/trends-image-features-1000/output
# - Analysis of optimal Cluster Size: https://www.kaggle.com/mks2192/trends-cluster-sfnc-groups
# - Using Schaefer 2018-features as time series: https://www.kaggle.com/kpriyanshu256/trends-time-series

# In[ ]:


import numpy as np
import pandas as pd

from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, StratifiedKFold

from fastai.tabular import * 
from fastai import *

import os, shutil
import sys


# In[ ]:


#https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/150697#845512
import logging
logging.getLogger().setLevel(logging.NOTSET)
# Notebook Settings

# np.set_printoptions(threshold=sys.maxsize)


# In[ ]:


kaggle_path = Path('.')
kaggle_input_path = Path('/kaggle/input/trends-assessment-prediction') # '/kaggle/input/trends-assessment-prediction'
cluster_input_path = Path('/kaggle/input/trends-cluster-sfnc-groups')
av_input_path = Path('/kaggle/input/trends-av-probs-of-test-and-s2')
img_53000_path = Path('/kaggle/input/fork-of-trends-image-features-53-100')

#https://www.kaggle.com/mks2192/trends-cluster-sfnc-groups/output
#for dirname, _, filenames in os.walk(kaggle_input_path):
#    print(dirname)#, filenames)


# # Parameters

# In[ ]:


INCLUDE_FNC_DATA = True

INCLUDE_FNC_CLUSTERS = False
INCLUDE_FNC_DIST_CCENTER = False

INCLUDE_FNC_CLUSTERS_2 = True
INCLUDE_FNC_DIST_CCENTER_2 = True

INCLUDE_IMG_53000 = False

INCLUDE_AV_DATA = False

IMPUTATION_STRAT = 'IGNORE_ON_TRAIN' # 'IGNORE_ON_TRAIN', 'MEAN' 
#LOSS_BASE = 'MIX' #'MSE' # 'MSE' # 'L1' 'MIX'
LOSS_WEIGHTS = [0.3, 0.175, 0.175, 0.175, 0.175] #[0.2,0.2,0.2,0.2,0.2] #[0,0,0,0,1]#
BS = 128

DEP_VAR = ['age','domain1_var1','domain1_var2', 'domain2_var1', 'domain2_var2']


# # Prepare data

# Including all tabular data but IC_20 (removing or adding IC_20 doesn't influence the public LB).

# In[ ]:


l_data = pd.read_csv(kaggle_input_path/'loading.csv').drop('IC_20',axis=1)

if INCLUDE_FNC_DATA:
    f_data = pd.read_csv(kaggle_input_path/'fnc.csv')
    l_data = l_data.merge(f_data, on='Id', how = 'inner')

if INCLUDE_FNC_CLUSTERS:
    c_data = pd.read_csv(cluster_input_path/'sfnc_group_clusters.csv')
    l_data = l_data.merge(c_data, on='Id', how = 'inner')

if INCLUDE_FNC_DIST_CCENTER:
    cc_data = pd.read_csv(cluster_input_path/'sfnc_dist_to_cluster_center.csv')
    l_data = l_data.merge(cc_data, on='Id', how = 'inner')
    

if INCLUDE_FNC_CLUSTERS_2:
    c2_data = pd.read_csv(cluster_input_path/'sfnc_group_clusters_2c.csv')
    temp_col = []
    for c in c2_data.columns:
        if c != 'Id':
            temp_col += [c+'_2']
        else:
            temp_col += [c]
    c2_data.columns = temp_col
    l_data = l_data.merge(c2_data, on='Id', how = 'inner')

if INCLUDE_FNC_DIST_CCENTER_2:
    cc2_data = pd.read_csv(cluster_input_path/'sfnc_dist_to_cluster_center_2c.csv')
    temp_col = []
    for c in cc2_data.columns:
        if c != 'Id':
            temp_col += [c+'_2']
        else:
            temp_col += [c]
    cc2_data.columns = temp_col
    l_data = l_data.merge(cc2_data, on='Id', how = 'inner')
    

if INCLUDE_IMG_53000:
    i_data = pd.read_csv(img_53000_path/'train_features.csv')
    i_data = i_data.append(pd.read_csv(img_53000_path/'test_features.csv'))
    l_data = l_data.merge(i_data, on='Id', how = 'inner')

    
if INCLUDE_AV_DATA:
    av_data = pd.read_csv(av_input_path/'test_s2_probs.csv')
    l_data = l_data.merge(av_data, on='Id', how = 'inner')



y_data = pd.read_csv(kaggle_input_path/'train_scores.csv')

idx_site2 = pd.read_csv(kaggle_input_path/'reveal_ID_site2.csv')
#submission = pd.read_csv(kaggle_input_path/'sample_submission.csv')


# In[ ]:


#i_data.info()


# In[ ]:


display(y_data.head())
display(y_data.describe())
y_data.shape


# In[ ]:


display(l_data.tail())
display(l_data.describe()),
l_data.shape


# In[ ]:


y_data.hist()


# ## Impute missing data

# Imputation strategie **IGNORE_ON_TRAIN**:
# - Some of the target values are empty. To ignore the empty one when calculating the loss the missing data is filled with 0 (0 is not in the target range). 0 is used later on as flag in the loss function as prediction to ignore.
# 
# Imputation strategie **MEAN**:
# - Impute missing targets with mean. They will be included in the loss function. This strategie is worse than IGNORE_ON_TRAIN.

# In[ ]:


if IMPUTATION_STRAT == 'IGNORE_ON_TRAIN':
    ## will later ignore the value when executing the loss function
    y_data = y_data.fillna(0)
else: #'MEAN'
    y_data = y_data.fillna(y_data.mean())
    
y_data


# In[ ]:


y_data.hist()

# lots of imputed data on domain1_var1/2


# ## Combine Xs and Ys

# In[ ]:


train_df = l_data.merge(y_data, on='Id', how='inner').sort_values(by='Id').reset_index(drop = True)
idx_train = train_df.pop('Id') 
train_df


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


# ## Prepare validation splits
# 
# ### Bin y data for stratified k fold.
# 
# The targets are classified in about ***n*** equaly sized buckets. The first bucket containing the smallest values and the n^th bucket containing the highest values. Later on the stratified split will treat these buckets as classes, so every split gets about the same target distribution.
# 
# A stratified k fold with 7 folds works best on ***bin_age_7***. 

# In[ ]:


for d in DEP_VAR:
    y_data['bin_'+d] = pd.qcut(y_data[d].rank(method='first'), q=4, labels=False)

y_data['bin_all'] = (y_data['bin_age']+
                    y_data['bin_domain1_var1']*10+
                    y_data['bin_domain1_var2']*100+
                    y_data['bin_domain2_var1']*1000+
                    y_data['bin_domain2_var2']*10000)

y_data['bin_age_10'] = pd.qcut(y_data['age'].rank(method='first'), q=10, labels=False)

y_data['bin_age_7'] = pd.qcut(y_data['age'].rank(method='first'), q=7, labels=False)

if INCLUDE_AV_DATA:
    y_data = y_data.merge(av_data[['Id','is_test_prob']])
    y_data['bin_test_7'] = pd.qcut(y_data['is_test_prob'].rank(method='first'), q=7, labels=False)

y_data


# # Model

# ## Metrics

# In[ ]:


def norm_absolute_error(preds, targs):
    # variation of https://github.com/fastai/fastai/blob/master/fastai/metrics.py#L85
    "Normalized absolute error between `pred` and `targ`."
    
    ## use sign for 0 imputeted empty targets, so they wan't be evaluated
    sg=targs.sign()
    y=targs*sg
        
    pred, targ = flatten_check(preds*sg, y)
    return torch.abs(targ - pred).sum() / targ.sum()

def weighted_nae(preds, targs, details = False):
    
    ## use sign for 0 imputeted empty targets, so they wan't be evaluated
    if IMPUTATION_STRAT == 'IGNORE_ON_TRAIN':
        sg = targs.float().sign()
        x0 = preds[:,0].float()*sg[:,0]
        x1 = preds[:,1].float()*sg[:,1]
        x2 = preds[:,2].float()*sg[:,2]
        x3 = preds[:,3].float()*sg[:,3]
        x4 = preds[:,4].float()*sg[:,4]
    else: # 'MEAN'
        sg = 1
        x0 = preds[:,0].float()
        x1 = preds[:,1].float()
        x2 = preds[:,2].float()
        x3 = preds[:,3].float()
        x4 = preds[:,4].float()
            
    y = targs.float()*sg
    
    return norm_absolute_error(x0,y[:,0]),            norm_absolute_error(x1,y[:,1]),            norm_absolute_error(x2,y[:,2]),            norm_absolute_error(x3,y[:,3]),            norm_absolute_error(x4,y[:,4]),            0.3 * norm_absolute_error(x0,y[:,0]) +            0.175 * norm_absolute_error(x1,y[:,1]) +            0.175 * norm_absolute_error(x2,y[:,2]) +            0.175 * norm_absolute_error(x3,y[:,3]) +            0.175 * norm_absolute_error(x4,y[:,4])

def plot_diff(y, y_truth):
    y_df = pd.DataFrame(y.numpy())
    y_df.columns = DEP_VAR
    y_df = y_df.melt()
    y_df['Id'] = y_df.index
    
    y_truth_df = pd.DataFrame(y_truth.numpy())
    y_truth_df.columns = DEP_VAR
    y_truth_df = y_truth_df.melt()
    y_truth_df['Id'] = y_truth_df.index

    plot_df = y_truth_df.merge(y_df, on=['variable', 'Id'], how='inner').drop('Id', axis = 1)
    plot_df.columns = ['Category', 'Target', 'Prediction']

    g = sns.relplot(x="Target", y="Prediction",
                  col="Category", hue="Category", style="Category",
                  kind="scatter", data=plot_df)


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
# The customized loss function which consideres five outputs is a variation of https://www.kaggle.com/iafoss/grapheme-fast-ai-starter-lb-0-964. It weights and combines the five single losses.
# 
# ***MSE*** works best. Best weights are 0.3, 4x0.175 (0.4 and 4x0.15 was promising too).

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
        
        ## use sign for 0 imputeted empty targets, so they wan't be evaluated
        if IMPUTATION_STRAT == 'IGNORE_ON_TRAIN':
            sg = target.float().sign()
            x0,x1,x2,x3,x4 = x0.float()*sg[:,0],x1.float()*sg[:,1],x2.float()*sg[:,2],x3.float()*sg[:,3],x4.float()*sg[:,4]
        else: # 'MEAN'
            sg = 1
            x0,x1,x2,x3,x4 = x0.float(),x1.float(),x2.float(),x3.float(),x4.float()
            
        y = target.float()*sg
        
        loss1 = 0
        loss2 = 0
        if self.loss_base in ('MSE','MIX'):
            loss_func = F.mse_loss 
            #reduction = 'sum'
            #loss1 = self.fw[0]*loss_func(x0,y[:,0],reduction=reduction)/sum(y[:,0]**2) + \
            #   self.fw[1]*loss_func(x1,y[:,1],reduction=reduction)/sum(y[:,1]**2) + \
            #   self.fw[2]*loss_func(x2,y[:,2],reduction=reduction)/sum(y[:,2]**2) + \
            #   self.fw[3]*loss_func(x3,y[:,3],reduction=reduction)/sum(y[:,3]**2) + \
            #   self.fw[4]*loss_func(x4,y[:,4],reduction=reduction)/sum(y[:,4]**2)
            
            loss1 = (self.fw*(F.mse_loss(input*sg,y,reduction='none').sum(dim=0))/((y**2.5).sum(dim=0)))*3
            loss1 = loss1.sum()
        
        if self.loss_base in ('L1','MIX'):
            loss_func = F.l1_loss 
            #reduction = 'mean'
            #loss2 =  self.fw[0]*loss_func(x0,y[:,0],reduction=reduction) + \
            #   self.fw[1]*loss_func(x1,y[:,1],reduction=reduction) + \
            #   self.fw[2]*loss_func(x2,y[:,2],reduction=reduction) + \
            #   self.fw[3]*loss_func(x3,y[:,3],reduction=reduction) + \
            #   self.fw[4]*loss_func(x4,y[:,4],reduction=reduction)
            
            reduction = 'sum'
            loss2 =  self.fw[0]*loss_func(x0,y[:,0],reduction=reduction)/sum(y[:,0]) +                self.fw[1]*loss_func(x1,y[:,1],reduction=reduction)/sum(y[:,1]) +                self.fw[2]*loss_func(x2,y[:,2],reduction=reduction)/sum(y[:,2]) +                self.fw[3]*loss_func(x3,y[:,3],reduction=reduction)/sum(y[:,3]) +                self.fw[4]*loss_func(x4,y[:,4],reduction=reduction)/sum(y[:,4])
            loss2 = loss2/4

        return loss1 + loss2 #/100


# ## Build the model

# Routing time series features through an LSTM and bypassing the other feature to the linear model.
# 
# Tutorials to build LSTM in pytorch: https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/, https://www.jessicayung.com/lstms-for-time-series-in-pytorch/

# In[ ]:


class TailBlock(nn.Module):
    
    def __init__(self, bs):
        super(TailBlock, self).__init__()
        
        self.bs = bs
        self.lstm = nn.GRU(input_size = 53, hidden_size = 20, num_layers = 64, bidirectional=False, dropout = 0.3) # input, layers
        self.linear = nn.Linear(40, 8)
        
        # init hidden cells
        #self.hidden_cell = (torch.zeros(64,bs, 20).cuda(), 
        #                    torch.zeros(64,bs, 20).cuda())  #h_n, c_n => num_layers, batch_size, hidden_dim
        self.hidden_cell = torch.zeros(128,bs, 20).cuda() #,
                           # torch.zeros(128,bs, 20).cuda())

    def forward(self,x_cont):# x_cat, x_cont):
        print(x_cont.size())
        #print(x_cat.size())
        
        x_non_lstm_feat = x_cont[:,:-5300]
        
        x53 = x_cont[:,-5300:].view(-1, 100, 53)
        print('x53',x53.shape)
        lstm_out, self.hidden_cell = self.lstm(x53.view(100, self.bs, -1), self.hidden_cell)
        print('lout',lstm_out.shape)
        x_lstm_out  = self.linear(torch.cat([x_non_lstm_feat.view(1,-1) , x_lstm_out.view(1,-1)], dim=1)) #lstm_out[-1].view(self.bs, -1))
        
        #print(torch.cat([x_non_lstm_feat , x_lstm_out.view(-1)], dim=1).shape)
        #return x_cat, 
        return x_lstm_out #,torch.cat([x_non_lstm_feat.view(1,-1) , x_lstm_out.view(1,-1)], dim=1)


# A neck or head module that separetes the "route" of one feature from four others. The idea is that ***age*** needs to be evaluated differntly than the ***domain_var*** values.
# 
# It only has a tiny impact on the model. I tried it as head and as neck. Head worked better. Not sure if it considered the target y_ranges (see **prep_learn**-definition below) if it is used as head.

# In[ ]:


class NeckBlock(nn.Module):
    
    def __init__(self, nf, ps=0.3):
        super(NeckBlock,self).__init__()
        
        self.sa = SelfAttention(256)
        
        # age
        self.bn1 = nn.BatchNorm1d(nf)
        self.d1 = nn.Dropout(ps)
        self.l1 = nn.Linear(nf, 1)
        #self.l1b = nn.Linear(nf, 1)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        
        # dom_var
        self.bn2 = nn.BatchNorm1d(nf)
        self.d2 = nn.Dropout(ps)
        self.l2 = nn.Linear(nf, 4)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)
        
        #self.bn2b = nn.BatchNorm1d(nf)
        #self.d2b = nn.Dropout(0.3)
        #self.l2b = nn.Linear(nf, 4)
        #self.act2b = nn.LeakyReLU(0.1, inplace=True)
        
    def forward(self, x):
        ident = x
        
        x0 = self.sa(x)
        
        x1 = self.bn1(x0)
        x1 = self.d1(x1)
        x1 = self.l1(x1)
        #x1 = self.l1b(x1)
        x1 = self.act1(x1+ident)
        
        x2 = self.bn2(x0) #+x1
        x2 = self.d2(x2)
        x2 = self.l2(x2)
        #x2 = self.l2b(x2)
        x2 = self.act2(x2+ident)
               
        return x1, xn
        


# Define the dataloader (wrapped in a fastai databunch).
# 
# Define the fastai learner. Surprisingly the best model was just 1 hidden layer with 8 neurons! ps=0.3 (DropOut) is used for regularization. y_range defines the range bounderies for the predictions and has some positive effect on some configurations.

# In[ ]:





# In[ ]:


def prep_data(bs=128, valid_idx=range(200, 400), train_df = train_df):
    procs = [FillMissing, Categorify, Normalize]
    
    cat_names = []
    if INCLUDE_FNC_CLUSTERS:
        cat_names = cat_names + list(set(c_data.columns)-set(['Id']))

    if INCLUDE_FNC_CLUSTERS_2:
        cat_names = cat_names + list(set(c2_data.columns)-set(['Id']))
    
    
    cont_names = list(set(train_df.columns) - set(cat_names) - set(['age','domain1_var1','domain1_var2', 'domain2_var1', 'domain2_var2'])-set(DEP_VAR))
    tlist = (TabularList.from_df(train_df, 
                                path=kaggle_path, 
                                cat_names=cat_names, 
                                cont_names=cont_names, 
                                procs=procs))

    if valid_idx == None:
        tlist = tlist.split_none()
    else:
        tlist = tlist.split_by_idx(valid_idx)

    data = (tlist.label_from_df(cols=DEP_VAR)
                 .add_test(TabularList.from_df(test, 
                                               cat_names=cat_names,
                                               cont_names=cont_names, 
                                               procs = procs))
                 .databunch(path = kaggle_path, bs = bs))
    
    return data


def prep_learn(data, loss_base = 'MSE'):
    
    learn = tabular_learner(data, 
                        #! only one hidden 8 neuron layer works best
                        layers = [8], #[64,8], # [256,128,256,128,64], #[1024,1024,128,1024,128,1024,1024],#[8], #
                        # Drop Out 0.3 works best
                        ps = 0.3, #.4
                        loss_func = Loss_combine(loss_weights = LOSS_WEIGHTS,  loss_base= loss_base),
                        metrics=[Metric_age(),
                                 Metric_domain1_var1(),
                                 Metric_domain1_var2(),
                                 Metric_domain2_var1(),
                                 Metric_domain2_var2(),
                                 Metric_total()],
                       # not sure if y_range is applied when using the NeckBlock module as head
                       y_range=(Tensor([12,12,0,0,0]).cuda(),Tensor([90,90,100,100,100]).cuda())
                       #y_range=(Tensor([0,0,0,0,0]).cuda(),Tensor([1,1,1,1,1]).cuda())
                       #y_range=(Tensor([12,12,0,0,0]).cuda(),Tensor([90,90,100,100,100]).cuda())     
                       )#.to_fp16()
    
    learn.clip_grad = 1.0
    # adding head/neck module
    learn.model.layers[4].add_module('NeckBlock', NeckBlock(5, 0.4))
    # learn.model = nn.Sequential(TailBlock(1),learn.model)
    # learn.model.bn_cont.add_module('TailBlock', TailBlock(1))
    #learn.model.layers[0].add_module('TailBlock', TailBlock(1))
    #learn.model.cuda()
    return learn
    


# ## Let's run a first training

# In[ ]:


#train500= train_df.head(500)#


# In[ ]:


#train500.shape


# In[ ]:


bs=128
valid_idx=range(200, 400)

data = prep_data(bs, valid_idx, train_df=train_df)
learn = prep_learn(data)


# In[ ]:


#learn.model #summary()


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
print(f'Weighted normalized absolute error (Valid): {weighted_nae(yv,yv_truth)[-1]}')
for i, dv in enumerate(DEP_VAR):
    print(f'NAE {dv} (Valid): {weighted_nae(yv,yv_truth)[i]}')
plot_diff(yv, yv_truth)


# The unseen validation set fitting best on the age target. The metrics in the epoch stats table are different than the final scores. But the tendencies in all the experiments I run where the same, so I didn't investigate further.

# I think it's also good to have a look after predicting your train data. It's not saying anything about unseen data. But it gives you a feeling how over-/underfit the model is.

# In[ ]:


print(f'Weighted normalized absolute error (Train): {weighted_nae(yt,yt_truth)[-1]}')
for i, dv in enumerate(DEP_VAR):
    print(f'NAE {dv} (Train): {weighted_nae(yt,yt_truth)[i]}')
plot_diff(yt, yt_truth)


# ### K-Fold Playground

# 10 Folds on ***bin_age_7*** with 10 epochs and max_lr = 4e-2 work the best.

# In[ ]:


LOSS_WEIGHTS = [.3, .175, .175, .175, .175] #[.4, .15, .15, .15, .15] #[1,0,0,0,0,] #

kf_split = 7
kf = StratifiedKFold(n_splits=kf_split, shuffle=True, random_state=2020)
y_oof = None
y_truth_oof = None
y_oof_g = None
y_truth_oof_g = None
val_ids = []

for i, (train_id, val_id) in enumerate(kf.split(train_df,y_data['bin_age_7'].values)):
    print('Fold #:',i)
    data = prep_data(BS, val_id.tolist())
    learn = prep_learn(data, loss_base = 'MSE')
    
    lr = 4e-2
    reduceLR = callbacks.ReduceLROnPlateauCallback(learn=learn, monitor = 'valid_loss', mode = 'auto', patience = 2, factor = 0.2, min_delta = 0)
    learn.fit_one_cycle(10, lr, callbacks=[reduceLR]) 
    
    val_ids = val_ids+val_id.tolist()

    yv, yv_truth= learn.get_preds(ds_type=DatasetType.Valid)
    if y_oof == None:
        y_oof = yv
        y_truth_oof = yv_truth
    else:
        y_oof=torch.cat((y_oof,yv),0)
        y_truth_oof=torch.cat((y_truth_oof,yv_truth),0)

    
    #print(f'Weighted normalized absolute error (Valid): {weighted_nae(yv,yv_truth)[-1]}')
    #for i, dv in enumerate(DEP_VAR):
    #    print(f'NAE {dv} (Valid): {weighted_nae(yv,yv_truth)[i]}')
    #plot_diff(yv, y_truthv)

print(f'### Total:')
print(f'Weighted normalized absolute error (Valid): {weighted_nae(y_oof,y_truth_oof)[-1]}')
for i, dv in enumerate(DEP_VAR):
    print(f'NAE {dv} (Valid): {weighted_nae(y_oof,y_truth_oof)[i]}')
plot_diff(y_oof, y_truth_oof) # 6: 155


# In[ ]:


#pd.DataFrame((y_oof+torch.tensor([9,16,19,11,12])-y_truth_oof).numpy()).hist()

offset=torch.tensor([.3,.3,.8,.8,.8])

print(f'### Total:')
print(f'Weighted normalized absolute error (Valid): {weighted_nae(y_oof+offset,y_truth_oof)[-1]}')
for i, dv in enumerate(DEP_VAR):
    print(f'NAE {dv} (Valid): {weighted_nae(y_oof+offset,y_truth_oof)[i]}')
plot_diff(y_oof+offset, y_truth_oof)


# ### Total:
# Weighted normalized absolute error (Valid): 0.15890708565711975
# NAE age (Valid): 0.144345223903656
# NAE domain1_var1 (Valid): 0.15140166878700256
# NAE domain1_var2 (Valid): 0.15128813683986664
# NAE domain2_var1 (Valid): 0.18171468377113342
# NAE domain2_var2 (Valid): 0.17618706822395325
# 
# ### Total:
# Weighted normalized absolute error (Valid): 0.15910734236240387
# NAE age (Valid): 0.14428390562534332
# NAE domain1_var1 (Valid): 0.15133097767829895
# NAE domain1_var2 (Valid): 0.15144872665405273
# NAE domain2_var1 (Valid): 0.18138742446899414
# NAE domain2_var2 (Valid): 0.17767387628555298

# # Prediction

# In[ ]:


def gather_preds(learn, ignore_vars=[], offset = torch.ones(5)):
    preds = learn.get_preds(ds_type=DatasetType.Test)[0]
    preds = preds + offset
    
    submission=None

    rec_flatt = pd.DataFrame(idx_test)
    rec_flatt.columns=['Id']
    
    for t, tcol in enumerate(DEP_VAR):
        if tcol not in ignore_vars:
            rec = pd.DataFrame(idx_test)
            rec['Id'] = rec['Id'].astype(str)+'_'+tcol
            rec['Predicted'] = preds[:,t]
            if isinstance(submission, pd.DataFrame):
                submission = submission.append(rec)
            else:
                submission = rec
            
        rec_flatt[tcol] = preds[:,t]
            
    return submission, rec_flatt


def make_submission(gathered_preds, filename = ''):
    #submission = gathered_preds.sort_values('Id').reset_index(drop=True)
    submission = gathered_preds.groupby('Id').mean().sort_values('Id').reset_index(drop=False)
    print('Shape of submission:', submission.shape)
    display(submission.head(10))
    submission.to_csv(filename, index=False)


# ## Predict without split (cross validation)

# In[ ]:


data = prep_data(BS, valid_idx = None)
learn = prep_learn(data)

lr = 4e-2
learn.fit_one_cycle(10, lr)


# In[ ]:


gath_pred_no_split, _ = gather_preds(learn)
make_submission(gath_pred_no_split, filename = 'submission_wo_split.csv')


# ## Predict K-Fold
# 10 Folds on ***bin_age_7*** with 10 epochs and max_lr = 4e-2 work the best.
# 
# ### 10-fold MIX-loss

# In[ ]:


LOSS_WEIGHTS = [.3, .175, .175, .175, .175]

kf_split = 10
kf = StratifiedKFold(n_splits=kf_split, shuffle=True, random_state=2020)
y_oof = None
y_truth_oof = None
val_ids = []
gathered_preds = None
gathered_preds_flat = None

for i, (train_id, val_id) in enumerate(kf.split(train_df,y_data['bin_age_7'].values)):    
    print('Fold #:',i)
    data = prep_data(BS, val_id.tolist())
    learn = prep_learn(data, loss_base = 'MIX')

    lr = 4e-2
    reduceLR = callbacks.ReduceLROnPlateauCallback(learn=learn, monitor = 'valid_loss', mode = 'auto', patience = 2, factor = 0.2, min_delta = 0)
    learn.fit_one_cycle(10, lr, callbacks=[reduceLR])

    val_ids = val_ids+val_id.tolist()

    yv, yv_truth= learn.get_preds(ds_type=DatasetType.Valid)
    if y_oof == None:
        y_oof = yv
        y_truth_oof = yv_truth
    else:
        y_oof=torch.cat((y_oof,yv),0)
        y_truth_oof=torch.cat((y_truth_oof,yv_truth),0)
    
    g_preds, g_preds_flat = gather_preds(learn) #, ignore_vars=['age'])
    if isinstance(gathered_preds, pd.DataFrame):
        gathered_preds = gathered_preds.append(g_preds)
        gathered_preds_flat = gathered_preds_flat.append(g_preds_flat)
    else:
        gathered_preds = g_preds
        gathered_preds_flat = g_preds_flat


y_oof_1 = y_oof
y_truth_oof_1 = y_oof
val_ids_1 = val_ids
print(f'K-Fold:',kf_split)
print(f'Weighted normalized absolute error (Valid): {weighted_nae(y_oof,y_truth_oof)[-1]}')
for i, dv in enumerate(DEP_VAR):
    print(f'NAE {dv} (Valid): {weighted_nae(y_oof,y_truth_oof)[i]}')
plot_diff(y_oof, y_truth_oof)

gathered_preds_1 = gathered_preds
gathered_preds_flat_1 = gathered_preds_flat

make_submission(gathered_preds_1.sort_values('Id').reset_index(drop=True), filename = 'submission_10fold_MIX.csv')
make_submission(gathered_preds_flat_1.sort_values('Id').reset_index(drop=True), filename = 'mix_test_preds_out.csv')


# In[ ]:


train_preds_out = pd.DataFrame(y_oof_1.numpy(), index=val_ids_1).sort_index()
train_preds_out.columns = DEP_VAR
train_preds_out['Id'] = idx_train
train_preds_out.to_csv('mix_train_preds_out.csv', index = False)

train_preds_out1 = train_preds_out.copy()

train_preds_out1


# ### 10 fold MSE-loss + offset

# In[ ]:


LOSS_WEIGHTS = [.3, .175, .175, .175, .175]

offset=torch.tensor([.3,.3,.8,.8,.8])

kf_split = 10
kf = StratifiedKFold(n_splits=kf_split, shuffle=True, random_state=2020)
y_oof = None
y_truth_oof = None
val_ids = []
gathered_preds = None
gathered_preds_flat = None

for i, (train_id, val_id) in enumerate(kf.split(train_df,y_data['bin_age_7'].values)):    #bin_age_7

    print('Fold #:',i)
    data = prep_data(BS, val_id.tolist())
    learn = prep_learn(data, loss_base = 'MSE')

    lr = 4e-2
    reduceLR = callbacks.ReduceLROnPlateauCallback(learn=learn, monitor = 'valid_loss', mode = 'auto', patience = 2, factor = 0.2, min_delta = 0)
    learn.fit_one_cycle(10, lr, callbacks=[reduceLR]) #MixUpCallback(learn, alpha=0.6),

    val_ids = val_ids+val_id.tolist()

    yv, yv_truth= learn.get_preds(ds_type=DatasetType.Valid)
    if y_oof == None:
        y_oof = yv
        y_truth_oof = yv_truth
    else:
        y_oof=torch.cat((y_oof,yv),0)
        y_truth_oof=torch.cat((y_truth_oof,yv_truth),0)
    
    g_preds, g_preds_flat = gather_preds(learn, offset = offset) #, ignore_vars = ['domain1_var1','domain1_var2', 'domain2_var1', 'domain2_var2'])
    if isinstance(gathered_preds, pd.DataFrame):
        gathered_preds_flat = gathered_preds_flat.append(g_preds_flat)
        gathered_preds = gathered_preds.append(g_preds)
    else:
        gathered_preds = g_preds
        gathered_preds_flat = g_preds_flat

        
y_oof_2 = y_oof
y_truth_oof_2 = y_oof
val_ids_2 = val_ids
print(f'K-Fold with offset:',kf_split)
print(f'Weighted normalized absolute error (Valid): {weighted_nae(y_oof,y_truth_oof)[-1]}')
for i, dv in enumerate(DEP_VAR):
    print(f'NAE {dv} (Valid): {weighted_nae(y_oof,y_truth_oof)[i]}')
plot_diff(y_oof, y_truth_oof)


gathered_preds_2 = gathered_preds
gathered_preds_flat_2 = gathered_preds_flat

make_submission(gathered_preds_2.sort_values('Id').reset_index(drop=True), filename = 'submission_10fold_MSE_offset.csv')
make_submission(gathered_preds_flat_2.sort_values('Id').reset_index(drop=True), filename = 'mse_test_preds_out.csv')


# In[ ]:


train_preds_out = pd.DataFrame(y_oof_2.numpy(), index=val_ids_2).sort_index()
train_preds_out.columns = DEP_VAR
train_preds_out['Id'] = idx_train
train_preds_out.to_csv('mse_train_preds_out.csv', index = False)

train_preds_out2 = train_preds_out.copy()

train_preds_out2


# ### Best of blended

# In[ ]:


gathered_preds_flat_1


# In[ ]:


offset = tensor([.3,.3,.3,.3,.3])

print(f'K-Fold blended')
print(f'Weighted normalized absolute error (Valid): {weighted_nae((y_oof_1 + y_oof_2)/2 + offset, y_truth_oof)[-1]}')
for i, dv in enumerate(DEP_VAR):
    print(f'NAE {dv} (Valid): {weighted_nae((y_oof_1 + y_oof_2)/2 + offset, y_truth_oof)[i]}')
plot_diff(y_oof, y_truth_oof)

# offset
gp1 = gathered_preds_1.copy()
gp2 = gathered_preds_2.copy()
gp1['Predicted']=gathered_preds_1['Predicted']+.3
gp2['Predicted']=gathered_preds_2['Predicted']+.3

gp_flat1 = gathered_preds_flat_1.copy()
gp_flat2 = gathered_preds_flat_2.copy()
for dv in DEP_VAR:
    gp_flat1[dv]=gathered_preds_flat_1[dv]+.3
    gp_flat2[dv]=gathered_preds_flat_2[dv]+.3

make_submission(gp1.append(gp2).sort_values('Id').reset_index(drop=True), filename = 'submission_10kfold_blended.csv')

make_submission(gp_flat1.append(gp_flat2).sort_values('Id').reset_index(drop=True), filename = 'blended_test_preds_out.csv')
make_submission(train_preds_out1.append(train_preds_out1).sort_values('Id').reset_index(drop=True), filename = 'blended_train_preds_out.csv')


# 
