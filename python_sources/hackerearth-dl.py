#!/usr/bin/env python
# coding: utf-8

# # Finalising The Notebook

# In[ ]:


# Load Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
get_ipython().system('pip install pretrainedmodels')
import pretrainedmodels
from fastai import *
from fastai.vision import *
from fastai.callbacks import * 
#from fastai.vision.models.cadene_models import *
from fastai.vision.models import *
from fastai.vision.learner import model_meta
import pandas as pd
from sklearn.model_selection import train_test_split
from fastai.torch_core import flatten_model
from fastai.layers import CrossEntropyFlat
from fastai.metrics import error_rate # 1 - accuracy
from sklearn.metrics import f1_score , roc_auc_score
import torch
from tqdm import tqdm


# # Setting Notebook Defaults

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
os.environ['FASTAI_TB_CLEAR_FRAMES']="1"
# Setting the path
path = "/kaggle/input/auto-tag-images-of-gala/dataset"


# # Util Functions

# In[ ]:


def arch_summary(arch):
    model = arch(False)
    tot = 0
    for i, l in enumerate(model.children()):
        n_layers = len(flatten_model(l))
        tot += n_layers
        print(f'({i}) {l.__class__.__name__:<12}: {n_layers:<4}layers (total: {tot})')


def get_groups(model, layer_groups):
    group_indices = [len(g) for g in layer_groups]
    curr_i = 0
    group = []
    for layer in model:
        group_indices[curr_i] -= len(flatten_model(layer))
        group.append(layer.__class__.__name__)
        if group_indices[curr_i] == 0:
            curr_i += 1
            print(f'Group {curr_i}:', group)   
            group = []


# # Training Functions

# In[ ]:


def get_data(valid_pct, img_size, batch_size):
    tfms = get_transforms()
    data = ImageDataBunch.from_csv(path,folder='Train Images',valid_pct=0.2,csv_labels='train.csv',ds_tfms=tfms,fn_col='Image',test='Test Images',label_col='Class',size=img_size,bs=batch_size).normalize(imagenet_stats)
#     if normalise:
#         data = data.normalise(imagenet_stats)
    return data

def train(arch,data,epoch,max_lr=None):
    learner = cnn_learner(data, arch,metrics=accuracy)
    learner.model_dir = '/kaggle/working/'
    if max_lr==None:
        learner.fit_one_cycle(epoch)
    else:
        learner.fit_one_cycle(epoch,max_lr=slice(max_lr))
    return learner

def prog_resize(arch,sz,n,k):
    sz=sz
    for i in range(n):
        data = get_data(0.2,sz,batch_size=64)
        if i>0:
            load_size=int(sz-k)
            learner = cnn_learner(data, arch,metrics=[FBeta(),accuracy],  wd=1e-1, opt_func= ranger).to_fp16()#, use_swa=True, callbacks=[SaveModelParams(learner.model)])
            RLR = ReduceLROnPlateauCallback(learner, monitor='f_beta',patience = 2)
            SAVEML = SaveModelCallback(learner, every='improvement', monitor='f_beta', name='best')
            learner.model_dir = '/kaggle/working/'
            #learner.lr_find()
            #learner.recorder.plot(suggestion=True)
            learner.load(f'learner_{load_size}')
            learner.fit_one_cycle(4,max_lr=slice(1e-4,1e-1),wd=3e-4,moms=(0.99-0.90),callbacks = [RLR, SAVEML, ShowGraph(learner)])
            #learner.unfreeze()
            #learner.fit_one_cycle(1, max_lr)
            learner.save(f'learner_{sz}')
        else:
            learner = train(arch, data, epoch=8)
            learner.model_dir = '/kaggle/working/'
            learner.save(f'learner_{sz}')
        sz+=k
#         params=[]
#         swa_model_params = [p.data.cpu().numpy() for p in learner.swa_model.parameters()]

#         for p_model1, p_model2, p_model3, p_swa_model in zip(*params, swa_model_params):
#             # check for equality up to a certain tolerance
#             print(np.isclose(p_swa_model, np.mean(np.stack([p_model1, p_model2, p_model3]), axis=0)))

    return learner


# # Normal Model (Both Cadene and Default) Runs

# ### Data Loaders with Transforms
# 
# **Hyperparameters**
# 1. Transformation Parameters -
#     1. Flip_vert,
#     2. max_lighting
#     3. max_zoom
#     4. max_warp
#         1. Default values - ["False",0.1,1.05,0.15]
#         2. Best - 
# 2. Data Loader Parameters -
#     1. Valid Pct (Def-0.2)
#     2. Image Size (Def- 140)
#     3. Batch Size
#     4. Normalize (Def-ImagenetStats)

# In[ ]:


tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.15)
data = ImageDataBunch.from_csv(path,folder='Train Images',valid_pct=0.2,csv_labels='train.csv',ds_tfms=tfms,fn_col='Image',test='Test Images',label_col='Class',size=224).normalize(imagenet_stats)


# ### Defining Custom Loss

# In[ ]:


class FocalLoss(nn.Module):
    def __init__(self, alpha=1., gamma=1.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, **kwargs):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * ((1-pt)**self.gamma) * CE_loss
        return F_loss.mean()

loss_func = FocalLoss(gamma=1.)


# ### Defining Custom Callbacks

# #### Example SWA Callback

# In[ ]:


class SWA(Callback):
    def __init__(self, model, swa_model, swa_start):
        super().__init__()
        self.model,self.swa_model,self.swa_start=model,swa_model,swa_start
        
    def on_train_begin(self):
        self.epoch = 0
        self.swa_n = 0

    def on_epoch_end(self, metrics):
        if (self.epoch + 1) >= self.swa_start:
            self.update_average_model()
            self.swa_n += 1
            
        self.epoch += 1
            
    def update_average_model(self):
        # update running average of parameters
        model_params = self.model.parameters()
        swa_params = self.swa_model.parameters()
        for model_param, swa_param in zip(model_params, swa_params):
            swa_param.data *= self.swa_n
            swa_param.data += model_param.data
            swa_param.data /= (self.swa_n + 1)

def collect_bn_modules(module, bn_modules):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        bn_modules.append(module)

def fix_batchnorm(swa_model, train_dl):
    """
    During training, batch norm layers keep track of a running mean and
    variance of the previous layer's activations. Because the parameters
    of the SWA model are computed as the average of other models' parameters,
    the SWA model never sees the training data itself, and therefore has no
    opportunity to compute the correct batch norm statistics. Before performing 
    inference with the SWA model, we perform a single pass over the training data
    to calculate an accurate running mean and variance for each batch norm layer.
    """
    bn_modules = []
    swa_model.apply(lambda module: collect_bn_modules(module, bn_modules))
    
    if not bn_modules: return

    swa_model.train()

    for module in bn_modules:
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)
    
    momenta = [m.momentum for m in bn_modules]

    inputs_seen = 0

    for (*x,y) in iter(train_dl):        
        xs = V(x)
        batch_size = xs[0].size(0)

        momentum = batch_size / (inputs_seen + batch_size)
        for module in bn_modules:
            module.momentum = momentum
                            
        res = swa_model(*xs)        
        
        inputs_seen += batch_size
                
    for module, momentum in zip(bn_modules, momenta):
        module.momentum = momentum 


# callback for storing the params of the model after each epoch
class SaveModelParams(Callback):
    def __init__(self, model):
        self.model = model

    def on_epoch_end(self, metrics):
        params.append([p.data.cpu().numpy() for p in self.model.parameters()])


# #### Probable Usage of SWA

# In[ ]:


# params = []

# learn2.fit(lr, 3, use_swa=True, callbacks=[SaveModelParams(learn2.model)])

# # grab the params from the SWA model
# swa_model_params = [p.data.cpu().numpy() for p in learn2.swa_model.parameters()]

# for p_model1, p_model2, p_model3, p_swa_model in zip(*params, swa_model_params):
#     # check for equality up to a certain tolerance
#     print(np.isclose(p_swa_model, np.mean(np.stack([p_model1, p_model2, p_model3]), axis=0)))


# # Getting The Big Guns out - One by one

# ## Se-Net 154 - with Custom Loss, Precision, MixUp, Weight Decay, and Callbacks

# In[ ]:


def senet154(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.senet154(pretrained=pretrained)
    return model

_se_resnet_meta = {'cut': -3, 'split': lambda m: (m[0][3], m[1]) }
model_meta[senet154] = _se_resnet_meta


# In[ ]:


learn = create_cnn(data, senet154, ps=0.5, wd=1e-1, loss_func=loss_func, metrics=[FBeta(),accuracy, auc_roc_score, MultiLabelFbeta(average='weighted')], pretrained=True, model_dir='/kaggle/working/').to_fp16()


# In[ ]:


RLR = ReduceLROnPlateauCallback(learn, monitor='f_beta',patience = 2)
SAVEML = SaveModelCallback(learn, every='improvement', monitor='f_beta', name='best')


# In[ ]:


learn.model_dir='/kaggle/working/'
learn.lr_find()

learn.recorder.plot(suggestion=True)


# In[ ]:


learn.model_dir='/kaggle/working/'
learn.fit_one_cycle(2)


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F  #(uncomment if needed,but you likely already have it)

#Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
#https://arxiv.org/abs/1908.08681v1
#implemented for PyTorch / FastAI by lessw2020 
#github: https://github.com/lessw2020/mish

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))


# 
# ## ResNext

# In[ ]:


def resnext101_32x4d(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.resnext101_32x4d(pretrained=pretrained)
    all_layers = list(model.children())
    return nn.Sequential(*all_layers[0], *all_layers[1:])


# In[ ]:


arch_summary(resnext101_32x4d)


# In[ ]:


import math
import torch
from torch.optim.optimizer import Optimizer, required
import itertools as it



class Ranger(Optimizer):

    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95,0.999), eps=1e-5, weight_decay=0):
        #parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        #parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        #N_sma_threshold of 5 seems better in testing than 4.
        #In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

        #prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas, N_sma_threshhold=N_sma_threshhold, eps=eps, weight_decay=weight_decay)
        super().__init__(params,defaults)

        #adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        #now we can get to work...
        #removed as we now use step from RAdam...no need for duplicate step counting
        #for group in self.param_groups:
        #    group["step_counter"] = 0
            #print("group step counter init")

        #look ahead params
        self.alpha = alpha
        self.k = k 

        #radam buffer for state
        self.radam_buffer = [[None,None,None] for ind in range(10)]

        #self.first_run_check=0

        #lookahead weights
        #9/2/19 - lookahead param tensors have been moved to state storage.  
        #This should resolve issues with load/save where weights were left in GPU memory from first load, slowing down future runs.

        #self.slow_weights = [[p.clone().detach() for p in group['params']]
        #                     for group in self.param_groups]

        #don't use grad for lookahead weights
        #for w in it.chain(*self.slow_weights):
        #    w.requires_grad = False

    def __setstate__(self, state):
        print("set state called")
        super(Ranger, self).__setstate__(state)


    def step(self, closure=None):
        loss = None
        #note - below is commented out b/c I have other work that passes back the loss as a float, and thus not a callable closure.  
        #Uncomment if you need to use the actual closure...

        #if closure is not None:
            #loss = closure()

        #Evaluate averages and grad, update param tensors
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Ranger optimizer does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]  #get state dict for this param

                if len(state) == 0:   #if first time to run...init dictionary with our desired entries
                    #if self.first_run_check==0:
                        #self.first_run_check=1
                        #print("Initializing slow buffer...should not see this at load from saved model!")
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                    #look ahead weight storage now in state dict 
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)

                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                #begin computations 
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                #compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                #compute mean moving avg
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1


                buffered = self.radam_buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

                #integrated look ahead...
                #we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer'] #get access to slow param tensor
                    slow_p.add_(self.alpha, p.data - slow_p)  #(fast weights - slow weights) * alpha
                    p.data.copy_(slow_p)  #copy interpolated weights to RAdam param tensor

        return loss


# In[ ]:


ranger = Ranger
resnext = cnn_learner(data, resnext101_32x4d, pretrained=True, metrics=[FBeta(),accuracy],  ps=0.5, wd=1e-1, loss_func=loss_func, opt_func= ranger,
                    cut=-2, split_on=lambda m: (m[0][6], m[1])).to_fp16()
RLR = ReduceLROnPlateauCallback(resnext, monitor='f_beta',patience = 2)
SAVEML = SaveModelCallback(resnext, every='improvement', monitor='f_beta', name='best')


# In[ ]:


resnext.model_dir='/kaggle/working/'
resnext.lr_find()

resnext.recorder.plot(suggestion=True)


# In[ ]:


resnext.fit_one_cycle(60, 2.75e-2, callbacks = [RLR, SAVEML, ShowGraph(resnext)])


# ## Dual Path Net

# In[ ]:


def dpn92(pretrained=False):
    pretrained = 'imagenet+5k' if pretrained else None
    model = pretrainedmodels.dpn92(pretrained=pretrained)
    return nn.Sequential(*list(model.children()))

dpn = cnn_learner(data, dpn92, pretrained=True, metrics=accuracy,
                    cut=-1, split_on=lambda m: (m[0][0][16], m[1])).to_fp16()
dpn.fit_one_cycle(60)


# # InceptionResnet

# In[ ]:


def inceptionresnetv2(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.inceptionresnetv2(pretrained=pretrained)
    return nn.Sequential(*model.children())

arch_summary(inceptionresnetv2)

inres = cnn_learner(data, inceptionresnetv2, pretrained=True, metrics=accuracy,
                    cut=-2, split_on=lambda m: (m[0][11], m[1]))
inres.fit_one_cycle(8)


# In[ ]:


# Needs lot more epochs to train
# def xception(pretrained=False):
#     pretrained = 'imagenet' if pretrained else None
#     model = pretrainedmodels.xception(pretrained=pretrained)
#     return nn.Sequential(*list(model.children()))

# arch_summary(xception)

# xcept = cnn_learner(data, xception, pretrained=True, metrics=accuracy,
#                     cut=-1, split_on=lambda m: (m[0][11], m[1]))
# xcept.fit_one_cycle(16)


# In[ ]:


# Needs more epochs 
# def se_resnet50(pretrained=False):
#     pretrained = 'imagenet' if pretrained else None
#     model = pretrainedmodels.se_resnet50(pretrained=pretrained)
#     return model
# arch_summary(se_resnet50)

# senet = cnn_learner(data, se_resnet50, pretrained=True,metrics=accuracy,
#                     cut=-2, split_on=lambda m: (m[0][3], m[1]))
# senet.fit_one_cycle(12)

#get_groups(nn.Sequential(*learn.model[0], *learn.model[1]), learn.layer_groups)


# In[ ]:


# def inceptionv4(pretrained=False):
#     pretrained = 'imagenet' if pretrained else None
#     model = pretrainedmodels.inceptionv4(pretrained=pretrained)
#     all_layers = list(model.children())
#     return nn.Sequential(*all_layers[0], *all_layers[1:])

# learn = cnn_learner(FakeData(), inceptionv4, pretrained=False,
#                     cut=-2, split_on=lambda m: (m[0][11], m[1]))
# get_groups(nn.Sequential(*learn.model[0], *learn.model[1]), learn.layer_groups)

# def inceptionresnetv2(pretrained=False):
#     pretrained = 'imagenet' if pretrained else None
#     model = pretrainedmodels.inceptionresnetv2(pretrained=pretrained)
#     return nn.Sequential(*model.children())

# arch_summary(inceptionresnetv2)

# learn = cnn_learner(FakeData(), inceptionresnetv2, pretrained=False,
#                     cut=-2, split_on=lambda m: (m[0][9], m[1]))
# get_groups(nn.Sequential(*learn.model[0], *learn.model[1]), learn.layer_groups)

# def xception(pretrained=False):
#     pretrained = 'imagenet' if pretrained else None
#     model = pretrainedmodels.xception(pretrained=pretrained)
#     return nn.Sequential(*list(model.children()))
# arch_summary(xception)
# learn = cnn_learner(FakeData(), xception, pretrained=False,
#                     cut=-1, split_on=lambda m: (m[0][11], m[1]))
# get_groups(nn.Sequential(*learn.model[0], *learn.model[1]), learn.layer_groups)

# def identity(x): return x

# def nasnetamobile(pretrained=False):
#     pretrained = 'imagenet' if pretrained else None
#     model = pretrainedmodels.nasnetamobile(pretrained=pretrained, num_classes=1000)  
#     model.logits = identity
#     return nn.Sequential(model)

# arch_summary(lambda _: nasnetamobile(False)[0])
# learn = cnn_learner(FakeData(), nasnetamobile, pretrained=False)


# def dpn92(pretrained=False):
#     pretrained = 'imagenet+5k' if pretrained else None
#     model = pretrainedmodels.dpn92(pretrained=pretrained)
#     return nn.Sequential(*list(model.children()))

# arch_summary(dpn92)

# arch_summary(lambda _: next(dpn92(False).children()))

# get_groups(nn.Sequential(*learn.model[0][0], *learn.model[1]), learn.layer_groups)
# learn = cnn_learner(FakeData(), dpn92, pretrained=False,
#                     cut=-1, split_on=lambda m: (m[0][0][16], m[1]))


# def pnasnet5large(pretrained=False):    
#     pretrained = 'imagenet' if pretrained else None
#     model = pretrainedmodels.pnasnet5large(pretrained=pretrained, num_classes=1000) 
#     model.logits = identity
#     return nn.Sequential(model)
# arch_summary(lambda _: pnasnet5large(False)[0])

# model_meta[pnasnet5large] =  { 'cut': noop, 
#                                'split': lambda m: (list(m[0][0].children())[8], m[1]) }

# All the metas

# Calling
#model_meta[fn_name] = _model_meta

# _resnext_meta = {'cut': -2, 'split': lambda m: (m[0][6], m[1]) }
# _se_resnet_meta = {'cut': -2, 'split': lambda m: (m[0][3], m[1]) }
# _inception_4_meta = { 'cut': -2, 'split': lambda m: (m[0][11], m[1]) }
# _squeezenet_meta = { 'cut': -1, 'split': lambda m: (m[0][0][7], m[1]) }
# _xception_meta = { 'cut': -1, 'split': lambda m: (m[0][11], m[1]) }
# model_meta[nasnetamobile] =  { 'cut': noop, 
#                                'split': lambda m: (list(m[0][0].children())[8], m[1]) }


# In[ ]:


# def get_ensemble(nmodels):
#     ens_model = [] # Empty List of ensemble model, I will store the trained learner object here 
#     learning_rate =[1e-3,1e-3,1e-3,1e-3] # List of learning rate for each model 
#     model_list = [models.resnet50,models.resnet152,models.densenet169,models.densenet201] ##List of Models . You can add resnet ones in the mix
#     for i in range(nmodels):
#         print(f'-----Training model: {i+1}--------')
             
#         data = get_data(0.2,80,64)
#         learn_resnet = cnn_learner(data, model_list[i],pretrained=True, metrics=[error_rate, accuracy,AUROC()],
#                                    model_dir="/tmp/model/")
#         learn_resnet.fit_one_cycle(6)
#         print('training for 120x120')
#         learn_resnet.set_data = get_data(0.2,120,64) # Train the model for imagesize 128
#         #learn_resnet.lr_find()
#         #learn_resnet.recorder.plot(suggestion=True)
#         learn_resnet.fit_one_cycle(4) # using the learning rate for the first model 
        
#         print('training for 160x160')
#         learn_resnet.set_data = get_data(0.2,160,64) # Train the model for imagesize 128
#         #learn_resnet.lr_find()
#         #learn_resnet.recorder.plot(suggestion=True)
#         learn_resnet.fit_one_cycle(3) # using the learning rate for the first model 
        
#         print('training for 200x200')
#         learn_resnet.set_data = get_data(0.2,200,64) # Train the model for imagesize 128
#         #learn_resnet.lr_find()
#         #learn_resnet.recorder.plot(suggestion=True)
#         learn_resnet.fit_one_cycle(2) # using the learning rate for the first model 
        
#         print('training for 240x240')
#         learn_resnet.set_data = get_data(0.2,240,64) #Train the model for imagesize 150
#         learn_resnet.fit_one_cycle(1)   # using the learning rate assigned for the first model   
        
#         learn_resnet.save(f'ensem_model_{i}.weights')
#         ens_model.append(learn_resnet)
#         print(f'-----Training of model {i+1} complete----')
#     return ens_model


# In[ ]:


#ens = get_ensemble(4)


# In[ ]:


# ens_test_preds = [] ## Creating a list of predictions 
# for mdl in ens:
#     preds,_ = mdl.TTA(ds_type=DatasetType.Test)
#     print(np.array(preds).shape)
#     ens_test_preds.append(np.array(preds)) ## create a list of prediction numpy arrays . 
    
# ens_preds = np.mean(ens_test_preds, axis =0) ## Average the prediction from various numpy arrays using numpy mean function
# test_df.has_cactus = ens_preds[:, 0] ##update the prediction in the test data
# test_df.head()


# In[ ]:


def ensemble_predition(img, ens):
    #img = open_image(test_path + test_img)
    preds=[]
    for i in ens:
        preds.append(i.predict(img))
        #densenet_predicition = dense.predict(img)
        #inc_predicition = inception.predict(img)
    
    #ensemble average
    sum_pred=torch.tensor([0,0,0,0],dtype=torch.float)
    for i in preds:
        #print(i[2])
        sum_pred += i[2] #+ densenet_predicition[2] #+ inc_predicition[2]
    prediction = sum_pred / len(preds)
    
    #prediction results
    predicted_label = torch.argmax(prediction).item()
    if predicted_label==0:
        predicted_label='Attire'
    elif predicted_label==1:
        predicted_label='Decorationandsignage'
    elif predicted_label==2:
        predicted_label='Food'
    elif predicted_label==3:
        predicted_label='misc'
    #predicted_label.map(val)
    return predicted_label, preds#resnet_predicition[0], densenet_predicition[0]#, inc_predicition[0]


# In[ ]:





# In[ ]:





# In[ ]:


res = prog_resize(models.resnet152,64,5,40)


# In[ ]:


dense = prog_resize(models.densenet201,64,5,40)


# In[ ]:


def train_resnext(arch,data,epoch,max_lr=None):
    learner = cnn_learner(data, arch, pretrained=True, metrics=accuracy,
                    cut=-1, split_on=lambda m: (m[0][0][16], m[1]))
    learner.model_dir = '/kaggle/working/'
    if max_lr==None:
        learner.fit_one_cycle(epoch)
    else:
        learner.fit_one_cycle(epoch,max_lr=slice(max_lr))
    return learner

def prog_resize_resnext(arch,sz,n,k, max_lr=None):
    sz=sz
    for i in range(n):
        data = get_data(0.2,sz,batch_size=64)
        if i>0:
            load_size=int(sz-k)
            learn = cnn_learner(data, arch, pretrained=True, metrics=accuracy,
                    cut=-1, split_on=lambda m: (m[0][0][16], m[1]))
            learner.model_dir = '/kaggle/working/'
            learner.load(f'learner_{load_size}')
            learner.fit_one_cycle(4)
#             learner.unfreeze()
#             learner.fit_one_cycle(1, max_lr)
            learner.save(f'learner_{sz}')
        else:
            learner = train_resnext(arch, data, epoch=8)
            learner.model_dir = '/kaggle/working/'
            learner.save(f'learner_{sz}')
        sz+=k
#         params=[]
#         swa_model_params = [p.data.cpu().numpy() for p in learner.swa_model.parameters()]

#         for p_model1, p_model2, p_model3, p_swa_model in zip(*params, swa_model_params):
#             # check for equality up to a certain tolerance
#             print(np.isclose(p_swa_model, np.mean(np.stack([p_model1, p_model2, p_model3]), axis=0)))

    return learner


# In[ ]:


dpn_pr = prog_resize_resnext(dpn92,80,5,40)


# In[ ]:


tfms = get_transforms()
data = ImageDataBunch.from_csv(path,folder='Train Images',valid_pct=0.02,csv_labels='train.csv',ds_tfms=tfms,fn_col='Image',test='Test Images',label_col='Class',size=220,bs=64)
dpn = train_dpn(dpn92,data,8)


# In[ ]:


# import pretrainedmodels
# def inceptionv4(pretrained=False):
#     pretrained = 'imagenet' if pretrained else None
#     model = pretrainedmodels.inceptionv4(pretrained=pretrained)
#     all_layers = list(model.children())
#     return nn.Sequential(*all_layers[0], *all_layers[1:])
# tfms = get_transforms()
# data = ImageDataBunch.from_csv(path,folder='Train Images',valid_pct=0.02,csv_labels='train.csv',ds_tfms=tfms,fn_col='Image',test='Test Images',label_col='Class',size=240)
# inception = cnn_learner(data, inceptionv4, pretrained=True, metrics=accuracy,
#                     cut=-2, split_on=lambda m: (m[0][11], m[1]))


# In[ ]:


#import pretrainedmodels
# def incres(pretrained=False):
#     pretrained = 'imagenet' if pretrained else None
#     model = inceptionresnetv2(num_classes=1000, pretrained=pretrained)
#     #all_layers = list(model.children())
#     #return nn.Sequential(*all_layers[0], *all_layers[1:])
#     return nn.Sequential(model)

# def nasnetalarge(pretrained:bool=False):
#     pretrained = 'imagenet' if pretrained else None
#     model = nasnetalarge(num_classes=1000, pretrained=pretrained)
#     model.logits = noop
#     return nn.Sequential(model)
# #model_meta[nasnetamobile] = {'cut': noop, 'split': lambda m: (list(m[0][0].children())[8], m[1])}

# def inceptionresnetv2(pretrained:bool=False):  return get_model('inceptionresnetv2', pretrained, seq=True)
# def dpn92(pretrained:bool=False):              return get_model('dpn92', pretrained, pname='imagenet+5k', seq=True)
# def xception_cadene(pretrained=False):         return get_model('xception', pretrained, seq=True)
# def se_resnet50(pretrained:bool=False):        return get_model('se_resnet50', pretrained)
# def se_resnet101(pretrained:bool=False):       return get_model('se_resnet101', pretrained)
# def se_resnext50_32x4d(pretrained:bool=False): return get_model('se_resnext50_32x4d', pretrained)
# def senet154(pretrained:bool=False):           return get_model('senet154', pretrained)

# model_meta[inceptionresnetv2] = {'cut': -2, 'split': lambda m: (m[0][9],     m[1])}
# model_meta[dpn92]             = {'cut': -1, 'split': lambda m: (m[0][0][16], m[1])}
# model_meta[xception_cadene]   = {'cut': -1, 'split': lambda m: (m[0][11],    m[1])}
# model_meta[senet154]          = {'cut': -3, 'split': lambda m: (m[0][3],     m[1])}
# _se_resnet_meta               = {'cut': -2, 'split': lambda m: (m[0][3],     m[1])}
# model_meta[se_resnet50]        = _se_resnet_meta
# model_meta[se_resnet101]       = _se_resnet_meta
# model_meta[se_resnext50_32x4d] = _se_resnet_meta


tfms = get_transforms()
data = ImageDataBunch.from_csv(path,folder='Train Images',valid_pct=0.02,csv_labels='train.csv',ds_tfms=tfms,fn_col='Image',test='Test Images',label_col='Class',size=180,bs=128)

#incres = cnn_learner(data, incres, pretrained=True, metrics=accuracy)#,
                    #cut=-2, split_on=lambda m: (m[0][11], m[1]))
#nas = cnn_learner(data, nasnetamobile, pretrained=True, metrics=accuracy)#,cut= 'noop', split_on = lambda m: (list(m[0][0].children())[8], m[1]))


# In[ ]:


#nas.fit_one_cycle(6)


# In[ ]:


from fastai.metrics import error_rate # 1 - accuracy
from sklearn.metrics import f1_score , roc_auc_score


# In[ ]:


# resnet = cnn_learner(data, models.resnet152, metrics=error_rate)#.mixup()
# resnet_withMixUp = cnn_learner(data, models.resnet152, metrics=error_rate).mixup()
# dnet = cnn_learner(data, models.densenet169, metrics=error_rate)#.mixup()
# dnet_withMixUp = cnn_learner(data, models.densenet169, metrics=error_rate).mixup()
# #resnext = cnn_learner(data, models.resnext152, metrics=error_rate)
# defaults.device = torch.device('cuda') # makes sure the gpu is used


# In[ ]:





# In[ ]:


interp_res = ClassificationInterpretation.from_learner(res)
interp_res.plot_confusion_matrix()


# In[ ]:


interp_dense = ClassificationInterpretation.from_learner(dense)
interp_dense.plot_confusion_matrix()


# In[ ]:


interp_in = ClassificationInterpretation.from_learner(inception)
interp_in.plot_confusion_matrix()


# In[ ]:


rnext.unfreeze() # must be done before calling lr_find
#res.unfreeze() # must be done before calling lr_find
#inception.unfreeze() # must be done before calling lr_find


# In[ ]:


learn.model_dir = '/kaggle/working/'
learn.lr_find()
learn.recorder.plot()


# In[ ]:


# dense.lr_find()
# dense.recorder.plot()


# In[ ]:


# inception.lr_find()
# inception.recorder.plot()


# In[ ]:


# tfms = get_transforms()
# data = ImageDataBunch.from_csv(path,folder='Train Images',valid_pct=0.02,csv_labels='train.csv',ds_tfms=tfms,fn_col='Image',test='Test Images',label_col='Class',size=240)
# res = cnn_learner(data, models.resnet152, metrics=accuracy)
# dense = cnn_learner(data, models.densenet201, metrics=accuracy)
# res.model_dir = '/kaggle/working/'
# dense.model_dir = '/kaggle/working/'
# res.load('reslearner_220')
# dense.load('denselearner_220')
# res.fit_one_cycle(2)


# In[ ]:


rnext.fit_one_cycle(8,max_lr=slice(1e-5, 1e-4),callbacks=[ShowGraph(rnext)])


# In[ ]:


#inception.fit_one_cycle(20,max_lr=slice(1e-5, 5e-3),callbacks=[ShowGraph(inception)])


# In[ ]:


res.fit_one_cycle(8,max_lr=slice(1e-5, 6e-5),callbacks=[ShowGraph(res)])


# In[ ]:


#learn.fit_one_cycle(20, max_lr=slice(3e-6, 6e-4),callbacks=[ShowGraph(learn)])


# In[ ]:


# from fastai.widgets import *

# ds, idxs = DatasetFormatter().from_toplosses(learn)
# ImageCleaner(ds, idxs, '/kaggle/working/')


# In[ ]:


# interp = ClassificationInterpretation.from_learner(learn)
# interp.plot_confusion_matrix()


# In[ ]:


# preds,y = learn.TTA()
# acc = accuracy(preds, y)
# print('The validation accuracy is {} %.'.format(acc * 100))


# In[ ]:


# #preds,y = learn.TTA()
# labels = np.argmax(preds, 1)
# score = f1_score(list(y),list(labels),average='weighted')
# print('The validation f1 score is {} %.'.format(score * 100))


# In[ ]:


res_preds,y = res.TTA()
res_acc = accuracy(res_preds, y)
print('The validation accuracy of Resnet is {} %.'.format(res_acc * 100))

#preds,y = learn.TTA()
res_labels = np.argmax(res_preds, 1)
res_score = f1_score(list(y),list(res_labels),average='weighted')
print('The validation f1 score of Resnet is {} %.'.format(res_score * 100))

# dense_preds,y = dense.TTA()
# dense_acc = accuracy(dense_preds, y)
# print('The validation accuracy of Densenet is {} %.'.format(dense_acc * 100))

# #preds,y = learn.TTA()
# dense_labels = np.argmax(dense_preds, 1)
# dense_score = f1_score(list(y),list(dense_labels),average='weighted')
# print('The validation f1 score of Densenet is {} %.'.format(dense_score * 100))

# inception_preds,y = inception.TTA()
# inception_acc = accuracy(inception_preds, y)
# print('The validation accuracy of InceptionNet is {} %.'.format(inception_acc * 100))

# #preds,y = learn.TTA()
# inception_labels = np.argmax(inception_preds, 1)
# inception_score = f1_score(list(y),list(inception_labels),average='weighted')
# print('The validation f1 score of InceptionNet is {} %.'.format(inception_score * 100))


# In[ ]:


ens = [res, dense, dpn]


# In[ ]:


test_path = path+'/Test Images/'
def generateSubmission():
    submissions = pd.read_csv(path+'/test.csv')
    #id_list = list(data.test_ds.x.items)
    id_list=list(submissions.Image)
    #predictions, *_ = learn.get_preds(DatasetType.Test,order=True)
    ensemble_label=[]
    #inception_label=[]
    #resnet1_label=[]
    resnet2_label=[]
    dense1_label=[]
    #dense2_label=[]
    dpn_label = []
    #resnext_label= []
    with tqdm(total=len(os.listdir(test_path))) as pbar:
        for iname in id_list:
            img=open_image(path+"/Test Images/"+iname)
#             resnet50_predicition = resnet50_learner.predict(img)
#             densenet121_predicition = densenet121_learner.predict(img)
#             vgg_predicition = vgg_learner.predict(img)

            ##ensemble average
#             sum_pred = resnet50_predicition[2] + densenet121_predicition[2] + vgg_predicition[2]
#             prediction = sum_pred / 3

            ##prediction results
#             label.append(torch.argmax(prediction).item())
            en,preds = ensemble_predition(img,ens)
            #dpn_pred = dpn.predict(img)
            ensemble_label.append(en)
            #resnet1_label.append(res.predict(img)[0])
            resnet2_label.append(preds[0][0])
            dense1_label.append(preds[1][0])
            #dense2_label.append(preds[3][0])
            dpn_label.append(preds[2][0])
            #resnext_label.append(resnext.predict(img)[0])
            #inception_label.append(inception.predict(img)[0])
            pbar.update(1)
    submissions = pd.DataFrame({'Image':id_list,'Class':ensemble_label})
    submissions.to_csv("submission_ensemble.csv",index = False)
    submissions_d1 = pd.DataFrame({'Image':id_list,'Class':dense1_label})
    submissions_d1.to_csv("submission_dense1.csv",index = False)
    #submissions_d2 = pd.DataFrame({'Image':id_list,'Class':dense2_label})
    #submissions_d2.to_csv("submission_dense2.csv",index = False)
    submissions_r1 = pd.DataFrame({'Image':id_list,'Class':resnet2_label})
    submissions_r1.to_csv("submission_res1.csv",index = False)
    #submissions_r2 = pd.DataFrame({'Image':id_list,'Class':resnet2_label})
    #submissions_r2.to_csv("submission_res2.csv",index = False)
    submissions_dpn = pd.DataFrame({'Image':id_list,'Class':dpn_label})
    submissions_dpn.to_csv("submission_dpn.csv",index = False)
    #submissions_resnext = pd.DataFrame({'Image':id_list,'Class':resnext_label})
    #submissions_resnext.to_csv("submission_resnext.csv",index = False)
#   
#   
#     submissions_i = pd.DataFrame({'Image':id_list,'Class':inception_label})
#     submissions_i.to_csv("submission_inception.csv",index = False)


# In[ ]:


generateSubmission()


# In[ ]:


# import os

# #to give np array the correct style
# submission_data = np.array([['dummy', 0]])

# #progress bar
# with tqdm(total=len(os.listdir(test_path))) as pbar:       
#     #test all test images
#     for img in os.listdir(test_path):
#         label = learn.predict(open_image(test_path+img))
#         new_np_array = np.array([[img, label]])
#         submission_data = np.concatenate((submission_data, new_np_array), axis=0)
#         pbar.update(1)

# #remove dummy
# submission_data = np.delete(submission_data, 0, 0)
# submissions = pd.DataFrame(submission_data, columns=['Image','Class'])
# submissions.to_csv("submission.csv",index = False)


# In[ ]:


# generateSubmission(learn)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,15))


# In[ ]:


res.model


# In[ ]:


import matplotlib.pyplot as plt
# Visualising Covolution Layers
layers = res.model
layer_ids = [1,4,7,11,15]
#plot the filters
fig,ax = plt.subplots(nrows=1,ncols=5)
for i in range(5):
    ax[i].imshow(layers[layer_ids[i]].get_weights()[0][:,:,:,0][:,:,0],cmap='gray')
    ax[i].set_title('block'+str(i+1))
    ax[i].set_xticks([])
    ax[i].set_yticks([])


# In[ ]:


#importing the required modules
from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations
from keras import applications
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (18,6)
#creating a VGG16 model using fully connected layers also because then we can 
#visualize the patterns for individual category
from keras.applications import VGG16
model = VGG16(weights='imagenet',include_top=True)

#finding out the layer index using layer name
#the find_layer_idx function accepts the model and name of layer as parameters and return the index of respective layer
layer_idx = utils.find_layer_idx(model,'predictions')
#changing the activation of the layer to linear
model.layers[layer_idx].activation = activations.linear
#applying modifications to the model
model = utils.apply_modifications(model)
#Indian elephant
img3 = visualize_activation(model,layer_idx,filter_indices=385,max_iter=5000,verbose=True)
plt.imshow(img3)


# In[ ]:


import numpy as np

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.activations import relu

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
def iter_occlusion(image, size=8):

    occlusion = np.full((size * 5, size * 5, 1), [0.5], np.float32)
    occlusion_center = np.full((size, size, 1), [0.5], np.float32)
    occlusion_padding = size * 2

    # print('padding...')
    image_padded = np.pad(image, (                         (occlusion_padding, occlusion_padding), (occlusion_padding, occlusion_padding), (0, 0)                         ), 'constant', constant_values = 0.0)

    for y in range(occlusion_padding, image.shape[0] + occlusion_padding, size):

        for x in range(occlusion_padding, image.shape[1] + occlusion_padding, size):
            tmp = image_padded.copy()

            tmp[y - occlusion_padding:y + occlusion_center.shape[0] + occlusion_padding,                 x - occlusion_padding:x + occlusion_center.shape[1] + occlusion_padding]                 = occlusion

            tmp[y:y + occlusion_center.shape[0], x:x + occlusion_center.shape[1]] = occlusion_center

            yield x - occlusion_padding, y - occlusion_padding,                   tmp[occlusion_padding:tmp.shape[0] - occlusion_padding, occlusion_padding:tmp.shape[1] - occlusion_padding]

            
from keras.preprocessing.image import load_img
# load an image from file
image = load_img('car.jpeg', target_size=(224, 224))
plt.imshow(image)
plt.title('ORIGINAL IMAGE')

from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# predict the probability across all output classes
yhat = model.predict(image)
temp = image[0]
print(temp.shape)
heatmap = np.zeros((224,224))
correct_class = np.argmax(yhat)
for n,(x,y,image) in enumerate(iter_occlusion(temp,14)):
    heatmap[x:x+14,y:y+14] = model.predict(image.reshape((1, image.shape[0], image.shape[1], image.shape[2])))[0][correct_class]
    print(x,y,n,' - ',image.shape)
heatmap1 = heatmap/heatmap.max()
plt.imshow(heatmap)

import skimage.io as io
#creating mask from the standardised heatmap probabilities
mask = heatmap1 < 0.85
mask1 = mask *256
mask = mask.astype(int)
io.imshow(mask,cmap='gray')

import cv2
#read the image
image = cv2.imread('car.jpeg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#resize image to appropriate dimensions
image = cv2.resize(image,(224,224))
mask = mask.astype('uint8')
#apply the mask to the image
final = cv2.bitwise_and(image,image,mask = mask)
final = cv2.cvtColor(final,cv2.COLOR_BGR2RGB)
#plot the final image
plt.imshow(final)


# # Saliency Maps

# In[ ]:


# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'predictions')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

#generating saliency map with unguided backprop
grads1 = visualize_saliency(model, layer_idx,filter_indices=None,seed_input=image)
#plotting the unguided saliency map
plt.imshow(grads1,cmap='jet')


# In[ ]:


#generating saliency map with guided backprop
grads2 =  visualize_saliency(model, layer_idx,filter_indices=None,seed_input=image,backprop_modifier='guided')
#plotting the saliency map as heatmap
plt.imshow(grads2,cmap='jet')


# # Layerwise Output

# In[ ]:


#importing required libraries and functions
from keras.models import Model
#defining names of layers from which we will take the output
layer_names = ['block1_conv1','block2_conv1','block3_conv1','block4_conv2']
outputs = []
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#extracting the output and appending to outputs
for layer_name in layer_names:
    intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(image)
    outputs.append(intermediate_output)
#plotting the outputs
fig,ax = plt.subplots(nrows=4,ncols=5,figsize=(20,20))

for i in range(4):
    for z in range(5):
        ax[i][z].imshow(outputs[i][0,:,:,z])
        ax[i][z].set_title(layer_names[i])
        ax[i][z].set_xticks([])
        ax[i][z].set_yticks([])
plt.savefig('layerwise_output.jpg')


# # Class Activation Maps (Gradient Weighted)
# 

# In[ ]:




