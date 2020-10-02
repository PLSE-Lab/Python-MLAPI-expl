#!/usr/bin/env python
# coding: utf-8

# **Memory Optimization with Checkpoints**
# 
# With DeepLearning, GPU memory has always been the handicap when it ccomes to handling larger sized images with deeper networks.
# Unless you have access to V100 GPU, you wouldn't need to be worrying about your model in a constrained batch size.
# 
# A typical flow of a backprogogation computiational graph can be represented by this figure.
# 
# ![](https://miro.medium.com/max/676/0*NARheCDvdoPc4A8z.)
# 
# 
# To optimize the memory usage, the idea is to free up the memory, by storing away the computations. This will enable reusage of the saved computation, rather than recalculating it everytime.
# 
# ![](https://miro.medium.com/max/676/0*udMSiPD0kZHum-sZ.)
# 
# 
# This is where the concept of checkpointing compes into play. Rather store or forget all the computations, we identify strategic nodes, to save, such that the memory usage is optimized.
# 
# ![](https://miro.medium.com/max/1355/0*VEYowymIqvNc2HzB.)

# **Visual Flow of a Computational Graph with checkpoint**
# 
# ![](https://miro.medium.com/max/676/0*s7U1QDfSXuVd1LrF.)

# Now to the code, the usual workflow for classifiying dogs, from lesson 1

# In[ ]:


from fastai import *
from fastai.vision import *

path = untar_data(URLs.PETS)
path_anno = path/'annotations'
path_img = path/'images'
fnames = get_image_files(path_img)
np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'


# **Training: resnet101 - No Checkpoint**

# In[ ]:


bs = 16*5


# *Note*: With bs = 16 * 6,  Kernel would fail with OOM on default workflow. 
# 

# In[ ]:


data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet101, metrics=error_rate)


# The Sequential layers starting from 5th position (containing multiple Basic Blocks) are the ones we are interested in.

# In[ ]:


learn.model


# In[ ]:


from fastai.utils.mem import GPUMemTrace
with GPUMemTrace():
  learn.fit_one_cycle(4)


# **Logic of the Process**
# 
# The original optimized code [for resnet](https://github.com/prigoyal/pytorch_memonger/blob/master/models/optimized/resnet_new.py) from *Priya Goyal* had only implementations for 'resnet1001'. The reason assumed is that the layers is not defined in the forward block, unlike reset 18 through 152.
# 
# However, this [implementation](https://github.com/eladhoffer/convNet.pytorch/blob/master/models/resnet.py) from *Elad Hoffer* overcame this by defining model at a feature method of class 'Resnet' module, and then inheriting this in class 'ResNet_imagenet' where layers are defined and checkpointed.
# 
# With FastAI, you have infinitesimal customization options, where one does not has to go to that level of code changes from the base.
# 
# We've used the [CheckpointModule](https://github.com/eladhoffer/convNet.pytorch/blob/master/models/modules/checkpoint.py) from the repository and repurposed to fit with FastAI code.

# **Training: resnet101 - with Checkpoint**

# The only change done with respect to the base code is on 'create_body1' where in instead returning the default **nn.Sequential(*list(model.children())[:cut]) ** , we are refactoring the sequential layers with **CheckpointModule**.
# 
# Everything else is same from current FastAI code repo.

# In[ ]:


########################################
## Defaults
########################################
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

from fastai.callbacks.hooks import *

def cnn_config(arch):
    "Get the metadata associated with `arch`."
    torch.backends.cudnn.benchmark = True
    return model_meta.get(arch, _default_meta)

def _default_split(m:nn.Module): return (m[1],)
def _resnet_split(m:nn.Module): return (m[0][6],m[1])

_default_meta    = {'cut':None, 'split':_default_split}
_resnet_meta     = {'cut':-2, 'split':_resnet_split }

model_meta = {
    models.resnet18 :{**_resnet_meta}, models.resnet34: {**_resnet_meta},
    models.resnet50 :{**_resnet_meta}, models.resnet101:{**_resnet_meta},
    models.resnet152:{**_resnet_meta}}


# In[ ]:


########################################
## Custom Checkpoint
########################################
class CheckpointModule(nn.Module):
    def __init__(self, module, num_segments=1):
        super(CheckpointModule, self).__init__()
        assert num_segments == 1 or isinstance(module, nn.Sequential)
        self.module = module
        self.num_segments = num_segments

    def forward(self, *inputs):
        if self.num_segments > 1:
            return checkpoint_sequential(self.module, self.num_segments, *inputs)
        else:
            return checkpoint(self.module, *inputs)

########################################
# Extract the sequential layers for resnet
########################################
def layer_config(arch):
    "Get the layers associated with `arch`."
    return model_layers.get(arch)

model_layers = {
    models.resnet18 :[2, 2, 2, 2], models.resnet34: [3, 4, 6, 3],
    models.resnet50 :[3, 4, 6, 3], models.resnet101:[3, 4, 23, 3],
    models.resnet152:[3, 8, 36, 3]}


# In[ ]:


########################################
## Send sequential layers in custom_body to Checkpoint
########################################
def create_body1(arch:Callable, pretrained:bool=True, cut:Optional[Union[int, Callable]]=None):
    "Cut off the body of a typically pretrained `model` at `cut` (int) or cut the model as specified by `cut(model)` (function)."
    model = arch(pretrained)
    cut = ifnone(cut, cnn_config(arch)['cut'])
    dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if   isinstance(cut, int):
    #Checkpoint - Changes Start
      if (arch.__name__).find("resnet")==0:       # Check if the Model is resnet                                                        
        n = 4                                     # Initial 4 Layers didn't have sequential and were not applicable with Checkpoint
        layers = layer_config(arch)               # Fetch the sequential layer split
        out = nn.Sequential(*list(model.children())[:cut][:n],
                            *[CheckpointModule(x, min(checkpoint_segments, layers[i])) for i, x in enumerate(list(model.children())[:cut][n:])])
        # Join the Initial 4 layers with Checkpointed sequential layers
      else:
        out = nn.Sequential(*list(model.children())[:cut])
      return out
    #Checkpoint - Changes End
    elif isinstance(cut, Callable): return cut(model)
    else:                           raise NamedError("cut must be either integer or a function")


# In[ ]:


## From base - function renamed
def create_head1(nf:int, nc:int, lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5,
                concat_pool:bool=True, bn_final:bool=False):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes."
    lin_ftrs = [nf, 512, nc] if lin_ftrs is None else [nf] + lin_ftrs + [nc]
    ps = listify(ps)
    if len(ps) == 1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
    layers = [pool, Flatten()]
    for ni,no,p,actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += bn_drop_lin(ni, no, True, p, actn)
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)

## From base - function renamed
def create_cnn1_model1(base_arch:Callable, nc:int, cut:Union[int,Callable]=None, pretrained:bool=True,
                     lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5, custom_head:Optional[nn.Module]=None,
                     bn_final:bool=False, concat_pool:bool=True):
    "Create custom convnet architecture"
    body = create_body1(base_arch, pretrained, cut)
    if custom_head is None:
        nf = num_features_model(nn.Sequential(*body.children())) * (2 if concat_pool else 1)
        head = create_head1(nf, nc, lin_ftrs, ps=ps, concat_pool=concat_pool, bn_final=bn_final)
    else: head = custom_head
    return nn.Sequential(body, head)

## From base - function renamed
def cnn_learner1(data:DataBunch, base_arch:Callable, cut:Union[int,Callable]=None, pretrained:bool=True,
                lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5, custom_head:Optional[nn.Module]=None,
                split_on:Optional[SplitFuncOrIdxList]=None, bn_final:bool=False, init=nn.init.kaiming_normal_,
                concat_pool:bool=True, **kwargs:Any)->Learner:
    "Build convnet style learner."
    meta = cnn_config(base_arch)
    model = create_cnn1_model1(base_arch, data.c, cut, pretrained, lin_ftrs, ps=ps, custom_head=custom_head,
        bn_final=bn_final, concat_pool=concat_pool)
    learn = Learner(data, model, **kwargs)
    learn.split(split_on or meta['split'])
    if pretrained: learn.freeze()
    if init: apply_init(model[1], init)
    return learn


# In[ ]:


## Clear redundant Memory
gc.collect()
import torch
torch.cuda.empty_cache()
learn.purge()
del data
del learn


# At times the kernel was able to execute 64 x 12. But to be at safer side, it was decided to place the batchsize 64 x 10 to prevent Kernel OOM

# In[ ]:


bs = bs * 2
checkpoint_segments = 4
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs).normalize(imagenet_stats)
learn = cnn_learner1(data, models.resnet101, metrics=error_rate)


# Notice all the Sequential layer after 4th Layer - MaxPool2d are now of type CheckPoint Module. And since we are not changing the architecture as such, pre-trained weights can be used as we usually do.

# In[ ]:


learn.model


# In[ ]:


from fastai.utils.mem import GPUMemTrace
with GPUMemTrace():
  learn.fit_one_cycle(4)


# There is still a warning with regards to the checkpoint usage -*UserWarning: None of the inputs have requires_grad=True. Gradients will be None*, which is discussed [here](https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/7). This will be updated in forth coming version.

# If you are constrained on only using Google Colab/Kaggle/in-house-limited memory for your experiments, this little trick might just help you.
# 
# **Please Upvote if you liked the kernel**

# **References:**
# * https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9 - Fitting Larger Networks in Memory
# * https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb - Original source depiciting benefits and usage of checkpointing
# * https://github.com/eladhoffer/convNet.pytorch/blob/master/models/modules/checkpoint.py - Code for checkpoint module
# * https://github.com/eladhoffer/convNet.pytorch/blob/master/models/resnet.py - Variation of base Resnet Model to enable Checkpointing
# * https://github.com/pnvijay/fastaiv3/blob/master/ConvLearner_Lesson1_Fastaiv3.ipynb - Nice explanation on internals of FastAI CNN Learner
# * https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11 - To overcome "UserWarning: None of the inputs have requires_grad=True. Gradients will be None"
