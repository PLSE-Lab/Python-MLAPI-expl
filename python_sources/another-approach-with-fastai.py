#!/usr/bin/env python
# coding: utf-8

# 
# # Introduction
# 

# Special thanks to Jeremy for the great class. Lots of ideas in this notebook is from fastai part2 2019.
# 
# In fastai part 1 2019, Lesson 5. We have seen that how Jeremey is wrapped the dataset in the dataloader, and used a signel linear layer nerual network to train MNIST set [Lesson 5: SGD with MNIST](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson5-sgd-mnist.ipynb). This can be easily done using pure pytorch.
# 
# 
# Fastai tranditional routine would be similar to the following:
# 
# 1. Create databunch using datablock API
# This ensures data argumentation, labeling / spilting data.
# 
# 2. Create Leaner with optim and arch, apply transfer learning, cut the model and attach with adapitveconcatpooling layer to maximize the gain from the CNN, then follow by couple linear layer to extract information before loss function (in MNIST the loss will be cross entropy loss, which is softmax + negative log likelyhood)
# Behind the scence, this is all done by the fastai library.
# 
# 3. Train your model using cos scheduler with high lr to have super convergence. 

# # Problem
# However, if one were trying to use transfer learning on MNIST, the first issue is how to clean the data. The most nerual network arch accepting 3 channels (RBG), but for MNIST dataset, it has 784 pixels in a row for single image. 
# 
# Fastai vision datablock API doesn't support this kind of data. It needs image files, one work around will be provide customized itemlist on top of ImageItemList. This can be done in couple line of codes, [FastAI 1.0 with customized Itemlist](https://www.kaggle.com/heye0507/fastai-1-0-with-customized-itemlist)

# # Another approach
# 
# Here I will explain another approach to hybrid pytorch and fastai. This approach has the following advantages:
# 
# 1. The model doesn't tight with fastai API's, one doesn't have to know how to use Datablock API. Therefore you can add fastai to the existing model. 
# 
# 2. Hybrid models, as described in the 2018 part 2, a object detection model can be done with 1 classifier model with another regression model (The classifier is to find right class in the image and the regression model is to work on bounding box coordinates). Therefore, ImageItemList datablock API won't suit for this case
# 
# 3. More control of the model. You can apply different customized layers into existing model, you can replace one layer with another (Later I will show how to use a subRelu() instead fo original Pytorch Relu())
# 
# However, this approach also has some downsides
# 
# 1. One has to know the boundries, since part of the work is done on fastai, part is done on Pytorch. Without proper support of the existing ItemList, some of the API is not going to work, for instance: show_batch(), predict()
# 
# 2. Hard to debug, it requires more time to figure out the root cause of the issus as this approach hybrids two things together. 
# 
# Therefore, this notebook is serverd as demonstration purpose, you can also wrap ItemList behavior to best suit the purpose. 

# # Things we want to take from fastai
# 
# 1. learning rate finder https://arxiv.org/abs/1803.09820
# 2. cos scheduler,discriminative learing rate https://arxiv.org/abs/1506.01186
# 3. fastai callback systems
# 

# # Prepare Data in Pytorch

# In[ ]:


import numpy as np
import pandas as pd
from fastai.callbacks import *
from fastai.callback import *
from fastai.basic_train import *
from fastai.basic_data import *
from fastai.basics import *
from fastai.metrics import *


# In[ ]:


path = Path('../input')
path.ls()


# In[ ]:


df_train = pd.read_csv(path/'train.csv')


# In[ ]:


df_train.head()


# In[ ]:


y = df_train['label'].values
X = df_train.drop(columns='label').values


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_valid,y_train,y_valid = train_test_split(X,y)
x_train.shape, x_valid.shape,y_train.max(),y_train.min()


# In[ ]:


x_train,y_train,x_valid,y_valid = map(tensor,(x_train,y_train,x_valid,y_valid))


# In[ ]:


def normalize_data(train,valid):
    train = train.float()
    valid = valid.float()
    m = train.mean()
    s = train.std()
    return (train - m)/s, (valid - m)/s


# In[ ]:


x_train, x_valid = normalize_data(x_train,x_valid)
x_train.shape,x_valid.shape


# check if the data is correct

# In[ ]:


plt.imshow(x_train[0].view(28,28),cmap='gray')
plt.title(str(y_train[0]))


# # Create databunch using Pytorch

# In[ ]:


train_ds = TensorDataset(x_train,y_train)
valid_ds = TensorDataset(x_valid,y_valid)


# In[ ]:


train_dl = DataLoader(
    dataset=train_ds,
    batch_size=64,
    shuffle=True,
    num_workers=2
)
valid_dl = DataLoader(
    dataset=valid_ds,
    batch_size=128,
    num_workers=2
)


# In[ ]:


data = DataBunch(train_dl,valid_dl)
data.c = len(y_train.unique())


# In[ ]:


plt.imshow(data.train_ds[0][0].view(28,28),cmap='gray')
plt.title(str(data.train_ds[0][1]))


# # Prepare own model

# As discusssed in 2019 fastai part 2, Kaiming indicates that we want to have 0 mean and 1 std for the initialization, however, during class we noticed that if you have 0 mean and std 1, after relu() layer, the less than 0 part are set to 0. The relu() layer will cause the activations to have roughtly 0.5 mean and 1 std, we can fix this by subtract 0.5 after relu()

# In[ ]:


def flatten(x):
    return x.view(x.shape[0],-1)

class Lambda(nn.Module):
    def __init__(self,func): 
        super().__init__()
        self.func = func
        
    def forward(self,xb): return self.func(xb)

class SubRelu(nn.Module):
    def __init__(self,sub=0.4):
        super().__init__()
        self.sub = sub
    
    def forward(self,xb):
        xb = F.relu(xb)
        xb.sub_(self.sub)
        return xb

def subConv2d(ni,nf,ks=3,stride=2):
    return nn.Sequential(nn.Conv2d(ni,nf,ks,padding=ks//2,stride=stride),SubRelu())

def get_subRelu_model():
    model = nn.Sequential(
        subConv2d(1,8),
        subConv2d(8,16),
        subConv2d(16,32),
        subConv2d(32,32),
        nn.AdaptiveAvgPool2d(1),
        Lambda(flatten),
        nn.Linear(32,10),
    )
    return model


# The simple CNN model built here always have 1 conv, 1 subRelu(), I simply pack them together.
# 
# Used Lambda layer to flatten the model, this could also have done using Lambda layer to reshape the input, but this can also be done using fastai callbacks, therefore in the future if we ever need to reshape the input, there is no need to change the model arch, using callbacks to do transformation will be much easier.

# In[ ]:


model = get_subRelu_model()
model


# # Prepare Learner

# Since not applying transfer learning, make sure the model is initialized
# https://arxiv.org/abs/1502.01852

# In[ ]:


def init_model(model):
    for layer in model:
        if isinstance(layer,nn.Sequential):
            nn.init.kaiming_normal_(layer[0].weight)
            layer[0].bias.detach().zero_()


# In[ ]:


model[0][0].bias


# In[ ]:


init_model(model)
model[0][0].bias #check the model is initialized


# Fastai Callback to handle outshape tensor.
# 
# Problem: 
# 
# Have input data shape (64,784), first conv layer has filter size (1,8,5,5). The input data needs to be reshaped to (64,1,28,28)
# 
# Resolve:
# 
# callback to do transformation on every batch. The process here is the following:
# 
# 1. grab a batch
# 2. apply transformation on training
# 
# Notice here, the transformation is done on run time, the data remains unchanged. This can also be applied to use other data argumentaion tenique other than the default fastai library. 
# 
# Also, fastai callbacks are controlled by _order
# 
# It will sort the callback list based on order, so make sure you use order in the right sequence
# Data during callback is packed in dict in fastai, check [fastai callback docs](https://docs.fast.ai/callback.html#Classes-for-callback-implementors)

# In[ ]:


class BatchTransFormXCallback(Callback):
    _order = 2 
    def __init__(self,tfm):
        #super().__init__(learn)
        self.tfm = tfm
    
    def on_batch_begin(self,**kwargs):
        xb = self.tfm(kwargs['last_input'])
        return {'last_input': xb.float()}


# In[ ]:


# wrap learner creation step, for easy re-use during build up this notebook
def get_learner(model):
    opt_func = optim.SGD
    loss_func = nn.CrossEntropyLoss()
    return Learner(data,model.cuda(),opt_func=opt_func,loss_func=loss_func,metrics=accuracy)


# In[ ]:


learn = get_learner(model)


# In[ ]:


learn.callbacks.append(BatchTransFormXCallback(lambda x: x.view(-1,1,28,28)))


# In[ ]:


learn.callback_fns.append(ActivationStats) #fastai build in hook to grab layer activation stats


# Before applying discriminative learnring rate, one have to tell fastai how to split the model to apply different lrs (this are done behind the scence in the fastai supported arch)
# Following the fastai way, we cut the model at the pooling layer, so CNN gets smaller lr, and Linear layer gets larger lr

# In[ ]:


learn.split(lambda m: m[4])


# In[ ]:


#double check the model is split
learn.layer_groups


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


#apply cos sched and discriminative lrs
learn.fit_one_cycle(8,slice(1e-1,4e-1),pct_start=0.3)


# In[ ]:


means = learn.activation_stats.stats[0]
for i in range(4):
    plt.plot(means[i][:800])
plt.legend(range(4))


# In[ ]:


std = learn.activation_stats.stats[1]
for i in range(4):
    plt.plot(std[i][:800])
plt.legend(range(4))


# Initialization seems working fine, in the first few epochs we have 0 mean and 1std. We can also check losses and lrs to see if the model is behave as we expected

# In[ ]:


learn.recorder.plot_lr()


# 30% up time, 70% down time, this ensures the model trains with high lr

# In[ ]:


learn.recorder.plot_losses()


# We can still train longer, but lets add batchnorm layer

# # BatchNorm V.S. Running Norm
# https://arxiv.org/pdf/1502.03167.pdf

# In[ ]:


class BatchNorm_layer(nn.Module):
    def __init__(self,nf,mom=0.1,eps=1e-6):
        super().__init__()
        self.nf = nf
        self.mom = mom
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(nf,1,1))
        self.beta = nn.Parameter(torch.zeros(nf,1,1))
        self.register_buffer('vars',torch.ones(1,nf,1,1))
        self.register_buffer('means',torch.zeros(1,nf,1,1))
    
    def batch_norm(self,xb):
        m = xb.mean(dim=(0,2,3),keepdim=True)
        #var = xb.var(dim=(0,2,3),keepdim=True)
        var = xb.detach().cpu().numpy() #kaggle torch.var only takes int for dim, not tuple
        var = var.var((0,2,3),keepdims=True)
        var = torch.from_numpy(var).cuda()
        self.means.lerp_(m,self.mom)
        self.vars.lerp_(var,self.mom)
        return m,var
    
    def forward(self,xb):
        if self.training:
            with torch.no_grad(): m,v = self.batch_norm(xb)
        else:
            m,v = self.means,self.vars
        xb = (xb - m) / (v+self.eps).sqrt()
        return self.gamma * xb + self.beta


# In[ ]:


class RunningBatchNorm(nn.Module):
    def __init__(self, nf, mom=0.1, eps=1e-5):
        super().__init__()
        self.mom, self.eps = mom, eps
        self.mults = nn.Parameter(torch.ones (nf,1,1))
        self.adds  = nn.Parameter(torch.zeros(nf,1,1))
        self.register_buffer('sums', torch.zeros(1,nf,1,1))
        self.register_buffer('sqrs', torch.zeros(1,nf,1,1))
        self.register_buffer('count', tensor(0.))
        self.register_buffer('factor', tensor(0.))
        self.register_buffer('offset', tensor(0.))
        self.batch = 0
        
    def update_stats(self, x):
        bs,nc,*_ = x.shape
        self.sums.detach_()
        self.sqrs.detach_()
        dims = (0,2,3)
        s    = x    .sum(dims, keepdim=True)
        ss   = (x*x).sum(dims, keepdim=True)
        c    = s.new_tensor(x.numel()/nc)
        mom1 = s.new_tensor(1 - (1-self.mom)/math.sqrt(bs-1))
        self.sums .lerp_(s , mom1)
        self.sqrs .lerp_(ss, mom1)
        self.count.lerp_(c , mom1)
        self.batch += bs
        means = self.sums/self.count
        varns = (self.sqrs/self.count).sub_(means*means)
        if bool(self.batch < 20): varns.clamp_min_(0.01)
        self.factor = self.mults / (varns+self.eps).sqrt()
        self.offset = self.adds - means*self.factor
        
    def forward(self, x):
        if self.training: self.update_stats(x)
        return x*self.factor + self.offset


# This code silightly modifies from fastai 2019 part 2 class. check the [link here](https://github.com/fastai/fastai_docs/blob/master/dev_course/dl2/07_batchnorm.ipynb)

# In[ ]:


def Conv2d_BN(ni,nf,ks=3,stride=2,BN=True):
    if BN:
        return nn.Sequential(nn.Conv2d(ni,nf,ks,padding=ks//2,stride=stride),SubRelu(),BatchNorm_layer(nf))
    else:
        return nn.Sequential(nn.Conv2d(ni,nf,ks,padding=ks//2,stride=stride),SubRelu(),RunningBatchNorm(nf))

def get_batchNorm_model(BN=True):
    model = nn.Sequential(
        #Lambda(lambda x: x.view(-1,1,28,28).float()),
        Conv2d_BN(1,8,BN=BN),
        Conv2d_BN(8,16,BN=BN),
        Conv2d_BN(16,32,BN=BN),
        Conv2d_BN(32,32,BN=BN),
        nn.AdaptiveAvgPool2d(1),
        Lambda(flatten),
        nn.Linear(32,10),
    )
    return model


# In[ ]:


bn_model = get_batchNorm_model()
opt_func = optim.SGD
loss_func = nn.CrossEntropyLoss
learn = Learner(data,bn_model.cuda(),opt_func=opt_func,loss_func=loss_func(),metrics=accuracy)
cb = BatchTransFormXCallback(tfm=lambda x: x.view(-1,1,28,28))
learn.callbacks.append(cb)
learn.callback_fns.append(ActivationStats)
init_model(learn.model)


# In[ ]:


learn.split(lambda m: m[4])


# In[ ]:


learn.fit_one_cycle(12,slice(1e-1,2.),pct_start=0.3)


# In[ ]:


learn.recorder.plot_lr()


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


means = learn.activation_stats.stats[0]
for i in range(4):
    plt.plot(means[i][:800])
plt.legend(range(4))


# In[ ]:


std = learn.activation_stats.stats[1]
for i in range(4):
    plt.plot(std[i][:800])
plt.legend(range(4))


# We peaked the learning rate to 2.0, and the model still converges. 
# 
# Also, from the mean and std we can see BatchNorm really smooth the activations cross all layers, therefore you can train the model in a very high learning rate and it still converges.
# 
# [How Does Batch Normalization Help Optimization](https://arxiv.org/pdf/1805.11604.pdf)
# 
# Now let's try using running batch norm from fastai class

# In[ ]:


bn_model = get_batchNorm_model(BN=False)
opt_func = optim.SGD
loss_func = nn.CrossEntropyLoss
learn = Learner(data,bn_model.cuda(),opt_func=opt_func,loss_func=loss_func(),metrics=accuracy)
cb = BatchTransFormXCallback(tfm=lambda x: x.view(-1,1,28,28))
learn.callbacks.append(cb)
learn.callback_fns.append(ActivationStats)
init_model(learn.model)


# In[ ]:


learn.split(lambda m: m[4])


# In[ ]:


learn.fit_one_cycle(12,slice(1e-1,2.),pct_start=0.3)


# Running Batch Norm seems really wroking well even for larger batch size (64), which Jeremey showed in class that you can use it with small BS, such as BS=2

# # Fin
# Now we just need to use our model to predict. Unfortunately, we didn't supply learner with a proper ItemList, therefore, we can't call learn.predict(). We will just use the model to do prediction, a possible work around would be supply the DataBunch with test_dl,then learn.predict() might work.

# This can be done easily just following Pytorch tutorial
# 
# 1. Wrap test data into Dataloader
# 2. Set the model to eval() mode to prevent updating weights
# 3. Grab data from dataloader, and use the model to predict

# In[ ]:


df_test = pd.read_csv(path/'test.csv')
df_test = df_test.values


# In[ ]:


test_train = torch.from_numpy(X) #just to get m and std
test = torch.from_numpy(df_test)
test.shape


# In[ ]:


_,test = normalize_data(test_train,test)


# Tensor dataset requires pair of input and output, so we will provide a dummy output to wrap the Dataset. 

# In[ ]:


dummy_y = torch.ones(test.shape[0])
dummy_y.shape


# In[ ]:


test_ds = TensorDataset(test,dummy_y)


# In[ ]:


test_dl = DataLoader(
    dataset = test_ds,
    batch_size = 64,
    num_workers = 2
)


# In[ ]:


learn.model.eval()


# In[ ]:


def get_preds(test_dl,model):
    preds = []
    model.cpu()
    for dl in test_dl:
        pred_batch = torch.argmax(model(dl[0].view(-1,1,28,28)),dim=1)
        preds += pred_batch.detach().tolist()
    return preds


# In[ ]:


preds = get_preds(test_dl,learn.model)
len(preds)


# In[ ]:


final = pd.Series(preds,name='Label')
submission = pd.concat([pd.Series(range(1,28001),name='ImageId'),final],axis=1)
submission.to_csv('fastai-pytorch-0.99.csv',index=False)


# In[ ]:




