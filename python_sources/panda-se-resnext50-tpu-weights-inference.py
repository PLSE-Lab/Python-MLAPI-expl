#!/usr/bin/env python
# coding: utf-8

# In this kernel, I finetune seresnext50 pretrained model on PANDA dataset. I used the tiles approach from [this kernel](http://https://www.kaggle.com/iafoss/panda-concat-tile-pooling-starter-0-79-lb) and learnt how to use TPU from [this kernel ](http://https://www.kaggle.com/tarunpaparaju/panda-challenge-resnet-multitask-8-fold-on-tpu). If you like the kernel please upvote this kernel and the above kernels as well.  

# In[ ]:


get_ipython().system('pip install ../input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/ > /dev/null')


# # Import Libraries

# In[ ]:


import sys
import os
import numpy as np
import pandas as pd 
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import GroupNorm
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
import torchvision.models as models
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from tqdm.notebook import tqdm
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import pretrainedmodels
import cv2
import skimage.io


# In[ ]:


IMG_DIR = '/kaggle/input/prostate-cancer-grade-assessment/test_images/'
DATA_DIR='/kaggle/input/prostate-cancer-grade-assessment/'
WEIGHTS_DIR = '/kaggle/input/tpu-weights-to-cpu/'

num_classes=6
N=12
sz=128
files=['SE_RNXT50_loss_1.pth','SE_RNXT50_loss_2.pth','SE_RNXT50_loss_3.pth','SE_RNXT50_loss_4.pth']
num_models=len(files)
device='cuda' if torch.cuda.is_available() else 'cpu'
seed=42

arch = pretrainedmodels.__dict__['se_resnext50_32x4d']
test = pd.read_csv(DATA_DIR+'test.csv')
sub=pd.read_csv(DATA_DIR+'sample_submission.csv')


# In[ ]:


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)


# In[ ]:


from torch.nn.parameter import Parameter

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
    
    
class Conv2d_ws(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(nn.Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,padding, dilation, bias=True ,padding_mode='zeros',
                                       groups=1, output_padding='zeros', transposed=False)



    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    
import math
def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_str = int(math.floor(previous_conv_size[0] / out_pool_size[i]))
        w_str = int(math.floor(previous_conv_size[1] / out_pool_size[i]))        
        h_pad = int((h_wid*out_pool_size[i] - previous_conv_size[0]+1)/2)
        w_pad = int((w_wid*out_pool_size[i] - previous_conv_size[1]+1)/2)
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_str, w_str), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if(i == 0):
            spp = x.view(num_sample,-1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
    return spp

class SPP(nn.Module):
    def __init__(self, n, shape, out):
        super(SPP,self).__init__()
        
        self.batch = n
        self.shape = shape
        self.out = out
        
    def forward(self, x):
        return spatial_pyramid_pool(x, self.batch, self.shape, self.out)
    
    
    
def convert_to_gem(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.AdaptiveAvgPool2d):
            setattr(model, child_name, GeM())
        else:
            convert_to_gem(child)
            
def convert_to_conv2d(model):
    for child_name, child in model.named_children():
        if child_name not in ['fc1','fc2']:
            if isinstance(child, nn.Conv2d):
                in_feat = child.in_channels
                out_feat = child.out_channels
                ker_size = child.kernel_size
                stride = child.stride
                padding = child.padding
                dilation = child.dilation
                groups = child.groups
                setattr(model, child_name, Conv2d_ws(in_channels=in_feat, out_channels=out_feat, kernel_size=ker_size, stride=stride,padding = padding, dilation=dilation, groups=groups))
            else:
                convert_to_conv2d(child)
                
def convert_to_groupnorm(model):
    for child_name, child in model.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_features = child.num_features
                setattr(model, child_name, GroupNorm(num_groups=32, num_channels=num_features))
            else:
                convert_to_groupnorm(child)
                
def convert_to_evonorm(model):
    for child_name, child in model.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_features = child.num_features
                setattr(model, child_name, EvoNorm2D(num_features))
            else:
                convert_to_evonorm(child)
                
                
def convert_to_identity(model):
    for child_name, child in model.named_children():
            if isinstance(child, nn.ReLU):
                setattr(model, child_name, nn.Identity())
            else:
                convert_to_identity(child)


# In[ ]:


class PANDA_MODEL(nn.Module):
    def __init__(self, pretrained=False, classes=num_classes, conv_ws=False):
        super(PANDA_MODEL, self).__init__()
        
        m = arch(pretrained='imagenet') if pretrained else arch(pretrained=None)
        in_feat = m.last_linear.in_features
        self.base = nn.Sequential(*list(m.children())[:-2]) 
        self.gem = GeM()
        self.linear1 = nn.Linear(in_features=in_feat, out_features= 512)
        self.relu1=nn.ReLU()
        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512,classes)
        
        if conv_ws:
            convert_to_conv2d(self.base)       
            convert_to_groupnorm(self.base)
            
    def forward(self, x):
        n=len(x)
        x=self.base(x)
        x=self.gem(x)  
        x=x.view(n,-1) 
        x=self.linear1(x)
        x=self.bn(x)
        x=self.relu1(x)
        x=self.dropout(x)
                
        out=self.linear2(x)
        
        return out


# In[ ]:


def tile(img):
    result = []
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=255)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    for i in range(len(img)):
        result.append({'img':img[i], 'idx':i})
    return result


# In[ ]:


class PANDA(Dataset):
    def __init__(self,df, transform, mode='train'):
        self.df = df['image_id'].values
        self.transform=transform
        self.mode=mode
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df[idx]
        tiles = tile(skimage.io.MultiImage(IMG_DIR+img_id+'.tiff')[-1])
        
        img=cv2.hconcat([cv2.vconcat([tiles[0]['img'], tiles[1]['img'], tiles[2]['img'], tiles[3]['img']]),
                        cv2.vconcat([tiles[4]['img'], tiles[5]['img'], tiles[6]['img'], tiles[7]['img']]),
                        cv2.vconcat([tiles[8]['img'], tiles[9]['img'], tiles[10]['img'], tiles[11]['img']])])
        
#         img=cv2.addWeighted( img,4, cv2.GaussianBlur( img , (0,0), sigmaX) ,-4 ,128)
        
        img = 255-img   # Very important. Background was 255. Now, background is zero. 

        if self.transform:
            img = self.transform(image=img)['image']
            
        
        if self.mode!='test':    
            label = self.df['isup_grade'][idx]
            
        img = img.transpose(2,0,1)
    
        if self.mode!='test':
            return {'image': torch.tensor(img, dtype=torch.float),
                'provider': provider,
               'label': torch.tensor(label, dtype=torch.long)}
        
        else:
            return {'image': torch.tensor(img, dtype=torch.float),
                    'img_id': img_id}            


# In[ ]:


def load_models(weight):
    model=PANDA_MODEL()
    model.load_state_dict(torch.load(WEIGHTS_DIR+weight))
    model.to(device)
    model.eval()
    return model

all_models=[load_models(files[i]) for i in range(num_models)]


# In[ ]:


test_tfm = A.Compose([A.Normalize(mean=[1-0.90949707,1-0.8188697,1-0.87795304],
                                   std=[0.36357649,0.49984502,0.40477625])])

test_dataset = PANDA(test, test_tfm, mode='test')
test_loader = DataLoader(test_dataset, batch_size=2, sampler=SequentialSampler(test_dataset))


# In[ ]:


if os.path.exists(IMG_DIR):
    all_preds=[]
    img_ids = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            img=batch['image'].to(device)
            img_id = batch['img_id']
            img_ids.append(img_id)
            pred_array=np.zeros((len(img),6))
    
            for model in all_models:
                prob=model(img)
                pred_array+=prob.detach().cpu().numpy()/num_models
        
            pred = np.argmax(pred_array,1)
            all_preds.append(pred)
            
    all_preds=np.concatenate(all_preds)
    img_ids = np.concatenate(img_ids)
    sub = pd.DataFrame()
    sub['image_id']=img_ids
    sub['isup_grade']=all_preds
    sub.to_csv('submission.csv', index=False)
else:
    sub.to_csv('submission.csv', index=False)


# In[ ]:


sub.head()

