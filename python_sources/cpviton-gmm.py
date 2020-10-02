#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os.path as osp
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision import utils
from PIL import ImageDraw
import time
import json
import random
import cv2
from matplotlib import pylab as plt
import torch.nn.functional as F
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


class mydataset(Dataset):
    def __init__(self,mode='train'):
        self.mode=mode
        self.size = (256, 192)
        self.transform = transforms.Compose([ 
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform_1d = transforms.Compose([ 
                transforms.ToTensor(), 
                transforms.Normalize((0.5,), (0.5,))])
        pair_list =[i.strip() for i in open('/kaggle/input/mvpdataset/mvp/data_pair2.txt', 'r').readlines()]
        if(self.mode=='train'):
            train_list = list(filter(lambda p: p.split('\t')[3] == 'train', pair_list))
            train_list = [i for i in train_list]
            self.img_list=train_list
        else:
            test_list = list(filter(lambda p: p.split('\t')[3] == 'test', pair_list))
            test_list = [i for i in test_list]
            self.img_list=test_list
    def __getitem__(self, index):
        img_source = self.img_list[index].split('\t')[0]
        img_target = self.img_list[index].split('\t')[1]
        cloth_img = self.img_list[index].split('\t')[2]

        source_splitext = os.path.splitext(img_source)[0]
        target_splitext = os.path.splitext(img_target)[0]
        cloth_splitext = os.path.splitext(cloth_img)[0]

        # png or jpg
        source_img_path = os.path.join('/kaggle/input/mvpdataset/mvp/all/', source_splitext + '.jpg')
        target_img_path = os.path.join('/kaggle/input/mvpdataset/mvp/all/', target_splitext + '.jpg')
        cloth_img_path = os.path.join('/kaggle/input/mvpdataset/mvp/all/', cloth_img)
        cloth_mask_path = os.path.join('/kaggle/input/mvpdataset/mvp/all/', cloth_splitext + '_mask.jpg')
        
        ### image
        source_img = Image.open(source_img_path)
        target_img = Image.open(target_img_path)
        cloth_img = Image.open(cloth_img_path)
        cloth_mask=Image.open(cloth_mask_path)
        source_img = self.transform(source_img)
        target_img = self.transform(target_img)
        cloth_img = self.transform(cloth_img)
        cloth_mask=self.transform_1d(cloth_mask)

        # parsing
        source_parse_path = os.path.join('/kaggle/input/mvpdataset/mvp/all_parsing/', source_splitext + '.png')
        target_parse_path = os.path.join('/kaggle/input/mvpdataset/mvp/all_parsing/', target_splitext + '.png')

        source_parse_shape = np.array(Image.open(source_parse_path))
        source_parse_shape = (source_parse_shape > 0).astype(np.float32)
        source_parse_shape = Image.fromarray((source_parse_shape*255).astype(np.uint8))
        source_parse_shape = source_parse_shape.resize((self.size[1]//16, self.size[0]//16), Image.BILINEAR) # downsample and then upsample
        source_parse_shape = source_parse_shape.resize((self.size[1], self.size[0]), Image.BILINEAR)
        source_parse_shape = self.transform_1d(source_parse_shape)

        target_parse_shape = np.array(Image.open(target_parse_path))
        target_parse_shape = (target_parse_shape > 0).astype(np.float32)
        target_parse_shape = Image.fromarray((target_parse_shape*255).astype(np.uint8))
        target_parse_shape = target_parse_shape.resize((self.size[1]//16, self.size[0]//16), Image.BILINEAR) # downsample and then upsample
        target_parse_shape = target_parse_shape.resize((self.size[1], self.size[0]), Image.BILINEAR)
        target_parse_shape = self.transform_1d(target_parse_shape)

        source_parse_head = (np.array(Image.open(source_parse_path)) == 1).astype(np.float32) +                     (np.array(Image.open(source_parse_path)) == 2).astype(np.float32) +                     (np.array(Image.open(source_parse_path)) == 4).astype(np.float32) +                     (np.array(Image.open(source_parse_path)) == 13).astype(np.float32)

        target_parse_head = (np.array(Image.open(target_parse_path)) == 1).astype(np.float32) +                     (np.array(Image.open(target_parse_path)) == 2).astype(np.float32) +                     (np.array(Image.open(target_parse_path)) == 4).astype(np.float32) +                     (np.array(Image.open(target_parse_path)) == 13).astype(np.float32)

        source_parse_cloth = (np.array(Image.open(source_parse_path)) == 5).astype(np.float32) +                 (np.array(Image.open(source_parse_path)) == 6).astype(np.float32) +                 (np.array(Image.open(source_parse_path)) == 7).astype(np.float32)

        target_parse_cloth = (np.array(Image.open(target_parse_path)) == 5).astype(np.float32) +                 (np.array(Image.open(target_parse_path)) == 6).astype(np.float32) +                 (np.array(Image.open(target_parse_path)) == 7).astype(np.float32)

        # prepare for warped cloth //target
        phead = torch.from_numpy(target_parse_head) # [0,1]
        pcm = torch.from_numpy(target_parse_cloth) # [0,1]
        im = target_img # [-1,1]
        im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts --> white same as GT ...
        im_h = target_img * phead - (1 - phead) # [-1,1], fill -1 for other parts, thus become black visual

        # pose heatmap embedding
        source_pose_path = os.path.join('/kaggle/input/mvpdataset/mvp/all_person_clothes_keypoints/', source_splitext +'_keypoints.json')
        with open(source_pose_path, 'r') as f:
            source_pose_label = json.load(f)
            source_pose_data = source_pose_label['people'][0]['pose_keypoints']
            source_pose_data = np.array(source_pose_data)
            source_pose_data = source_pose_data.reshape((-1,3))
        source_point_num = source_pose_data.shape[0]
        source_pose_map = torch.zeros(source_point_num, 256, 192)
        r = 3
        source_im_pose = Image.new('L', (192, 256))
        source_pose_draw = ImageDraw.Draw(source_im_pose)
        for i in range(source_point_num):
            source_one_map = Image.new('L', (192, 256))
            source_draw = ImageDraw.Draw(source_one_map)
            source_pointx = source_pose_data[i,0]
            source_pointy = source_pose_data[i,1]
            if source_pointx > 1 and source_pointy > 1:
                source_draw.rectangle((source_pointx-r, source_pointy-r, source_pointx+r, source_pointy+r), 'white', 'white')
                source_pose_draw.rectangle((source_pointx-r, source_pointy-r, source_pointx+r, source_pointy+r), 'white', 'white')
            source_one_map = self.transform_1d(source_one_map)
            source_pose_map[i] = source_one_map[0]
        
        source_agnostic = torch.cat([source_parse_shape, im_h, source_pose_map], 0)

        # pose heatmap embedding
        target_pose_path = os.path.join('/kaggle/input/mvpdataset/mvp/all_person_clothes_keypoints/', target_splitext +'_keypoints.json')
        with open(target_pose_path, 'r') as f:
            target_pose_label = json.load(f)
            target_pose_data = target_pose_label['people'][0]['pose_keypoints']
            target_pose_data = np.array(target_pose_data)
            target_pose_data = target_pose_data.reshape((-1,3))
        target_point_num = target_pose_data.shape[0]
        target_pose_map = torch.zeros(target_point_num, 256, 192)
        r = 3
        target_im_pose = Image.new('L', (192, 256))
        target_pose_draw = ImageDraw.Draw(target_im_pose)
        for i in range(target_point_num):
            target_one_map = Image.new('L', (192, 256))
            target_draw = ImageDraw.Draw(target_one_map)
            target_pointx = target_pose_data[i,0]
            target_pointy = target_pose_data[i,1]
            if target_pointx > 1 and target_pointy > 1:
                target_draw.rectangle((target_pointx-r, target_pointy-r, target_pointx+r, target_pointy+r), 'white', 'white')
                target_pose_draw.rectangle((target_pointx-r, target_pointy-r, target_pointx+r, target_pointy+r), 'white', 'white')
            target_one_map = self.transform_1d(target_one_map)
            target_pose_map[i] = target_one_map[0]

        target_agnostic = torch.cat([target_parse_shape, im_h, target_pose_map], 0)

        result = {
                'source_image': source_img, 
                'target_image': target_img,
                'cloth_image': cloth_img,
                'im_h': im_h, # source image head and hair
                'im_c': im_c, # target_cloth_image_warped
                'source_parse_shape': source_parse_shape,
                'target_parse_shape':target_parse_shape,  
                'source_pose': source_pose_map,  
                'target_pose': target_pose_map,   
                'target_agnostic':target_agnostic,
                'source_agnostic':source_agnostic,
                'mask_image':cloth_mask,
                'mask_label':pcm.unsqueeze_(0),
        }
        return result

    def __len__(self):
        return len(self.img_list)


# In[ ]:


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
dataset=mydataset(mode='train')
print(len(dataset))


# In[ ]:


# fig=plt.subplots(10,10,figsize=(20,20))
# for i in range(100):
#     data=dataset.__getitem__(i)["target_image"]
#     plt.subplot(10,10,i+1)
#     plt.axis('off')
#     plt.imshow(np.transpose(data*0.5+0.5,(1,2,0)))


# In[ ]:


data0=dataset.__getitem__(67)["source_image"].unsqueeze(0)
data1=dataset.__getitem__(67)["target_image"].unsqueeze(0)
data2=dataset.__getitem__(67)["cloth_image"].unsqueeze(0)
data3=dataset.__getitem__(67)["im_h"].unsqueeze(0)
data4=dataset.__getitem__(67)["im_c"].unsqueeze(0)
data5=dataset.__getitem__(67)["source_parse_shape"].unsqueeze(0)
data6=dataset.__getitem__(67)["target_parse_shape"].unsqueeze(0)
data7=dataset.__getitem__(67)['source_pose'].unsqueeze(0)
data8=dataset.__getitem__(67)['target_pose'].unsqueeze(0)
data9=dataset.__getitem__(67)['source_agnostic'].unsqueeze(0)
data10=dataset.__getitem__(67)['target_agnostic'].unsqueeze(0)
data11=dataset.__getitem__(67)["mask_image"].unsqueeze(0)
data12=dataset.__getitem__(67)['mask_label'].unsqueeze(0)
plt.figure(figsize=(12,12))
plt.subplot(4,5,1)
plt.axis('off')
plt.imshow(np.transpose(data0[0]*0.5+0.5,(1,2,0)))
plt.subplot(4,5,2)
plt.axis('off')
plt.imshow(np.transpose(data1[0]*0.5+0.5,(1,2,0)))
plt.subplot(4,5,3)
plt.axis('off')
plt.imshow(np.transpose(data2[0]*0.5+0.5,(1,2,0)))
plt.subplot(4,5,4)
plt.axis('off')
plt.imshow(np.transpose(data3[0]*0.5+0.5,(1,2,0)))
plt.subplot(4,5,5)
plt.axis('off')
plt.imshow(np.transpose(data4[0]*0.5+0.5,(1,2,0)))

plt.subplot(4,5,6)
plt.axis('off')
plt.imshow(data5[0].squeeze(0),cmap='gray')
plt.subplot(4,5,7)
plt.axis('off')
plt.imshow(data6[0].squeeze(0),cmap='gray')

a=torch.zeros(256,192)
for i in range(18):
    a+=np.transpose(data7[0],(1,2,0))[:,:,i]
plt.subplot(4,5,8)
plt.axis('off')
plt.imshow(a,cmap='gray')

b=torch.zeros(256,192)
for i in range(18):
    b+=np.transpose(data8[0],(1,2,0))[:,:,i]
plt.subplot(4,5,9)
plt.axis('off')
plt.imshow(b,cmap='gray')


c=torch.zeros(256,192)
for i in range(22):
    c+=np.transpose(data9[0],(1,2,0))[:,:,i]
plt.subplot(4,5,10)
plt.axis('off')
plt.imshow(c,cmap='gray')


d=torch.zeros(256,192)
for i in range(22):
    d+=np.transpose(data10[0],(1,2,0))[:,:,i]
plt.subplot(4,5,11)
plt.axis('off')
plt.imshow(d,cmap='gray')

plt.subplot(4,5,12)
plt.axis('off')
plt.imshow(data11[0].squeeze(0),cmap='gray')
plt.subplot(4,5,13)
plt.axis('off')
plt.imshow(data12[0].squeeze(0),cmap='gray')


# In[ ]:


data01=dataset.__getitem__(134)["source_image"].unsqueeze(0)
data11=dataset.__getitem__(134)["target_image"].unsqueeze(0)
data21=dataset.__getitem__(134)["cloth_image"].unsqueeze(0)
data31=dataset.__getitem__(134)["im_h"].unsqueeze(0)
data41=dataset.__getitem__(134)["im_c"].unsqueeze(0)
data51=dataset.__getitem__(134)["source_parse_shape"].unsqueeze(0)
data61=dataset.__getitem__(134)["target_parse_shape"].unsqueeze(0)
data71=dataset.__getitem__(134)['source_pose'].unsqueeze(0)
data81=dataset.__getitem__(134)['target_pose'].unsqueeze(0)
data91=dataset.__getitem__(134)['source_agnostic'].unsqueeze(0)
data101=dataset.__getitem__(134)['target_agnostic'].unsqueeze(0)
data111=dataset.__getitem__(134)["mask_image"].unsqueeze(0)
data121=dataset.__getitem__(134)['mask_label'].unsqueeze(0)
plt.figure(figsize=(12,12))
plt.subplot(4,5,1)
plt.axis('off')
plt.imshow(np.transpose(data01[0]*0.5+0.5,(1,2,0)))
plt.subplot(4,5,2)
plt.axis('off')
plt.imshow(np.transpose(data11[0]*0.5+0.5,(1,2,0)))
plt.subplot(4,5,3)
plt.axis('off')
plt.imshow(np.transpose(data21[0]*0.5+0.5,(1,2,0)))
plt.subplot(4,5,4)
plt.axis('off')
plt.imshow(np.transpose(data31[0]*0.5+0.5,(1,2,0)))
plt.subplot(4,5,5)
plt.axis('off')
plt.imshow(np.transpose(data41[0]*0.5+0.5,(1,2,0)))

plt.subplot(4,5,6)
plt.axis('off')
plt.imshow(data51[0].squeeze(0),cmap='gray')
plt.subplot(4,5,7)
plt.axis('off')
plt.imshow(data61[0].squeeze(0),cmap='gray')

a=torch.zeros(256,192)
for i in range(18):
    a+=np.transpose(data71[0],(1,2,0))[:,:,i]
plt.subplot(4,5,8)
plt.axis('off')
plt.imshow(a,cmap='gray')

b=torch.zeros(256,192)
for i in range(18):
    b+=np.transpose(data81[0],(1,2,0))[:,:,i]
plt.subplot(4,5,9)
plt.axis('off')
plt.imshow(b,cmap='gray')


c=torch.zeros(256,192)
for i in range(22):
    c+=np.transpose(data91[0],(1,2,0))[:,:,i]
plt.subplot(4,5,10)
plt.axis('off')
plt.imshow(c,cmap='gray')


d=torch.zeros(256,192)
for i in range(22):
    d+=np.transpose(data101[0],(1,2,0))[:,:,i]
plt.subplot(4,5,11)
plt.axis('off')
plt.imshow(d,cmap='gray')

plt.subplot(4,5,12)
plt.axis('off')
plt.imshow(data111[0].squeeze(0),cmap='gray')
plt.subplot(4,5,13)
plt.axis('off')
plt.imshow(data121[0].squeeze(0),cmap='gray')


# In[ ]:


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=1,
    drop_last=True
)


# In[ ]:


#coding=utf-8
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import os
import torch.nn.functional as F
import numpy as np

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

class FeatureExtraction(nn.Module):
    def __init__(self, input_nc, ngf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(FeatureExtraction, self).__init__()
        downconv = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        model = [downconv, nn.ReLU(True), norm_layer(ngf)]
        for i in range(n_layers):
            in_ngf = 2**i * ngf if 2**i * ngf < 512 else 512
            out_ngf = 2**(i+1) * ngf if 2**i * ngf < 512 else 512
            downconv = nn.Conv2d(in_ngf, out_ngf, kernel_size=4, stride=2, padding=1)
            model += [downconv, nn.ReLU(True)]
            model += [norm_layer(out_ngf)]
        model += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(True)]
        model += [norm_layer(512)]
        model += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(True)]
        
        self.model = nn.Sequential(*model)
        init_weights(self.model, init_type='normal')

    def forward(self, x):
        return self.model(x)

class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)
    
class FeatureCorrelation(nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()
    
    def forward(self, feature_A, feature_B):
        b,c,h,w = feature_A.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
        feature_B = feature_B.view(b,c,h*w).transpose(1,2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B,feature_A)
        correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
        return correlation_tensor
    
class FeatureRegression(nn.Module):
    def __init__(self, input_nc=512,output_dim=6, use_cuda=True):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(64 * 4 * 3, output_dim)
        self.tanh = nn.Tanh()
        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()
            self.tanh.cuda()

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        x = self.tanh(x)
        return x

class AffineGridGen(nn.Module):
    def __init__(self, out_h=256, out_w=192, out_ch = 3):
        super(AffineGridGen, self).__init__()        
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch
        
    def forward(self, theta):
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size,self.out_ch,self.out_h,self.out_w))
        return F.affine_grid(theta, out_size)
        
class TpsGridGen(nn.Module):
    def __init__(self, out_h=256, out_w=192, use_regular_grid=True, grid_size=3, reg_factor=0, use_cuda=True):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda

        # create grid in numpy
        self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X,self.grid_Y = np.meshgrid(np.linspace(-1,1,out_w),np.linspace(-1,1,out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1,1,grid_size)
            self.N = grid_size*grid_size
            P_Y,P_X = np.meshgrid(axis_coords,axis_coords)
            P_X = np.reshape(P_X,(-1,1)) # size (N,1)
            P_Y = np.reshape(P_Y,(-1,1)) # size (N,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.P_X_base = P_X.clone()
            self.P_Y_base = P_Y.clone()
            self.Li = self.compute_L_inverse(P_X,P_Y).unsqueeze(0)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
            if use_cuda:
                self.P_X = self.P_X.cuda()
                self.P_Y = self.P_Y.cuda()
                self.P_X_base = self.P_X_base.cuda()
                self.P_Y_base = self.P_Y_base.cuda()

            
    def forward(self, theta):
        warped_grid = self.apply_transformation(theta,torch.cat((self.grid_X,self.grid_Y),3))
        
        return warped_grid
    
    def compute_L_inverse(self,X,Y):
        N = X.size()[0] # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N,N)
        Ymat = Y.expand(N,N)
        P_dist_squared = torch.pow(Xmat-Xmat.transpose(0,1),2)+torch.pow(Ymat-Ymat.transpose(0,1),2)
        P_dist_squared[P_dist_squared==0]=1 # make diagonal 1 to avoid NaN in log computation
        K = torch.mul(P_dist_squared,torch.log(P_dist_squared))
        # construct matrix L
        O = torch.FloatTensor(N,1).fill_(1)
        Z = torch.FloatTensor(3,3).fill_(0)       
        P = torch.cat((O,X,Y),1)
        L = torch.cat((torch.cat((K,P),1),torch.cat((P.transpose(0,1),Z),1)),0)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li
        
    def apply_transformation(self,theta,points):
        if theta.dim()==2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords  
        # and points[:,:,:,1] are the Y coords  
        
        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X=theta[:,:self.N,:,:].squeeze(3)
        Q_Y=theta[:,self.N:,:,:].squeeze(3)
        Q_X = Q_X + self.P_X_base.expand_as(Q_X)
        Q_Y = Q_Y + self.P_Y_base.expand_as(Q_Y)
        
        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]
        
        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = self.P_X.expand((1,points_h,points_w,1,self.N))
        P_Y = self.P_Y.expand((1,points_h,points_w,1,self.N))
        
        # compute weigths for non-linear part
        W_X = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_X)
        W_Y = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_Y)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        # compute weights for affine part
        A_X = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_X)
        A_Y = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        
        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = points[:,:,:,0].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,0].size()+(1,self.N))
        points_Y_for_summation = points[:,:,:,1].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,1].size()+(1,self.N))
        
        if points_b==1:
            delta_X = points_X_for_summation-P_X
            delta_Y = points_Y_for_summation-P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation-P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation-P_Y.expand_as(points_Y_for_summation)
            
        dist_squared = torch.pow(delta_X,2)+torch.pow(delta_Y,2)
        # U: size [1,H,W,1,N]
        dist_squared[dist_squared==0]=1 # avoid NaN in log computation
        U = torch.mul(dist_squared,torch.log(dist_squared)) 
        
        # expand grid in batch dimension if necessary
        points_X_batch = points[:,:,:,0].unsqueeze(3)
        points_Y_batch = points[:,:,:,1].unsqueeze(3)
        if points_b==1:
            points_X_batch = points_X_batch.expand((batch_size,)+points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,)+points_Y_batch.size()[1:])
        
        points_X_prime = A_X[:,:,:,:,0]+                        torch.mul(A_X[:,:,:,:,1],points_X_batch) +                        torch.mul(A_X[:,:,:,:,2],points_Y_batch) +                        torch.sum(torch.mul(W_X,U.expand_as(W_X)),4)
                    
        points_Y_prime = A_Y[:,:,:,:,0]+                        torch.mul(A_Y[:,:,:,:,1],points_X_batch) +                        torch.mul(A_Y[:,:,:,:,2],points_Y_batch) +                        torch.sum(torch.mul(W_Y,U.expand_as(W_Y)),4)
        
        return torch.cat((points_X_prime,points_Y_prime),3)
        

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self, layids = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class GMM(nn.Module):
    """ Geometric Matching Module
    """
    def __init__(self):
        super(GMM, self).__init__()
        self.extractionA = FeatureExtraction(22, ngf=64, n_layers=3, norm_layer=nn.BatchNorm2d) 
        self.extractionB = FeatureExtraction(3, ngf=64, n_layers=3, norm_layer=nn.BatchNorm2d)
        self.l2norm = FeatureL2Norm()
        self.correlation = FeatureCorrelation()
        self.regression = FeatureRegression(input_nc=192, output_dim=2*5**2, use_cuda=True)
        self.gridGen = TpsGridGen(256, 192, use_cuda=True, grid_size=5)
        
    def forward(self, inputA, inputB):
        featureA = self.extractionA(inputA)
        featureB = self.extractionB(inputB)
        featureA = self.l2norm(featureA)
        featureB = self.l2norm(featureB)
        correlation = self.correlation(featureA, featureB)
        theta = self.regression(correlation)
        grid = self.gridGen(theta)
        return grid, theta


# In[ ]:


G=GMM().to(device)
lrG=0.0001
optG=torch.optim.Adam(G.parameters(),lr=lrG,betas=(0.5,0.999))
G.load_state_dict(torch.load('/kaggle/input/gmmtom/gmm_final.pth'))
G.cuda()


# In[ ]:


def init_weights(net,gain=0.02):
    def init_func(m):
        classname=m.__class__.__name__
        if hasattr(m,'weight') and (classname.find('Conv')!=-1):
            nn.init.normal_(m.weight.data,0.0,gain)
            if hasattr(m,'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data,0.0)
        elif classname.find('Linear')!=-1:
            nn.init.xavier_normal_(m.weight.data)
        elif classname.find('BatchNorm2d')!=-1:
            nn.init.normal_(m.weight.data,1.0,gain)
            nn.init.constant_(m.bias.data,0.0)
    print('init network...')
    net.apply(init_func)

def init_net(net,gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net,0.02)
    return net


# In[ ]:


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc*2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# In[ ]:


# # Define a resnet block
# class ResnetBlock(nn.Module):
#     def __init__(self, dim, norm_layer):
#         super(ResnetBlock, self).__init__()
#         self.conv_block = self.build_conv_block(dim, norm_layer)

#     def build_conv_block(self, dim, norm_layer):
#         conv_block = []
#         p = 0
#         conv_block += [nn.ReflectionPad2d(1)]

#         conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
#                        norm_layer(dim),
#                        nn.ReLU(True)]
#         p = 0
#         conv_block += [nn.ReflectionPad2d(1)]
#         conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
#                        norm_layer(dim)]

#         return nn.Sequential(*conv_block)

#     def forward(self, x):
#         out = x + self.conv_block(x)
#         return out

# class ResnetDiscriminator(nn.Module):
#     def __init__(self, input_nc, ngf=64, norm_layer=nn.InstanceNorm2d, n_blocks=3, n_downsampling=2):
#         super(ResnetDiscriminator, self).__init__()
#         self.input_nc = input_nc
#         self.ngf = ngf
#         model = [nn.ReflectionPad2d(3),
#                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
#                  norm_layer(ngf),
#                  nn.ReLU(True)]

#         # n_downsampling = 2
#         for i in range(n_downsampling):
#             mult = 2 ** i
#             model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
#                                 stride=2, padding=1),
#                         norm_layer(ngf * mult * 2),
#                         nn.ReLU(True)]

#         mult = 2 ** n_downsampling

#         for i in range(n_blocks):
#             model += [ResnetBlock(ngf * mult, norm_layer=norm_layer)]

#         self.model = nn.Sequential(*model)

#     def forward(self, input):
#         return self.model(input)


# In[ ]:


L1_Loss=nn.L1Loss()
# MSE_Loss=nn.MSELoss()
criterionVGG = VGGLoss()
T = UnetGenerator(24, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
T=init_net(T,[0])
# D=ResnetDiscriminator(21)
# D=init_net(D,[0])
lrT=0.0001
# lrD=0.0002
optT=torch.optim.Adam(T.parameters(),lr=lrT,betas=(0.5,0.999))
# optD=torch.optim.Adam(D.parameters(),lr=lrD,betas=(0.5,0.999))
T.load_state_dict(torch.load('/kaggle/input/viton-compare/viton_20.pth'))
T.cuda()
# D.load_state_dict(torch.load('/kaggle/input/viton-compare/networks/viton_5.pth'))
# D.cuda()


# In[ ]:


get_ipython().system('mkdir -p /kaggle/working/networks/')
get_ipython().system('mkdir -p /kaggle/working/output1/')


# In[ ]:


def plot_images(img,epoch,save=True):
    fig,axs=plt.subplots(2,7,figsize=(12,6))
    imgs=[img[0].cpu(),img[1].cpu(),img[2].cpu(),img[3].cpu(),img[4].cpu(),img[5].cpu(),img[6].cpu(),
          img[7].cpu(),img[8].cpu(),img[9].cpu(),img[10].cpu(),img[11].cpu(),img[12].cpu(),img[13].cpu()]
    for i,(ax,img) in enumerate(zip(axs.flatten(),imgs)):
        ax.axis('off')
        img=img.squeeze()
        if(i==3 or i==4 or i==10 or i==11):
            ax.imshow(img.squeeze(0),cmap='gray')
        else:
            img=np.transpose((img*0.5)+0.5,(1,2,0))
            ax.imshow(img)
    title='Epoch {}'.format(epoch+1)
    fig.text(0.5,0.04,title,ha='center')
    if save:
        if not os.path.exists(os.path.dirname('/kaggle/working/output1/')):
            os.makedirs(os.path.dirname('/kaggle/working/output1/'))
        plt.savefig('/kaggle/working/output1/%d.jpg'%(epoch+1))
    plt.show()

def sample_images(epoch):
    image=data0.float().cuda()
    target_pose=data8.float().cuda()
    target_agnostic=data10.float().cuda()
    cloth=data2.float().cuda()
    mask_img=data11.float().cuda()
    label=data1.float().cuda()
    mask_label=data12.float().cuda()
    grid,theta=G(target_agnostic,cloth)
    grid=grid.detach()
    theta=theta.detach()
    warped_cloth = F.grid_sample(cloth, grid, padding_mode='border')
    warped_mask = F.grid_sample(mask_img, grid, padding_mode='zeros')
    output=T(torch.cat([image,target_pose,warped_cloth],1)).detach()
    gen_img,mask=torch.split(output, 3, 1)
    gen_img=F.tanh(gen_img)
    mask=F.sigmoid(mask)
    p_tryon = warped_cloth * mask + gen_img * (1 - mask)

    image1=data01.float().cuda()
    target_pose1=data81.float().cuda()
    target_agnostic1=data101.float().cuda()
    cloth1=data21.float().cuda()
    mask_img1=data111.float().cuda()
    label1=data11.float().cuda()
    mask_label1=data121.float().cuda()
    grid1,theta1=G(target_agnostic1,cloth1)
    grid1=grid1.detach()
    theta1=theta1.detach()
    warped_cloth1 = F.grid_sample(cloth1, grid1, padding_mode='border')
    warped_mask1 = F.grid_sample(mask_img1, grid1, padding_mode='zeros')
    output1=T(torch.cat([image1,target_pose1,warped_cloth1],1)).detach()
    gen_img1,mask1=torch.split(output1, 3, 1)
    gen_img1=F.tanh(gen_img1)
    mask1=F.sigmoid(mask1)
    p_tryon1 = warped_cloth1 * mask1 + gen_img1 * (1 - mask1)
    
    plot_images([image,cloth,warped_cloth,mask,mask_label,p_tryon,label,
                 image1,cloth1,warped_cloth1,mask1,mask_label1,p_tryon1,label1],epoch,save=True)


# In[ ]:


sample_images(1)


# In[ ]:


import time
import sys
import datetime
prev_time=time.time()
for epoch in range(20,35):
    start=time.time()
    for i,batch in enumerate(dataloader):
        optT.zero_grad()
        image=batch['source_image'].float().cuda()
        target_pose=batch['target_pose'].float().cuda()
        target_agnostic=batch['target_agnostic'].float().cuda()
        cloth=batch['cloth_image'].float().cuda()
        mask_img=batch['mask_image'].float().cuda()
        label=batch['target_image'].float().cuda()
        mask_label=batch['mask_label'].float().cuda()
        grid,theta=G(target_agnostic,cloth)
        grid=grid.detach()
        theta=theta.detach()
        warped_cloth = F.grid_sample(cloth, grid, padding_mode='border')
        warped_mask = F.grid_sample(mask_img, grid, padding_mode='zeros')
        output=T(torch.cat([image,target_pose,warped_cloth],1))
        gen_img,mask=torch.split(output, 3, 1)
        gen_img=F.tanh(gen_img)
        mask=F.sigmoid(mask)
        p_tryon = warped_cloth * mask + gen_img * (1 - mask)
        t_loss=0.2*L1_Loss(p_tryon,label)+0.8*criterionVGG(label,p_tryon)+0.1*L1_Loss(mask,mask_label)
        t_loss.backward()
        optT.step()
        #log
        batches_done=epoch*len(dataloader)+i
        batches_left=35*len(dataloader)-batches_done
        time_left=datetime.timedelta(seconds=batches_left*(time.time()-prev_time))
        prev_time=time.time()
        
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [T loss: %f] ETA:%s"
             %(
                epoch,
                35,
                i,
                len(dataloader),
                t_loss.item(),
                time_left
             )
        )
        if batches_done%50==0:
            sample_images(epoch)
    print('Time for one epoch is {} sec'.format(time.time()-start))
    if((epoch+1)%3==0):
        torch.save(T.state_dict(),'/kaggle/working/networks/viton_{}.pth'.format(epoch+1))

