#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        pass

# Any results you write to the current directory are saved as output.


# In[ ]:


# coding: utf-8

# As far as we know, I/O operations are most expensive in training Neural Networks.<br> In this kernel I wrote pytorch Dataset loader that loads and preprocess all images once and stores them into RAM.

# ### Benchmark 
# As a baseline I took <a href="https://www.kaggle.com/speedwagon/ralsgan-dogs">this</a> kernel 
#  - dataloader with 1 worker takes 2 min for 1 epoch
#  - dataloader with 4 workers takes 75s for 1 epoch
#  - RAMdataloader (this kernel) with 4 workers takes 30s for 1 epoch
#  
# With this dataloader you can make much more experiments and epochs!

# In[1]:


import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import xml.etree.ElementTree as ET # for parsing XML

import pprint
import argparse
import cv2
import copy
import random

###
# utility
import time
def tic():
    start_time = time.time()
    return start_time

def toc(start_time = 0):
    elapsed_time = time.time() - start_time
    return elapsed_time

      
class EasyDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(EasyDict, self).pop(k, d)

def _get_default_config():
  c = EasyDict()

  # dataset
  c.data = EasyDict()
  c.data.n_images = -1
  c.data.sbox = True
  c.data.real_label = [0.6, 0.8]
  c.data.fake_label = [0.0, 0.2]

  # model
  c.modelG = EasyDict()
  c.modelG.name = 'generator_nroyUS'
  c.modelG.params = EasyDict()
  c.modelG.params.winit = 'default'
  c.modelG.params.nz = 128
  c.modelG.params.gchs = [1024, 512, 256, 128, 64]
  c.modelG.params.upsample = 'nearest'
  c.modelG.params.kernel = 3
  
  c.modelD = EasyDict()
  c.modelD.name = 'discriminator_PFN'
  c.modelD.params = EasyDict()
  c.modelD.params.winit = 'default'
  c.modelD.params.dchs = [32, 64, 128, 256]
  c.modelD.params.bns = [False, False, True, True, True]
  c.modelD.params.acti = True
  c.modelD.params.lrelu_slope = 0.2
  c.modelD.params.bn = False
  c.modelD.params.mbstd = True
  c.modelD.params.nfilt = 32
  c.modelD.params.kernel = 5

  # train
  c.train = EasyDict()
  c.train.batch_size = 32
  c.train.epoch = 2000
  c.train.log_step = 100
  c.train.params = EasyDict()
  c.train.params.use_averaging_model = True
  c.train.params.ema_alpha = 0.9999
  c.train.params.ma_start = 100
  c.train.params.ema = True # False in submitted version

  # evaluation
  c.eval = EasyDict()
  c.eval.batch_size = 50

  # optimizer
  c.optimizerG = EasyDict()
  c.optimizerG.name = 'adam'
  c.optimizerG.params = EasyDict()
  c.optimizerG.params.lr = 0.0005
  c.optimizerG.params.beta1 = 0.5
  c.optimizerG.params.beta2 = 0.999

  c.optimizerD = EasyDict()
  c.optimizerD.name = 'adam'
  c.optimizerD.params = EasyDict()
  c.optimizerD.params.lr = 0.0005
  c.optimizerD.params.beta1 = 0.5
  c.optimizerD.params.beta2 = 0.999

  # loss
  c.loss = EasyDict()
  c.loss.name = 'sGAN'


  # scheduler
  c.scheduler = EasyDict()
  c.scheduler.name = 'none'
  c.scheduler.params = EasyDict()

  return c

config = _get_default_config()
pprint.PrettyPrinter(indent=2).pprint(config)

from tqdm import tqdm_notebook as tqdm


# In[ ]:


# ## Parameters of GAN

if True:
    root_dir = '../input/generative-dog-images'
else:
    root_dir = '../input'

eval_do = False
use_scheduler = False

# In[2]:

if config.data.sbox:
    exrw = 1
    exrh = 1
else:
    exrw = 0
    exrh = 0
n_images = config.data.n_images    


g_model = config.modelG.name
d_model = config.modelD.name

g_params = config.modelG.params
d_params = config.modelD.params

nz = g_params['nz']

print(g_params)
print(d_params)

epochs = config.train.epoch
batch_size = config.train.batch_size
lrG = config.optimizerG.params.lr
G_beta1 = config.optimizerG.params.beta1
G_beta2 = config.optimizerD.params.beta2

lrD = config.optimizerD.params.lr
D_beta1 = config.optimizerD.params.beta1
D_beta2 = config.optimizerD.params.beta2

use_averaging_model = config.train.params.use_averaging_model

# labels
#label_type = 1
#if label_type==0:
#    # default
#    real_label = +1
#    fake_label = -1
#elif label_type==1:
#    # smooth labeling
#    real_label = +0.5
#    fake_label = -0.5
#else:
#    # smooth labeling with random values
#    real_label = [0.7, 1.2]
#    fake_label = [-1.2, -0.7]

real_label = config.data.real_label
fake_label = config.data.fake_label

#
num_classes = 120
sz = 64

#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


# ## Pytorch Dataset and DataLoader

# In[3]:

def get_list_objects_recursively(tdir):
    obj_list = []
    for root, dirnames, filenames in os.walk(tdir):
        for filename in filenames:
            obj_list.append(root + '/' + filename)
    obj_list.sort()
    return obj_list


def get_bboxes(anno_list, img_name):
    basename = os.path.splitext(img_name)[0]
    anno_fn = None
    for fn in anno_list:
        fn_base = os.path.basename(fn)
        if basename==fn_base:
            #print(img_name, fn)
            if anno_fn is None:
                anno_fn = fn
            else:
                print('filename error')
                sys.exit()
                
    dog_label = anno_fn.split('/')[-2]
    tree = ET.parse(anno_fn)
    root = tree.getroot()
    objects = root.findall('object')
    bboxes = []
    dog_labels = []
    for o in objects:
        bndbox = o.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bbox = (xmin, ymin, xmax, ymax)
        bboxes.append(bbox)
        dog_labels.append(dog_label)
    return bboxes, dog_labels
    
class DogDataset(Dataset):
    def __init__(self, img_dir, anno_dir, transform1=None, transform2=None, exrw=0, exrh=0, n_images=-1):
    
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.anno_dir = anno_dir
        self.anno_list = get_list_objects_recursively(anno_dir)
        self.transform1 = transform1
        self.transform2 = transform2
        
        self.dog_label_dict = {}
        
        self.imgs = []
        self.dog_classes = []
        
        
        if n_images>0:
            self.n_images = n_images
            seed = 190804
            random.seed(seed)
            random.shuffle(self.anno_list)
        else:
            self.n_images = len(self.anno_list)
        
        cnt = 0
        for img_name in tqdm(self.img_names):
            cnt += 1
            if cnt > self.n_images:
                break
            img_org = Image.open(os.path.join(img_dir, img_name))
            width, height = img_org.size
            bboxes, dog_labels = get_bboxes(self.anno_list, img_name)
            for bbox, dog_label in zip(bboxes, dog_labels):
                xmin, ymin, xmax, ymax = bbox
                if False:
                    # exact square
                    if width < height:
                        d = (height - width)//2
                        xmin -= d
                        xmax += d
                    else:
                        d = (width - height)//2
                        ymin -= d
                        ymax += d
                else:
                    # rectangular
                    if width < height:
                        dmax = height - width
                        dxl = xmin - int(max(xmin*(1-exrw), 0))
                        dxr = int(min(xmax*(1+exrw), width)) - xmax
                        dx = min(dxl, dxr, dmax)
                        xmin -= dx
                        xmax += dx
                    else:
                        dmax = width - height
                        dyt = ymin - int(max(ymin*(1-exrh), 0)) 
                        dyb = int(min(ymax*(1+exrh), height)) - ymax
                        dy = min(dyt, dyb, dmax)
                        ymin -= dy
                        ymax += dy
    
                   
                bbox = (xmin, ymin, xmax, ymax)
                img = img_org.crop(bbox)
            
                if self.transform1 is not None:
                    img = self.transform1(img)
                
                if dog_label not in self.dog_label_dict:
                    self.dog_label_dict[dog_label] = len(self.dog_label_dict)
                dog_class = self.dog_label_dict[dog_label]

                self.imgs.append(img)
                self.dog_classes.append(dog_class)
                #print(img_name, dog_label, dog_class)

    def __getitem__(self, index):
        img = self.imgs[index]
        dog_class = self.dog_classes[index]
        
        if self.transform2 is not None:
            img = self.transform2(img)
        
        return img, dog_class

    def __len__(self):
        return len(self.imgs)


# In[ ]:


# First preprocessing of data
transform1 = transforms.Compose([transforms.Resize(sz),
                                transforms.CenterCrop(sz)])

# Data augmentation and converting to tensors
random_transforms = [transforms.RandomRotation(degrees=5)]
transform2 = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.RandomApply(random_transforms, p=0.3), 
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = DogDataset(img_dir=root_dir + '/all-dogs/all-dogs/',
                           anno_dir=root_dir + '/annotation/Annotation',
                           transform1=transform1,
                           transform2=transform2,
                           exrw=exrw,
                           exrh=exrh,
                           n_images=n_images)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4)


# In[ ]:


x, y = next(iter(train_loader))


# In[6]:

fig = plt.figure(figsize=(25, 16))
for ii, img in enumerate(x):
    if ii>31:
        break
    ax = fig.add_subplot(4, 8, ii + 1, xticks=[], yticks=[])
    
    img = img.numpy().transpose(1, 2, 0)
    if img.shape[0]<64:
        img = cv2.resize(img, (64, 64))
    plt.imshow((img+1)/2)
    plt.pause(0.1)


# In[ ]:


# taken from https://www.kaggle.com/speedwagon/ralsgan-dogs
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 64, 64)
    return x
    
    
    
### This is to show one sample image for iteration of chosing
def show_generated_img(epoch=0):
    if not eval_do:
        return
    if epoch % 10 == 0 or epoch==epochs-1:
        noise = fixed_noise
        labels = fixed_labels
        gen_tensors = netG(noise, labels).to("cpu").clone().detach().squeeze(0)
        gen_tensors = gen_tensors/2+0.5
        gen_tensors = gen_tensors.clamp(0, 1)
        gen_images = gen_tensors.numpy().transpose(0, 2, 3, 1)
        plt.subplot(1,2,1)
        plt.imshow(gen_images[0])
        plt.subplot(1,2,2)
        plt.imshow(gen_images[1])
        plt.pause(0.1)
        print('gen_image: (%f, %f)' % (gen_images.min(), gen_images.max()))

# In[10]:

def generate_samples(outimg_dir, netG, n_images=10000):
    if not os.path.exists(outimg_dir):
        os.mkdir(outimg_dir)
    im_batch_size = 50
    #n_images=10000
    for i_batch in range(0, n_images, im_batch_size):
        gen_z = torch.randn(im_batch_size, nz, 1, 1, device=device)
        gen_labels = torch.randint(num_classes, (im_batch_size,), device=device)
        gen_images = (netG(gen_z, gen_labels) + 1)/2
        images = gen_images.to("cpu").clone().detach()
        images = images.numpy().transpose(0, 2, 3, 1)
        for i_image in range(gen_images.size(0)):
            save_image(gen_images[i_image, :, :, :], os.path.join(outimg_dir, f'image_{i_batch+i_image:05d}.png'))


from scipy.stats import truncnorm
def truncated_noise_sample(batch_size=1, dim_z=128, truncation=1., seed=None):
    """ Create a truncated noise vector.
        Params:
            batch_size: batch size.
            dim_z: dimension of z
            truncation: truncation value to use
            seed: seed for the random generator
        Output:
            array of shape (batch_size, dim_z)
    """
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation * values


def generate_samples_truncated(outimg_dir, netG, th=1, n_images=10000):
    if not os.path.exists(outimg_dir):
        os.mkdir(outimg_dir)
    im_batch_size = 50
    #n_images=10000
    for i_batch in range(0, n_images, im_batch_size):
        if th>0:
            #print('truncated')
            tns = truncated_noise_sample(im_batch_size, nz, th)
            tns = np.expand_dims(tns, -1)
            tns = np.expand_dims(tns, -1)
            gen_z = torch.from_numpy(tns).to(device)
        else:
            gen_z = torch.randn(im_batch_size, nz, 1, 1, device=device)
        gen_labels = torch.randint(num_classes, (im_batch_size,), device=device)
        gen_images = (netG(gen_z, gen_labels) + 1)/2
        images = gen_images.to("cpu").clone().detach()
        images = images.numpy().transpose(0, 2, 3, 1)
        for i_image in range(gen_images.size(0)):
            save_image(gen_images[i_image, :, :, :], os.path.join(outimg_dir, f'image_{i_batch+i_image:05d}.png'))


# In[ ]:


# ## Initialize models and optimizers
import torch as th
from torch.nn.utils import spectral_norm

def init_weights_normal(m):
    w_mean = 0.0
    w_std = 0.02
    if type(m) == nn.ConvTranspose2d:
        print('gaussian initialization')
        torch.nn.init.normal_(m.weight, w_mean, w_std)
    elif type(m) == nn.Conv2d:
        print('gaussian initialization')
        torch.nn.init.normal_(m.weight, w_mean, w_std)


def init_weights_xavier(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        print('xavier_uniform initialization')
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.)        


class PixelwiseNorm(nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()
        self.alpha = 1e-8

    def forward(self, x):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        if False:
            y = x.pow(2.).mean(dim=1, keepdim=True).add(self.alpha).sqrt()  # [N1HW]
            y = x / y  # normalize the input x volume
        else:
            # https://github.com/deepsound-project/pggan-pytorch/blob/master/network.py
            mean = torch.mean(x * x, 1, keepdim=True)
            dom = torch.rsqrt(mean + self.alpha)
            y = x * dom
        return y
        

class MinibatchStdDev(nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    """

    def __init__(self):
        """
        derived class constructor
        """
        super(MinibatchStdDev, self).__init__()
        self.alpha = 1e-8

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape
        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)
        # [1 x C x H x W]  Calc standard deviation over batch
        y = th.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + self.alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size,1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = th.cat([x, y], 1)
        # return the computed values:
        return y


class UpsamplingNN(nn.Module):
    def __init__(self, n_input, n_output, k_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        layers = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=bias),
            ]
        self.convlayer = nn.Sequential(*layers)
        self.weight = self.convlayer[1].weight
        
    def forward(self, x):
        y = self.convlayer(x)
        return y
        
class Generator(nn.Module):
    def __init__(self, nz, nfeats, nchannels, upsample='nearest', kernel=3):
        super().__init__()
        if upsample=='pixelshuffle_conv':
            upsample = ConvPS
        elif upsample=='conv_pixelshuffle':
            upsample = ConvPS_convfirst
        elif upsample=='nearest':
            upsample = UpsamplingNN
        elif upsample=='bilinear':
            upsample = UpsamplingBL
        else:
            print('unknown upsample: ', upsample)
            sys.exit()

        print(self.__class__)
        print('upsample: ', upsample)
        print('nfeats: ', nfeats)
        print('kernel: ', kernel)
        pad = kernel//2
        # input is Z, going into a convolution
        self.conv1 = spectral_norm(nn.ConvTranspose2d(nz, nfeats * 8, 4, 1, 0, bias=False))
        #self.bn1 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*8) x 4 x 4
        
        self.conv2 = spectral_norm(upsample(nfeats * 8, nfeats * 8, kernel, 1, pad, bias=False))
        #self.bn2 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*8) x 8 x 8
        
        self.conv3 = spectral_norm(upsample(nfeats * 8, nfeats * 4, kernel, 1, pad, bias=False))
        #self.bn3 = nn.BatchNorm2d(nfeats * 4)
        # state size. (nfeats*4) x 16 x 16
        
        self.conv4 = spectral_norm(upsample(nfeats * 4, nfeats * 2, kernel, 1, pad, bias=False))
        #self.bn4 = nn.BatchNorm2d(nfeats * 2)
        # state size. (nfeats * 2) x 32 x 32
        
        self.conv5 = spectral_norm(upsample(nfeats * 2, nfeats, kernel, 1, pad, bias=False))
        #self.bn5 = nn.BatchNorm2d(nfeats)
        # state size. (nfeats) x 64 x 64
        
        self.conv6 = spectral_norm(nn.Conv2d(nfeats, nchannels, 5, 1, 2, bias=False))
        # state size. (nchannels) x 64 x 64
        self.pixnorm = PixelwiseNorm()
    def forward(self, x, labels=None):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.pixnorm(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.pixnorm(x)
        x = F.leaky_relu(self.conv4(x))
        x = self.pixnorm(x)
        x = F.leaky_relu(self.conv5(x))
        x = self.pixnorm(x)
        x = torch.tanh(self.conv6(x))
        return x

def get_generator(nz, channels=3, nfilt=32, winit=None, upsample='nearest', kernel=3, **_):
    model = Generator(nz, nfeats=nfilt, nchannels=channels, upsample=upsample, kernel=kernel)
    if winit=='normal':
        model.apply(init_weights_normal)
    elif winit=='xavier':
        model.apply(init_weights_xavier)
    return model


# In[ ]:


# https://github.com/pfnet-research/chainer-gan-lib/blob/master/common/net.py
def snconv(chin, chout, kernel, stride, pad, bias=False, bn=False):
    layers = []
    layers.append(spectral_norm(nn.Conv2d(chin, chout, kernel, stride, pad, bias=bias)))
    if bn:
        layers.append(nn.BatchNorm2d(chout))
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    def __init__(self, nchannels, nfeats, acti=False, lrelu_slope=0.2, bn=False, mbstd=False, kernel=3):
        super().__init__()
        bias = True
        pad = kernel//2

        # input is (nchannels) x 64 x 64
        self.snconv1_1 = snconv(nchannels, nfeats//2, kernel, 1, pad, bias=bias, bn=bn)
        self.snconv1_2 = snconv(nfeats//2, nfeats*1, 4, 2, 1, bias=bias, bn=bn)
        self.snconv2_1 = snconv(nfeats*1,  nfeats*1, kernel, 1, pad, bias=bias, bn=bn)
        self.snconv2_2 = snconv(nfeats*1,  nfeats*2, 4, 2, 1, bias=bias, bn=bn)
        self.snconv3_1 = snconv(nfeats*2,  nfeats*2, kernel, 1, pad, bias=bias, bn=bn)
        self.snconv3_2 = snconv(nfeats*2,  nfeats*4, 4, 2, 1, bias=bias, bn=bn)
        self.snconv4_1 = snconv(nfeats*4,  nfeats*4, kernel, 1, pad, bias=bias, bn=bn)
        self.snconv4_2 = snconv(nfeats*4,  nfeats*8, 4, 2, 1, bias=bias, bn=bn)        
        # state size. (nfeats*8) x 4 x 4
        self.batch_discriminator = MinibatchStdDev()
        self.pixnorm = PixelwiseNorm()
        # state size. (nfeats*8) x 4 x 4
        self.snconv5 = snconv(nfeats*8+mbstd, 1, 4, 1, 0, bias=True, bn=False)
        self.conv5 = nn.Conv2d(nfeats*8+mbstd, 1, 4, 1, 0, bias=True)
        # state size. 1 x 1 x 1
        
        self.acti = acti
        self.lrelu_slope = lrelu_slope
        self.bn = bn
        self.mbstd = mbstd
        print('D: last sigmoid: ', self.acti)
        print('   lrelu_slope: ', self.lrelu_slope)
        print('   batch_norm: ', self.bn)
        print('   kernel_size: ', kernel)
        
    def forward(self, x, labels=None):
        x = F.leaky_relu(self.snconv1_1(x), self.lrelu_slope)
        x = F.leaky_relu(self.snconv1_2(x), self.lrelu_slope)
        x = F.leaky_relu(self.snconv2_1(x), self.lrelu_slope)
        x = F.leaky_relu(self.snconv2_2(x), self.lrelu_slope)
        x = F.leaky_relu(self.snconv3_1(x), self.lrelu_slope)
        x = F.leaky_relu(self.snconv3_2(x), self.lrelu_slope)
        x = F.leaky_relu(self.snconv4_1(x), self.lrelu_slope)
        x = F.leaky_relu(self.snconv4_2(x), self.lrelu_slope)
        if self.mbstd:
            x = self.batch_discriminator(x)
        #x = torch.sigmoid(self.conv5(x))
        x= self.snconv5(x)
        if self.acti:
            x = torch.sigmoid(x)
        return x.view(-1, 1)



def get_discriminator(channels=3, nfilt=32, winit=None, acti=True, lrelu_slope=0.2, bn=False, mbstd=False, kernel=3, **_):
    model = Discriminator(nchannels=channels, nfeats=nfilt, acti=acti, lrelu_slope=lrelu_slope, bn=bn, mbstd=mbstd, kernel=kernel)
    if winit=='normal':
        model.apply(init_weights_normal)
    elif winit=='xavier':
        model.apply(init_weights_xavier)
    return model


# In[ ]:


netG = get_generator(**g_params).to(device)
netD = get_discriminator(**d_params).to(device)

optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(D_beta1, D_beta2))
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(G_beta1, G_beta2))

lr_schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=100, gamma=0.5)
lr_schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=100, gamma=0.5)

fixed_noise = torch.randn(25, nz, 1, 1, device=device)
#fixed_labels = torch.randint(num_classes, (25,), device=device)
fixed_labels = torch.randint(1, (25,), device=device)


# In[ ]:


def ema_model(target_model, source_model, alpha):
    target_params = target_model.state_dict()    
    source_params = source_model.state_dict()
    for param_name, param in source_params.items():
        #print(param_name)
        #target_params[param_name] *= (1-alpha)
        #target_params[param_name] += alpha * param
        target_params[param_name] *= alpha
        target_params[param_name] += (1-alpha) * param


def ma_model(target_model, source_model, n):
    target_params = target_model.state_dict()    
    source_params = source_model.state_dict()
    for param_name, param in source_params.items():
        #print(param_name)
        target_params[param_name] *= (n / (n+1))
        target_params[param_name] += (1/(n+1)) * param
        
def copy_model_params(target_model, source_model):
    target_model.load_state_dict(source_model.state_dict())

if use_averaging_model:
    ma_count = 0
    netG_ma = copy.deepcopy(netG)


# In[ ]:


# loss function
# 1=sGAN
def get_lossD_sGAN():
    criterion = nn.BCELoss()
    def lossD_real(outputR, real_labels):
        return criterion(outputR, real_labels)
    def lossD_fake(outputF, fake_labels):        
        return criterion(outputF, fake_labels)
    return lossD_real, lossD_fake

def get_lossG_sGAN():
    criterion = nn.BCELoss()
    def lossG(outputR, outputF, real_labels, fake_labels):
        return criterion(outputF, real_labels)
    return lossG
    

# 7=RaLSGAN
def get_lossD_RaLSGAN():
    def lossD_rerative(outputR, outputF, real_labels, fake_labels):
        errD = (torch.mean((outputR - torch.mean(outputF) - real_labels) ** 2) +
                torch.mean((outputF - torch.mean(outputR) - fake_labels) ** 2))/2
        return errD
    return lossD_rerative

def get_lossG_RaLSGAN():
    def lossG_rerative(outputR, outputF, real_labels, fake_labels):
        errG = (torch.mean((outputR - torch.mean(outputF) + real_labels) ** 2) +
                torch.mean((outputF - torch.mean(outputR) + fake_labels) ** 2))/2
        return errG
    return lossG_rerative



loss_type = config.loss.name
if loss_type=='sGAN':
    print('loss type: sGAN')
    criterionD_real, criterionD_fake = get_lossD_sGAN()
    criterionG = get_lossG_sGAN()
    relative_loss = False
else:
    print('loss type: RaLSGAN')
    criterionD = get_lossD_RaLSGAN()
    criterionG = get_lossG_RaLSGAN()
    relative_loss = True


# In[ ]:


#epochs = 150


# In[ ]:


# ## Training loop
output_dir = '../tmp'
outimg_dir = '../tmp/output_images'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(outimg_dir):
    os.mkdir(outimg_dir)

start_time = tic()    
    
lossD = []
lossG = []
for epoch in range(epochs):
    
    elasped_time = toc(start_time)
    print('elasped_time[h]: ', elasped_time/3600)
    if elasped_time/3600 > 8.5:
        break
    
    lossD_epoch = 0
    lossG_epoch = 0
    for ii, (real_images, bleed_labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        ###########################
        # input
        ###########################
        real_images = real_images.to(device)
        bleed_labels = bleed_labels.to(device)
        bleed_labels[:] = 0
        #print(bleed_labels.shape)
        batch_size = real_images.size(0)
        #labels = torch.full((batch_size, 1), real_label, device=device)

        ###########################
        # labels
        ###########################
        if isinstance(real_label, list):
            real_labels = torch.empty((batch_size, 1), device=device).uniform_(real_label[0], real_label[1])
            fake_labels = torch.empty((batch_size, 1), device=device).uniform_(fake_label[0], fake_label[1])
            #for p in np.random.choice(batch_size, size=np.random.randint((batch_size//8)), replace=False):
            #    real_labels[p], fake_labels[p] = fake_labels[p].clone(), real_labels[p].clone() # swap labels
        else:
            # simple
            real_labels = torch.full((batch_size, 1), real_label, device=device)
            fake_labels = torch.full((batch_size, 1), fake_label, device=device)
    
    
        ###########################
        # (1) Update D network
        ###########################
        netD.zero_grad()
        outputR = netD(real_images, bleed_labels)
        
        if relative_loss:
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise, bleed_labels)
            outputF = netD(fake.detach(), bleed_labels)
            errD = criterionD(outputR, outputF, real_labels, fake_labels)
            errD.backward(retain_graph=True)
        else:
            errD_real = criterionD_real(outputR, real_labels)
            errD_real.backward()
            
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise, bleed_labels)
            outputF = netD(fake.detach(), bleed_labels)
            errD_fake = criterionD_fake(outputF, fake_labels)
            errD_fake.backward()
            errD = errD_real + errD_fake
            
        optimizerD.step()
        ###########################
        # (2) Update G network
        ###########################
        netG.zero_grad()
        outputF = netD(fake, bleed_labels)
        errG = criterionG(outputR, outputF, real_labels, fake_labels)
                
        errG.backward()
        optimizerG.step()
        
        # log
        lossD_epoch += errD.item()
        lossG_epoch += errG.item()
        
        if (ii+1) % (len(train_loader)//2) == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                  % (epoch + 1, epochs, ii+1, len(train_loader),
                     errD.item(), errG.item()))
                     
        # averaging
        if use_averaging_model:
            if epoch >= config.train.params.ma_start:
                if ma_count==0:
                    #ma_model(netG_ma, netG_ema, ma_count)
                    copy_model_params(netG_ma, netG)
                else:
                    if config.train.params.ema:
                        ema_model(netG_ma, netG, config.train.params.ema_alpha)
                    else:
                        ma_model(netG_ma, netG, ma_count)
                ma_count += 1
                
                     

    # epoch end
    if use_scheduler:
        lr_schedulerD.step()
        lr_schedulerG.step()

    lossD.append(lossD_epoch)
    lossG.append(lossG_epoch)
    show_generated_img(epoch)


# In[ ]:


# for debug


# In[ ]:


# plot
plt.figure()
plt.plot(lossD, label='lossD')
plt.plot(lossG, label='lossG')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
#plt.show()
plt.pause(0.01)


# In[ ]:


# ## Let's visualise generated results

# In[11]:

netG.eval()
gen_z = torch.randn(32, nz, 1, 1, device=device)
gen_labels = torch.randint(num_classes, (32,), device=device)
gen_images = netG(gen_z, gen_labels).to("cpu").clone().detach()
gen_images = gen_images.numpy().transpose(0, 2, 3, 1)


fig = plt.figure(figsize=(25, 16))
for ii, img in enumerate(gen_images):
    if ii>31:
        break
    ax = fig.add_subplot(4, 8, ii + 1, xticks=[], yticks=[])
    if img.shape[0]<64:
        img = cv2.resize(img, (64, 64))
    plt.imshow((img+1)/2)
plt.pause(0.1)


# In[ ]:


# ## Make predictions and submit
# final_evaluation

th = 1.0
if use_averaging_model:
    # ma
    generate_samples_truncated(outimg_dir, netG_ma)
else:
    generate_samples_truncated(outimg_dir, netG)
    

import shutil
shutil.make_archive('images', 'zip', outimg_dir)

print('last epoch: ', epoch)
elapsed_time = toc(start_time)
print('elapsed time[h] :', elapsed_time/3600)

