#!/usr/bin/env python
# coding: utf-8

# Architecture was taken from https://github.com/ozanciga/gans-with-pytorch

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import shutil
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from tqdm import tqdm_notebook as tqdm


# # Generator and Discriminator

# In[ ]:


import torch
from torch import nn 
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import shutil

# TODO for ScaledCrossReplicaBatchNorm2d
class _BatchNorm(nn.Module):
    _version = 2

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, '                'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_BatchNorm, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

# TODO for ScaledCrossReplicaBatchNorm2d
class ScaledCrossReplicaBatchNorm2d(_BatchNorm):
    r"""Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .
    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size).
    By default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.
    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.
    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momemtum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.
    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.
    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    Examples::
        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm2d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


# In[ ]:


import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


# In[ ]:



import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F

import functools
from torch.autograd import Variable


class Spectral_Norm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()
        sigma = u @ weight_mat @ v
        weight_sn = weight / sigma
        # weight_sn = weight_sn.view(*size)

        return weight_sn, u

    @staticmethod
    def apply(module, name):
        fn = Spectral_Norm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)


def spectral_norm(module, name='weight'):
    Spectral_Norm.apply(module, name)

    return module


def spectral_init(module, gain=1):
    init.xavier_uniform_(module.weight, gain)
    if module.bias is not None:
        module.bias.data.zero_()

    return spectral_norm(module)

def init_linear(linear):
    init.xavier_uniform_(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


def leaky_relu(input):
    return F.leaky_relu(input, negative_slope=0.2)

class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation=F.relu):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

        init_conv(self.query_conv)
        init_conv(self.key_conv)
        init_conv(self.value_conv)
        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out



class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, n_condition=148):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channel, affine=False)

        self.embed = nn.Linear(n_condition, in_channel* 2)
        self.embed.weight.data[:, :in_channel] = 1
        self.embed.weight.data[:, in_channel:] = 0

    def forward(self, input, class_id):
        out = self.bn(input)
        # print(class_id.dtype)
        # print('class_id', class_id.size()) # torch.Size([4, 148])
        # print(out.size()) #torch.Size([4, 128, 4, 4])
        # class_id = torch.randn(4,1)
        # print(self.embed)
        embed = self.embed(class_id)
        # print('embed', embed.size())
        gamma, beta = embed.chunk(2, 1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        # print(beta.size())
        out = gamma * out + beta

        return out


class GBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=[3, 3],
                 padding=1, stride=1, n_class=None, bn=True,
                 activation=F.relu, upsample=True, downsample=False):
        super().__init__()

        gain = 2 ** 0.5

        self.conv0 = SpectralNorm(nn.Conv2d(in_channel, out_channel,
                                             kernel_size, stride, padding,
                                             bias=True if bn else True))
        self.conv1 = SpectralNorm(nn.Conv2d(out_channel, out_channel,
                                             kernel_size, stride, padding,
                                             bias=True if bn else True))

        self.skip_proj = False
        if in_channel != out_channel or upsample or downsample:
            self.conv_sc = SpectralNorm(nn.Conv2d(in_channel, out_channel,
                                                   1, 1, 0))
            self.skip_proj = True

        self.upsample = upsample
        self.downsample = downsample
        self.activation = activation
        self.bn = bn
        if bn:
            self.HyperBN = ConditionalNorm(in_channel, 148)
            self.HyperBN_1 = ConditionalNorm(out_channel, 148)

    def forward(self, input, condition=None):
        out = input

        if self.bn:
            # print('condition',condition.size()) #condition torch.Size([4, 148])
            out = self.HyperBN(out, condition)
        out = self.activation(out)
        if self.upsample:
            # TODO different form papers
            out = F.upsample(out, scale_factor=2)
        out = self.conv0(out)
        if self.bn:
            out = self.HyperBN_1(out, condition)
        out = self.activation(out)
        out = self.conv1(out)

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        if self.skip_proj:
            skip = input
            if self.upsample:
                # TODO different form papers
                skip = F.upsample(skip, scale_factor=2)
            skip = self.conv_sc(skip)
            if self.downsample:
                skip = F.avg_pool2d(skip, 2)

        else:
            skip = input

        return out + skip


class Generator(nn.Module):
    def __init__(self, code_dim=100, n_class=1000, chn=96, debug=False):
        super().__init__()

        self.linear = SpectralNorm(nn.Linear(n_class, 128, bias=False))
        
        if debug:
            chn = 8

        self.first_view = 16 * chn

        self.G_linear = SpectralNorm(nn.Linear(20, 4 * 4 * 16 * chn))

        self.conv = nn.ModuleList([
                                #GBlock(16*chn, 16*chn, n_class=n_class),
                                GBlock(16*chn, 8*chn, n_class=n_class),
                                GBlock(8*chn, 4*chn, n_class=n_class),
                                GBlock(4*chn, 2*chn, n_class=n_class),
                                SelfAttention(2*chn),
                                GBlock(2*chn, 1*chn, n_class=n_class)])

        # TODO impl ScaledCrossReplicaBatchNorm 
        self.ScaledCrossReplicaBN = ScaledCrossReplicaBatchNorm2d(1*chn)
        self.colorize = SpectralNorm(nn.Conv2d(1*chn, 3, [3, 3], padding=1))

    def forward(self, input, class_id):
        codes = torch.split(input, 20, 1)
        class_emb = self.linear(class_id)  # 128

        out = self.G_linear(codes[0])
        # out = out.view(-1, 1536, 4, 4)
        out = out.view(-1, self.first_view, 4, 4)
        ids = 1
        for i, conv in enumerate(self.conv):
            if isinstance(conv, GBlock):
                
                conv_code = codes[ids]
                ids = ids+1
                condition = torch.cat([conv_code, class_emb], 1)
                # print('condition',condition.size()) #torch.Size([4, 148])
                out = conv(out, condition)

            else:
                out = conv(out)

        out = self.ScaledCrossReplicaBN(out)
        out = F.relu(out)
        out = self.colorize(out)

        return F.tanh(out)


class Discriminator(nn.Module):
    def __init__(self, n_class=1000, chn=96, debug=False):
        super().__init__()

        def conv(in_channel, out_channel, downsample=True):
            return GBlock(in_channel, out_channel,
                          bn=False,
                          upsample=False, downsample=downsample)

        gain = 2 ** 0.5
        

        if debug:
            chn = 8
        self.debug = debug

        self.pre_conv = nn.Sequential(SpectralNorm(nn.Conv2d(3, 1*chn, 3,padding=1),),
                                      nn.ReLU(),
                                      SpectralNorm(nn.Conv2d(1*chn, 1*chn, 3,padding=1),),
                                      nn.AvgPool2d(2))
        self.pre_skip = SpectralNorm(nn.Conv2d(3, 1*chn, 1))

        self.conv = nn.Sequential(conv(1*chn, 1*chn, downsample=True),
                                  SelfAttention(1*chn),
                                  conv(1*chn, 2*chn, downsample=True),    
                                  conv(2*chn, 4*chn, downsample=True),
                                  conv(4*chn, 8*chn, downsample=True),
                                  conv(8*chn, 16*chn, downsample=True)
                                  #,conv(16*chn, 16*chn, downsample=False)
                                 )

        self.linear = SpectralNorm(nn.Linear(16*chn, 1))

        self.embed = nn.Embedding(n_class, 16*chn)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.embed = spectral_norm(self.embed)

    def forward(self, input, class_id):
        
        out = self.pre_conv(input)
        out = out + self.pre_skip(F.avg_pool2d(input, 2))
        # print(out.size())
        out = self.conv(out)
        out = F.relu(out)
        out = out.view(out.size(0), out.size(1), -1)
        out = out.sum(2)
        out_linear = self.linear(out).squeeze(1)
        embed = self.embed(class_id)

        prod = (out * embed).sum(1)

        # if self.debug == debug:
        #     print('class_id',class_id.size())
        #     print('out_linear',out_linear.size())
        #     print('embed', embed.size())
        #     print('prod', prod.size())

        return out_linear + prod


# In[ ]:





# In[ ]:



import os
import time
import torch
import datetime

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from utils import *

class Trainer(object):
    def __init__(self, data_loader, config):

 # Data loader
        self.data_loader = data_loader

        # exact model and loss
        
        self.model = config['model']
        self.adv_loss = config['adv_loss']

        # Model hyper-parameters
        self.imsize = config['imsize']
        self.g_num = config['g_num']
        self.z_dim = config['z_dim']
        self.g_conv_dim = config['g_conv_dim']
        self.d_conv_dim = config['d_conv_dim']
        self.parallel = config['parallel']
        self.gpus = config['gpus']

        self.lambda_gp = config['lambda_gp']
        self.total_step = config['total_step']
        self.d_iters = config['d_iters']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.g_lr = config['g_lr']
        self.d_lr = config['d_lr']
        self.lr_decay = config['lr_decay']
        self.beta1 = config['beta1']
        self.beta2 = config['beta2']
        self.pretrained_model = config['pretrained_model']

        self.dataset = config['dataset']
        self.use_tensorboard = config['use_tensorboard']
        self.image_path = config['image_path']
        self.log_path = config['log_path']
        self.model_save_path = config['model_save_path']
        self.sample_path = config['sample_path']
        self.log_step = config['log_step']
        self.sample_step = config['sample_step']
        self.model_save_step = config['model_save_step']
        self.version = config['version']

        self.n_class = config['n_class']
        self.chn = config['chn']
        class_id = config['class_id']

        # Path
        self.log_path = os.path.join(config['log_path'], self.version)
        self.sample_path = os.path.join(config['sample_path'], self.version)
        self.model_save_path = os.path.join(config['model_save_path'], self.version)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('build_model...')
        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            print('load_pretrained_model...')
            self.load_pretrained_model()


    def label_sampel(self):
        label = torch.LongTensor(self.batch_size, 1).random_()%self.n_class
        one_hot= torch.zeros(self.batch_size, self.n_class).scatter_(1, label, 1)
        return label.squeeze(1).to(self.device), one_hot.to(self.device)       

    def train(self):

        # Data iterator
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)
        model_save_step = int(self.model_save_step * step_per_epoch)

        # Fixed input for debugging
        fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        # Start time
        print('Start   ======  training...')
        start_time = time.time()
        for step in range(start, self.total_step):

            # ================== Train D ================== #
            self.D.train()
            self.G.train()

            try:
                real_images, real_labels = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                real_images, real_labels = next(data_iter)
            
            #print(real_labels)
            #print(real_labels.shape)
            #print(type(real_labels))
            # Compute loss with real images

            real_labels = real_labels.to(self.device)
            real_images = real_images.to(self.device)

            d_out_real = self.D(real_images, real_labels)
            if self.adv_loss == 'wgan-gp':
                d_loss_real = - torch.mean(d_out_real)
            elif self.adv_loss == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            # apply Gumbel Softmax
            z = torch.randn(self.batch_size, self.z_dim).to(self.device)

            z_class, z_class_one_hot = self.label_sampel()
 
            fake_images = self.G(z, z_class_one_hot)
            d_out_fake = self.D(fake_images, z_class)

            if self.adv_loss == 'wgan-gp':
                d_loss_fake = d_out_fake.mean()
            elif self.adv_loss == 'hinge':
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()


            # Backward + Optimize
            d_loss = d_loss_real + d_loss_fake
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()


            if self.adv_loss == 'wgan-gp':
                # Compute gradient penalty
                alpha = torch.rand(real_images.size(0), 1, 1, 1).to(self.device).expand_as(real_images)
                interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
                out = self.D(interpolated, z_class)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).to(self.device),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                # Backward + Optimize
                d_loss = self.lambda_gp * d_loss_gp

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

            # ================== Train G and gumbel ================== #
            # Create random noise
            z = torch.randn(self.batch_size, self.z_dim).to(self.device)
            z_class, z_class_one_hot = self.label_sampel()
            
            fake_images = self.G(z, z_class_one_hot)

            # Compute loss with fake images
            g_out_fake = self.D(fake_images, z_class)  # batch x n
            if self.adv_loss == 'wgan-gp':
                g_loss_fake = - g_out_fake.mean()
            elif self.adv_loss == 'hinge':
                g_loss_fake = - g_out_fake.mean()

            self.reset_grad()
            g_loss_fake.backward()
            self.g_optimizer.step()


            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, d_out_fake: {:.4f}, g_loss_fake: {:.4f}".
                      format(elapsed, step + 1, self.total_step, (step + 1),
                             self.total_step , d_loss_real.item(), d_loss_fake.item(), g_loss_fake.item()))
                
                if self.use_tensorboard:
                    self.writer.add_scalar('data/d_loss_real', d_loss_real.item(),(step + 1))
                    self.writer.add_scalar('data/d_loss_fake', d_loss_fake.item(),(step + 1))
                    self.writer.add_scalar('data/d_loss', d_loss.item(), (step + 1))

                    self.writer.add_scalar('data/g_loss_fake', g_loss_fake.item(), (step + 1))

            # Sample images
            if (step + 1) % self.sample_step == 0:
                print('Sample images {}_fake.png'.format(step + 1))
                fake_images= self.G(fixed_z, z_class_one_hot)
                save_image(denorm(fake_images.data),
                           os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)))
                #fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))
                #fake_images2= self.G(fixed_z, z_class_one_hot)
                #fake_images2=denorm(fake_images2.data)
                #plt.imshow(fake_images2[0, :, :, :])
                #shutil.make_archive('learning', 'zip', '../learning/dogs')
                
                
            if (step+1) % model_save_step==0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
                torch.save(self.D.state_dict(),
                           os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))
            
        if not os.path.exists('../output_images'):
            os.mkdir('../output_images')
    
        im_batch_size = self.batch_size
        n_images=10000

        for i_batch in range(0, n_images, im_batch_size):
            #print(i_batch)
            fixed_z = tensor2var(torch.randn(im_batch_size, self.z_dim))
            fake_images= self.G(fixed_z, z_class_one_hot)
            fake_images=denorm(fake_images.data)
            for i_image in range(fake_images.size(0)):
                #print(i_image)
                save_image(fake_images[i_image, :, :, :],os.path.join('../output_images', f'image_{i_batch+i_image:05d}.png'))
     
        import shutil
        shutil.make_archive('images', 'zip', '../output_images')

    def build_model(self):
        # code_dim=100, n_class=1000
        self.G = Generator(self.z_dim, self.n_class, chn=self.chn).to(self.device)
        self.D = Discriminator(self.n_class, chn=self.chn).to(self.device)
        if self.parallel:
            print('use parallel...')
            print('gpuids ', self.gpus)
            gpus = [int(i) for i in self.gpus.split(',')]
    
            self.G = nn.DataParallel(self.G, device_ids=gpus)
            self.D = nn.DataParallel(self.D, device_ids=gpus)

        # self.G.apply(weights_init)
        # self.D.apply(weights_init)

        # Loss and optimizer
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

        self.c_loss = torch.nn.CrossEntropyLoss()
        # print networks
        print(self.G)
        print(self.D)

    def build_tensorboard(self):
        from tensorboardX import SummaryWriter
        # from logger import Logger
        # self.logger = Logger(self.log_path)
        
        tf_logs_path = os.path.join(self.log_path, 'tf_logs')
        self.writer = SummaryWriter(log_dir=tf_logs_path)


    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
        
    


# In[ ]:





# In[ ]:


import argparse

def str2bool(v):
    return v.lower() in ('true')

    # Model hyper-parameters


# In[ ]:


import torch
import torchvision.datasets as dsets
from torchvision import transforms
# required imports
import os
import xml.etree.ElementTree as ET
import torchvision
from tqdm import tqdm_notebook as tqdm
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from tqdm import tqdm_notebook as tqdm

def doggo_loader(path):
    img = torchvision.datasets.folder.default_loader(path) # default loader

        # Get bounding box
    annotation_basename = os.path.splitext(os.path.basename(path))[0]
    annotation_dirname = next(dirname for dirname in os.listdir('../input/annotation/Annotation/') if dirname.startswith(annotation_basename.split('_')[0]))
    annotation_filename = os.path.join('../input/annotation/Annotation', annotation_dirname, annotation_basename)
    tree = ET.parse(annotation_filename)
    root = tree.getroot()
    objects = root.findall('object')
    for o in objects:
        bndbox = o.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
    bbox = (xmin, ymin, xmax, ymax)
    
        # return cropped image
    return img.crop(bbox)
random_transforms = [transforms.ColorJitter(), transforms.RandomRotation(degrees=0)]

class Data_Loader():
    def __init__(self, train, dataset, image_path, image_size, batch_size, shuf=True):
        self.dataset = dataset
        self.path = image_path
        self.imsize = image_size
        self.batch = batch_size
        self.shuf = shuf
        self.train = train

    def transform(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def load_lsun(self, classes=['church_outdoor_train','classroom_train']):
        transforms = self.transform(True, True, True, False)
        dataset = dsets.LSUN(self.path, classes=classes, transform=transforms)
        return dataset
    
    def load_imagenet(self):
        transforms = self.transform(True, True, True, True)
        dataset = dsets.ImageFolder(self.path+'/imagenet', transform=transforms)
        return dataset

    def load_celeb(self):
        transforms = self.transform(True, True, True, True)
        dataset = dsets.ImageFolder(self.path+'/CelebA', transform=transforms)
        return dataset



    def load_off(self):
        #transforms = self.transform(True, True, True, False)
        dataset = torchvision.datasets.ImageFolder(
                '../input/all-dogs/',
                loader=doggo_loader, # THE CUSTOM LOADER
                transform=torchvision.transforms.Compose([transforms.Resize(64),
                                            transforms.CenterCrop(64),
                                            transforms.RandomHorizontalFlip(p=0),
                                            transforms.RandomApply(random_transforms, p=0),
                                            transforms.ToTensor()
                                            ,transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ]) # some transformations, add your data preprocessing here
            )
                                                  
        return dataset

    def loader(self):
        if self.dataset == 'lsun':
            dataset = self.load_lsun()
        elif self.dataset == 'imagenet':
            dataset = self.load_imagenet()
        elif self.dataset == 'celeb':
            dataset = self.load_celeb()
        elif self.dataset == 'off':
            dataset = self.load_off()

        print('dataset',len(dataset))
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=self.batch,
                                              shuffle=self.shuf,
                                              num_workers=2,
                                              drop_last=True)
        return loader


# In[ ]:


import os
import torch
from torch.autograd import Variable
from torch.nn import init

def make_folder(path, version):
        if not os.path.exists(os.path.join(path, version)):
            os.makedirs(os.path.join(path, version))


def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)

def var2tensor(x):
    return x.data.cpu()

def var2numpy(x):
    return x.data.cpu().numpy()

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)


# In[ ]:


batch_size = 8
image_size = 64

config = { 'model':'sagan',
                        'adv_loss':'hinge',
                        'imsize':image_size,
                        'g_num':5,
                        'chn':24,
                        'z_dim':120,
                        'g_conv_dim':32,
                        'd_conv_dim':32,
                        'lambda_gp':10,
                        'version':'dogs',
                        'n_class':2,    
                        'class_id':1,

                            # Training setting
                        'total_step':180000,
                        'd_iters':5,
                        'batch_size':batch_size,
                        'num_workers':12,
                        'g_lr':0.00001,
                        'd_lr':0.00004,
                        'lr_decay':0.95,
                        'beta1':0,
                        'beta2':0.9,

                            # using pretrained
                        'pretrained_model':None,

                            # Misc
                        'train':True,
                        'parallel':True,
                        'gpus':'0',
                        'dataset':'off',
                        'use_tensorboard':True,

                            # Path
                        'image_path':'../input/all-dogs/',
                        'log_path':'../logs',
                        'model_save_path':'../models',
                        'sample_path':'../learning',
                        'attn_path':'../attn',

                            # Step size
                        'log_step':2000,
                        'sample_step':2000,
                        'model_save_step':20000}


# In[ ]:


if not os.path.exists('../learning'):
    os.mkdir('../learning')
if not os.path.exists('../learning/dogs'):
    os.mkdir('../learning/dogs')
if not os.path.exists('../models'):
    os.mkdir('../models')
if not os.path.exists('../models/dogs'):
    os.mkdir('../models/dogs')
if not os.path.exists('../attn'):
    os.mkdir('../attn')
if not os.path.exists('../attn/dogs'):
    os.mkdir('../attn/dogs')
if not os.path.exists('../logs'):
    os.mkdir('../logs')
if not os.path.exists('../logs/dogs'):
    os.mkdir('../logs/dogs')

import os
os.listdir('../learning')
import shutil

data_loader = Data_Loader(config['train'], config['dataset'], config['image_path'], config['imsize'],
                             config['batch_size'], shuf=config['train'])
trainer = Trainer(data_loader.loader(), config)

trainer.train()

shutil.make_archive('learning', 'zip', '../learning/dogs')
shutil.make_archive('attn', 'zip', '../attn/dogs')
shutil.make_archive('models', 'zip', '../models/dogs')

