#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime
import gc
import json
import math
import os
import time
import shutil
import sys

from functools import partial
from numba import jit, njit
from PIL import Image
from tqdm import tqdm, trange

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torch.utils.data as data
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
import torchvision.transforms as T

import albumentations as A
from albumentations.pytorch import ToTensor

import xml.etree.ElementTree as ET


# In[ ]:


INPUT_PATH = '../input/all-dogs'
ANNOT_PATH = '../input/annotation/Annotation'
OUT_DIR = '../output_images'

IMG_SIZE = 64
IMG_SIZE_2 = IMG_SIZE * 2
IMG_CHANNELS = 3
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


# In[ ]:


# Layers

# Projection of x onto y
def proj(x, y):
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
    for y in ys:
        x = x - proj(x, y)
    return x


# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
    # Lists holding singular vectors and values
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        # Run one step of the power iteration
        with torch.no_grad():
            v = torch.matmul(u, W)
            # Run Gram-Schmidt to subtract components of all other singular vectors
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            # Add to the list
            vs += [v]
            # Update the other singular vector
            u = torch.matmul(v, W.t())
            # Run Gram-Schmidt to subtract components of all other singular vectors
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            # Add to the list
            us += [u]
            if update:
                u_[i][:] = u
        # Compute this singular value and add it to the list
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
        # svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
    return svs, us, vs


# Convenience passthrough function
class identity(nn.Module):
    def forward(self, input):
        return input


# Spectral normalization base class
class SN(object):
    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        # Number of power iterations per step
        self.num_itrs = num_itrs
        # Number of singular values
        self.num_svs = num_svs
        # Transposed?
        self.transpose = transpose
        # Epsilon value for avoiding divide-by-0
        self.eps = eps
        # Register a singular vector for each sv
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))

    # Singular vectors (u side)
    @property
    def u(self):
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    # Singular values;
    # note that these buffers are just for logging and are not used in training.
    @property
    def sv(self):
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

    # Compute the spectrally-normalized weight
    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        # Apply num_itrs power iterations
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
            # Update the svs
        if self.training:
            with torch.no_grad():  # Make sure to do this in a no_grad() context or you'll get memory leaks!
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]


# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                           padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
    def __init__(self, in_features, out_features, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)

    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)


# Embedding layer with spectral norm
# We use num_embeddings as the dim instead of embedding_dim here
# for convenience sake
class SNEmbedding(nn.Embedding, SN):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, _weight=None,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, padding_idx,
                              max_norm, norm_type, scale_grad_by_freq,
                              sparse, _weight)
        SN.__init__(self, num_svs, num_itrs, num_embeddings, eps=eps)

    def forward(self, x):
        return F.embedding(x, self.W_())


# A non-local block as used in SA-GAN
# Note that the implementation as described in the paper is largely incorrect;
# refer to the released code for the actual implementation.
# class Attention(nn.Module):
#     def __init__(self, ch, which_conv=SNConv2d, name='attention'):
#         super(Attention, self).__init__()
#         # Channel multiplier
#         self.ch = ch
#         self.which_conv = which_conv
#         self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
#         self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
#         self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
#         self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
#         # Learnable gain parameter
#         self.gamma = P(torch.tensor(0.), requires_grad=True)

#     def forward(self, x, y=None):
#         # Apply convs
#         theta = self.theta(x)
#         phi = F.max_pool2d(self.phi(x), [2, 2])
#         g = F.max_pool2d(self.g(x), [2, 2])
#         # Perform reshapes
#         theta = theta.view(-1, self.ch // 8, x.shape[2] * x.shape[3])
#         phi = phi.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
#         g = g.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
#         # Matmul and softmax to get attention maps
#         beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
#         # Attention map times g path
#         o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
#         return self.gamma * o + x
    
    
class Attention(nn.Module):
    def __init__(self, ch, which_conv=SNConv2d, name='attention'):
        super(Attention, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.which_conv = which_conv
        self.theta = self.which_conv(self.ch, self.ch // 4, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(self.ch, self.ch // 4, kernel_size=1, padding=0, bias=False)
        self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self.ch // 4, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch // 4, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


# Fused batchnorm op
def fused_bn(x, mean, var, gain=None, bias=None, eps=1e-5):
    # Apply scale and shift--if gain and bias are provided, fuse them here
    # Prepare scale
    scale = torch.rsqrt(var + eps)
    # If a gain is provided, use it
    if gain is not None:
        scale = scale * gain
    # Prepare shift
    shift = mean * scale
    # If bias is provided, use it
    if bias is not None:
        shift = shift - bias
    return x * scale - shift
    # return ((x - mean) / ((var + eps) ** 0.5)) * gain + bias # The unfused way.


# Manual BN
# Calculate means and variances using mean-of-squares minus mean-squared
def manual_bn(x, gain=None, bias=None, return_mean_var=False, eps=1e-5):
    # Cast x to float32 if necessary
    float_x = x.float()
    # Calculate expected value of x (m) and expected value of x**2 (m2)
    # Mean of x
    m = torch.mean(float_x, [0, 2, 3], keepdim=True)
    # Mean of x squared
    m2 = torch.mean(float_x ** 2, [0, 2, 3], keepdim=True)
    # Calculate variance as mean of squared minus mean squared.
    var = (m2 - m ** 2)
    # Cast back to float 16 if necessary
    var = var.type(x.type())
    m = m.type(x.type())
    # Return mean and variance for updating stored mean/var if requested
    if return_mean_var:
        return fused_bn(x, m, var, gain, bias, eps), m.squeeze(), var.squeeze()
    else:
        return fused_bn(x, m, var, gain, bias, eps)


# My batchnorm, supports standing stats
class myBN(nn.Module):
    def __init__(self, num_channels, eps=1e-5, momentum=0.1):
        super(myBN, self).__init__()
        # momentum for updating running stats
        self.momentum = momentum
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Register buffers
        self.register_buffer('stored_mean', torch.zeros(num_channels))
        self.register_buffer('stored_var', torch.ones(num_channels))
        self.register_buffer('accumulation_counter', torch.zeros(1))
        # Accumulate running means and vars
        self.accumulate_standing = False

    # reset standing stats
    def reset_stats(self):
        self.stored_mean[:] = 0
        self.stored_var[:] = 0
        self.accumulation_counter[:] = 0

    def forward(self, x, gain, bias):
        if self.training:
            out, mean, var = manual_bn(x, gain, bias, return_mean_var=True, eps=self.eps)
            # If accumulating standing stats, increment them
            if self.accumulate_standing:
                self.stored_mean[:] = self.stored_mean + mean.data
                self.stored_var[:] = self.stored_var + var.data
                self.accumulation_counter += 1.0
            # If not accumulating standing stats, take running averages
            else:
                self.stored_mean[:] = self.stored_mean * (1 - self.momentum) + mean * self.momentum
                self.stored_var[:] = self.stored_var * (1 - self.momentum) + var * self.momentum
            return out
        # If not in training mode, use the stored statistics
        else:
            mean = self.stored_mean.view(1, -1, 1, 1)
            var = self.stored_var.view(1, -1, 1, 1)
            # If using standing stats, divide them by the accumulation counter
            if self.accumulate_standing:
                mean = mean / self.accumulation_counter
                var = var / self.accumulation_counter
            return fused_bn(x, mean, var, gain, bias, self.eps)


# Simple function to handle groupnorm norm stylization
def groupnorm(x, norm_style):
    # If number of channels specified in norm_style:
    if 'ch' in norm_style:
        ch = int(norm_style.split('_')[-1])
        groups = max(int(x.shape[1]) // ch, 1)
    # If number of groups specified in norm style
    elif 'grp' in norm_style:
        groups = int(norm_style.split('_')[-1])
    # If neither, default to groups = 16
    else:
        groups = 16
    return F.group_norm(x, groups)


# Class-conditional bn
# output size is the number of channels, input size is for the linear layers
# Andy's Note: this class feels messy but I'm not really sure how to clean it up
# Suggestions welcome! (By which I mean, refactor this and make a pull request
# if you want to make this more readable/usable).
class ccbn(nn.Module):
    def __init__(self, output_size, input_size, which_linear, eps=1e-5, momentum=0.1,
                 cross_replica=False, mybn=False, norm_style='bn', ):
        super(ccbn, self).__init__()
        self.output_size, self.input_size = output_size, input_size
        # Prepare gain and bias layers
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Use cross-replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # Norm style?
        self.norm_style = norm_style

        if self.mybn:
            self.bn = myBN(output_size, self.eps, self.momentum)
        elif self.norm_style in ['bn', 'in']:
            self.register_buffer('stored_mean', torch.zeros(output_size))
            self.register_buffer('stored_var', torch.ones(output_size))

    def forward(self, x, y):
        # Calculate class-conditional gains and biases
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        # If using my batchnorm
        if self.mybn:
            return self.bn(x, gain=gain, bias=bias)
        # else:
        else:
            if self.norm_style == 'bn':
                out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
                                   self.training, 0.1, self.eps)
            elif self.norm_style == 'in':
                out = F.instance_norm(x, self.stored_mean, self.stored_var, None, None,
                                      self.training, 0.1, self.eps)
            elif self.norm_style == 'gn':
                out = groupnorm(x, self.normstyle)
            elif self.norm_style == 'nonorm':
                out = x
            return out * gain + bias

    def extra_repr(self):
        s = 'out: {output_size}, in: {input_size},'
        s += ' cross_replica={cross_replica}'
        return s.format(**self.__dict__)


# Normal, non-class-conditional BN
class bn(nn.Module):
    def __init__(self, output_size, eps=1e-5, momentum=0.1,
                 cross_replica=False, mybn=False):
        super(bn, self).__init__()
        self.output_size = output_size
        # Prepare gain and bias layers
        self.gain = P(torch.ones(output_size), requires_grad=True)
        self.bias = P(torch.zeros(output_size), requires_grad=True)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Use cross-replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn

        if mybn:
            self.bn = myBN(output_size, self.eps, self.momentum)
        # Register buffers if neither of the above
        else:
            self.register_buffer('stored_mean', torch.zeros(output_size))
            self.register_buffer('stored_var', torch.ones(output_size))

    def forward(self, x, y=None):
        if self.mybn:
            gain = self.gain.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
            return self.bn(x, gain=gain, bias=bias)
        else:
            return F.batch_norm(x, self.stored_mean, self.stored_var, self.gain,
                                self.bias, self.training, self.momentum, self.eps)


# Generator blocks
# Note that this class assumes the kernel size and padding (and any other
# settings) have been selected in the main generator module and passed in
# through the which_conv arg. Similar rules apply with which_bn (the input
# size [which is actually the number of channels of the conditional info] must
# be preselected)
class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 which_conv=nn.Conv2d, which_bn=bn, activation=None,
                 upsample=None):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(in_channels)
        self.bn2 = self.which_bn(out_channels)
        # upsample layers
        self.upsample = upsample

    def forward(self, x, y):
        h = self.activation(self.bn1(x, y))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h, y))
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


# Residual block for the discriminator
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True,
                 preactivation=False, activation=None, downsample=None, ):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)

    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
        if self.preactivation:
            # h = self.activation(x) # NOT TODAY SATAN
            # Andy's note: This line *must* be an out-of-place ReLU or it
            #              will negatively affect the shortcut connection.
            h = F.relu(x)
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)

        return h + self.shortcut(x)


# In[ ]:


# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.
def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}
    arch[512] = {'in_channels': [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
                 'out_channels': [ch * item for item in [16, 8, 8, 4, 2, 1, 1]],
                 'upsample': [True] * 7,
                 'resolution': [8, 16, 32, 64, 128, 256, 512],
                 'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                               for i in range(3, 10)}}
    arch[256] = {'in_channels': [ch * item for item in [16, 16, 8, 8, 4, 2]],
                 'out_channels': [ch * item for item in [16, 8, 8, 4, 2, 1]],
                 'upsample': [True] * 6,
                 'resolution': [8, 16, 32, 64, 128, 256],
                 'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                               for i in range(3, 9)}}
    arch[128] = {'in_channels': [ch * item for item in [16, 16, 8, 4, 2]],
                 'out_channels': [ch * item for item in [16, 8, 4, 2, 1]],
                 'upsample': [True] * 5,
                 'resolution': [8, 16, 32, 64, 128],
                 'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                               for i in range(3, 8)}}
    arch[64] = {'in_channels': [ch * item for item in [16, 16, 8, 4]],
                'out_channels': [ch * item for item in [16, 8, 4, 2]],
                'upsample': [True] * 4,
                'resolution': [8, 16, 32, 64],
                'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                              for i in range(3, 7)}}
    arch[32] = {'in_channels': [ch * item for item in [4, 4, 4]],
                'out_channels': [ch * item for item in [4, 4, 4]],
                'upsample': [True] * 3,
                'resolution': [8, 16, 32],
                'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                              for i in range(3, 6)}}

    return arch


class Generator(nn.Module):
    def __init__(self, G_ch=64, dim_z=128, bottom_width=4, resolution=128,
                 G_kernel_size=3, G_attn='64', n_classes=1000,
                 num_G_SVs=1, num_G_SV_itrs=1,
                 G_shared=True, shared_dim=0, hier=False,
                 cross_replica=False, mybn=False,
                 G_activation=nn.ReLU(inplace=False),
                 G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
                 BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False,
                 G_init='ortho', skip_init=False, no_optim=False,
                 G_param='SN', norm_style='bn',
                 **kwargs):
        super(Generator, self).__init__()
        # Channel width mulitplier
        self.ch = G_ch
        # Dimensionality of the latent space
        self.dim_z = dim_z
        # The initial spatial dimensions
        self.bottom_width = bottom_width
        # Resolution of the output
        self.resolution = resolution
        # Kernel size?
        self.kernel_size = G_kernel_size
        # Attention?
        self.attention = G_attn
        # number of classes, for use in categorical conditional generation
        self.n_classes = n_classes
        # Use shared embeddings?
        self.G_shared = G_shared
        # Dimensionality of the shared embedding? Unused if not using G_shared
        self.shared_dim = shared_dim if shared_dim > 0 else dim_z
        # Hierarchical latent space?
        self.hier = hier
        # Cross replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # nonlinearity for residual blocks
        self.activation = G_activation
        # Initialization style
        self.init = G_init
        # Parameterization style
        self.G_param = G_param
        # Normalization style
        self.norm_style = norm_style
        # Epsilon for BatchNorm?
        self.BN_eps = BN_eps
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # fp16?
        self.fp16 = G_fp16
        # Architecture dict
        self.arch = G_arch(self.ch, self.attention)[resolution]

        # If using hierarchical latents, adjust z
        if self.hier:
            # Number of places z slots into
            self.num_slots = len(self.arch['in_channels']) + 1
            self.z_chunk_size = (self.dim_z // self.num_slots)
            # Recalculate latent dimensionality for even splitting into chunks
            self.dim_z = self.z_chunk_size * self.num_slots
        else:
            self.num_slots = 1
            self.z_chunk_size = 0

        # Which convs, batchnorms, and linear layers to use
        if self.G_param == 'SN':
            self.which_conv = partial(SNConv2d,
                kernel_size=3, padding=1,
                num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                eps=self.SN_eps)
            self.which_linear = partial(SNLinear,
                num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                eps=self.SN_eps)
        else:
            self.which_conv = partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear

        # We use a non-spectral-normed embedding here regardless;
        # For some reason applying SN to G's embedding seems to randomly cripple G
        self.which_embedding = nn.Embedding
        bn_linear = (partial(self.which_linear, bias=False) if self.G_shared else self.which_embedding)
        self.which_bn = partial(ccbn,
            which_linear=bn_linear,
            cross_replica=self.cross_replica,
            mybn=self.mybn,
            input_size=(self.shared_dim + self.z_chunk_size if self.G_shared else self.n_classes),
            norm_style=self.norm_style,
            eps=self.BN_eps)

        # Prepare model
        # If not using shared embeddings, self.shared is just a passthrough
        self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared else identity())
        # First linear layer
        self.linear = self.which_linear(
            self.dim_z // self.num_slots,
            self.arch['in_channels'][0] * (self.bottom_width ** 2))

        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        # while the inner loop is over a given block
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[GBlock(
                in_channels=self.arch['in_channels'][index],
                out_channels=self.arch['out_channels'][index],
                which_conv=self.which_conv,
                which_bn=self.which_bn,
                activation=self.activation,
                upsample=(partial(F.interpolate, scale_factor=2) if self.arch['upsample'][index] else None)
            )]]

            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [Attention(self.arch['out_channels'][index], self.which_conv)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # output layer: batchnorm-relu-conv.
        # Consider using a non-spectral conv here
        self.output_layer = nn.Sequential(
            bn(self.arch['out_channels'][-1], cross_replica=self.cross_replica, mybn=self.mybn),
            self.activation,
            self.which_conv(self.arch['out_channels'][-1], 3)
        )

        # Initialize weights. Optionally skip init for testing.
        if not skip_init:
            self.init_weights()

        # Set up optimizer
        # If this is an EMA copy, no need for an optim, so just return now
        if no_optim:
            return
        self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
        if G_mixed_precision:
            print('Using fp16 adam in G...')
            self.optim = Adam16(params=self.parameters(), lr=self.lr,
                                betas=(self.B1, self.B2), weight_decay=0,
                                eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                    betas=(self.B1, self.B2), weight_decay=0,
                                    eps=self.adam_eps)

        # LR scheduling, left here for forward compatibility
        # self.lr_sched = {'itr' : 0}# if self.progressive else {}
        # self.j = 0

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for G''s initialized parameters: %d' % self.param_count)

    # Note on this forward function: we pass in a y vector which has
    # already been passed through G.shared to enable easy class-wise
    # interpolation later. If we passed in the one-hot and then ran it through
    # G.shared in this forward function, it would be harder to handle.
    def forward(self, z, y):
        # If hierarchical, concatenate zs and ys
        if self.hier:
            zs = torch.split(z, self.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            ys = [y] * len(self.blocks)

        # First linear layer
        h = self.linear(z)
        # Reshape
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                h = block(h, ys[index])

        # Apply batchnorm-relu-conv-tanh at output
        return torch.tanh(self.output_layer(h))


# Discriminator architecture, same paradigm as G's above
def D_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}
    arch[256] = {'in_channels': [3] + [ch * item for item in [1, 2, 4, 8, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
                 'downsample': [True] * 6 + [False],
                 'resolution': [128, 64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[128] = {'in_channels': [3] + [ch * item for item in [1, 2, 4, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 16]],
                 'downsample': [True] * 5 + [False],
                 'resolution': [64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[64] = {'in_channels': [3] + [ch * item for item in [1, 2, 4, 8]],
                'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
                'downsample': [True] * 4 + [False],
                'resolution': [32, 16, 8, 4, 4],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 7)}}
    arch[32] = {'in_channels': [3] + [item * ch for item in [4, 4, 4]],
                'out_channels': [item * ch for item in [4, 4, 4, 4]],
                'downsample': [True, True, False, False],
                'resolution': [16, 16, 16, 16],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 6)}}
    return arch


class Discriminator(nn.Module):

    def __init__(self, D_ch=64, D_wide=True, resolution=128,
                 D_kernel_size=3, D_attn='64', n_classes=1000,
                 num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
                 D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
                 SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False,
                 D_init='ortho', skip_init=False, D_param='SN', **kwargs):
        super(Discriminator, self).__init__()
        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16
        # Architecture
        self.arch = D_arch(self.ch, self.attention)[resolution]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == 'SN':
            self.which_conv = partial(SNConv2d,
                kernel_size=3, padding=1,
                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                eps=self.SN_eps)
            self.which_linear = partial(SNLinear,
                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                eps=self.SN_eps)
            self.which_embedding = partial(SNEmbedding,
                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                eps=self.SN_eps)
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[DBlock(
                in_channels=self.arch['in_channels'][index],
                out_channels=self.arch['out_channels'][index],
                which_conv=self.which_conv,
                wide=self.D_wide,
                activation=self.activation,
                preactivation=(index > 0),
                downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None)
            )]]
            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [Attention(self.arch['out_channels'][index], self.which_conv)]
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
        # Embedding for projection discrimination
        self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

        # Initialize weights
        if not skip_init:
            self.init_weights()

        # Set up optimizer
        self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
        if D_mixed_precision:
            print('Using fp16 adam in D...')
            self.optim = Adam16(params=self.parameters(), lr=self.lr,
                                betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                    betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        # LR scheduling, left here for forward compatibility
        # self.lr_sched = {'itr' : 0}# if self.progressive else {}
        # self.j = 0

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for D''s initialized parameters: %d' % self.param_count)

    def forward(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        # Apply global sum pooling as in SN-GAN
        h = torch.sum(self.activation(h), [2, 3])
        # Get initial class-unconditional output
        out = self.linear(h)
        # Get projection of final featureset onto class vectors and add to evidence
        out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
        return out


# Parallelized G_D to minimize cross-gpu communication
# Without this, Generator outputs would get all-gathered and then rebroadcast.
class G_D(nn.Module):
    def __init__(self, G, D):
        super(G_D, self).__init__()
        self.G = G
        self.D = D

    def forward(self, z, gy, x=None, dy=None, train_G=False, return_G_z=False,
                split_D=False):
        # If training G, enable grad tape
        with torch.set_grad_enabled(train_G):
            # Get Generator output given noise
            G_z = self.G(z, self.G.shared(gy))
            # Cast as necessary
            if self.G.fp16 and not self.D.fp16:
                G_z = G_z.float()
            if self.D.fp16 and not self.G.fp16:
                G_z = G_z.half()
        # Split_D means to run D once with real data and once with fake,
        # rather than concatenating along the batch dimension.
        if split_D:
            D_fake = self.D(G_z, gy)
            if x is not None:
                D_real = self.D(x, dy)
                return D_fake, D_real
            else:
                if return_G_z:
                    return D_fake, G_z
                else:
                    return D_fake
        # If real data is provided, concatenate it with the Generator's output
        # along the batch dimension for improved efficiency.
        else:
            D_input = torch.cat([G_z, x], 0) if x is not None else G_z
            D_class = torch.cat([gy, dy], 0) if dy is not None else gy
            # Get Discriminator output
            D_out = self.D(D_input, D_class)
            if x is not None:
                return torch.split(D_out, [G_z.shape[0], x.shape[0]])  # D_fake, D_real
            else:
                if return_G_z:
                    return D_out, G_z
                else:
                    return D_out


# In[ ]:


class MyImg():
    def __init__(self, img, tfm):
        self.px = np.array(img)
        self.tfm = tfm
        
    @property
    def size(self):
        h, w, _ = self.px.shape
        return min(w, h)


def pad(img, padding_mode='reflect'):
    p = math.ceil((max(img.size) - min(img.size)) / 2)
    p_horr = p if img.width < img.height else 0
    p_vert = p if img.height < img.width else 0
    img = T.Pad((p_horr, p_vert), padding_mode=padding_mode)(img)
    if img.width != img.height:
        s = min(img.size)
        img = img.crop((0, 0, s, s))
    return img


def take_top(img):
    size = min(img.size)
    bbox = (0, 0, size, size)
    return img.crop(bbox)


def take_diagonal(img):
    w, h = img.size
    size = min(w, h)
    bbox_l = (0, 0, size, size)
    bbox_r = (w - size, h - size, w, h)
    return [img.crop(bbox_l), img.crop(bbox_r)]
    

resize = T.Resize(IMG_SIZE, interpolation=Image.LANCZOS)
resize2x = T.Resize(IMG_SIZE_2, interpolation=Image.LANCZOS)

center_crop = T.Compose([resize, T.CenterCrop(IMG_SIZE)])
center_crop2x = T.Compose([resize2x, T.CenterCrop(IMG_SIZE_2)])

top_crop = T.Compose([T.Lambda(take_top), resize])
top_crop2x = T.Compose([T.Lambda(take_top), resize2x])

two_crops = T.Compose([
    resize, T.FiveCrop(IMG_SIZE),
    T.Lambda(lambda imgs: [imgs[i] for i in [0, 2]])
])
two_crops2x = T.Compose([resize2x, T.Lambda(take_diagonal)])
pad_only = T.Compose([T.Lambda(pad), resize])
pad_only2x = T.Compose([T.Lambda(pad), resize2x])


@jit(nopython=True)
def pad_one_dim(clow, chigh, pad, cmax):
    clow = max(0, clow - pad)
    chigh = min(cmax, chigh + pad)
    return clow, chigh, chigh - clow


def calc_bbox(obj, img_w, img_h, zoom=0.0, try_square=True):
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    
    # occasionally i get bboxes which exceed img size
    xmin, xmax, obj_w = pad_one_dim(xmin, xmax, 0, img_w)
    ymin, ymax, obj_h = pad_one_dim(ymin, ymax, 0, img_h)
    
    if zoom != 0.0:
        pad_w = obj_w * zoom / 2
        pad_h = obj_h * zoom / 2
        xmin, xmax, obj_w = pad_one_dim(xmin, xmax, pad_w, img_w)
        ymin, ymax, obj_h = pad_one_dim(ymin, ymax, pad_h, img_h)
    
    if try_square:
        # try pad both sides equaly
        if obj_w > obj_h:
            pad = (obj_w - obj_h) / 2
            ymin, ymax, obj_h = pad_one_dim(ymin, ymax, pad, img_h)
        elif obj_h > obj_w:
            pad = (obj_h - obj_w) / 2
            xmin, xmax, obj_w = pad_one_dim(xmin, xmax, pad, img_w)

        # if it's still not square, try pad where possible
        if obj_w > obj_h:
            pad = obj_w - obj_h
            ymin, ymax, obj_h = pad_one_dim(ymin, ymax, pad, img_h)
        elif obj_h > obj_w:
            pad = obj_h - obj_w
            xmin, xmax, obj_w = pad_one_dim(xmin, xmax, pad, img_w)
            
    return int(xmin), int(ymin), int(xmax), int(ymax)


@jit(nopython=True)
def bb2wh(bbox):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height


def make_x2res(img, bbox):
    if min(bb2wh(bbox)) < IMG_SIZE_2: return
    ar = img.width / img.height
    if ar == 1.0:          
        tfm_img = resize2x(img)
    elif 1.0 < ar < 1.15:   
        tfm_img = center_crop2x(img)
    elif 1.15 < ar < 1.25:
        tfm_img = pad_only2x(img)
    elif 1.25 < ar < 1.5:
        tfm_img = two_crops2x(img)
    elif 1.0 < 1/ar < 1.6: 
        tfm_img = top_crop2x(img)
    else:
        tfm_img = None
    return tfm_img
    
    
def add_sample(samples, label, tfm, imgs, labels):
    if not samples: return
    elif isinstance(samples, Image.Image):
        imgs.append(MyImg(samples, tfm))
        labels.append(label)
    elif isinstance(samples, list):
        imgs.extend([MyImg(s, tfm) for s in samples])
        labels.extend([label] * len(samples))
    else: assert False

    
def is_valid_file(x):
    return datasets.folder.has_file_allowed_extension(x, IMG_EXTENSIONS)
    

class DogsDataSet(datasets.vision.VisionDataset):
    def __init__(self, root, transforms, target_transform=None, max_samples=None):
        super().__init__(root, transform=None)
        assert isinstance(transforms, list) and len(transforms) == 3
        self.transforms = transforms
        self.target_transform = target_transform
        self.max_samples = max_samples
        self.classes = {}
        
        imgs, labels = self._load_subfolders_images(self.root)   
        assert len(imgs) == len(labels)
        if len(imgs) == 0:
            raise RuntimeError(f'Found 0 files in subfolders of: {self.root}')
        self.imgs    = imgs
        self.labels  = labels
        
    def _create_or_get_class(self, name):
        try:
            label = self.classes[name]
        except KeyError:
            label = len(self.classes)
            self.classes[name] = label
        return label
            
    def _load_subfolders_images(self, root):
        NO_ZOOM, LIGHT_ZOOM, MEDIUM_ZOOM = 0.0, 0.08, 0.12
        n_pad, n_center, n_top, n_2crops, n_skip, n_dup, n_noop = 0, 0, 0, 0, 0, 0, 0
        imgs, labels, paths = [], [], []
        
        add_sample_ = partial(add_sample, imgs=imgs, labels=labels)

        for root, _, fnames in sorted(os.walk(root)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                paths.append(path)
        if self.max_samples:
            paths = paths[:self.max_samples]

        for path in paths:
            if not is_valid_file(path): continue
            img = datasets.folder.default_loader(path)
            annotation_basename = os.path.splitext(os.path.basename(path))[0]
            annotation_dirname = next(dirname for dirname in os.listdir(ANNOT_PATH) if dirname.startswith(annotation_basename.split('_')[0]))
            annotation_filename = os.path.join(ANNOT_PATH, annotation_dirname, annotation_basename)
            tree = ET.parse(annotation_filename)
            root = tree.getroot()
            objects = root.findall('object')
            for o in objects:
                name = o.find('name').text
                label = self._create_or_get_class(name)
                prev_bbox, tfm_imgs = None, None
                
                bbox = calc_bbox(o, img_w=img.width, img_h=img.height, zoom=LIGHT_ZOOM)
                obj_img = img.crop(bbox)
                add_sample_(make_x2res(obj_img, bbox), label, 2)

                bbox = calc_bbox(o, img_w=img.width, img_h=img.height)                
                if min(bb2wh(bbox)) < IMG_SIZE:
                    # don't want pixel mess in gen imgs
                    n_skip += 1; continue
                obj_img = img.crop(bbox)
                ar = obj_img.width / obj_img.height
                if ar == 1.0:
                    tfm_imgs = [resize(obj_img)]; n_noop += 1
                elif 1.0 < ar < 1.3:
                    tfm_imgs = [center_crop(obj_img), pad_only(obj_img)]
                    n_center += 1; n_pad += 1
                elif 1.3 <= ar < 1.5:
                    tfm_imgs = two_crops(obj_img) + [pad_only(obj_img)]
                    n_2crops += 2; n_pad += 1
                elif 1.5 <= ar < 2.0:
                    tfm_imgs = two_crops(obj_img); n_2crops += 2
                elif 1.0 < 1/ar < 1.5:
                    tfm_imgs = [top_crop(obj_img), pad_only(obj_img)]
                    n_top += 1; n_pad += 1
                elif 1.5 <= 1/ar < 1.8:
                    tfm_imgs = [top_crop(obj_img)]; n_top += 1
                else:
                    tfm_imgs = None; n_skip += 1
                add_sample_(tfm_imgs, label, 0)
                add_sample_(make_x2res(obj_img, bbox), label, 1)
                prev_bbox = bbox

                bbox = calc_bbox(o, img_w=img.width, img_h=img.height, zoom=MEDIUM_ZOOM, try_square=False)
                if bbox == prev_bbox:
                    n_dup += 1; continue
                if min(bb2wh(bbox)) < IMG_SIZE_2: continue
                obj_img = img.crop(bbox)
                ar = obj_img.width / obj_img.height
                if 1.3 < ar < 1.5:
                    tfm_imgs = two_crops(obj_img); n_2crops += 2
                elif 1.05 < 1/ar < 1.6: # maybe tall
                    tfm_imgs = top_crop(obj_img); n_top += 1
                else: continue
                add_sample_(tfm_imgs, label, 0)
                add_sample_(make_x2res(obj_img, bbox), label, 1)
                prev_bbox = bbox
        
        n_x1, n_x2 = 0, 0
        for i, img in enumerate(imgs):
            if img.size == IMG_SIZE: 
                n_x1 +=1
            else: 
                n_x2 +=1
                
        print(f'Found {len(self.classes)} classes\nLoaded 64x64 {n_x1} images\n'
              f'Loaded 128x128 {n_x2} images\n')
        print(f'Pad only: {n_pad}\nCrop center: {n_center}\n'
              f'Crop top: {n_top}\nCrop 2 times: {n_2crops}\n'
              f'Take as-is: {n_noop}\nSkip: {n_skip}\nSame bbox: {n_dup}')
        return imgs, labels
    
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        tfms = self.transforms[img.tfm]
        img = tfms(image=img.px)['image']          
        if self.target_transform:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.imgs)
    
    
def create_runtime_tfms():
    mean, std = [0.5]*3, [0.5]*3
    resize_to_64 = A.SmallestMaxSize(IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
    out = [A.HorizontalFlip(p=0.5), A.Normalize(mean=mean, std=std), ToTensor()]

    rand_crop = A.Compose([
        A.SmallestMaxSize(IMG_SIZE + 8, interpolation=cv2.INTER_LANCZOS4),
        A.RandomCrop(IMG_SIZE, IMG_SIZE)
    ])

    affine_1 = A.ShiftScaleRotate(
        shift_limit=0, scale_limit=0.1, rotate_limit=8, 
        interpolation=cv2.INTER_CUBIC, 
        border_mode=cv2.BORDER_REFLECT_101, p=1.0)
    affine_1 = A.Compose([affine_1, resize_to_64])

    affine_2 = A.ShiftScaleRotate(
        shift_limit=0.06, scale_limit=(-0.06, 0.18), rotate_limit=6, 
        interpolation=cv2.INTER_CUBIC, 
        border_mode=cv2.BORDER_REFLECT_101, p=1.0)
    affine_2 = A.Compose([affine_2, resize_to_64])

    tfm_0 = A.Compose(out)
    tfm_1 = A.Compose([A.OneOrOther(affine_1, rand_crop, p=1.0), *out])
    tfm_2 = A.Compose([affine_2, *out])
    return [tfm_0, tfm_1, tfm_2]


def get_data_loaders(data_root=None, batch_size=32, num_workers=2, shuffle=True,
                     pin_memory=True, drop_last=True):
    print('Using dataset root location %s' % data_root)
    # tfms_x1, tfms_x2 = create_runtime_tfms()
    train_set = DogsDataSet(data_root, create_runtime_tfms())
    # Prepare loader; the loaders list is for forward compatibility with
    # using validation / test splits.
    loaders = []
    loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory,
                     'drop_last': drop_last}  # Default, drop last incomplete batch
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, **loader_kwargs)
    loaders.append(train_loader)
    return loaders


# In[ ]:


def seed_rng(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


# Utility to peg all roots to a base root
# If a base root folder is provided, peg all other root folders to it.
def update_config_roots(config):
    if config['base_root']:
        print('Pegging all root folders to base root %s' % config['base_root'])
        for key in ['data', 'weights', 'logs', 'samples']:
            config['%s_root' % key] = '%s/%s' % (config['base_root'], key)
    return config


def prepare_root(config):
    for key in ['weights_root', 'logs_root', 'samples_root']:
        if not os.path.exists(config[key]):
            print('Making directory %s for %s...' % (config[key], key))
            os.mkdir(config[key])


# Function to join strings or ignore them
# Base string is the string to link "strings," while strings
# is a list of strings or Nones.
def join_strings(base_string, strings):
    return base_string.join([item for item in strings if item])


# In[ ]:


# A highly simplified convenience class for sampling from distributions
# One could also use PyTorch's inbuilt distributions package.
# Note that this class requires initialization to proceed as
# x = Distribution(torch.randn(size))
# x.init_distribution(dist_type, **dist_kwargs)
# x = x.to(device,dtype)
# This is partially based on https://discuss.pytorch.org/t/subclassing-torch-tensor/23754/2
class Distribution(torch.Tensor):
    # Init the params of the distribution
    def init_distribution(self, dist_type, **kwargs):
        self.dist_type = dist_type
        self.dist_kwargs = kwargs
        if self.dist_type == 'normal':
            self.mean, self.var = kwargs['mean'], kwargs['var']
        elif self.dist_type == 'categorical':
            self.num_categories = kwargs['num_categories']

    def sample_(self):
        if self.dist_type == 'normal':
            self.normal_(self.mean, self.var)
        elif self.dist_type == 'categorical':
            self.random_(0, self.num_categories)
            # return self.variable

    # Silly hack: overwrite the to() method to wrap the new object
    # in a distribution as well
    def to(self, *args, **kwargs):
        new_obj = Distribution(self)
        new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
        new_obj.data = super().to(*args, **kwargs)
        return new_obj


def prepare_z_y(G_batch_size, dim_z, nclasses, device='cuda', fp16=False, z_var=1.0):
    z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
    z_.init_distribution('normal', mean=0, var=z_var)
    z_ = z_.to(device, torch.float16 if fp16 else torch.float32)

    if fp16:
        z_ = z_.half()

    y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
    y_.init_distribution('categorical', num_categories=nclasses)
    y_ = y_.to(device, torch.int64)
    return z_, y_


# Sample function for use with inception metrics
def sample_fn(G, z_, y_, config):
    with torch.no_grad():
        z_.sample_()
        y_.sample_()
        G_z = G(z_, G.shared(y_))
        return G_z, y_


def toggle_grad(model, on_or_off):
    for param in model.parameters():
        param.requires_grad = on_or_off
        
        
# Simple wrapper that applies EMA to a model. COuld be better done in 1.0 using
# the parameters() and buffers() module functions, but for now this works
# with state_dicts using .copy_
class apply_ema(object):
    def __init__(self, source, target, decay=0.9999, start_itr=0):
        self.source = source
        self.target = target
        self.decay = decay
        # Optional parameter indicating what iteration to start the decay at
        self.start_itr = start_itr
        # Initialize target's params to be source's
        self.source_dict = self.source.state_dict()
        self.target_dict = self.target.state_dict()
        print('Initializing EMA parameters to be source parameters...')
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.source_dict[key].data)
                # target_dict[key].data = source_dict[key].data # Doesn't work!

    def update(self, itr=None):
        # If an iteration counter is provided and itr is less than the start itr,
        # peg the ema weights to the underlying weights.
        if itr and itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.target_dict[key].data * decay
                                                 + self.source_dict[key].data * (1 - decay))


# In[ ]:


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def plot_imgs(imgs, cols=8, size=4):
    n = imgs.shape[0]
    rows = n // cols
    _, axes = plt.subplots(figsize=(cols * size, rows * size), ncols=cols, nrows=rows)
    for i, ax in enumerate(axes.flatten()):
        img = denorm(imgs[i]).numpy().transpose(1, 2, 0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(img)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.show()
    
    
def save_and_sample(G, G_ema, fixed_z, fixed_y, config):
    which_G = G_ema if config['ema'] and config['use_ema'] else G
    with torch.no_grad():
        fixed_Gz = which_G(fixed_z, which_G.shared(fixed_y))
        sample_images = fixed_Gz.float().cpu()
        plot_imgs(sample_images, size=2)


def create_train_fn(G, D, GD, z_, y_, ema, state_dict, config):
    def train(x, y):
        G.optim.zero_grad()
        D.optim.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(x, config['batch_size'])
        y = torch.split(y, config['batch_size'])
        counter = 0

        # Optionally toggle D and G's "require_grad"
        if config['toggle_grads']:
            toggle_grad(D, True)
            toggle_grad(G, False)

        for step_index in range(config['num_D_steps']):
            # If accumulating gradients, loop multiple times before an optimizer step
            D.optim.zero_grad()
            for accumulation_index in range(config['num_D_accumulations']):
                z_.sample_()
                y_.sample_()
                D_fake, D_real = GD(z_[:config['batch_size']], y_[:config['batch_size']],
                                    x[counter], y[counter], train_G=False,
                                    split_D=config['split_D'])

                # Compute components of D's loss, average them, and divide by
                # the number of gradient accumulations
                D_loss_real, D_loss_fake = loss_hinge_dis(D_fake, D_real)
                D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
                D_loss.backward()
                counter += 1

            # Optionally apply ortho reg in D
            if config['D_ortho'] > 0.0:
                ortho(D, config['D_ortho'])

            D.optim.step()

        # Optionally toggle "requires_grad"
        if config['toggle_grads']:
            toggle_grad(D, False)
            toggle_grad(G, True)

        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()

        # If accumulating gradients, loop multiple times
        for accumulation_index in range(config['num_G_accumulations']):
            z_.sample_()
            y_.sample_()
            D_fake = GD(z_, y_, train_G=True, split_D=config['split_D'])
            G_loss = loss_hinge_gen(D_fake) / float(config['num_G_accumulations'])
            G_loss.backward()

        # Optionally apply modified ortho reg in G
        if config['G_ortho'] > 0.0:
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            ortho(G, config['G_ortho'], blacklist=[param for param in G.shared.parameters()])
        G.optim.step()

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
            ema.update(state_dict['itr'])

        out = {'G_loss': float(G_loss.item()),
               'D_loss_real': float(D_loss_real.item()),
               'D_loss_fake': float(D_loss_fake.item())}
        # Return G's loss and the components of D's loss.
        return out

    return train


def loss_hinge_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake


def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss


# Apply modified ortho reg to a model
# This function is an optimized version that directly computes the gradient,
# instead of computing and then differentiating the loss.
def ortho(model, strength=1e-4, blacklist=[]):
    with torch.no_grad():
        for param in model.parameters():
            # Only apply this to parameters with at least 2 axes, and not in the blacklist
            if len(param.shape) < 2 or any([param is item for item in blacklist]):
                continue
            w = param.view(param.shape[0], -1)
            grad = (2 * torch.mm(torch.mm(w, w.t())
                                 * (1. - torch.eye(w.shape[0], device=w.device)), w))
            param.grad.data += strength * grad.view(param.shape)


# Default ortho reg
# This function is an optimized version that directly computes the gradient,
# instead of computing and then differentiating the loss.
def default_ortho(model, strength=1e-4, blacklist=[]):
    with torch.no_grad():
        for param in model.parameters():
            # Only apply this to parameters with at least 2 axes & not in blacklist
            if len(param.shape) < 2 or param in blacklist:
                continue
            w = param.view(param.shape[0], -1)
            grad = (2 * torch.mm(torch.mm(w, w.t())
                                 - torch.eye(w.shape[0], device=w.device), w))
            param.grad.data += strength * grad.view(param.shape)


# In[ ]:


def run(G, D, GD, loaders, state_dict, config, device, G_ema=None, ema=None):
    # Prepare noise and randomly sampled label arrays
    # Allow for different batch sizes in G
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    z_, y_ = prepare_z_y(
        G_batch_size, G.dim_z, config['n_classes'], device=device, fp16=config['G_fp16'])
    # Prepare a fixed z & y to see individual sample evolution throghout training
    fixed_z, fixed_y = prepare_z_y(
        G_batch_size, G.dim_z, config['n_classes'], device=device, fp16=config['G_fp16'])
    fixed_z.sample_()
    fixed_y.sample_()
    # Loaders are loaded, prepare the training function
    train = create_train_fn(G, D, GD, z_, y_, ema, state_dict, config)

    print('Beginning training at epoch %d...' % state_dict['epoch'])
    start_time = time.perf_counter()
    total_iters = config['num_epochs'] * len(loaders[0])

    # Train for specified number of epochs, although we mostly track G iterations.
    for epoch in range(state_dict['epoch'], config['num_epochs']):
        for i, (x, y) in enumerate(loaders[0]):
            # Increment the iteration counter
            state_dict['itr'] += 1
            # Make sure G and D are in training mode, just in case they got set to eval
            # For D, which typically doesn't have BN, this shouldn't matter much.
            G.train()
            D.train()
            if config['ema']:
                G_ema.train()
            if config['D_fp16']:
                x, y = x.to(device).half(), y.to(device)
            else:
                x, y = x.to(device), y.to(device)
            metrics = train(x, y)

            if not (state_dict['itr'] % config['log_interval']):
                curr_time = time.perf_counter()
                curr_time_str = datetime.datetime.fromtimestamp(curr_time).strftime('%H:%M:%S')
                elapsed = str(datetime.timedelta(seconds=(curr_time - start_time)))
                log = (
                    "[{}] [{}] [{} / {}] Ep {}, ".format(curr_time_str, elapsed, state_dict['itr'], total_iters, epoch) + 
                    ', '.join(['%s : %+4.3f' % (key, metrics[key]) for key in metrics])
                )
                print(log)

            # Save weights and copies as configured at specified interval
            if not (state_dict['itr'] % config['save_every']):
                if config['G_eval_mode']:
                    print('Switchin G to eval mode...')
                    G.eval()
                    # if config['ema']:
                        # G_ema.eval()
                save_and_sample(G, G_ema, fixed_z, fixed_y, config)
                
            if config['stop_after'] > 0 and int(time.perf_counter() - start_time) > config['stop_after']:
                print("Time limit reached! Stopping training!") 
                return

        # Increment epoch counter at end of epoch
        state_dict['epoch'] += 1


# In[ ]:


def trunc_trick(bs, z_dim, bound=2.0):
    z = torch.randn(bs, z_dim)
    while z.min() < -bound or bound < z.max():
        z = z.where((-bound < z) & (z < bound), torch.randn_like(z))
    return z


def collect_bn_stats(G, n_samples, config):
    im_batch_size = config['n_classes']
    G.train()

    for i_batch in range(0, n_samples, im_batch_size):
        with torch.no_grad():
            z = torch.randn(im_batch_size, G.dim_z, device=device)
            y = torch.arange(im_batch_size).to(device)
            images = G(z, G.shared(y)).float().cpu()


def generate_images(out_dir, G, n_images, config):
    im_batch_size = config['n_classes']
    z_bound = config['trunc_z']
    if z_bound > 0.0: print(f'Truncating z to (-{z_bound}, {z_bound})')

    for i_batch in range(0, n_images, im_batch_size):
        with torch.no_grad():
            if z_bound > 0.0:
                z = trunc_trick(im_batch_size, G.dim_z, bound=z_bound).to(device)
            else:
                z = torch.randn(im_batch_size, G.dim_z, device=device)
            y = torch.arange(im_batch_size).to(device)
            images = G(z, G.shared(y)).float().cpu()

        if i_batch + im_batch_size > n_images:
            n_last_images = n_images - i_batch
            print(f'Taking only {n_last_images} images from the last batch...')
            images = images[:n_last_images]

        for i_image, image in enumerate(images):
            fname = os.path.join(out_dir, f'image_{i_batch+i_image:05d}.png')
            image = denorm(image)
            if config['denoise']:
                image = image * 256
                image = image.numpy().transpose(1, 2, 0).astype('uint8')
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = cv2.fastNlMeansDenoisingColored(
                    src=image, dst=None, 
                    h=config['denoise_str_lum'], 
                    hColor=config['denoise_str_col'], 
                    templateWindowSize=config['denoise_kernel_size'], 
                    searchWindowSize=config['denoise_search_window']
                )
                cv2.imwrite(fname, image)
            else:
                torchvision.utils.save_image(image, fname)


def show_saved_samples(n_images=256, cols=8, size=3):
    rows = math.ceil(n_images / cols)
    _, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size))
    for i, ax in enumerate(axes.flatten()):
        img = Image.open(f'{OUT_DIR}/image_{i:05d}.png')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(img)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.show()


# In[ ]:


activation_dict = {
    'inplace_relu': nn.ReLU(inplace=True),
    'relu': nn.ReLU(inplace=False),
    'ir': nn.ReLU(inplace=True),
}

config = {
    'num_workers': 2,
    'pin_memory': True,
    'shuffle': True,
    'use_multiepoch_sampler': False,

    # Model
    'model': 'BigGAN',
    'G_param': 'SN',
    'D_param': 'SN',
    'G_ch': 64,
    'D_ch': 64,
    'G_depth': 1,
    'D_depth': 1,
    'D_wide': True,
    'G_shared': False,
    'shared_dim': 0,
    'dim_z': 128,
    'z_var': 1.0,
    'hier': False,
    'cross_replica': False,
    'mybn': False,
    'G_nl': 'relu',
    'D_nl': 'relu',
    'G_attn': '64',
    'D_attn': '64',
    'norm_style': 'bn',

    ### Model init stuff ###
    'seed': 0,
    'G_init': 'ortho',
    'D_init': 'ortho',
    'skip_init': False,

    ### Optimizer stuff ###
    'G_lr': 5e-5,
    'D_lr': 2e-4,
    'G_B1': 0.0,
    'D_B1': 0.0,
    'G_B2': 0.999,
    'D_B2': 0.999,

    ### Batch size, parallel, and precision stuff ###
    'batch_size': 64,
    'G_batch_size': 0,
    'num_G_accumulations': 1,
    'num_D_steps': 2,
    'num_D_accumulations': 1,
    'split_D': False,
    'num_epochs': 100,
    'parallel': False,
    'G_fp16': False,
    'D_fp16': False,
    'D_mixed_precision': False,
    'G_mixed_precision': False,
    'accumulate_stats': False,
    'num_standing_accumulations': 16,

    ### Bookkeping stuff ###
    'G_eval_mode': False,
    'save_every': 2000,
    'num_save_copies': 2,
    'num_best_copies': 2,
    'which_best': 'IS',
    'no_fid': False,
    'test_every': 5000,
    'num_inception_images': 50000,
    'hashname': False,
    'base_root': '',
    'data_root': '../data',
    'weights_root': '../weights',
    'logs_root': '../logs',
    'samples_root': '../samples',
    'pbar': 'mine',
    'name_suffix': '',
    'experiment_name': '',
    'config_from_name': False,

    ### EMA Stuff ###
    'ema': False,
    'ema_decay': 0.9999,
    'use_ema': False,
    'ema_start': 0,

    ### Numerical precision and SV stuff ###
    'adam_eps': 1e-8,
    'BN_eps': 1e-5,
    'SN_eps': 1e-8,
    'num_G_SVs': 1,
    'num_D_SVs': 1,
    'num_G_SV_itrs': 1,
    'num_D_SV_itrs': 1,

    ### Ortho reg stuff ###
    'G_ortho': 0.0,  # 1e-4 is default for BigGAN
    'D_ortho': 0.0,
    'toggle_grads': True,

    ### Which train function ###
    'which_train_fn': 'GAN',

    ### Resume training stuff
    'load_weights': '',
    'resume': False,

    ### Log stuff ###
    'logstyle': '%3.3e',
    'log_G_spectra': False,
    'log_D_spectra': False,
    'sv_log_interval': 10,
    'log_interval': 100,
}

config['resolution'] = 64
config['n_classes'] = 120
config['data_root'] = INPUT_PATH


# In[ ]:


# Current experiment
config['num_epochs'] = 100
config['num_D_steps'] = 1
config['G_lr'] = 1e-4 * 0.85
config['D_lr'] = 4e-4 * 0.85
config['D_B2'] = 0.999
config['G_B2'] = 0.999
config['G_ch'] = 32
config['D_ch'] = 64
config['G_attn'] = '32'
config['D_attn'] = '32'
config['G_nl'] = 'inplace_relu'
config['D_nl'] = 'inplace_relu'
config['G_shared'] = True
config['shared_dim'] = 128
config['dim_z'] = 120
config['hier'] = True
config['SN_eps'] = 1e-8
config['BN_eps'] = 1e-5
config['adam_eps'] = 1e-8
config['G_ortho'] = 0.0
config['G_init'] = 'ortho'
config['D_init'] = 'ortho'
config['G_eval_mode'] = True
config['save_every'] = 5000
config['log_interval'] = 500
config['batch_size'] = 32

config['ema'] = True
config['use_ema'] = True
config['ema_decay'] = 0.9999
config['ema_start'] = 50_000

config['stop_after'] = 31_500

config['trunc_z'] = 0.0
config['denoise'] = True
config['denoise_str_lum'] = 2.65
config['denoise_str_col'] = 1.0
config['denoise_kernel_size'] = 7
config['denoise_search_window'] = 11


# In[ ]:


config = update_config_roots(config)
device = 'cuda'
seed_rng(1234)
prepare_root(config)
torch.backends.cudnn.benchmark = True


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Prepare data; the Discriminator's batch size is all that needs to be passed\n# to the dataloader, as G doesn't require dataloading.\n# Note that at every loader iteration we pass in enough data to complete\n# a full D iteration (regardless of number of D steps and accumulations)\nD_batch_size = (config['batch_size'] * config['num_D_steps'] * config['num_D_accumulations'])\nloaders = get_data_loaders(\n    data_root=INPUT_PATH,\n    batch_size=D_batch_size,\n    num_workers=config['num_workers'],\n    shuffle=config['shuffle'],\n    pin_memory=config['pin_memory'],\n    drop_last=True\n)")


# In[ ]:


config['G_activation'] = activation_dict[config['G_nl']]
config['D_activation'] = activation_dict[config['D_nl']]

G = Generator(**config).to(device)
D = Discriminator(**config).to(device)

# If using EMA, prepare it
if config['ema']:
    print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
    G_ema = Generator(**{**config, 'skip_init': True, 'no_optim': True}).to(device)
    ema = apply_ema(G, G_ema, config['ema_decay'], config['ema_start'])
else:
    G_ema, ema = None, None

GD = G_D(G, D)
print('Number of params in G: {} D: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [G, D]]))


# In[ ]:


state_dict = {
    'itr': 0,
    'epoch': 0, 
    'config': config
}


# In[ ]:


run(G, D, GD, loaders, state_dict, config, device, G_ema=G_ema, ema=ema)


# In[ ]:


get_ipython().run_cell_magic('time', '', "if not os.path.exists(OUT_DIR):\n    os.mkdir(OUT_DIR)\n    \nwhich_G = G_ema if config['ema'] and config['use_ema'] else G\n\nif config['use_ema']:\n    collect_bn_stats(G_ema, 20_000, config)\n\ngc.collect()\n# G.eval()\ngenerate_images(OUT_DIR, which_G, 10_000, config)\nshutil.make_archive('images', 'zip', OUT_DIR)")


# In[ ]:


get_ipython().system('ls -1 $OUT_DIR | wc -l')


# In[ ]:


show_saved_samples()


# In[ ]:


get_ipython().system('rm -rf $OUT_DIR')

