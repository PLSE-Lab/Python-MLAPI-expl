#!/usr/bin/env python
# coding: utf-8

# ## This model uses drhabib's L2 model and my L1 model. Predictions from both models are taken using TTA. The final output is obtained by taking the average of two model outputs and hand tuned class boundaries.  

# In[ ]:


import os
os.listdir('../input/kaggle-public')


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import StratifiedKFold
from joblib import load, dump
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from torchvision import models as md
from torch import nn
from torch.nn import functional as F
import re
import math
import collections
from functools import partial
from torch.utils import model_zoo
from sklearn import metrics
from collections import Counter
import json

import sys
package_path = '../input/efficientnet-pytorch/efficientnet-pytorch/EfficientNet-PyTorch-master'
sys.path.append(package_path)


# ### The following cell is needed to run drhabib's pretrained model

# In[ ]:


#I could not figure out how to install package in local kernel so i just stole from github =)
#code stolen from https://github.com/lukemelas/EfficientNet-PyTorch


"""
This file contains helper functions for building the model and for loading model parameters.
These helper functions are built to mirror those in the official TensorFlow implementation.
"""



# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size'])


# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])


# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def relu_fn(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)


def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def get_same_padding_conv2d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)

class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""
    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class Identity(nn.Module):
    def __init__(self,):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


########################################################################
############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############
########################################################################


def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    return params_dict[model_name]


class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.

        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2,
                 drop_connect_rate=0.2, image_size=None, num_classes=1000):
    """ Creates a efficientnet model. """

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        # data_format='channels_last',  # removed, this is always true in PyTorch
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


url_map = {
    'efficientnet-b0': 'http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth',
    'efficientnet-b1': 'http://storage.googleapis.com/public-models/efficientnet-b1-dbc7070a.pth',
    'efficientnet-b2': 'http://storage.googleapis.com/public-models/efficientnet-b2-27687264.pth',
    'efficientnet-b3': 'http://storage.googleapis.com/public-models/efficientnet-b3-c8376fa2.pth',
    'efficientnet-b4': 'http://storage.googleapis.com/public-models/efficientnet-b4-e116e8b3.pth',
    'efficientnet-b5': 'http://storage.googleapis.com/public-models/efficientnet-b5-586e6cc6.pth',
}

def load_pretrained_weights(model, model_name, load_fc=True):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    state_dict = model_zoo.load_url(url_map[model_name])
    if load_fc:
        model.load_state_dict(state_dict)
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        res = model.load_state_dict(state_dict, strict=False)
        assert str(res.missing_keys) == str(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'
    print('Loaded pretrained weights for {}'.format(model_name))
    
    
class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._dropout = self._global_params.dropout_rate
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = relu_fn(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = relu_fn(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """

        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self._dropout:
            x = F.dropout(x, p=self._dropout, training=self.training)
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = EfficientNet.from_name(model_name, override_params={'num_classes': num_classes})
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet_b'+str(i) for i in range(num_models)]
        if model_name.replace('-','_') not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))


# In[ ]:


#making model
md_ef = EfficientNet.from_pretrained('efficientnet-b5', num_classes=1)


# In[ ]:


#copying weighst to the local directory 
get_ipython().system('mkdir models')
get_ipython().system("cp '../input/kaggle-public/abcdef.pth' 'models'")


# In[ ]:


import multiprocessing
from tqdm import tqdm
import numpy as np
import glob
import cv2
import sys
import os


class ImagePreprocessor(object):
    def __init__(self, root_dir: str, save_dir: str, img_size: int, tolerance: int = 10, remove_outer_pixels: float = 0.0):
        """
        Preprocess images for kaggle competitions and general training purposes.

        args:
            root_dir  = absolute path to images folder
            save_dir  = folder in which to store processed images
            img_size  = final image dimensions, common values : 224, 512
            tolerance = tolerance value for pitch_black_remover func
            remove_outer_pixels = remove boundary pixels of image
        """
        if remove_outer_pixels > 0.50:
            print("ERROR: eroding more than 50% of image")
            raise InterruptedError

        self.root_dir = root_dir
        self.img_size = img_size
        self.tolerance = tolerance
        self.remove_outer_pixels = remove_outer_pixels

        self.images = glob.glob(f"{self.root_dir}/*.png") + glob.glob(
            f"{self.root_dir}/*.jpeg") + glob.glob(f"{self.root_dir}/*.jpg")
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.total_count = len(self.images)

    # counter decorator
    @staticmethod
    def _counter(func):
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
        return wrapper

    # different preprocessing methods

    @staticmethod
    def light_sensitivity_reducer(img: np.ndarray, alpha: int = 4, beta: int = -4, gamma: int = 128):
        """smooth image and apply ben's preprocessing"""
        return cv2.addWeighted(img, alpha, cv2.GaussianBlur(img, (0, 0), 30), beta, gamma)

    @staticmethod
    def outer_pixels_remover(img: np.ndarray, scale: float):
        """remove outer/boundary pixels of image"""
        scale_2 = scale / 2.0
        miny = int(img.shape[0]*scale_2)
        maxy = int(img.shape[0]-miny)
        minx = int(img.shape[1]*scale_2)
        maxx = int(img.shape[1]-minx)
        return img[miny:maxy, minx:maxx]

    @staticmethod
    def scale_image(img: np.ndarray, img_size: int):
        """resize image based on given scale"""
        return cv2.resize(img, (img_size, img_size))

    @staticmethod
    def pitch_black_remover(img: np.ndarray, tolerance: int = 10):
        """remove black pixels in image edges"""
        if img.ndim == 2:
            img_mask = img > tolerance
            return img[np.ix_(img_mask.any(1), img_mask.any(0))]
        elif img.ndim == 3:
            greyed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_mask = greyed > tolerance
            img_1 = img[:, :, 0][np.ix_(img_mask.any(1), img_mask.any(0))]
            if img_1.shape[0] == 0:
                return img
            img_2 = img[:, :, 1][np.ix_(img_mask.any(1), img_mask.any(0))]
            img_3 = img[:, :, 2][np.ix_(img_mask.any(1), img_mask.any(0))]
            return np.stack([img_1, img_2, img_3], axis=-1)
        else:
            print("Image has more than 3 dimensions")
            raise InterruptedError

    # collaging different methods together and preprocessing images

    def replace_existing(self):
        import shutil
        shutil.rmtree(self.root_dir)
        os.makedirs(self.root_dir)
        processed_images = [i.path for i in os.scandir(
            self.save_dir) if i.is_file()]
        processor_pool = multiprocessing.Pool(64)
        for counter, _ in enumerate(processor_pool.imap_unordered(lambda x: shutil.move(x, self.root_dir), processed_images), 1):
            sys.stdout.write(
                f"\rMoving : {(counter/self.total_count)*100:3.2f}% \t[ {counter}/{self.total_count} ]")
        os.rmdir(self.save_dir)
        sys.stdout.write("\n\n")

    def forward(self, image: str):
        """take a single image path, preprocesse image and store preprocessed image"""
        img = cv2.imread(image)
        img = self.pitch_black_remover(img, tolerance=self.tolerance)
        img = self.scale_image(img, img_size=self.img_size)
#         img = self.light_sensitivity_reducer(img)
        if self.remove_outer_pixels > 0.0:
            img = self.outer_pixels_remover(img, self.remove_outer_pixels)
        cv2.imwrite(os.path.join(self.save_dir, image.split('/')[-1]), img)

    def run(self, replace: bool = False):
        """process all images in root_dir in an iterative way"""
        for image in tqdm(self.images):
            # add logging if required
            self.forward(image)
        if replace:
            self.replace_existing()

    def parallel_run(self, workers: int = multiprocessing.cpu_count(), replace: bool = False):
        """process all images in root_dir parallely using python's multiprocessing library"""
        # haven't figured out a stable way for logging in case of multiprocessing
        processor_pool = multiprocessing.Pool(workers)

        for counter, _ in enumerate(processor_pool.imap_unordered(self.forward, self.images), 1):
            sys.stdout.write(
                f"\rProgress : {(counter/self.total_count)*100:3.2f}% \t[ {counter}/{self.total_count} ]")
        if replace:
            self.replace_existing()
        sys.stdout.write("\n\n")
        
preprocessor = ImagePreprocessor(root_dir='../input/aptos2019-blindness-detection/test_images/', 
                                 save_dir='test_processed', 
                                 img_size=256,
                                 tolerance=10, 
                                 remove_outer_pixels=0
                                )
preprocessor.parallel_run(multiprocessing.cpu_count(),False)


# In[ ]:


def get_df():
    base_image_dir = os.path.join('..', 'input/aptos2019-blindness-detection/')
    train_dir = os.path.join(base_image_dir,'train_images/')
    df = pd.read_csv(os.path.join(base_image_dir, 'train.csv'))
    df['path'] = df['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))
    df = df.drop(columns=['id_code'])
    df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
    test_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
    return df, test_df

df, test_df = get_df()


# In[ ]:


#you can play around with tfms and image sizes
bs = 64
sz = 224
tfms = get_transforms(do_flip=True,flip_vert=True)


# In[ ]:


data = (ImageList.from_df(df=df,path='./',cols='path') 
        .split_by_rand_pct(0.2) 
        .label_from_df(cols='diagnosis',label_cls=FloatList) 
        .transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') 
        .databunch(bs=bs,num_workers=4) 
        .normalize(imagenet_stats)  
       )


# In[ ]:


def qk(y_pred, y):
    return torch.tensor(cohen_kappa_score(torch.round(y_pred), y, weights='quadratic'), device='cuda:0')


# In[ ]:


learn = Learner(data, 
                md_ef, 
                metrics = [qk], 
                model_dir="models")
learn.data.add_test(ImageList.from_df(test_df,
                             'test_processed/',
                             folder='',
                             suffix='.png',
                             ))
learn.load('abcdef')
pass


# In[ ]:


learn.to_fp32()
pass


# In[ ]:


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >=coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        y = y.cpu().apply_(lambda x: 0 if x==0 else 1 if x==0.25 else 2 if x==0.5 else 3 if x==0.75 else 4)
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.125,0.375,0.625,0.875]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
        print(-loss_partial(self.coef_['x']))

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']

opt = OptimizedRounder()


# In[ ]:


__all__ = []

def _tta_only(learn:Learner, ds_type:DatasetType=DatasetType.Valid, activ:nn.Module=None, scale:float=1.35) -> Iterator[List[Tensor]]:
    "Computes the outputs for several augmented inputs for TTA"
    dl = learn.dl(ds_type)
    ds = dl.dataset
    old = ds.tfms
#     activ = ifnone(activ, _loss_func2activ(learn.loss_func))
    augm_tfm = [o for o in learn.data.train_ds.tfms]
    try:
        pbar = master_bar(range(4))
        for i in pbar:
            row = 1 if i&1 else 0
            col = 1 if i&2 else 0
#             flip = i&4
            d = {'row_pct':row, 'col_pct':col, 'is_random':False}
            tfm = [*augm_tfm, zoom(scale=scale, **d), crop_pad(**d)]
#             if flip: tfm.append(flip_lr(p=1.))
            ds.tfms = tfm
            yield get_preds(learn.model, dl, pbar=pbar, activ=activ)[0]
    finally: ds.tfms = old

# Learner.tta_only = _tta_only

def _TTA(learn:Learner, scale:float=1.2, ds_type:DatasetType=DatasetType.Valid, activ:nn.Module=None, with_loss:bool=False) -> Tensors:
    "Applies TTA to predict on `ds_type` dataset."
    preds_normal,y = learn.get_preds(ds_type)
    preds_normal_tta = list(_tta_only(learn,ds_type=ds_type, scale=scale))
    preds_normal_tta_avg = torch.stack(preds_normal_tta).mean(0)
    
    learn.dl(ds_type).dataset.tfms = [flip_lr(p=1.)]
    preds_flipped = get_preds(learn.model,learn.dl(ds_type))[0]
    preds_flipped_tta = list(_tta_only(learn,ds_type=ds_type, scale=scale))
    preds_flipped_tta_avg = torch.stack(preds_flipped_tta).mean(0)
    
#     if beta is None: return preds,avg_preds,y
#     else:
#         final_preds = preds*beta + avg_preds*(1-beta)
#         if with_loss:
#             with NoneReduceOnCPU(learn.loss_func) as lf: loss = lf(final_preds, y)
#             return final_preds, y, loss
#         return final_preds, y

    return preds_normal,preds_normal_tta_avg,preds_flipped,preds_flipped_tta_avg


# In[ ]:


learn.dl(DatasetType.Test).dataset.tfms=[]
op = _TTA(learn,ds_type=DatasetType.Test)
op_1 = [2*p-4 for p in op]


# In[ ]:


get_ipython().system('cp ../input/l1-aptos/L1.pkl export.pkl')
learn = load_learner('',test=ImageList.from_df(test_df,
                             'test_processed/',
                             folder='',
                             suffix='.png',
                             ))
learn.dl(DatasetType.Test).dataset.tfms=[]
op_2 = _TTA(learn,ds_type=DatasetType.Test)


# In[ ]:


p = [0] * len(op_1)
for i in range(len(op_1)):
    p[i] = (op_1[i]+op_2[i])/2


# In[ ]:


opt = OptimizedRounder()
op = 0.3*p[0]+0.2*p[1]+0.3*p[2]+0.2*p[3]
op_preds = torch.Tensor(opt.predict(op,[-2.8031, -1.0877, 1.02 , 2.92]))
print(type(op_preds))
# preds = torch.stack(op_preds).median(0).values 
test_df.diagnosis = op_preds.numpy().astype(int)
test_df.to_csv('submission.csv',index=False)


# In[ ]:


torch.stack(op_preds).median(0).values


# In[ ]:


test_df.hist()


# In[ ]:


test_df.groupby('diagnosis').size()


# In[ ]:


get_ipython().system('rm -r test_processed')

