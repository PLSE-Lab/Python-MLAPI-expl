#!/usr/bin/env python
# coding: utf-8

# ## Classifier

# In[ ]:


get_ipython().system('pip install --no-deps ../input/segmentations-models-pytorch-7fa1020/pretrainedmodels-0.7.4-py3-none-any.whl')
import sys
sys.path.append('../input/smp-latest/segmentation_models.pytorch-master/')
sys.path.append('../input/efficientnet-pytorch/efficientnet_pytorch-0.4.0/')


# In[ ]:


# CLASSIFICATION_THRES = 0.00


# In[ ]:


# import sys
# sys.path.insert(0, "../input/timm-models/pytorch-image-models/pytorch-image-models")
# from fastai.vision import * 
# from fastai import *
# import timm
# df=pd.read_csv('../input/steel-classifier-csv/train_classes.csv')
# df.Label = (df.Label!='0').astype('int')
# bs=64 ## Batch size
# tfms = get_transforms() ## Transformation
# stats =([0.334,0.334,0.334],[0.199,0.199,0.199])
# data = (
#         ImageList.from_df(df=df ,path='',cols='Image_Id', folder='../input/severstal-steel-defect-detection/train_images') 
#         .split_by_rand_pct(0.1)
#         .label_from_df(cols='Label')
#         .transform(tfms)
#         .databunch(bs=bs,num_workers=4)
#         .normalize(stats)
#        )
# learn_res34 = cnn_learner(data ,timm.models.resnet34,pretrained=False, metrics =[accuracy, FBeta(beta= 1)], callback_fns=ShowGraph)
# learn_wrn50 = cnn_learner(data ,timm.models.wide_resnet50_2,pretrained=False, metrics =[accuracy, FBeta(beta= 1)], callback_fns=ShowGraph)
# !mkdir models/ && cp ../input/ubambamodels/rn34-stage-5.pth models/
# !cp ../input/ubambamodels/wrn50-stage-4.pth models/
# learn_res34.load('rn34-stage-5')
# learn_wrn50.load('wrn50-stage-4')
# !rm -r models/
# sample_submission_path = '../input/severstal-steel-defect-detection/sample_submission.csv'
# test_df= pd.read_csv(sample_submission_path)
# def changename(x):
#     x=x[:-2]
#     return x
# test_df.ImageId_ClassId=test_df.ImageId_ClassId.apply(changename)
# test_df=test_df.drop_duplicates(['ImageId_ClassId'])
# learn_res34.data.add_test(
#     ImageList.from_df(df=test_df ,path='',cols='ImageId_ClassId', folder='../input/severstal-steel-defect-detection/test_images')
# )
# learn_wrn50.data.add_test(
#     ImageList.from_df(df=test_df ,path='',cols='ImageId_ClassId', folder='../input/severstal-steel-defect-detection/test_images')
# )
# x1,y1=learn_res34.get_preds(DatasetType.Test)
# x2,y2=learn_wrn50.get_preds(DatasetType.Test)
# arr1 = np.array(x1[:,1]>CLASSIFICATION_THRES).astype(int)
# arr2 = np.array(x2[:,1]>CLASSIFICATION_THRES).astype(int)
# arr3 = (arr1+arr2)/2
# test_df.EncodedPixels = np.ceil(arr3).astype(int)


# In[ ]:


from fastai.vision import * 
from fastai import *
df=pd.read_csv('../input/steel-classifier-csv/train_classes.csv')
df.Label = (df.Label!='0').astype('int')
sample_submission_path = '../input/severstal-steel-defect-detection/sample_submission.csv'
test_df= pd.read_csv(sample_submission_path)
def changename(x):
    x=x[:-2]
    return x
test_df.ImageId_ClassId=test_df.ImageId_ClassId.apply(changename)
test_df=test_df.drop_duplicates(['ImageId_ClassId'])


# In[ ]:


test_df.EncodedPixels.value_counts()


# ## Infernce

# In[ ]:


import pdb
import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from albumentations import (Normalize, Compose, HorizontalFlip)
from albumentations.pytorch import ToTensor
import torch.utils.data as data


# In[ ]:


from torch.hub import load_state_dict_from_url
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
device = torch.device("cuda")


# In[ ]:


#https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# In[ ]:


class TestDataset(Dataset):
    '''Dataset for test prediction'''
    def __init__(self, root, df, mean, std):
        self.root = root
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)
        self.transform = Compose(
            [
                Normalize(mean=mean, std=std, p=1),
                ToTensor()
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname)
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return fname, images

    def __len__(self):
        return self.num_samples


# In[ ]:


def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


# In[ ]:


sample_submission_path = '../input/severstal-steel-defect-detection/sample_submission.csv'
test_data_folder = "../input/severstal-steel-defect-detection/test_images"


# In[ ]:


test_df_= pd.read_csv(sample_submission_path)


# In[ ]:


# initialize test dataloader
fold_csv = pd.read_csv(sample_submission_path)
num_workers = 4
batch_size = 4
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
df = pd.read_csv(sample_submission_path)
testset = DataLoader(
    TestDataset(test_data_folder, test_df_, mean, std),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)


# In[ ]:


import re
from collections import namedtuple


GlobalParams = namedtuple('GlobalParams', ['batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'num_classes',
                                           'width_coefficient', 'depth_coefficient', 'depth_divisor', 'min_depth',
                                           'drop_connect_rate'])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = namedtuple('BlockArgs', ['kernel_size', 'num_repeat', 'input_filters', 'output_filters', 'expand_ratio',
                                     'id_skip', 'strides', 'se_ratio'])
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


IMAGENET_WEIGHTS = {
    'efficientnet-b0': 'http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth',
    'efficientnet-b1': 'http://storage.googleapis.com/public-models/efficientnet-b1-dbc7070a.pth',
    'efficientnet-b2': 'http://storage.googleapis.com/public-models/efficientnet-b2-27687264.pth',
    'efficientnet-b3': 'http://storage.googleapis.com/public-models/efficientnet-b3-c8376fa2.pth',
    'efficientnet-b4': 'http://storage.googleapis.com/public-models/efficientnet-b4-e116e8b3.pth',
    'efficientnet-b5': 'http://storage.googleapis.com/public-models/efficientnet-b5-586e6cc6.pth'
}


def round_filters(filters, global_params):
    """Round number of filters
    """
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Round number of repeats
    """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def get_efficientnet_params(model_name, override_params=None):
    """Get efficientnet params based on model name
    """
    params_dict = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        # Note: the resolution here is just for reference, its values won't be used.
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    if model_name not in params_dict.keys():
        raise KeyError('There is no model named {}.'.format(model_name))

    width_coefficient, depth_coefficient, _, dropout_rate = params_dict[model_name]

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=0.2,
        num_classes=1000,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None)

    if override_params:
        global_params = global_params._replace(**override_params)

    decoder = BlockDecoder()
    return decoder.decode(blocks_args), global_params


class BlockDecoder(object):
    """Block Decoder for readability
    """

    @staticmethod
    def _decode_block_string(block_string):
        """Gets a block through a string notation of arguments."""
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            strides=[int(options['s'][0]), int(options['s'][1])]
        )

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

    def decode(self, string_list):
        """Decodes a list of string notations to specify blocks inside the network.
        Args:
          string_list: a list of strings, each string is a notation of block.
        Returns:
          A list of namedtuples to represent blocks arguments.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(self._decode_block_string(block_string))
        return blocks_args

    def encode(self, blocks_args):
        """Encodes a list of Blocks to a list of strings.
        Args:
          blocks_args: A list of namedtuples to represent blocks arguments.
        Returns:
          a list of strings, each string is a notation of block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(self._encode_block_string(block))
        return block_strings

#########################################

class EfficientNet(nn.Module):

    def __init__(self, block_args_list, global_params):
        super().__init__()

        self.block_args_list = block_args_list
        self.global_params = global_params

        # Batch norm parameters
        batch_norm_momentum = 1 - self.global_params.batch_norm_momentum
        batch_norm_epsilon = self.global_params.batch_norm_epsilon

        # Stem
        in_channels = 3
        out_channels = round_filters(32, self.global_params)
        self._conv_stem = Conv2dSamePadding(in_channels,
                                            out_channels,
                                            kernel_size=3,
                                            stride=2,
                                            bias=False,
                                            name='stem_conv')
        self._bn0 = BatchNorm2d(num_features=out_channels,
                                momentum=batch_norm_momentum,
                                eps=batch_norm_epsilon,
                                name='stem_batch_norm')

        self._swish = Swish(name='swish')

        # Build _blocks
        idx = 0
        self._blocks = nn.ModuleList([])
        for block_args in self.block_args_list:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self.global_params),
                output_filters=round_filters(block_args.output_filters, self.global_params),
                num_repeat=round_repeats(block_args.num_repeat, self.global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self.global_params, idx=idx))
            idx += 1

            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, strides=1)

            # The rest of the _blocks
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self.global_params, idx=idx))
                idx += 1

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self.global_params)
        self._conv_head = Conv2dSamePadding(in_channels,
                                            out_channels,
                                            kernel_size=1,
                                            bias=False,
                                            name='head_conv')
        self._bn1 = BatchNorm2d(num_features=out_channels,
                                momentum=batch_norm_momentum,
                                eps=batch_norm_epsilon,
                                name='head_batch_norm')

        # Final linear layer
        self.dropout_rate = self.global_params.dropout_rate
        self._fc = nn.Linear(out_channels, self.global_params.num_classes)

    def forward(self, x):
        # Stem
        x = self._conv_stem(x)
        x = self._bn0(x)
        x = self._swish(x)

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self.global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= idx / len(self._blocks)
            x = block(x, drop_connect_rate)

        # Head
        x = self._conv_head(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Pooling and Dropout
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Fully-connected layer
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, *, n_classes=1000, pretrained=False):
        return _get_model_by_name(model_name, classes=n_classes, pretrained=pretrained)

    @classmethod
    def encoder(cls, model_name, *, pretrained=False):
        model = cls.from_name(model_name, pretrained=pretrained)

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()

                self.name = model_name

                self.global_params = model.global_params

                self.stem_conv = model._conv_stem
                self.stem_batch_norm = model._bn0
                self.stem_swish = Swish(name='stem_swish')
                self.blocks = model._blocks
                self.head_conv = model._conv_head
                self.head_batch_norm = model._bn1
                self.head_swish = Swish(name='head_swish')

            def forward(self, x):
                # Stem
                x = self.stem_conv(x)
                x = self.stem_batch_norm(x)
                x = self.stem_swish(x)

                # Blocks
                for idx, block in enumerate(self.blocks):
                    drop_connect_rate = self.global_params.drop_connect_rate
                    if drop_connect_rate:
                        drop_connect_rate *= idx / len(self.blocks)
                    x = block(x, drop_connect_rate)

                # Head
                x = self.head_conv(x)
                x = self.head_batch_norm(x)
                x = self.head_swish(x)
                return x

        return Encoder()

    @classmethod
    def custom_head(cls, model_name, *, n_classes=1000, pretrained=False):
        if n_classes == 1000:
            return cls.from_name(model_name, n_classes=n_classes, pretrained=pretrained)
        else:
            class CustomHead(nn.Module):
                def __init__(self, out_channels):
                    super().__init__()
                    self.encoder = cls.encoder(model_name, pretrained=pretrained)
                    self.custom_head = custom_head(self.n_channels * 2, out_channels)

                @property
                def n_channels(self):
                    n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                                       'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                                       'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
                    return n_channels_dict[self.encoder.name]

                def forward(self, x):
                    x = self.encoder(x)
                    mp = nn.AdaptiveMaxPool2d(output_size=(1, 1))(x)
                    ap = nn.AdaptiveAvgPool2d(output_size=(1, 1))(x)
                    x = torch.cat([mp, ap], dim=1)
                    x = x.view(x.size(0), -1)
                    x = self.custom_head(x)

                    return x

            return CustomHead(n_classes)


def _get_model_by_name(model_name, classes=1000, pretrained=False):
    block_args_list, global_params = get_efficientnet_params(model_name, override_params={'num_classes': classes})
    model = EfficientNet(block_args_list, global_params)
    try:
        if pretrained:
            pretrained_state_dict = load_state_dict_from_url(IMAGENET_WEIGHTS[model_name])

            if classes != 1000:
                random_state_dict = model.state_dict()
                pretrained_state_dict['_fc.weight'] = random_state_dict['_fc.weight']
                pretrained_state_dict['_fc.bias'] = random_state_dict['_fc.bias']

            model.load_state_dict(pretrained_state_dict)

    except KeyError as e:
        print(f"NOTE: Currently model {e} doesn't have pretrained weights, therefore a model with randomly initialized"
              " weights is returned.")

    return model

################################
class Swish(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv2dSamePadding(nn.Conv2d):
    """2D Convolutions with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, name=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation, groups=groups,
                         bias=bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        self.name = name

    def forward(self, x):
        input_h, input_w = x.size()[2:]
        kernel_h, kernel_w = self.weight.size()[2:]
        stride_h, stride_w = self.stride
        output_h, output_w = math.ceil(input_h / stride_h), math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * self.stride[0] + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * self.stride[1] + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, name=None):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine,
                         track_running_stats=track_running_stats)
        self.name = name


def drop_connect(inputs, drop_connect_rate, training):
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1.0 - drop_connect_rate
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block
    """

    def __init__(self, block_args, global_params, idx):
        super().__init__()

        block_name = 'blocks_' + str(idx) + '_'

        self.block_args = block_args
        self.batch_norm_momentum = 1 - global_params.batch_norm_momentum
        self.batch_norm_epsilon = global_params.batch_norm_epsilon
        self.has_se = (self.block_args.se_ratio is not None) and (0 < self.block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip

        self.swish = Swish(block_name + '_swish')

        # Expansion phase
        in_channels = self.block_args.input_filters
        out_channels = self.block_args.input_filters * self.block_args.expand_ratio
        if self.block_args.expand_ratio != 1:
            self._expand_conv = Conv2dSamePadding(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=1,
                                                  bias=False,
                                                  name=block_name + 'expansion_conv')
            self._bn0 = BatchNorm2d(num_features=out_channels,
                                    momentum=self.batch_norm_momentum,
                                    eps=self.batch_norm_epsilon,
                                    name=block_name + 'expansion_batch_norm')

        # Depth-wise convolution phase
        kernel_size = self.block_args.kernel_size
        strides = self.block_args.strides
        self._depthwise_conv = Conv2dSamePadding(in_channels=out_channels,
                                                 out_channels=out_channels,
                                                 groups=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=strides,
                                                 bias=False,
                                                 name=block_name + 'depthwise_conv')
        self._bn1 = BatchNorm2d(num_features=out_channels,
                                momentum=self.batch_norm_momentum,
                                eps=self.batch_norm_epsilon,
                                name=block_name + 'depthwise_batch_norm')

        # Squeeze and Excitation layer
        if self.has_se:
            num_squeezed_channels = max(1, int(self.block_args.input_filters * self.block_args.se_ratio))
            self._se_reduce = Conv2dSamePadding(in_channels=out_channels,
                                                out_channels=num_squeezed_channels,
                                                kernel_size=1,
                                                name=block_name + 'se_reduce')
            self._se_expand = Conv2dSamePadding(in_channels=num_squeezed_channels,
                                                out_channels=out_channels,
                                                kernel_size=1,
                                                name=block_name + 'se_expand')

        # Output phase
        final_output_channels = self.block_args.output_filters
        self._project_conv = Conv2dSamePadding(in_channels=out_channels,
                                               out_channels=final_output_channels,
                                               kernel_size=1,
                                               bias=False,
                                               name=block_name + 'output_conv')
        self._bn2 = BatchNorm2d(num_features=final_output_channels,
                                momentum=self.batch_norm_momentum,
                                eps=self.batch_norm_epsilon,
                                name=block_name + 'output_batch_norm')

    def forward(self, x, drop_connect_rate=None):
        identity = x
        # Expansion and depth-wise convolution
        if self.block_args.expand_ratio != 1:
            x = self._expand_conv(x)
            x = self._bn0(x)
            x = self.swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self.swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self.swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self.block_args.input_filters, self.block_args.output_filters
        if self.id_skip and self.block_args.strides == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, drop_connect_rate=drop_connect_rate, training=self.training)
            x = x + identity
        return x


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=2, stride=2
    )


def custom_head(in_channels, out_channels):
    return nn.Sequential(
        nn.Dropout(),
        nn.Linear(in_channels, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(512, out_channels)
    )

################################################

from collections import OrderedDict


__all__ = ['EfficientUnet', 'get_efficientunet_b0', 'get_efficientunet_b1', 'get_efficientunet_b2',
           'get_efficientunet_b3', 'get_efficientunet_b4', 'get_efficientunet_b5', 'get_efficientunet_b6',
           'get_efficientunet_b7']


def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0

    def register_hook(module):

        def hook(module, input, output):
            try:
                nonlocal count
                if module.name == f'blocks_{count}_output_batch_norm':
                    count += 1
                    shape = output.size()[-2:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[module.name] = output

                elif module.name == 'head_swish':
                    # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                    # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                    # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                    # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                    # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                    # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                    blocks.popitem()
                    blocks[module.name] = output

            except AttributeError:
                pass

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # register hook
    model.apply(register_hook)

    # make a forward pass to trigger the hooks
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return blocks


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_xa = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,xa):
        g1 = self.W_g(g)
        x1 = self.W_xa(xa)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return xa*psi




class EfficientUnet(nn.Module):
    def __init__(self, encoder, out_channels=2, concat_input=True):
        super().__init__()

        self.encoder = encoder
        self.concat_input = concat_input

        self.up_conv1 = up_conv(self.n_channels, 512)
        self.Att1 = Attention_block(F_g=512,F_l=592-512,F_int=56)
        self.double_conv1 = double_conv(self.size[0], 512)
        self.up_conv2 = up_conv(512, 256)
        self.Att2 = Attention_block(F_g=256,F_l=296-256,F_int=28)
        self.double_conv2 = double_conv(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.Att3 = Attention_block(F_g=128,F_l=152-128,F_int=16)
        self.double_conv3 = double_conv(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.Att4 = Attention_block(F_g=64,F_l=80-64,F_int=12)
        self.double_conv4 = double_conv(self.size[3], 64)

        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = double_conv(self.size[4], 32)

        self.final_conv = nn.Conv2d(self.size[5], out_channels, kernel_size=1)

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    def forward(self, x):
        input_ = x

        blocks = get_blocks_to_be_concat(self.encoder, x)
        
      
        _, x = blocks.popitem()

        x1 = self.up_conv1(x)        
        _, y = blocks.popitem()   
        x = self.Att1(g=x1,xa=y)      
        x = torch.cat([x, x1], dim=1)
        x = self.double_conv1(x)

        x1 = self.up_conv2(x)
        y = blocks.popitem()[1]
        x = self.Att2(g=x1,xa=y)
        x = torch.cat([x, x1], dim=1)
        x = self.double_conv2(x)

        x1 = self.up_conv3(x)
        y = blocks.popitem()[1]
        x = self.Att3(g=x1,xa=y)
        x = torch.cat([x, x1], dim=1)
        x = self.double_conv3(x)

        x1 = self.up_conv4(x)
        y = blocks.popitem()[1]
        x = self.Att4(g=x1,xa=y) 
        x = torch.cat([x, x1], dim=1)
        x = self.double_conv4(x)

        if self.concat_input:
            x = self.up_conv_input(x)
            x = torch.cat([x, input_], dim=1)
            x = self.double_conv_input(x)

        x = self.final_conv(x)

        return x


def get_efficientunet_b0(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b1(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b1', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b2(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b2', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b3(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b3', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b4(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b4', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b5(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b5', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b6(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b6', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b7(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b7', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


# In[ ]:


unet_b0 = get_efficientunet_b0(out_channels=4, concat_input=True, pretrained=False)
unet_b1 = get_efficientunet_b1(out_channels=4, concat_input=True, pretrained=False)
import segmentation_models_pytorch as smp
FPN_res34 = smp.FPN('resnet34', classes=4, activation=None,encoder_weights=None)
PSP_seresx50 = smp.PSPNet('se_resnext50_32x4d', classes=4, activation=None,encoder_weights=None)
FPN_seresx50 = smp.FPN('se_resnext50_32x4d', classes=4, activation=None,encoder_weights=None)
FPN_b0 = smp.FPN('efficientnet-b0', classes=4, activation=None,encoder_weights=None)
FPN_b1 = smp.FPN('efficientnet-b1', classes=4, activation=None,encoder_weights=None)
FPN_b2 = smp.FPN('efficientnet-b2', classes=4, activation=None,encoder_weights=None)
FPN_b3 = smp.FPN('efficientnet-b3', classes=4, activation=None,encoder_weights=None)
FPN_b4 = smp.FPN('efficientnet-b4', classes=4, activation=None,encoder_weights=None)


# In[ ]:


# Initialize mode and load trained weights
def weight_loader(model,path):
    ckpt_path = path
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])
    return model

def make_eval(models):
    for i,m in enumerate(models):
        models[i]=m.eval()
    return models  

unet_b0 = weight_loader(unet_b0,'../input/b0fulldata/b0att_complete.pth')
FPN_seresx50 = weight_loader(FPN_seresx50,'../input/b0fulldata/fpnserex50complete.pth')
FPN_b1 = weight_loader(FPN_b1,'../input/fpn-b1/fpn_b1-stage-4.pth')
FPN_b0 = weight_loader(FPN_b0,'../input/b0fulldata/fpn_b0-complete-stage-2.pth')
unet_b1 = weight_loader(unet_b1,'../input/classres50/effb1unetbceloss_lrreduce_full_2.pth')
# PSP_seresx50 = weight_loader(PSP_seresx50,'../input/pspresnxt50/psp_se_resnext50_32x4d_alldata_reducelr.pth')
# FPN_res34 = weight_loader(FPN_res34,'../input/pspfpn_res34/fpn_res34-stage-2lrreduce.pth')
unet_se_resnext50_32x4d = torch.jit.load('/kaggle/input/severstalmodels/unet_se_resnext50_32x4d.pth').cuda().eval()
# unet_mobilenet2 = torch.jit.load('/kaggle/input/severstalmodels/unet_mobilenet2.pth').cuda().eval()
unet_resnet34 = torch.jit.load('/kaggle/input/severstalmodels/unet_resnet34.pth').cuda().eval()
FPN_b2 = weight_loader(FPN_b2,'../input/ubambamodels/fpn_b2-complete.pth')
FPN_b4 = weight_loader(FPN_b4,'../input/fpnb41024/fpn-b4-crop-stage-5-2-all.pth')
FPN_b3 = weight_loader(FPN_b3,'../input/fpnb3stage4/fpn-b3_crop-stage-4-all.pth')


# In[ ]:


class Model_sig:
    def __init__(self, models):
        self.models = models
    def __call__(self, x):
        res = []
        x = x.cuda()
        with torch.no_grad():
            for m in self.models:
                res.append(m(x))
        x = torch.flip(x,dims = [-1])
        with torch.no_grad():
            for m in self.models:
                flipped_mask = m(x)
                mask = torch.flip(flipped_mask, dims = [-1])
                res.append(mask)
        res = (torch.stack(res))

        return torch.mean(torch.sigmoid(res), dim=0)

class Model:
    def __init__(self, models):
        self.models = models
    def __call__(self, x):
        res = []
        x = x.cuda()
        with torch.no_grad():
            for m in self.models:
                res.append(m(x))
        x = torch.flip(x,dims = [-1])
        with torch.no_grad():
            for m in self.models:
                flipped_mask = m(x)
                mask = torch.flip(flipped_mask,dims = [-1])
                res.append(mask)
        res = torch.stack(res)
        return torch.sigmoid(torch.mean(res, dim=0))

# model = Model([unet_b0, FPN_seresx50, FPN_b0,FPN_b1,FPN_b2,unet_b1,PSP_seresx50,FPN_res34, unet_se_resnext50_32x4d, unet_mobilenet2, unet_resnet34])
model = Model([FPN_b3,unet_b0,unet_b1, FPN_seresx50, FPN_b0, FPN_b1, FPN_b2, unet_se_resnext50_32x4d, unet_resnet34,FPN_b4])
# model_sig = Model_sig([FPN_b2, unet_se_resnext50_32x4d, unet_resnet34])


# In[ ]:


# start prediction
thresholds_max=[0.7,0.7,0.7,0.7]
thresholds_min=[0.2,0.2,0.3,0.3]
min_area=[350, 500, 750, 1000]
res = []
predictions = []
fnames_all = []

## IF USING CLASSIFIER
for i, batch in enumerate(tqdm(testset)):
    fnames, images = batch    
    batch_preds = model(images.to(device)).detach().cpu().numpy()
#     batch_preds += 3 * model_sig(images.to(device)).detach().cpu().numpy()
#     batch_preds = batch_preds/7

    # Batch post processing
    for p, file in zip(batch_preds, fnames):
        file = os.path.basename(file)
        # Image postprocessing
        for i in range(4):
            p_channel = p[i]
            p_channel_ = p_channel
            imageid_classid = file+'_'+str(i+1)
            p_channel = (p_channel>thresholds_max[i]).astype(np.uint8)
            if p_channel.sum() < min_area[i]:
                p_channel = np.zeros(p_channel.shape, dtype=p_channel.dtype)
            else:
                p_channel = (p_channel_>thresholds_min[i]).astype(np.uint8)
            
            res.append({
                'ImageId_ClassId': imageid_classid,
                'EncodedPixels': mask2rle(p_channel)
            })
        
df = pd.DataFrame(res)


# In[ ]:


# save predictions to submission.csv
# df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
df.to_csv("submission.csv", index=False)


# In[ ]:


def change(x):
    x=x[-1:]
    return x
df_=df.copy()
df_.ImageId_ClassId=df_.ImageId_ClassId.apply(change)
df_[df_.EncodedPixels!=''].ImageId_ClassId.value_counts()


# In[ ]:


df.head()


# # Visualisation

# In[ ]:


# train_path = '../input/severstal-steel-defect-detection/train.csv'
# train_df = pd.read_csv(train_path)


# In[ ]:


# for i in range(20):
#     NUMBER=train_df.dropna().index[i]
#     train_data_folder = '../input/severstal-steel-defect-detection/train_images'
#     num_workers = 4
#     batch_size = 1
#     mean = (0.485, 0.456, 0.406)
#     std = (0.229, 0.224, 0.225)
#     trainset = DataLoader(
#         TestDataset(train_data_folder, train_df[NUMBER:NUMBER+1], mean, std),
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True
#     )
#     thresholds=[0.2,0.5,0.5,0.5]
#     min_area=[100, 600, 1000, 2000]
#     res = []
#     predictions = []
#     fnames_all = []
    
#     ## IF USING CLASSIFIER
#     for i, batch in enumerate(tqdm(trainset)):
#         fnames, images = batch    
#         batch_preds = torch.sigmoid(model(images.to(device))).detach().cpu().numpy()
    
#         # Batch post processing
#         for p, file in zip(batch_preds, fnames):
#             file = os.path.basename(file)
#             # Image postprocessing
#             for i in range(4):
#                 p_channel = p[i]
#                 imageid_classid = file+'_'+str(i+1)
#                 p_channel = (p_channel>thresholds[i]).astype(np.uint8)
#                 if p_channel.sum() < min_area[i]:
#                     p_channel = np.zeros(p_channel.shape, dtype=p_channel.dtype)
#                 a = p_channel
#                 if(i==int(train_df.ImageId_ClassId.values[NUMBER][-1:])-1):
#                     print(int(train_df.ImageId_ClassId.values[NUMBER][-1:]))
#                     break
#     kernel = np.ones((10,10),np.uint8)
# #     a_er=cv2.erode(a,kernel,5)
#     kernel = np.ones((5,5),np.uint8)
#     a_er=cv2.dilate(a,kernel,1)
#     plt.imshow(a,cmap='gray')
#     plt.pause(0.00001)
#     plt.imshow(a_er,cmap='gray')
#     plt.pause(0.00001)
#     plt.imshow(rle2mask(train_df.EncodedPixels.values[NUMBER]),cmap='gray')
#     plt.pause(0.00001)


# In[ ]:


# a.sum()


# In[ ]:


# a.shape


# In[ ]:


# def rle2mask(mask_rle, shape=(1600,256)):
#     '''
#     mask_rle: run-length as string formated (start length)
#     shape: (width,height) of array to return 
#     Returns numpy array, 1 - mask, 0 - background

#     '''
#     s = mask_rle.split()
#     starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
#     starts -= 1
#     ends = starts + lengths
#     img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
#     for lo, hi in zip(starts, ends):
#         img[lo:hi] = 1
#     return img.reshape(shape).T

