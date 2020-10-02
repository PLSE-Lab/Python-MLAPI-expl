#!/usr/bin/env python
# coding: utf-8

# # Bengali.AI: Self-supervised pretraining with Context Encoders (fastai2)
# 
# This notebook implements the paper [Context Encoders: Feature Learning by Inpainting](https://arxiv.org/abs/1604.07379) by Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell and Alexei A. Efros.
# 
# My hypothesis is that a model that can successfully fill in the blanks with Bengali handwritten characters, has a pretty darn good understanding of the Bengali language and thus can be repurposed to classify constituent elements of the characters.
# 
# 
# ## 0. Table of contents
# 
# 1. Context Encoders overview
# 2. Libraries and hyperparams
# 3. Generating input and targets
# 4. Building the model
# 5. Training
# 6. Visualising the results

# ## 1. Context Encoders overview
# 
# The Context Encoders paper describes a simple pre-training task: remove stuff from images and have the model try to predict what was removed.
# 
# [![image.png](https://i.postimg.cc/xTdGCDZC/image.png)](https://postimg.cc/K4d3qVxS)
# 
# The paper describes a number of different techniques for removing stuff. The simplest is to just extract a region from the centre of the image which is what I'm doing in this kernel. In future, I might try the other techniques - I am a little concerned that the task may be a little too easy.
# 
# I have made a slight modification to the paper by using EfficientNet-B0 instead of AlexNet for the encoder. Also, I'm only concerning myself with Reconstruction Loss (L2) as I don't mind blurry reconstructions just that the weights can be transferred to the classification task.
# 
# It uses the development version of fastai2.

# ## 2. Libraries and hyperparams

# The library makes use of the development version of [fastai2](https://github.com/fastai/fastai2). Since it isn't available in kernels yet, I'll install it alongside [EfficientNet-PyTorch](https://github.com/zhoudaxia233/EfficientUnet-PyTorch).

# In[ ]:


get_ipython().system('pip install git+https://github.com/fastai/fastai2 > /dev/null')
get_ipython().system('pip install efficientnet-pytorch > /dev/null')


# In[ ]:


from pathlib import Path

import pandas as pd

import torch
from efficientnet_pytorch import EfficientNet
from torch.utils import model_zoo

from fastai2.basics import *
from fastai2.data.all import *
from fastai2.callback.all import *
from fastai2.vision.all import *


# In[ ]:


DATA_PATH = Path('/kaggle/input/bengaliai-cv19')
IMAGE_DATA_PATH = Path('/kaggle/input/grapheme-imgs-128x128')
OUTPUT_PATH = Path('/kaggle/working')

VALID_PCT = 0.2
SEED = 420
BATCH_SIZE = 64
CROP_SIZE = 32
IMG_SIZE = 128


# In[ ]:


train_df = pd.read_csv(DATA_PATH/'train.csv')


# ## 3. Generating input and targets

# I'm using 2 transforms: one to remove the centre of an image (the `X`) and another to return just the centre (the `y`).

# In[ ]:


class ImageWithCenterRemoved(Transform):
    """Transform that removes the center part of an image."""
    
    order = 6

    def __init__(self, crop_size=CROP_SIZE):
        self.crop_size = crop_size

    def encodes(self, x:PILImageBW) -> PILImageBW:
        x = array(x)
    
        start_height = tuple(IMG_SIZE // 2 - (CROP_SIZE // 2))
        start_width = tuple(IMG_SIZE // 2 - (CROP_SIZE // 2)) 
        
        x[
            ...,
            start_height:start_height+self.crop_size,
            start_width:start_width+self.crop_size
        ] = 0
    
        return PILImageBW(Image.fromarray(x))
    
    def encodes(self, x:TensorImage):
        start_height = IMG_SIZE // 2 - (CROP_SIZE // 2)
        start_width = IMG_SIZE // 2 - (CROP_SIZE // 2)
        
        x[
            ...,
            start_height:start_height+self.crop_size,
            start_width:start_width+self.crop_size
        ] = 0
        
        return TensorImage(x)
    
    
class ImageWithOnlyCenter(Transform):
    """Transform that keeps only the center part of an image."""
    
    order = 6
    
    def __init__(self, crop_size=CROP_SIZE):
        self.crop_size = crop_size

    def encodes(self, x:TensorImage) -> PILImageBW:
        start_height = IMG_SIZE // 2 - (CROP_SIZE // 2)
        start_width = IMG_SIZE // 2 - (CROP_SIZE // 2)
        
        output = x[
            ...,
            start_height:start_height + self.crop_size,
            start_width:start_width + self.crop_size
        ]

        return TensorImage(output)


# In[ ]:


items = get_image_files(IMAGE_DATA_PATH)


# Next I create a `Datasets` instance splitting the data into a 80/20 train/val split.

# In[ ]:


x_tfms = [PILImageBW.create, ToTensor, ImageWithCenterRemoved()]
y_tfms = [PILImageBW.create, ToTensor, ImageWithOnlyCenter()]
tfms = [x_tfms, y_tfms]

splitter = RandomSplitter(VALID_PCT, seed=SEED)

tds = Datasets(items, tfms, splits=splitter(items))


# In[ ]:


imagenet_stats


# Lastly, I use the ImageNet stats to normalise the data, since I will be using a pretrained EfficientNet-B0 model.

# In[ ]:


dl_tfms = [IntToFloatTensor,  Normalize(mean=0.485, std=0.229)]

train_dl = TfmdDL(tds.train, bs=BATCH_SIZE, after_batch=dl_tfms)
valid_dl = TfmdDL(tds.valid, bs=BATCH_SIZE, after_batch=dl_tfms)


# As you can see, we're now successfully cutting the centre out of an image and have the centre crop as the label.

# In[ ]:


train_dl.show_batch()


# Lastly, I'll put the data into a `DataLoaders` class (formally `DataBunch`).

# In[ ]:


data = DataLoaders(train_dl, valid_dl)


# ## 4. Building the model

# The model described below is very similar to a [Unet](https://arxiv.org/abs/1505.04597) model. In that it has an encoder which is responsible for generating a series of downsampled features, then a decoder which upsamples the generates features using a series of fractionally-strided convolution operations.
# 
# [![image.png](https://i.postimg.cc/BZ2DnZN4/image.png)](https://postimg.cc/t7C7rjLM)
# 
# Note that I am omitting the Adversarial Loss, which is more concerned with real looking results - I only want transferability.
# 
# For the encoder, I simply repurpose the [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) model by removing the final classification layers of the model.

# In[ ]:


class EfficientNetEncoder(EfficientNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # the initial layer to convolve into 3 channels
        # idea from https://www.kaggle.com/aleksandradeis/bengali-ai-efficientnet-pytorch-starter
        self.input_conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)

    def forward(self, inputs):
        x = self.input_conv(inputs)
        return self.extract_features(x)
    
    @classmethod
    def load_pretrained(cls):
        model_name = 'efficientnet-b0'
        model = cls.from_name(model_name, override_params={'num_classes': 1})
        model_dict = model.state_dict()

        state_dict = model_zoo.load_url('https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b0-355c32eb.pth')
        state_dict_no_fc = {k: v for k, v in state_dict.items() if not k.startswith('_fc')}
        model_dict.update(state_dict_no_fc)
        
        model.load_state_dict(model_dict)

        return model


# For the decoder, I create a series of fractional convolutions that upsamples the image to the size of the crop. The paper uses BatchNorm and ReLU in the decoder, so I'm doing the same here.

# In[ ]:


def up_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


# In[ ]:


class Decoder(nn.Module):

    def __init__(self, encoder, n_channels, out_channels=1):
        super().__init__()

        self.encoder = encoder

        self.up_conv1 = up_conv(n_channels, 256)    
        self.up_conv2 = up_conv(256, 128)    # 8x8
        self.up_conv3 = up_conv(128, 64)    # 16x16
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.encoder(x)     # input: 1x128x128, output: 1280x4x4
        x = self.up_conv1(x)    # input: 1280x4x4, output: 256x8x8
        x = self.up_conv2(x)    # input: 256x8x8, output: 128x16x16
        x = self.up_conv3(x)    # input: 128x16x16, output: 64x32x32
        x = self.final_conv(x)  # input: 64x32x32, output: 1x32x32
        
        return x


# In[ ]:


encoder = EfficientNetEncoder.load_pretrained()
model = Decoder(encoder, n_channels=1280)  # 1280: EfficientNet b0 output. To do: don't hardcode this.


# ## 5. Training

# With everything in place, the model can just be trained like we normally would in Fast.ai. I'm using standard [OneCycle](https://mc.ai/finding-good-learning-rate-and-the-one-cycle-policy/) training as is familiar to people who have done the Fast.ai course or used the library.

# In[ ]:


if torch.cuda.is_available():
    print('Cuda available')
    model = model.cuda()
    data = data.cuda()


# In[ ]:


learner = Learner(data, model, loss_func=nn.MSELoss())


# In[ ]:


# learner.lr_find()


# In[ ]:


learner.fit_one_cycle(4, 1e-3)


# In[ ]:


learner.recorder.plot_loss()


# In[ ]:


learner.validate()


# ## 6. Visualising results

# In[ ]:


learner.show_results(ds_idx=1)


# That looks pretty good.
# 
# In an upcoming kernel, I'll do a mini-ableation study to see if these weights are useful for transfer learning on the competition's multilabel classification problem.

# In[ ]:


learner.save('model_cycle_1')

