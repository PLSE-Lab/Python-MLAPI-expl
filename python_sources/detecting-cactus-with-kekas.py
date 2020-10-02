#!/usr/bin/env python
# coding: utf-8

# ## General information
# 
# ![](https://jivasquez.files.wordpress.com/2017/08/cactus_0163.jpg?w=616)
# 
# Researchers in Mexico have created the VIGIA project, aiming to build a system for autonomous surveillance of protected areas. One of the first steps is being able to recognize the vegetation in the area. In this competition we are trying to identify whether there is a cactus in the image.
# 
# In this kernel I use kekas (https://github.com/belskikh/kekas) as a wrapper for Pytorch.
# 
# Most of the code is taken from my other kernel: https://www.kaggle.com/artgor/cancer-detection-with-kekas

# In[ ]:


# libraries
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import time 
from PIL import Image
train_on_gpu = True
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import accuracy_score
import cv2


# Some of good libraries for DL aren't available in Docker with GPU by default, so it is necessary to install them. (don't forget to turn on internet connection in kernels).

# In[ ]:


get_ipython().system('pip install albumentations > /dev/null 2>&1')
get_ipython().system('pip install pretrainedmodels > /dev/null 2>&1')
get_ipython().system('pip install kekas > /dev/null 2>&1')
get_ipython().system('pip install adabound > /dev/null 2>&1')


# In[ ]:


# more imports
import albumentations
from albumentations import torch as AT
import pretrainedmodels
import adabound

from kekas import Keker, DataOwner, DataKek
from kekas.transformations import Transformer, to_torch, normalize
from kekas.metrics import accuracy
from kekas.modules import Flatten, AdaptiveConcatPool2d
from kekas.callbacks import Callback, Callbacks, DebuggerCallback
from kekas.utils import DotDict


# ## Data overview

# In[ ]:


labels = pd.read_csv('../input/train.csv')
fig = plt.figure(figsize=(25, 8))
train_imgs = os.listdir("../input/train/train")
for idx, img in enumerate(np.random.choice(train_imgs, 20)):
    ax = fig.add_subplot(4, 20//4, idx+1, xticks=[], yticks=[])
    im = Image.open("../input/train/train/" + img)
    plt.imshow(im)
    lab = labels.loc[labels['id'] == img, 'has_cactus'].values[0]
    ax.set_title(f'Label: {lab}')


# Images were resized, so I can see almost nothing in them...

# Kekas accepts pandas DataFrame as an input and iterates over it to get image names and labels

# In[ ]:


test_img = os.listdir('../input/test/test')
test_df = pd.DataFrame(test_img, columns=['id'])
test_df['has_cactus'] = -1
test_df['data_type'] = 'test'

labels['has_cactus'] = labels['has_cactus'].astype(int)
labels['data_type'] = 'train'

labels.head()


# In[ ]:


labels.loc[labels['data_type'] == 'train', 'has_cactus'].value_counts()


# We have some disbalance in the data, but it isn't too big.

# In[ ]:


# splitting data into train and validation
train, valid = train_test_split(labels, stratify=labels.has_cactus, test_size=0.2)


# ### Reader function
# 
# At first it is necessary to create a reader function, which will open images. It accepts i and row as input (like from pandas iterrows). The function should return a dictionary with image and label.
# [:,:,::-1] - is a neat trick which converts BGR images to RGB, it works faster that converting to RGB by usual means.
# 

# In[ ]:


def reader_fn(i, row):
    image = cv2.imread(f"../input/{row['data_type']}/{row['data_type']}/{row['id']}")[:,:,::-1] # BGR -> RGB
    label = torch.Tensor([row["has_cactus"]])
    return {"image": image, "label": label}


# ### Data transformation
# 
# Next step is defining data transformations and augmentations. This differs from standard PyTorch way. We define resizing, augmentations and normalizing separately, this allows to easily create separate transformers for train and valid/test data.
# 
# At first we define augmentations. We create a function with a list of augmentations (I prefer albumentation library: https://github.com/albu/albumentations)

# In[ ]:


def augs(p=0.5):
    return albumentations.Compose([
        albumentations.HorizontalFlip(),
        albumentations.VerticalFlip(),
        albumentations.RandomBrightness(),
    ], p=p)


# Now we create a transforming function. It heavily uses Transformer from kekas.
# 
#   * The first step is defining resizing. You can change arguments of function if you want images to have different height and width, otherwis you can leave it as it is.
#   * Next step is defining augmentations. Here we provide the key of image which is defined in reader_fn;
#   * The third step is defining final transformation to tensor and normalizing;
#   * After this we can compose separate transformations for train and valid/test data;

# In[ ]:


def get_transforms(dataset_key, size, p):

    PRE_TFMS = Transformer(dataset_key, lambda x: cv2.resize(x, (size, size)))

    AUGS = Transformer(dataset_key, lambda x: augs()(image=x)["image"])

    NRM_TFMS = transforms.Compose([
        Transformer(dataset_key, to_torch()),
        Transformer(dataset_key, normalize())
    ])
    
    train_tfms = transforms.Compose([PRE_TFMS, AUGS, NRM_TFMS])
    val_tfms = transforms.Compose([PRE_TFMS, NRM_TFMS])
    
    return train_tfms, val_tfms


# In[ ]:


train_tfms, val_tfms = get_transforms("image", 32, 0.5)


# Now we can create a DataKek, which is similar to creating dataset in Pytorch. We define the data, reader function and transformation.Then we can define standard PyTorch DataLoader.

# In[ ]:


train_dk = DataKek(df=train, reader_fn=reader_fn, transforms=train_tfms)
val_dk = DataKek(df=valid, reader_fn=reader_fn, transforms=val_tfms)

batch_size = 64
workers = 0

train_dl = DataLoader(train_dk, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=True)
val_dl = DataLoader(val_dk, batch_size=batch_size, num_workers=workers, shuffle=False)


# In[ ]:


test_dk = DataKek(df=test_df, reader_fn=reader_fn, transforms=val_tfms)
test_dl = DataLoader(test_dk, batch_size=batch_size, num_workers=workers, shuffle=False)


# ### Building a neural net
# 
# Here we define the architecture of the neural net.
# 
# * Pre-trained backbone is taken from pretrainedmodels: https://github.com/Cadene/pretrained-models.pytorch Here I take densenet169
# * We also define changes to the architecture. For example, we take off the last layer and add a custom head with nn.Sequential. AdaptiveConcatPool2d is a layer in kekas, which concats AdaptiveMaxPooling and AdaptiveAveragePooling

# In[ ]:


class Net(nn.Module):
    def __init__(
            self,
            num_classes: int,
            p: float = 0.2,
            pooling_size: int = 2,
            last_conv_size: int = 1664,
            arch: str = "densenet169",
            pretrained: str = "imagenet") -> None:
        """A simple model to finetune.
        
        Args:
            num_classes: the number of target classes, the size of the last layer's output
            p: dropout probability
            pooling_size: the size of the result feature map after adaptive pooling layer
            last_conv_size: size of the flatten last backbone conv layer
            arch: the name of the architecture form pretrainedmodels
            pretrained: the mode for pretrained model from pretrainedmodels
        """
        super().__init__()
        net = pretrainedmodels.__dict__[arch](pretrained=pretrained)
        modules = list(net.children())[:-1]  # delete last layer
        # add custom head
        modules += [nn.Sequential(
            # AdaptiveConcatPool2d is a concat of AdaptiveMaxPooling and AdaptiveAveragePooling 
            # AdaptiveConcatPool2d(size=pooling_size),
            Flatten(),
            nn.BatchNorm1d(1664),
            nn.Dropout(p),
            nn.Linear(1664, num_classes)
        )]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        logits = self.net(x)
        return logits


# The data for training needs to be transformed one more time - we define DataOwner, which contains all the data. For now let's define it for train and valid.
# Next we define model and loss. As I choose BCEWithLogitsLoss, we can set the number of classes for output to 1.

# In[ ]:


dataowner = DataOwner(train_dl, val_dl, None)
model = Net(num_classes=1)
criterion = nn.BCEWithLogitsLoss()


# And now we define what will the model do with the data. For example we could slice the output and take only a part of it. For now we will simply return the output of the model.

# In[ ]:


def step_fn(model: torch.nn.Module,
            batch: torch.Tensor) -> torch.Tensor:
    """Determine what your model will do with your data.

    Args:
        model: the pytorch module to pass input in
        batch: the batch of data from the DataLoader

    Returns:
        The models forward pass results
    """
    
    inp = batch["image"]
    return model(inp)


# Defining custom metrics

# In[ ]:


def bce_accuracy(target: torch.Tensor,
                 preds: torch.Tensor,
                 thresh: bool = 0.5) -> float:
    target = target.cpu().detach().numpy()
    preds = (torch.sigmoid(preds).cpu().detach().numpy() > thresh).astype(int)
    return accuracy_score(target, preds)
  
def roc_auc(target: torch.Tensor,
                 preds: torch.Tensor) -> float:
    target = target.cpu().detach().numpy()
    preds = torch.sigmoid(preds).cpu().detach().numpy()
    return roc_auc_score(target, preds)


# ### Keker
# 
# Now we can define the Keker - the core Kekas class for training the model.
# 
# Here we define everything which is necessary for training:
# 
# * the model which was defined earlier;
# * dataowner containing the data for training and validation;
# * criterion;
# * step function;
# * the key of labels, which was defined in the reader function;
# * the dictionary with metrics (there can be several of them);
# * The optimizer and its parameters;
# 

# In[ ]:


keker = Keker(model=model,
              dataowner=dataowner,
              criterion=criterion,
              step_fn=step_fn,
              target_key="label",
              metrics={"acc": bce_accuracy, 'auc': roc_auc},
              opt=torch.optim.SGD,
              opt_params={"momentum": 0.99})


# In[ ]:


keker.unfreeze(model_attr="net")

layer_num = -1
keker.freeze_to(layer_num, model_attr="net")


# In[ ]:


keker.kek_one_cycle(max_lr=1e-2,                  # the maximum learning rate
                    cycle_len=4,                  # number of epochs, actually, but not exactly
                    momentum_range=(0.95, 0.85),  # range of momentum changes
                    div_factor=25,                # max_lr / min_lr
                    increase_fraction=0.3,        # the part of cycle when learning rate increases
                    logdir='train_logs')
keker.plot_kek('train_logs')


# In[ ]:



keker.kek_one_cycle(max_lr=1e-3,                  # the maximum learning rate
                    cycle_len=4,                  # number of epochs, actually, but not exactly
                    momentum_range=(0.95, 0.85),  # range of momentum changes
                    div_factor=25,                # max_lr / min_lr
                    increase_fraction=0.2,        # the part of cycle when learning rate increases
                    logdir='train_logs1')
keker.plot_kek('train_logs1')


# ### Predicting and TTA
# 
# Simply predicting on test data is okay, but it is better to use TTA - test time augmentation. Let's see how it can be done with Kekas.
# 
# * define augmentations;
# * define augmentation function;
# * create objects with these augmentations;
# * put these objects into a single dictionary;

# In[ ]:


preds = keker.predict_loader(loader=test_dl)


# In[ ]:


# flip_ = albumentations.HorizontalFlip(always_apply=True)
# transpose_ = albumentations.Transpose(always_apply=True)

# def insert_aug(aug, dataset_key="image", size=224):    
#     PRE_TFMS = Transformer(dataset_key, lambda x: cv2.resize(x, (size, size)))
    
#     AUGS = Transformer(dataset_key, lambda x: aug(image=x)["image"])
    
#     NRM_TFMS = transforms.Compose([
#         Transformer(dataset_key, to_torch()),
#         Transformer(dataset_key, normalize())
#     ])
    
#     tfm = transforms.Compose([PRE_TFMS, AUGS, NRM_TFMS])
#     return tfm

# flip = insert_aug(flip_)
# transpose = insert_aug(transpose_)

# tta_tfms = {"flip": flip, "transpose": transpose}

# # third, run TTA
# keker.TTA(loader=test_dl,                # loader to predict on 
#           tfms=tta_tfms,                # list or dict of always applying transforms
#           savedir="tta_preds1",  # savedir
#           prefix="preds")               # (optional) name prefix. default is 'preds'


# In[ ]:


# prediction = np.zeros((test_df.shape[0], 1))
# for i in os.listdir('tta_preds1'):
#     pr = np.load('tta_preds1/' + i)
#     prediction += pr
# prediction = prediction / len(os.listdir('tta_preds1'))


# In[ ]:


test_preds = pd.DataFrame({'imgs': test_df.id.values, 'preds': preds.reshape(-1,)})
test_preds.columns = ['id', 'has_cactus']
test_preds.to_csv('sub.csv', index=False)
test_preds.head()

