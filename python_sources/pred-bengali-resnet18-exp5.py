#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import albumentations
import argparse
import collections
import cv2
import datetime
import gc
import glob
import logging
import math
import operator
import os 
import pickle
import pkg_resources
import random
import re
import scipy.stats as stats
import seaborn as sns
import shutil
import sys
import time
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms
from contextlib import contextmanager
from collections import OrderedDict
# from nltk.stem import PorterStemmer
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import KFold, GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.nn import CrossEntropyLoss, MSELoss
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import (Dataset,DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import tensorflow as tf
from PIL import Image

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
# from tqdm import tqdm, tqdm_notebook, trange
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
tqdm.pandas()
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings
warnings.filterwarnings('ignore')
# from apex import amp

# import sys
# sys.path.append("drive/My Drive/transformers")

from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

SEED = 1129

def seed_everything(seed=1129):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(SEED)


# In[ ]:


import logging
import sys

LOGGER = logging.getLogger()
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


def setup_logger(out_file=None, stderr=True, stderr_level=logging.INFO, file_level=logging.DEBUG):
    LOGGER.handlers = []
    LOGGER.setLevel(min(stderr_level, file_level))

    if stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(FORMATTER)
        handler.setLevel(stderr_level)
        LOGGER.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(FORMATTER)
        handler.setLevel(file_level)
        LOGGER.addHandler(handler)

    LOGGER.info("logger set up")
    return LOGGER

# ===============
# Settings
# ===============
# SEED = np.random.randint(100000)
device = "cuda"
OUT_DIR = '/kaggle/working'

batch_size = 4
accumulation_steps = 8
# fold_id = 0
# epochs = 5
EXP_ID = "exp1"
LOGGER_PATH = f"log_{EXP_ID}.txt"
model_path = None

setup_logger(out_file=LOGGER_PATH)
LOGGER.info("seed={}".format(SEED))


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info('[{}] done in {} s'.format(name, round(time.time() - t0, 2)))


# In[ ]:


# !ls ../input/bengali-preprocessed-zip-resize128128-otsumethod/


# In[ ]:


import albumentations
from albumentations import pytorch as AT
import torchvision.transforms as transforms

def image_to_tensor(image, normalize=None):
    tensor = torch.from_numpy(np.moveaxis(image / (255. if image.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
    if normalize is not None:
        return F.normalize(tensor)
    return tensor

data_transforms = albumentations.Compose([
    albumentations.Resize(128, 128),
    # albumentations.Flip(p=0.5),
    # albumentations.Normalize(),
    # AT.ToTensor()
    ])

data_transforms_test = albumentations.Compose([
    albumentations.Resize(128, 128),
    # albumentations.Normalize(),
    # AT.ToTensor()
    ])


# In[ ]:


class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,kernel_size=3,padding=1,bias=False):
        super(ResidualBlock,self).__init__()
        self.cnn1 =nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size,1,padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
            
    def forward(self,x):
        residual = x
        x = self.cnn1(x)
        x = self.cnn2(x)
        x += self.shortcut(residual)
        x = nn.ReLU(True)(x)

        return x


# In[ ]:


hidden_size = 64
channel_size = 1

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(channel_size,hidden_size,kernel_size=2,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True)
        )
        
        self.block2 = nn.Sequential(
            nn.MaxPool2d(1,1),
            ResidualBlock(hidden_size,hidden_size),
            ResidualBlock(hidden_size,hidden_size,2)
        )
        
        self.block3 = nn.Sequential(
            ResidualBlock(hidden_size,hidden_size*2),
            ResidualBlock(hidden_size*2,hidden_size*2,2)
        )
        
        self.block4 = nn.Sequential(
            ResidualBlock(hidden_size*2,hidden_size*4),
            ResidualBlock(hidden_size*4,hidden_size*4,2)
        )
        self.block5 = nn.Sequential(
            ResidualBlock(hidden_size*4,hidden_size*8),
            ResidualBlock(hidden_size*8,hidden_size*8,2)
        )
        
        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(512*4,512)  
        # vowel_diacritic
        self.fc1 = nn.Linear(512,11)
        # grapheme_root
        self.fc2 = nn.Linear(512,168)
        # consonant_diacritic
        self.fc3 = nn.Linear(512,7)
        
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        return x1,x2,x3


# In[ ]:


ls ../input/bengali-exp5/


# In[ ]:


with timer('create model'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = torchvision.models.resnet50(pretrained=True)
    # model.load_state_dict(torch.load("../input/pytorch-pretrained-models/resnet101-5d3b4d8f.pth"))
    
    model = ResNet18()
    model.load_state_dict(torch.load("../input/bengali-exp5/exp1_fold0.pth"))
    model = model.to(device)


# In[ ]:


import PIL
def threshold_image(img):
    '''
    Helper function for thresholding the images
    '''
    gray = PIL.Image.fromarray(np.uint8(img), 'L')
    ret,th = cv2.threshold(np.array(gray),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th


# In[ ]:


class BengaliAIDatasetTest(torch.utils.data.Dataset):
    def __init__(self,df,transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        
        input_dic = {}
        image = self.df.iloc[idx][1:].values.reshape(128,128).astype(np.float)
        image = threshold_image(image)
        image = self.transform(image=image)['image']
        image = (image.astype(np.float32) - 0.0692) / 0.2051
        image = image_to_tensor(image, normalize=False) 
        
        input_dic['image'] = image
        
        return input_dic


# In[ ]:


#Credits: https://www.kaggle.com/phoenix9032/pytorch-efficientnet-starter-code/data

SIZE = 128
HEIGHT=137
WIDTH=236

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=SIZE, pad=16):
    
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))

def Resize(df,size=128):
    resized = {} 
    df = df.set_index('image_id')
    
    for i in tqdm(range(df.shape[0])): 
        image0 = 255 - df.loc[df.index[i]].values.reshape(137,236).astype(np.uint8)
        
        #normalize each image by its max val
        img = (image0*(255.0/image0.max())).astype(np.uint8)
        image = crop_resize(img)
        resized[df.index[i]] = image.reshape(-1)
    resized = pd.DataFrame(resized).T.reset_index()
    resized.columns = resized.columns.astype(str)
    resized.rename(columns={'index':'image_id'},inplace=True)
    return resized


# In[ ]:


# !ls ../input/bengaliaicv19feather/

with timer('predict'):
    model.eval()
    test_data = [
        '../input/bengaliaicv19feather/test_image_data_0.feather',
        '../input/bengaliaicv19feather/test_image_data_1.feather',
        '../input/bengaliaicv19feather/test_image_data_2.feather',
        '../input/bengaliaicv19feather/test_image_data_3.feather',
    ]
    predictions = []
    batch_size=1
    for fname in tqdm(test_data):
        data = pd.read_feather(fname)
        data = Resize(data,size=128)
        test_dataset = BengaliAIDatasetTest(data, transform=data_transforms_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        with torch.no_grad():
            for step, (input_dic) in tqdm(enumerate(test_loader), total=len(test_loader)):
                for k in input_dic.keys():
                    input_dic[k] = input_dic[k].to(device)

                    outputs1,outputs2,outputs3 = model(input_dic["image"].unsqueeze(1))
                    predictions.append(outputs3.argmax(1).cpu().detach().numpy())
                    predictions.append(outputs2.argmax(1).cpu().detach().numpy())
                    predictions.append(outputs1.argmax(1).cpu().detach().numpy())


# In[ ]:


submission = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
submission.target = np.hstack(predictions)


# In[ ]:


LOGGER.info('submission head10 : {}'.format(submission.head(10)))

LOGGER.info('target value_counts : {}'.format(submission.target.value_counts()))


# In[ ]:


submission.to_csv('submission.csv',index=False)

