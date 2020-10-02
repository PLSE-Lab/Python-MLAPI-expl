#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ../input/nvidiaapex/repository/NVIDIA-apex-39e153a')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


import cv2
import matplotlib.pyplot as plt
from os.path import isfile
import torch.nn.init as init
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim import Adam, SGD, RMSprop
import time
from torch.autograd import Variable
import torch.functional as F
from tqdm import tqdm
from sklearn import metrics
import urllib
import pickle
import cv2
import torch.nn.functional as F
from torchvision import models
import seaborn as sns
import random
import sys


# In[ ]:


package_path = '../input/efficientnet/efficientnet-pytorch/EfficientNet-PyTorch/'
sys.path.append(package_path)
from efficientnet_pytorch import EfficientNet
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(1234)
TTA         = 5
num_classes = 1
IMG_SIZE    = 256
test = '../input/aptos2019-blindness-detection/test_images/'
def expand_path(p):
    p = str(p)
    if isfile(test + p + ".png"):
        return test + (p + ".png")
    return p

def p_show(imgs, label_name=None, per_row=3):
    n = len(imgs)
    rows = (n + per_row - 1)//per_row
    cols = min(per_row, n)
    fig, axes = plt.subplots(rows,cols, figsize=(15,15))
    for ax in axes.flatten(): ax.axis('off')
    for i,(p, ax) in enumerate(zip(imgs, axes.flatten())): 
        img = Image.open(expand_path(p))
        ax.imshow(img)
        ax.set_title(train_df[train_df.id_code == p].diagnosis.values)
def crop_image1(img,tol=7):
    # img is image data
    # tol  is tolerance
        
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img
    

def circle_crop_v2(img):

        #img = cv2.imread(img)
        img = crop_image_from_gray(img)

        height, width, depth = img.shape
        largest_side = np.max((height, width))
        img = cv2.resize(img, (largest_side, largest_side))

        height, width, depth = img.shape

        x = int(width / 2)
        y = int(height / 2)
        r = np.amin((x, y))

        circle_img = np.zeros((height, width), np.uint8)
        cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
        img = cv2.bitwise_and(img, img, mask=circle_img)
        img = crop_image_from_gray(img)

        return img    
    
class MyDataset(Dataset):
    
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        label = self.df.diagnosis.values[idx]
        label = np.expand_dims(label, -1)
        
        p = self.df.id_code.values[idx]
        p_path = expand_path(p)
        image = cv2.imread(p_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = circle_crop_v2(image)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 10) ,-4 ,128)
        image = transforms.ToPILImage()(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
test_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation((-120, 120)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

testset        = MyDataset(pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv'), 
                 transform=test_transform)
test_loader    = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)
model = EfficientNet.from_name('efficientnet-b4')
in_features = model._fc.in_features
model._fc = nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load('../input/efficientnet-mytest/weight_best.pt'))
model.cuda()
for param in model.parameters():
    param.requires_grad = False
sample = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
test_pred = np.zeros((len(sample), 1))
model.eval()

for _ in range(TTA):
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            images, _ = data
            images = images.cuda()
            pred = model(images)
            test_pred[i * 16:(i + 1) * 16] += pred.detach().cpu().squeeze().numpy().reshape(-1, 1)

output = test_pred / TTA
preds1 = output.copy()
coef = [0.5, 1.5, 2.5, 3.5]
for i, pred in enumerate(output):
    if pred < coef[0]:
        output[i] = 0
    elif pred >= coef[0] and pred < coef[1]:
        output[i] = 1
    elif pred >= coef[1] and pred < coef[2]:
        output[i] = 2
    elif pred >= coef[2] and pred < coef[3]:
        output[i] = 3
    else:
        output[i] = 4
submission1 = pd.DataFrame({'id_code':pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv').id_code.values,
                          'diagnosis':np.squeeze(output).astype(int)})


# In[ ]:


submission1.diagnosis = output.astype(int)
submission1.to_csv('submission.csv', index=False)

