#!/usr/bin/env python
# coding: utf-8

# # README
# * Pre-trained EfficientNet-B6
# * Ben's Preprocessing + CenterCrop
# * 10 times TTA

# # 1. Setup

# In[ ]:


import cv2
import matplotlib.pyplot as plt
from os.path import isfile
import torch.nn.init as init
import torch
import torch.nn as nn
import numpy as np
import pandas as pd 
import os
from PIL import Image, ImageFilter
print(os.listdir("../input"))
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms
import torch.optim as optim
import time
from tqdm import tqdm
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

get_ipython().run_line_magic('matplotlib', 'inline')


# Loading EfficientNet-B6

# In[ ]:


package_path = '../input/efficientnet/efficientnet-pytorch/EfficientNet-PyTorch/'
sys.path.append(package_path)

from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_name('efficientnet-b6')


# Seed setting

# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Option setting

# In[ ]:


seed_everything(1234)
num_classes = 1
IMG_SIZE = 256
lr = 1e-3
batch_size = 32
num_TTA = 10


# In[ ]:


# DataFrame for Train / Test set
df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
df_test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
df_sample = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

y_train = df['diagnosis']

# Image data
train = '../input/aptos2019-blindness-detection/train_images/'
test = '../input/aptos2019-blindness-detection/test_images/'


# # 2. Image Preprocessing

# In[ ]:


def expand_path(p):
    p = str(p)
    if isfile(train + p + ".png"):
        return train + (p + ".png")
    if isfile(test + p + ".png"):
        return test + (p + ".png")
    return p


# In[ ]:


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


# In[ ]:


from albumentations import *
import time

# IMG_SIZE = (256, 256)

def albaugment(aug0, img):
    return aug0(image=img)['image']

# idx=8
# image1 = x_test[idx]

# # 1. Rotate or Flip
# aug1 = OneOf([Rotate(p=0.99, limit=160, border_mode=0, value=0), Flip(p=0.5)], p=1)

# # 2. Adjust Brightness or Contrast
# aug2 = RandomBrightnessContrast(brightness_limit=0.45, contrast_limit=0.45, p=1)
# h_min = np.round(IMG_SIZE*0.72).astype(int)
# h_max = np.round(IMG_SIZE*0.9).astype(int)
# # print(h_min, h_max)

# # 3. Random Crop and then Resize
# aug3 = RandomSizedCrop((h_min, h_max), IMG_SIZE, IMG_SIZE, w2h_ratio=IMG_SIZE/IMG_SIZE, p=1)

# # 4. CutOut Augumentation
# max_hole_size = int(IMG_SIZE/10)
# aug4 = Cutout(p=1, max_h_size=max_hole_size, max_w_size=max_hole_size, num_holes=8)

# # 5. SunFlare Augmentation
# aug5 = RandomSunFlare(src_radius=max_hole_size, 
#                       num_flare_circles_lower=10,
#                       num_flare_circles_upper=20, 
#                       p=1)

# # 6. Ultimate Augmentation
# final_aug = Compose([aug1, aug2, aug3, aug4, aug5], p=1)

# 7. Center to zoom
aug6 = CenterCrop(height=180, width=180, p=1)


# # 3. Dataset & DataLoader setting

# In[ ]:


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
        # Ben's Preprocess
        image = cv2.imread(p_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image_from_gray(image)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), sigmaX=30), -4, 128)
        # CenterCrop
        image = albaugment(aug6, image)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = transforms.ToPILImage()(image)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


# In[ ]:


test_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((-120, 120)),
    transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

testset = MyDataset(df_sample, transform=test_transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


# # 4. Define Network

# In[ ]:


in_features = model._fc.in_features
model._fc = nn.Linear(in_features=in_features, out_features=num_classes)
model.load_state_dict(torch.load('../input/efficientnetb6-weight-0831/efficientnet-b6_weight_best_0831.pt'))
model.cuda()


# # 5. Inference with TTA

# In[ ]:


for param in model.parameters():
    param.requires_grad = False


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntest_pred = np.zeros((len(df_sample), 1))\n\nmodel.eval()\n\nfor i in range(num_TTA):\n    with torch.no_grad():\n        for i, data in enumerate(tqdm(test_loader)):\n            images, _ = data\n            images = images.cuda()\n            pred = model(images)\n            test_pred[i * batch_size:(i + 1) * batch_size] += pred.cpu().squeeze().numpy().reshape(-1, 1)\n\noutput = test_pred / num_TTA')


# In[ ]:


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


# # 6. Submission

# In[ ]:


submission = pd.DataFrame({'id_code':df_sample.id_code.values,
                           'diagnosis':np.squeeze(output).astype(int)})
print(submission.head())


# In[ ]:


submission.to_csv('submission.csv', index=False)
print(os.listdir('./'))


# In[ ]:


submission['diagnosis'].value_counts()


# In[ ]:




