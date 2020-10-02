#!/usr/bin/env python
# coding: utf-8

# # SIIM-ISIC SE_ResNeXT50
# 
# Hello everyone,
# 
# This notebook is a baseline for future experiments with SE_ResNeXT50. 
# 
# Here are the tips/tricks present in the training pipeline: 
# - Use of meta-features
# - Differential learning rates for SE_ResNeXT and head for meta-features
# - Use of awesome Alex Shonenkov's dataset
# - BalanceClassSampler
# - HairAugmentation
# - SoftMarginFocalLoss
# 
# Credits go to:
# - Alex Shonenkov for his great starter notebook: https://www.kaggle.com/shonenkov/training-cv-melanoma-starter
# - Roman's hair augmentation: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/159176
# 
# This notebook is also inspired from my EDA notebook: https://www.kaggle.com/rftexas/siim-isic-melanoma-analysis-eda-efficientnetb1

# # Installing dependencies

# In[ ]:


get_ipython().system('pip install -q pretrainedmodels')


# In[ ]:


import os, re, random, gc
from tqdm import tqdm
from collections import OrderedDict
from glob import glob

from datetime import datetime
import time

import numpy as np
import pandas as pd

import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam, lr_scheduler
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Dataset

from catalyst.data.sampler import BalanceClassSampler

import pretrainedmodels

import albumentations
from albumentations.pytorch.transforms import ToTensorV2

import cv2

from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')


# # Config

# In[ ]:


class TrainConfig:
    num_workers = 8
    batch_size = 256
    
    lr_cnn = 1e-3
    lr_meta = 1e-2
    
    num_epochs = 8
    seed = 2020
    
    verbose = True
    verbose_step = 1
    
    step_scheduler = True
    
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1, 
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-8
    )


# In[ ]:


DATA_PATH = '../input/melanoma-merged-external-data-512x512-jpeg/'
TRAIN_DATA_PATH = DATA_PATH + '512x512-dataset-melanoma/512x512-dataset-melanoma/'
TEST_CSV = '/kaggle/input/siim-isic-melanoma-classification/test.csv'

WIDTH = 128
HEIGHT = 128


# In[ ]:


# Loading data

df = pd.read_csv(DATA_PATH + 'folds.csv')


# In[ ]:


df.head()


# # Utils

# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(TrainConfig.seed)


# In[ ]:


class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[ ]:


class RocAucMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.y_true = np.array([0, 1])
        self.y_pred = np.array([0.5, 0.5])
        self.score = 0
        
    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().argmax(axis=1).astype(int)
        y_pred = 1 - F.softmax(y_pred, dim=1).data.cpu().numpy()[:, 0]
        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))
        self.score = roc_auc_score(self.y_true, self.y_pred)
        
    @property
    def avg(self):
        return self.score


# # Augmentations

# In[ ]:


class HairAugmentation(albumentations.ImageOnlyTransform):
    def __init__(self, 
                 max_hairs:int = 4, 
                 hairs_folder: str = "/kaggle/input/melanoma-hairs", 
                 p=0.5):
        
        super().__init__(p=p)
        self.max_hairs = max_hairs
        self.hairs_folder = hairs_folder
    
    def apply(self, img, **params):
        n_hairs = random.randint(0, self.max_hairs)

        if not n_hairs:
            return img

        height, width, _ = img.shape  # target image width and height
        hair_images = [im for im in os.listdir(self.hairs_folder) if 'png' in im]

        for _ in range(n_hairs):
            hair = cv2.imread(os.path.join(self.hairs_folder, random.choice(hair_images)))
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))
            
            h_height, h_width, _ = hair.shape  # hair image width and height
            hair = cv2.resize(hair, (int(h_width*0.8), int(h_height*0.8)))
            
            h_height, h_width, _ = hair.shape  # hair image width and height
            roi_ho = random.randint(0, img.shape[0] - hair.shape[0])
            roi_wo = random.randint(0, img.shape[1] - hair.shape[1])
            roi = img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv).astype(np.float32)
            hair_fg = cv2.bitwise_and(hair, hair, mask=mask).astype(np.float32)

            dst = cv2.add(img_bg, hair_fg)
            img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst

        return img


# In[ ]:


def get_train_transforms():

    return albumentations.Compose([

        HairAugmentation(p=0.5),

        albumentations.ShiftScaleRotate(p=0.9),       
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),


        albumentations.OneOf([

            albumentations.CLAHE(p=0.5),
            albumentations.RandomBrightnessContrast(p=0.9),
            albumentations.HueSaturationValue(p=0.5),

        ]),

        albumentations.OneOf([

            albumentations.GridDistortion(p=0.5),
            albumentations.ElasticTransform(p=0.5),

        ]),

        albumentations.CoarseDropout(p=0.5),

        albumentations.Resize(width=WIDTH, height=HEIGHT, p=1.0),
        
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            p=1.0
        ),
        
        ToTensorV2(p=1.0),

    ])
    

def get_valid_transforms():

    return albumentations.Compose([

        albumentations.Resize(width=WIDTH, height=HEIGHT, p=1.0),
        
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            p=1.0
        ),
        
        ToTensorV2(p=1.0),

    ])


# # Dataset

# ### Generating meta-features

# In[ ]:


test_df = pd.read_csv(TEST_CSV)


# In[ ]:


# One-hot encoding of anatom_site_general_challenge feature
concat = pd.concat([df['anatom_site_general_challenge'], 
                    test_df['anatom_site_general_challenge']], ignore_index=True)
dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
df = pd.concat([df, dummies.iloc[:df.shape[0]]], axis=1)
test_df = pd.concat([test_df, dummies.iloc[df.shape[0]:].reset_index(drop=True)], axis=1)

# Sex features
df['sex'] = df['sex'].map({'male': 1, 'female': 0})
test_df['sex'] = test_df['sex'].map({'male': 1, 'female': 0})
df['sex'] = df['sex'].fillna(-1)
test_df['sex'] = test_df['sex'].fillna(-1)

# Age features
df['age_approx'] /= df['age_approx'].max()
test_df['age_approx'] /= test_df['age_approx'].max()
df['age_approx'] = df['age_approx'].fillna(-99)
test_df['age_approx'] = test_df['age_approx'].fillna(-99)

df['patient_id'] = df['patient_id'].fillna(-999)


# In[ ]:


meta_features = ['sex', 'age_approx'] + [col for col in df.columns if 'site_' in col]
meta_features.remove('anatom_site_general_challenge')


# In[ ]:


test_df.to_csv('test.csv', index=False)


# In[ ]:


df.head()


# ### Dataset class

# In[ ]:


class MelanomaDataset(Dataset):
    def __init__(self, 
                 image_ids, 
                 targets, 
                 meta_features, 
                 augmentations=None):
        super().__init__()
        self.image_ids = image_ids
        self.targets = targets
        self.meta_features = meta_features
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, item):
        
        # Image
        path = TRAIN_DATA_PATH + self.image_ids[item] + '.jpg'
        image = cv2.imread(path)
                        
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']
                                            
        # Meta-features
        patient_info = np.array(self.meta_features.iloc[item].values, dtype=np.float32)
        
        return {
            'image': image,
            'target': self.one_hot(2, self.targets[item]),
            'meta': torch.tensor(patient_info, dtype=torch.float),
        }
    
    def get_targets(self):
        return list(self.targets)
    
    @staticmethod
    def one_hot(size, target):
        tensor = torch.zeros(size, dtype=torch.float32)
        tensor[target] = 1.
        return tensor


# # Loss

# In[ ]:


class SoftMarginFocalLoss(nn.Module):
    def __init__(self, margin=0.2, gamma=2):
        super(SoftMarginFocalLoss, self).__init__()
        self.gamma = gamma
        self.margin = margin
                
        self.weight_pos = 2
        self.weight_neg = 1
    
    def forward(self, inputs, targets):
        em = np.exp(self.margin)
        
        log_pos = -F.logsigmoid(inputs)
        log_neg = -F.logsigmoid(-inputs)
        
        log_prob = targets*log_pos + (1-targets)*log_neg
        prob = torch.exp(-log_prob)
        margin = torch.log(em + (1-em)*prob)
        
        weight = targets*self.weight_pos + (1-targets)*self.weight_neg
        loss = self.margin + weight * (1 - prob) ** self.gamma * log_prob
        
        loss = loss.mean()
        
        return loss


# # Model

# In[ ]:


class MelanomaModel(nn.Module):
    def __init__(self, n_meta_features):
        super(MelanomaModel, self).__init__()
        
        self.encoder = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained=None)
        self.encoder.load_state_dict(
            torch.load(
                "../input/pretrained-model-weights-pytorch/se_resnext50_32x4d-a260b3a4.pth"
            )
        ) 
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(2048+250, 2, bias=True)
        
        self.n_meta_features = n_meta_features
        
        self.meta = nn.Sequential(OrderedDict([
            ('meta_l1', nn.Linear(self.n_meta_features, 500, bias=True)),
            ('meta_bn1', nn.BatchNorm1d(500)),
            ('meta_a1', nn.ReLU()),
            ('meta_d1', nn.Dropout(p=0.2)),
            ('meta_l2', nn.Linear(500, 250, bias=True)),  
            ('meta_bn2', nn.BatchNorm1d(250)),
            ('meta_a2', nn.ReLU()),
            ('meta_d2', nn.Dropout(p=0.2)),
        ]))
    
    def forward(self, image, meta_features):
        batch_size, _, _, _ = image.shape
        
        cnn_features = self.encoder.features(image)
        cnn_features = F.adaptive_avg_pool2d(cnn_features, 1).reshape(batch_size, -1)
        
        meta_features = self.meta(meta_features)
        
        features = torch.cat((cnn_features, meta_features), dim=1)
        logit = self.head(self.dropout(features))
        
        return logit


# # Fitter

# In[ ]:


class Fitter:
    def __init__(self, model, device, config):
        self.config = config
        self.model = model
        self.device = device
        
        self.epoch = 0
        
        self.base_dir = './'
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_loss = float('inf')
            
        param_optimizer = list(model.named_parameters())
        image_parameters = [p for n, p in param_optimizer if 'meta_' not in n]
        meta_parameters = [p for n, p in param_optimizer if 'meta_' in n]
    
        self.optimizer = torch.optim.Adam([
            
            {'params': image_parameters, 'lr': config.lr_cnn},
            {'params': meta_parameters, 'lr': config.lr_meta},
            
        ], lr=config.lr_cnn)
    
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            **self.config.scheduler_params
        )
        
        
        self.criterion = SoftMarginFocalLoss().to(self.device)
        self.log(f'Fitter prepared. Training on {self.device}')
    
    def fit(self, train_loader, valid_loader):
        for epoch in range(self.config.num_epochs):
            
            if self.config.verbose:
                lr_cnn = self.optimizer.param_groups[0]['lr']
                lr_meta = self.optimizer.param_groups[1]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR_encoder: {lr_cnn}\nLR_meta: {lr_meta}')
            
            t = time.time()
            train_loss, train_auc = self.train_one_epoch(train_loader)
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, ' +                      f'loss: {train_loss.avg:.5f}, auc: {train_auc.avg:.5f}, ' +                      f'time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')
            
            t = time.time()
            val_loss, val_auc = self.validation_one_epoch(valid_loader)
            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, ' +                      f'val_loss: {val_loss.avg:.5f}, val_auc: {val_auc.avg:.5f}, ' +                      f'time: {(time.time() - t):.5f}')
            
            if self.config.step_scheduler:
                self.scheduler.step(val_loss.avg)
            
            if val_loss.avg < self.best_loss:
                self.best_loss = val_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)
            
            self.epoch += 1 
    
    def train_one_epoch(self, train_loader):
        self.model.train()
        
        loss_score = AverageMeter()
        auc_score = RocAucMeter()
        
        t = time.time()
        
        for step, data in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'loss: {loss_score.avg:.5f}, auc: {auc_score.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
                
            images = data['image']
            meta_features = data['meta']
            targets = data['target']
                
            images = images.to(self.device)
            meta_features = meta_features.to(self.device)
            targets = targets.to(self.device).float()
                
            batch_size = images.shape[0]
            self.model.zero_grad()
                
            outputs = self.model(images, meta_features)
                
            loss = self.criterion(outputs, targets)
            loss.backward()
                
            auc_score.update(targets, outputs)
            loss_score.update(loss.detach().item(), batch_size)
                
            self.optimizer.step()
        
        return loss_score, auc_score

    def validation_one_epoch(self, valid_loader):
        self.model.eval()
        
        loss_score = AverageMeter()
        auc_score = RocAucMeter()
        
        t = time.time()
        
        for step, data in enumerate(valid_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(valid_loader)}, ' + \
                        f'loss: {loss_score.avg:.5f}, auc: {auc_score.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            
            images = data['image']
            meta_features = data['meta']
            targets = data['target']
            
            images = images.to(self.device)
            meta_features = meta_features.to(self.device)
            targets = targets.to(self.device).float()
            
            batch_size = images.shape[0]
            
            with torch.no_grad():
                outputs = self.model(images, meta_features)
                loss = self.criterion(outputs, targets)
                auc_score.update(targets, outputs)
                loss_score.update(loss.detach().item(), batch_size)
        
        return loss_score, auc_score
    
    def save(self, path):
        self.model.eval()
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'epoch': self.epoch,
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_loss']
        self.epoch = checkpoint['epoch'] + 1
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


# # Engine

# In[ ]:


def run_fold(fold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MelanomaModel(len(meta_features)).to(device)
    
    # Selecting fold
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    valid_df = df[(df['fold'] == fold) & (df['source'] == 'ISIC20')].reset_index(drop=True)
    
    # Loading data
    train_dataset = MelanomaDataset(
        image_ids=train_df['image_id'],
        targets=train_df['target'],
        meta_features=train_df[meta_features],
        augmentations=get_train_transforms(),
    )
    
    valid_dataset = MelanomaDataset(
        image_ids=valid_df['image_id'],
        targets=valid_df['target'],
        meta_features=valid_df[meta_features],
        augmentations=get_valid_transforms(),
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=BalanceClassSampler(labels=train_dataset.get_targets(), mode="downsampling"),
        batch_size=TrainConfig.batch_size,
        pin_memory=True,
        drop_last=True,
        num_workers=TrainConfig.num_workers
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=TrainConfig.batch_size,
        num_workers=TrainConfig.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    
    fitter = Fitter(
        model=model,
        device=device,
        config=TrainConfig
    )
    
    fitter.fit(train_loader, valid_loader)


# # Training

# In[ ]:


run_fold(0)


# In[ ]:


run_fold(1)


# In[ ]:


run_fold(2)


# In[ ]:


run_fold(3)


# In[ ]:


run_fold(4)

