#!/usr/bin/env python
# coding: utf-8

# # Important: For full training, use DEBUG=False and increase epochs.

# Came to know of the colored library from https://www.kaggle.com/tarunpaparaju/twitter-challenge-roberta-sentiment-predictor
# I even learnt to use TPU from his kernels. 

# In this kernel:
# 
# 1. Finetune vanilla seresnext50 pretrained model on PANDA dataset. 
# 2. MultilabelStratifiedKFold used.
# 3. Tiles approach with 16 tiles.
# 4. No head
# 5. Loss: MSELoss
# 6. Epochs 4
# 7. Optimizer: Adam
# 8. LR: 1e-4
# 9. Scheduler: ReduceLROnPlateau
# 
# GPU 1 epoch per fold = 7 mins, 
# TPU 1 epoch simultaneously 4 folds = 15 mins

# In[ ]:


get_ipython().system('pip install ../input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/ > /dev/null')


# In[ ]:


get_ipython().system('pip install iterative-stratification')


# # Pytorch XLA

# In[ ]:


get_ipython().system('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
get_ipython().system('python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev')
get_ipython().system('export XLA_USE_BF16=1')


# In[ ]:


get_ipython().system('pip install -q colored')


# # Import Libraries

# In[ ]:


import os
import sys
import numpy as np
import pandas as pd 
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import GroupNorm
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import torchvision.models as models
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from tqdm.notebook import tqdm
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import pretrainedmodels
import cv2
import tarfile
from functools import partial
import scipy as sp

from colored import fg, bg, attr
import time
from joblib import Parallel, delayed
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

print(os.listdir('/kaggle/input/'))


# In[ ]:


DATA_DIR = '/kaggle/input/prostate-cancer-grade-assessment/'
IMG_DIR = '/kaggle/input/panda-td-tiles-16x128x128/'
MARKER_DIR = '/kaggle/input/marker-images/marker_images/'

n_folds=4
seed=42
sigmaX=10
bs=16
device='cuda' if torch.cuda.is_available() else 'cpu'
num_epochs=6
LR=1e-4      # important. Performance significantly worse on 1e-3
num_tiles=16
num_classes=1
DEBUG=True
arch = pretrainedmodels.__dict__['se_resnext50_32x4d']
if DEBUG:
    num_epochs=1
    debug_samples=100


# In[ ]:


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)


# In[ ]:


train = pd.read_csv(DATA_DIR+'train.csv')
print(train.shape)


# In[ ]:


valid_ids = list(set([id.split('_')[0] for id in os.listdir(IMG_DIR)]))
# marker_ids = list(set([id.split('.')[0] for id in os.listdir(MARKER_DIR)]))
train = train[train['image_id'].isin(valid_ids)].reset_index(drop=True)
# train = train[~train['image_id'].isin(marker_ids)].reset_index(drop=True)

if DEBUG:
    train = train.sample(n=debug_samples, random_state=seed).reset_index(drop=True)


# In[ ]:


print('New shape: ', train.shape)


# # Multilabel Stratified Folds

# In[ ]:


Y = train[['isup_grade','data_provider']].values
skf = MultilabelStratifiedKFold(n_splits=n_folds,shuffle=True, random_state=seed)
train['fold'] = -1

for i,(trn_idx, val_idx) in enumerate(skf.split(train,Y)):
    train.loc[val_idx,'fold'] = i


# In[ ]:


train.head()


# # DataSet

# In[ ]:


class PANDA(Dataset):
    def __init__(self,df, transform, mode='train'):
        self.df = df
        self.transform=transform
        self.mode='train'
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df['image_id'].values[idx]
        provider = self.df['data_provider'].values[idx]
        img = [cv2.imread(IMG_DIR+img_id+f'_{16-i-1}.png') for i in range(num_tiles)]
        
        img=cv2.hconcat([cv2.vconcat([img[0], img[1], img[2], img[3]]),
                        cv2.vconcat([img[4], img[5], img[6], img[7]]),
                        cv2.vconcat([img[8], img[9], img[10], img[11]]),
                        cv2.vconcat([img[12], img[13], img[14], img[15]])])
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          
        if self.transform:
            img = self.transform(image=img)['image']
            
        
        if self.mode!='test':    
            label = self.df['isup_grade'][idx]
            
    
        return {'image': img,
                'provider': provider,
               'label': torch.tensor(label, dtype=torch.float)}


# # New Layers

# In[ ]:


from torch.nn.parameter import Parameter

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
    
    
class Conv2d_ws(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(nn.Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,padding, dilation, bias=True ,padding_mode='zeros',
                                       groups=1, output_padding='zeros', transposed=False)



    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
def convert_to_gem(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.AdaptiveAvgPool2d):
            setattr(model, child_name, GeM())
        else:
            convert_to_gem(child)
            
def convert_to_conv2d(model):
    for child_name, child in model.named_children():
        if child_name not in ['fc1','fc2']:
            if isinstance(child, nn.Conv2d):
                in_feat = child.in_channels
                out_feat = child.out_channels
                ker_size = child.kernel_size
                stride = child.stride
                padding = child.padding
                dilation = child.dilation
                groups = child.groups
                setattr(model, child_name, Conv2d_ws(in_channels=in_feat, out_channels=out_feat, kernel_size=ker_size, stride=stride,padding = padding, dilation=dilation, groups=groups))
            else:
                convert_to_conv2d(child)
                
def convert_to_groupnorm(model):
    for child_name, child in model.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_features = child.num_features
                setattr(model, child_name, GroupNorm(num_groups=32, num_channels=num_features))
            else:
                convert_to_groupnorm(child)


# # Model

#  I have replaced the final avg_pool layer with GeM layer. Using Conv2d with Weight Standardization (WS) increases each epoch time by almost 4 minutes. 
#  GroupNorm paper: https://arxiv.org/pdf/1803.08494.pdf . 
#  Conv2d with WS paper: https://arxiv.org/abs/1903.10520

# In[ ]:


class PANDA_MODEL(nn.Module):
    def __init__(self, pretrained=False, classes=num_classes):
        super(PANDA_MODEL, self).__init__()
        
        self.model = arch(pretrained=None)
        if pretrained:
            self.model.load_state_dict(torch.load('../input/pytorch-se-resnext/se_resnext50_32x4d-a260b3a4.pth'))
        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x           


# # OptimizedRounder

# In[ ]:


class OptimizedRounder():
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
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5

        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

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
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5
        return X_p

    def coefficients(self):
        return self.coef_['x']


# # Train and Validation Functions

# In[ ]:


def train_model(dataloader, model, device, optimizer, criterion):

    model.train()
    train_loss=0
    length = len(dataloader)
    optimizer.zero_grad()
    iterator = tqdm(enumerate(dataloader), total=length)
    prediction, truth=[], []
    
    for i, batch in iterator:
        img=batch['image'].to(device)
        label=batch['label'].to(device)
        
        output=model(img)
        loss=criterion(output.view(-1), label)
        loss.backward()
            
#         optimizer.step()
        xm.optimizer_step(optimizer, barrier=True)
        optimizer.zero_grad()
        
        
        prediction.append(output.detach().cpu().numpy())
        truth.append(label.detach().cpu().numpy())
        train_loss+=loss.item()/length
    
    prediction = np.concatenate(prediction)
    truth = np.concatenate(truth)
    
    return train_loss, prediction, truth


def validate(dataloader, model, device, criterion):

    model.eval()
    val_loss=0
    length = len(dataloader)
    iterator = tqdm(enumerate(dataloader), total=length)
    prediction, truth=[], []
    kprediction, ktruth=[], []
    rprediction, rtruth=[], []
    
    with torch.no_grad():
        for i, batch in iterator:
            img=batch['image'].to(device)
            provider=batch['provider']
            label=batch['label'].to(device)
        
            output=model(img)
            loss=criterion(output.view(-1), label)
            
            pred = output.detach().cpu().numpy()
            prediction.append(pred)
            
            kindex=[i for i,pro in enumerate(provider) if pro=='karolinska']
            kprediction.append(pred[kindex])
            ktruth.append(label.detach().cpu().numpy()[kindex])

            rindex=[i for i,pro in enumerate(provider) if pro=='radboud']
            rprediction.append(pred[rindex])
            rtruth.append(label.detach().cpu().numpy()[rindex]) 
            
            truth.append(label.detach().cpu().numpy())
            val_loss+=loss.item()/length

    prediction = np.concatenate(prediction)
    truth = np.concatenate(truth)

    kprediction = np.concatenate(kprediction)
    ktruth = np.concatenate(ktruth)

    rprediction = np.concatenate(rprediction)
    rtruth = np.concatenate(rtruth)
    
    return val_loss, prediction, truth, kprediction, ktruth, rprediction, rtruth


# In[ ]:


train_tfm = A.Compose([A.HorizontalFlip(p=0.5),
                       A.VerticalFlip(p=0.5),
                      A.Normalize(mean=[0.485, 0.456, 0.406],   
                                  std=[0.229, 0.224, 0.225]),
                      ToTensorV2()])

valid_tfm = A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
                      ToTensorV2()])


# # Visualize images

# In[ ]:


import matplotlib.pyplot as plt

train_fold = train[train['fold']!=0].reset_index(drop=True)

train_dataset = PANDA(train_fold, train_tfm, mode='train')
    
plt.figure(figsize=(18,18))
for i in range(9):
    plt.subplot(3,3,i+1)
    index=np.random.randint(len(train_fold))
    img, provider, label=train_dataset[index]['image'], train_dataset[index]['provider'], train_dataset[index]['label']
    plt.title('{}, {}'.format(provider,label))
    plt.imshow(img.numpy().transpose(1,2,0))
    plt.axis('off')


# # Main Training

# In[ ]:


fonts = [(fg(82),attr('reset')), (fg(169),attr('reset'))]

def train_fold(fold):
    
    
    device = xm.xla_device(fold + 1)
    
    optimized_rounder = OptimizedRounder()
        
    val_fold = train[train['fold']==fold].reset_index(drop=True)
    train_fold = train[train['fold']!=fold].reset_index(drop=True)

    train_dataset = PANDA(train_fold, train_tfm, mode='train')
    val_dataset = PANDA(val_fold, valid_tfm, mode='validate')
    
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=4)


    model = PANDA_MODEL(pretrained=True)
    model = model.to(device)


    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, verbose=True)
    
    best_loss=np.inf
    best_score=0
    best_opt_score=0


    for epoch in range(num_epochs):
        start = time.time()
        
        train_loss, train_pred, train_truth = train_model(train_loader, model, device, optimizer, criterion)
        train_score = cohen_kappa_score(np.round(train_pred), train_truth, weights='quadratic')
        
        val_loss, val_pred, val_truth, kpre, ktru, rpre, rtru = validate(val_loader, model, device, criterion)
        val_score = cohen_kappa_score(np.round(val_pred), val_truth, weights='quadratic')
        
        optimized_rounder.fit(val_pred, val_truth)
        coefficients = optimized_rounder.coefficients()
        final_preds = optimized_rounder.predict(val_pred, coefficients)
        
        opt_val_score = cohen_kappa_score(final_preds, val_truth, weights='quadratic')
        
        scheduler.step(val_loss)
        
        end = time.time()
        total = np.round(end-start,2)
        
        pre = 'Fold: %s{}%s'.format(fold+1)%fonts[0] + ' Epoch: %s{}%s'.format(epoch+1)%fonts[0]
        content =   'Train Loss: %s{:.4f}%s'.format(train_loss) % fonts[1]+                    ' Val Loss: %s{:.4f}%s'.format(val_loss) % fonts[1]+                    ' Train Kappa: %s{:.4f}%s'.format(train_score) % fonts[1]+                    ' Val Kappa: %s{:.4f}%s'.format(val_score) % fonts[1]+                    ' Optimized Val Kappa: %s{:.4f}%s'.format(opt_val_score) % fonts[1]+                    ' Coefficients: %s{}%s \n'.format(coefficients) % fonts[1]
        time_ = 'Time: %s{}%s s'.format(total) % fonts[1]

        print(pre+'\n'+content+'\n'+time_+'\n')
        
        with open(f'log_{fold+1}.txt', 'a') as appender:
                appender.write(content + '\n')

        if val_loss<best_loss:
            torch.save(model.state_dict(), f'SE_RNXT50_loss_{fold}.pt')
            best_loss=val_loss
            loss_kpre, loss_ktru, loss_rpre, loss_rtru = kpre, ktru, rpre, rtru
        
        if opt_val_score>best_opt_score:
            torch.save(model.state_dict(), f'SE_RNXT50_opt_qwk_{fold}.pt')
            best_opt_score=opt_val_score
            opt_coefs = coefficients
            optqwk_kpre, optqwk_ktru, optqwk_rpre, optqwk_rtru = kpre, ktru, rpre, rtru
        
        if val_score>best_score:
            torch.save(model.state_dict(), f'SE_RNXT50_qwk_{fold}.pt')
            best_score=val_score
            qwk_kpre, qwk_ktru, qwk_rpre, qwk_rtru = kpre, ktru, rpre, rtru
    
        recorder={'model': model.state_dict(), 'optim': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
        torch.save(recorder, 'recorder.pth')
    
    # In each fold we have saved 3 models by monitoring on Validation loss, QWK score and Optimized QWK score.
    # Here we find QWK score for each of it and their confusion matrices with Karolinska and Radboud separately. 
    
    l_kscore = cohen_kappa_score(np.round(loss_kpre), loss_ktru, weights='quadratic')
    l_rscore = cohen_kappa_score(np.round(loss_rpre), loss_rtru, weights='quadratic')
    print('Fold %s{}%s, LOSS MODEL \n'.format(fold+1) % fonts[0]+          'Karolinska QWK: %s{:.4f}%s \n'.format(l_kscore) % fonts[1]+          'Radboud QWK: %s{:.4f}%s \n'.format(l_rscore) % fonts[1])
    
    print('Confusion matrix for Karolinska, Fold {}, Loss model \n'.format(fold+1)+          '{} \n'.format(confusion_matrix(loss_ktru, np.round(loss_kpre))))
    
    print('Confusion matrix for Radboud, Fold {}, Loss model \n'.format(fold+1)+          '{} \n'.format(confusion_matrix(loss_rtru, np.round(loss_rpre))))
    
    q_kscore = cohen_kappa_score(np.round(qwk_kpre), qwk_ktru, weights='quadratic')
    q_rscore = cohen_kappa_score(np.round(qwk_rpre), qwk_rtru, weights='quadratic')
    print('Fold %s{}%s, QWK MODEL \n'.format(fold+1) % fonts[0]+          'Karolinska QWK: %s{:.4f}%s \n'.format(q_kscore) % fonts[1]+          'Radboud QWK: %s{:.4f}%s \n'.format(q_rscore) % fonts[1])
    
    print('Confusion matrix for Karolinska, Fold {}, QWK model \n'.format(fold+1)+          '{} \n'.format(confusion_matrix(qwk_ktru, np.round(qwk_kpre))))
    
    print('Confusion matrix for Radboud, Fold {}, QWK model \n'.format(fold+1)+          '{} \n'.format(confusion_matrix(qwk_rtru, np.round(qwk_rpre))))
    
    optqwk_kpre = optimized_rounder.predict(optqwk_kpre, opt_coefs)
    optqwk_rpre = optimized_rounder.predict(optqwk_rpre, opt_coefs)
    oq_kscore = cohen_kappa_score(optqwk_kpre, optqwk_ktru, weights='quadratic')
    oq_rscore = cohen_kappa_score(optqwk_rpre, optqwk_rtru, weights='quadratic')
    print('Fold %s{}%s Optimized QWK MODEL \n'.format(fold+1) % fonts[0]+          'Karolinska QWK: %s{:.4f}%s \n'.format(oq_kscore) % fonts[1]+          'Radboud QWK: %s{:.4f}%s \n'.format(oq_rscore) % fonts[1])
    
    print('Confusion matrix for Karolinska, Fold {}, Optimized QWK model \n'.format(fold+1)+          '{} \n'.format(confusion_matrix(optqwk_ktru, optqwk_kpre)))
    
    print('Confusion matrix for Radboud, Fold {}, Optimized QWK model \n'.format(fold+1)+          '{} \n'.format(confusion_matrix(optqwk_rtru, optqwk_rpre))) 


# In[ ]:


Parallel(n_jobs=n_folds, backend="threading")(delayed(train_fold)(i) for i in range(n_folds))

