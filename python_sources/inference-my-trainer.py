#!/usr/bin/env python
# coding: utf-8

# # Melanoma inference kernel by [@shonenkov](https://www.kaggle.com/shonenkov)

# # Main Idea:
# 
# Inference for single model

# # Dependencies

# In[ ]:


get_ipython().system('pip install -q efficientnet_pytorch > /dev/null')


# In[ ]:


from glob import glob
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import GroupKFold
import cv2
from skimage import io
import albumentations as A
import torch
import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.nn import functional as F
from glob import glob
import sklearn
from torch import nn
import warnings

warnings.filterwarnings("ignore") 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)


# # External data
# 
# I have prepared [kernel with merging data](https://www.kaggle.com/shonenkov/merge-external-data). Don't forget to read this kernel ;)

# In[ ]:


DATA_PATH = '../input/melanoma-merged-external-data-512x512-jpeg'


# In[ ]:


TEST_ROOT_PATH = f'{DATA_PATH}/512x512-test/512x512-test'

def get_valid_transforms():
    return A.Compose([
            A.Resize(height=768, width=768, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

class DatasetRetriever(Dataset):

    def __init__(self, image_ids,age,sex,anatom,transforms=None):
        super().__init__()
        self.image_ids = image_ids
        self.age = age
        self.sex = sex
        self.anatom = anatom
        self.transforms = transforms

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image = cv2.imread(f'{TEST_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = image.astype(np.float32) / 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
        meta = np.array([self.age[idx], self.sex[idx], self.anatom[idx]]).astype('float')
        meta = torch.Tensor(meta)
        return image, image_id,meta

    def __len__(self) -> int:
        return self.image_ids.shape[0]


# In[ ]:


df_test = pd.read_csv(f'../input/preprocessed-siim/test_final.csv', index_col='image_name')
df_test.head()


# In[ ]:


get_ipython().system('pip install timm')
import timm


# In[ ]:


model = timm.create_model('efficientnet_b3',pretrained = True)
class cnn_model(nn.Module):
    def __init__(self,model):
        super(cnn_model,self).__init__()
        self.image = model
    def forward(self,x):
        return self.image(x)


class meta_model(nn.Module):
    def __init__(self):
        super(meta_model,self).__init__()
        self.lin1 = nn.Linear(3,64)
        self.lin2 = nn.Linear(64,128)
        self.lin3 = nn.Linear(128,256)
        self.bn1 = nn.BatchNorm1d(num_features = 64)
        self.bn2 = nn.BatchNorm1d(num_features = 128)
        
    def forward(self,x):
        x = self.bn1(self.lin1(x))
        x = F.relu(x)
        x = self.bn2(self.lin2(x))
        x = F.relu(x)
        x = self.lin3(x)
        return x


class get_net(nn.Module):
    def __init__(self,image_model,meta_model):
        super(get_net,self).__init__()
        self.image_model = image_model
        self.meta_model = meta_model
        self.bn_1 = nn.BatchNorm1d(num_features = 1256)
        self.lin_1 = nn.Linear(1256,512)
        self.bn_2 = nn.BatchNorm1d(num_features = 512)
        self.lin_2 = nn.Linear(512,256)
        self.bn_3 = nn.BatchNorm1d(num_features = 256)
        self.lin_3 = nn.Linear(256,2)
        
    def forward(self,image,meta_data):
        image_feat = self.image_model(image)
        meta_feat = self.meta_model(meta_data)
        x = torch.cat((image_feat,meta_feat),dim =1)
        x = self.bn_1(x)
        x = self.bn_2(self.lin_1(x))
        x = F.relu(x)
        x = self.bn_3(self.lin_2(x))
        x = F.relu(x)
        x = self.lin_3(x)
        return x   


# In[ ]:


modelA = cnn_model(model)
modelB = meta_model()
net = get_net(modelA,modelB)


# In[ ]:


net.cuda()


# In[ ]:


checkpoint = torch.load(f'../input/b3-meta-f4-768/fold4/best-score-checkpoint-008epoch.bin')
net.load_state_dict(checkpoint);


# In[ ]:



test_dataset = DatasetRetriever(
    image_ids=df_test.index.values,
    age=df_test.age_approx.values,
    sex=df_test.sex.values,
    anatom=df_test.anatom_site_general_challenge.values,
    transforms=get_valid_transforms(),
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=8,
    num_workers=2,
    shuffle=False,
    sampler=SequentialSampler(test_dataset),
    pin_memory=False,
    drop_last=False,
)


# In[ ]:


TTA = 3
net.eval()
preds = np.zeros((len(df_test),1))
for j in range(TTA):
        print(j) 
        for i, x in enumerate(test_loader):
            images = x[0]
            image_names = x[1]
            meta_data = x[2]
            images = images.cuda().float()
            meta_data = meta_data.cuda().float()
            outputs = net(images,meta_data)
            y_pred = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()[:,1]
            preds[i*y_pred.shape[0]:i*y_pred.shape[0] + y_pred.shape[0]] += y_pred.reshape((-1,1))
preds /= TTA
    


# In[ ]:


preds


# In[ ]:


result = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')
result['target'] = preds
result.head()


# In[ ]:





# In[ ]:


net.eval();

result = {'image_name': [], 'target': []}
for images, image_names,meta_data in tqdm(test_loader, total=len(test_loader)):
    with torch.no_grad():
        images = images.cuda().float()
        meta_data = meta_data.cuda().float()
        outputs = net(images,meta_data)
        y_pred = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()[:,1]
    
    result['image_name'].extend(image_names)
    result['target'].extend(y_pred)

submission = pd.DataFrame(result)


# In[ ]:


submission.to_csv('b3metaf4768.csv', index=False)
submission['target'].hist(bins=100);


# # Thank you for reading my kernel
# 
# Don't forget to read my other kernels about this competition:
# 
# - [[Training CV] Melanoma Starter](https://www.kaggle.com/shonenkov/training-cv-melanoma-starter)
# - [[Merge External Data]](https://www.kaggle.com/shonenkov/merge-external-data)
