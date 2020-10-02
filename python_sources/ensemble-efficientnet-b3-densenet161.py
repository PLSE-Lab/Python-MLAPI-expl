#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## TODO :
# Run for 80 epochs with no cut mix and mixup, acc after 60 epochs is .95


# In[ ]:


get_ipython().system('pip install ../input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/ > /dev/null # no output')

get_ipython().system('pip install /kaggle/input/efficientnet-pytorch/efficientnet_pytorch-0.6.1-py3-none-any.whl > /dev/null')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import cv2
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import sys

sys.path.insert(0, "/kaggle/input/ghostnetbengali")


import albumentations
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data.sampler import SequentialSampler
# from torchvision import transforms as torchtransforms
import torch.nn as nn
from tqdm import tqdm_notebook as tqdm
import torchvision
import torch.nn.functional as F
import time
from efficientnet_pytorch import EfficientNet
from ghost_net import ghost_net


# In[ ]:



# ======================
# Params

# I did not optimize the batch size
# If the GPU has more memory available
# you can increase this for faster inference
BATCH_SIZE = 96
N_WORKERS = 4

HEIGHT = 137
WIDTH = 236

INPUT_PATH="/kaggle/input/bengaliai-cv19"
MODEL_BASE_PATH_EFNET_B3  = '/kaggle/input/bengaliaieffnetb3'
MODEL_BASE_PATH_DN161FULL  = '/kaggle/input/densenet161full'


# In[ ]:



# These are from my experiments (I did not upload the weights)
# For this demo I used equal weights, but feel free to modify them.
# Make sure the sum of the labels are equals to 1 (per label)
ENSEMBLES = [
    {
        'model': 'efficientnet_b3',
        'model_state_file': MODEL_BASE_PATH_EFNET_B3 + '/efficientNet_fold2.pth',
    # LB: 0.9635
    },
    {
        'model': 'efficientnet_b3',
        'model_state_file': MODEL_BASE_PATH_EFNET_B3 + '/efficientNet_fold4.pth',
        # 0.9626
    },
    {
        'model': 'densenet161',
        'model_state_file': MODEL_BASE_PATH_DN161FULL + '/densenet161_b30',
    # with cutout and Scalerotate
    #  LB 0.9607
    },
]


# In[ ]:


NUM_ENSEMBLE = len(ENSEMBLES)


# In[ ]:


for ensemble in ENSEMBLES:
    ensemble['w_grapheme'] = 1 / NUM_ENSEMBLE
    ensemble['w_vowel'] = 1 / NUM_ENSEMBLE
    ensemble['w_conso'] = 1 / NUM_ENSEMBLE


# In[ ]:


test_df = pd.read_csv(INPUT_PATH + ('/test.csv'))
submission_df = pd.read_csv(INPUT_PATH + '/sample_submission.csv')


# In[ ]:


import pretrainedmodels
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class EfficientNetWrapper(nn.Module):
    def __init__(self):
        super(EfficientNetWrapper, self).__init__()
        
        # Load imagenet pre-trained model 
        self.effNet = EfficientNet.from_name('efficientnet-b3')
        
        # Appdend output layers based on our date
        self.fc_root = nn.Linear(in_features=1000, out_features=168)
        self.fc_vowel = nn.Linear(in_features=1000, out_features=11)
        self.fc_consonant = nn.Linear(in_features=1000, out_features=7)
        
    def forward(self, X):
        output = self.effNet(X)
        output_root = self.fc_root(output)
        output_vowel = self.fc_vowel(output)
        output_consonant = self.fc_consonant(output)
        
        return output_vowel, output_root, output_consonant
    
    
class DenseNet161(nn.Module):
    def __init__(self, pretrained):
        super(DenseNet161, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__["densenet161"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["densenet161"](pretrained=None)

        # self.model.features.gem = GeM()
        self.model.last_linear = Identity()

        self.l0 = nn.Linear(2208, 11)
        self.l1 = nn.Linear(2208, 168)
        self.l2 = nn.Linear(2208, 7)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)

        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        return l0, l1, l2


# In[ ]:


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=128, pad=16):
    HEIGHT = 137
    WIDTH = 236
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


# In[ ]:


class ClsTestDataset(Dataset):

    def __init__(self, num_samples=1, model='densenet161'):
        
        self.num_samples = num_samples
        self.images = np.zeros([num_samples, WIDTH * HEIGHT], dtype=np.uint8)
    
        self.aug = albumentations.Compose([
            albumentations.Resize(int(HEIGHT), int(WIDTH), always_apply=True),
            albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], always_apply=True)
        ])
    
        img_id = 0

        for i in tqdm(range(4)):
            datafile = INPUT_PATH + '/test_image_data_{}.parquet'.format(i)
            parq = pq.read_pandas(datafile, columns=[str(x) for x in range(32332)]).to_pandas()
            parq =  parq.iloc[:, :].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
            
            for idx, image in enumerate(parq):
                self.images[img_id, ...] = image.reshape(-1).astype(np.uint8)
                img_id = img_id + 1
                
        del parq

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        img = self.images[idx]
        img = img.reshape(HEIGHT, WIDTH).astype(float)
        img = Image.fromarray(img).convert("RGB")
        img = self.aug(image=np.array(img))["image"]
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = torch.tensor(img, dtype=torch.float)

        return {
            "image": img,
            "image_id": idx
        }


# In[ ]:


bengali_dataset = ClsTestDataset(num_samples = test_df.shape[0] // 3)


# In[ ]:


# If you'd like to use different batch size for
# different size models (tip #12)
data_loader_test = torch.utils.data.DataLoader(
    bengali_dataset,
    batch_size=BATCH_SIZE,
    num_workers=N_WORKERS,
    sampler=SequentialSampler(bengali_dataset),
    shuffle=False
)


# In[ ]:


# Predictions
size = submission_df.shape[0] // 3
results = {
    'grapheme_root': np.zeros((len(ENSEMBLES), size, 168), dtype=np.float),
    'vowel_diacritic': np.zeros((len(ENSEMBLES), size, 11), dtype=np.float),
    'consonant_diacritic': np.zeros((len(ENSEMBLES), size, 7), dtype=np.float),
}


# In[ ]:


for model_idx, ensemble in enumerate(ENSEMBLES):
        
    if ensemble['model'].lower() == 'densenet161':
        model = DenseNet161(pretrained=False)
    elif ensemble['model'].lower() == 'efficientnet_b3':
        model = EfficientNetWrapper()
    else:
        raise ValueError
    

    model_state = torch.load(ensemble['model_state_file'], map_location='cuda:0')
    model.load_state_dict(model_state)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    del model_state
    
    for batch_idx, data in enumerate(data_loader_test):
        images = data['image']
        image_idx = data['image_id']

        if torch.cuda.is_available():
            images = images.float().cuda()
        else:
            images = images.float()
        
        with torch.no_grad():
            out_vowel, out_graph, out_conso = model(images)
                              
                    
        out_graph = F.softmax(out_graph, dim=1).data.cpu().numpy() * ensemble['w_grapheme']
        out_vowel = F.softmax(out_vowel, dim=1).data.cpu().numpy() * ensemble['w_vowel']
        out_conso = F.softmax(out_conso, dim=1).data.cpu().numpy() * ensemble['w_conso']

        start = batch_idx * BATCH_SIZE
        end = min((batch_idx + 1) * BATCH_SIZE, submission_df.shape[0] // 3)

        results['grapheme_root'][model_idx, start:end, :] = out_graph
        results['vowel_diacritic'][model_idx, start:end, :] = out_vowel
        results['consonant_diacritic'][model_idx, start:end, :] = out_conso
        
        del images
        del out_graph, out_vowel, out_conso
            
    del model


# In[ ]:


# Clean-up
del data_loader_test
del bengali_dataset
del test_df

gc.collect()
get_ipython().run_line_magic('reset', '-f out')


# In[ ]:


submission_df = pd.read_csv(INPUT_PATH + '/sample_submission.csv')
submission_df.head()


# In[ ]:


for l in ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']:
    idx = submission_df[submission_df['row_id'].str.contains(l)].index
    submission_df.iloc[idx, 1] = results[l].sum(axis=0).argmax(axis=1)


# In[ ]:


submission_df.to_csv('./submission.csv', index=False)


# In[ ]:


submission_df.head(100)


# In[ ]:




