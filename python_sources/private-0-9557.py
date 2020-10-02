#!/usr/bin/env python
# coding: utf-8

# # Description
# This kernel performs inference for [Grapheme fast.ai starter](https://www.kaggle.com/iafoss/grapheme-fast-ai-starter) kernel. Check it for more training details. The image preprocessing pipline is provided [here](https://www.kaggle.com/iafoss/image-preprocessing-128x128).

# In[ ]:


get_ipython().system('pip install ../input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/ > /dev/null')


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import cv2
import torch
from tqdm import tqdm_notebook as tqdm
import fastai
from fastai.vision import *
import os
import pretrainedmodels
print(os.listdir('../input'))
from mish_activation import *
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


HEIGHT = 137
WIDTH = 236
SIZE = 128
bs = 128
stats = (0.0692, 0.2051)
arch = pretrainedmodels.__dict__['se_resnext50_32x4d']
MODEL = '../input/weights/seresnext.pth'
nworkers = 2

TEST = ['/kaggle/input/bengaliai-cv19/test_image_data_0.parquet',
        '/kaggle/input/bengaliai-cv19/test_image_data_1.parquet',
        '/kaggle/input/bengaliai-cv19/test_image_data_2.parquet',
        '/kaggle/input/bengaliai-cv19/test_image_data_3.parquet']

LABELS = '../input/bengaliai-cv19/train.csv'

df = pd.read_csv(LABELS)
nunique = list(df.nunique())[1:-1]


# In[ ]:


nunique


# # Model

# In[ ]:


class Head(nn.Module):
    def __init__(self, nc, n, ps=0.5):
        super().__init__()
        layers = [AdaptiveConcatPool2d(), Mish(), Flatten()] +             bn_drop_lin(nc*2, 512, True, ps, Mish()) +             bn_drop_lin(512, n, True, ps)
        self.fc = nn.Sequential(*layers)
        self._init_weight()
        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
        
    def forward(self, x):
        return self.fc(x)

#change the first conv to accept 1 chanel input
class Dnet_1ch(nn.Module):
    def __init__(self, arch=arch, n=nunique, pre=True, ps=0.5):
        super().__init__()
        m = arch(pretrained='imagenet') if pre else arch(pretrained=None)
        
        conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        w = (m.layer0.conv1.weight.sum(1)).unsqueeze(1)
        conv.weight = nn.Parameter(w)
        
        self.layer0 = nn.Sequential(conv, m.layer0.bn1, m.layer0.relu1, m.layer0.pool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = nn.Sequential(m.layer4[0], m.layer4[1], m.layer4[2])

        
        nc = self.layer4[-1].se_module.fc2.out_channels #changes as per architecture
        self.head1 = Head(nc,n[0])
        self.head2 = Head(nc,n[1])
        self.head3 = Head(nc,n[2])
        #to_Mish(self.layer0), to_Mish(self.layer1), to_Mish(self.layer2)
        #to_Mish(self.layer3), to_Mish(self.layer4)
        
    def forward(self, x):    
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x1 = self.head1(x)
        x2 = self.head2(x)
        x3 = self.head3(x)
        
        return x1,x2,x3


# In[ ]:


model = Dnet_1ch(pre=False).cuda()
model.load_state_dict(torch.load(MODEL, map_location=torch.device('cpu')));
model.eval();


# # Data

# In[ ]:


#check https://www.kaggle.com/iafoss/image-preprocessing-128x128

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


# In[ ]:


class GraphemeDataset(Dataset):
    def __init__(self, fname):
        self.df = pd.read_parquet(fname)
        self.data = 255 - self.df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.df.iloc[idx,0]
        #normalize each image by its max val
        img = (self.data[idx]*(255.0/self.data[idx].max())).astype(np.uint8)
        img = crop_resize(img)
        img = (img.astype(np.float32)/255.0 - stats[0])/stats[1]
        return img, name


# # Prediction

# In[ ]:


EXP = -1.2
G_logits = []
V_logits = []
C_logits = []
G_preds = []
V_preds=[]
C_preds = []


# In[ ]:


import scipy


# In[ ]:


row_id,target = [],[]
for fname in TEST:
    ds = GraphemeDataset(fname)
    dl = DataLoader(ds, batch_size=bs, num_workers=nworkers, shuffle=False)
    with torch.no_grad():
        for x,y in tqdm(dl):
            x = x.unsqueeze(1).cuda()
            p1,p2,p3 = model(x)
            p1 = torch.nn.functional.softmax(p1)
            p2 = torch.nn.functional.softmax(p2)
            p3 = torch.nn.functional.softmax(p3)
            G_logits.append(p1.cpu())
            V_logits.append(p2.cpu())
            C_logits.append(p3.cpu())
            p1 = p1.argmax(-1).view(-1).cpu()
            p2 = p2.argmax(-1).view(-1).cpu()
            p3 = p3.argmax(-1).view(-1).cpu()
            G_preds.append(p1)
            V_preds.append(p2)
            C_preds.append(p3)
           
        


# In[ ]:


G_preds = np.concatenate(G_preds)
V_preds = np.concatenate(V_preds)
C_preds = np.concatenate(C_preds)


# In[ ]:


s = pd.Series(G_preds)
vc = s.value_counts().sort_index()
df = pd.DataFrame({'a':np.arange(168),'b':np.ones(168)})
df.b = df.a.map(vc)
df.fillna(df.b.min(),inplace=True)
mat1 = np.diag(df.b.astype('float32')**EXP)

s = pd.Series(V_preds)
vc = s.value_counts().sort_index()
df = pd.DataFrame({'a':np.arange(11),'b':np.ones(11)})
df.b = df.a.map(vc)
df.fillna(df.b.min(),inplace=True)
mat2 = np.diag(df.b.astype('float32')**EXP)

s = pd.Series(C_preds)
vc = s.value_counts().sort_index()
df = pd.DataFrame({'a':np.arange(7),'b':np.ones(7)})
df.b = df.a.map(vc)
df.fillna(df.b.min(),inplace=True)
mat3 = np.diag(df.b.astype('float32')**EXP)

G_logits = np.concatenate(G_logits)
V_logits = np.concatenate(V_logits)
C_logits = np.concatenate(C_logits)

G_pred = np.argmax( G_logits.dot(mat1), axis=1)
V_pred = np.argmax( V_logits.dot(mat2), axis=1)
C_pred = np.argmax( C_logits.dot(mat3), axis=1)


# In[ ]:


all_list = [G_pred, V_pred, C_pred]


# In[ ]:


labels = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']

sub = pd.DataFrame({'row_id': [f'Test_{i}_{j}' for i in range(len(G_pred)) for j in labels ],'target':[data[i].item() for i in range(len(G_pred)) for data in all_list]} )
sub.to_csv('submission.csv', index=False)
sub

