#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install /kaggle/input/pretrained-models/pretrainedmodels-0.7.4-py3-none-any.whl')
get_ipython().system('pip install /kaggle/input/pretrained-models/efficientnet_pytorch-0.6.1-py3-none-any.whl')


# In[ ]:


## Ref https://youtu.be/uZalt-weQMM
import sys
import pretrainedmodels
from torchvision import transforms,models
from efficientnet_pytorch import EfficientNet
import glob
import torch
import albumentations
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
from tqdm import tqdm
from PIL import Image
import joblib
import torch.nn as nn
from torch.nn import functional as F


# In[ ]:


TEST_BATCH_SIZE = 32
MODEL_MEAN=(0.485,0.465,0.406)
MODEL_STD=(0.229,0.224,0.225)
IMG_HEIGHT=224
IMG_WIDTH=224
DEVICE="cuda"


# In[ ]:


class EfficientNetB1(nn.Module):
    def __init__(self, pretrained):
        super(EfficientNetB1, self).__init__()

        if pretrained is True:
            self.model = EfficientNet.from_pretrained("efficientnet-b1")
        else:
            self.model = EfficientNet.from_name('efficientnet-b1')
        
        self.l0 = nn.Linear(1280, 168)
        self.l1 = nn.Linear(1280, 11)
        self.l2 = nn.Linear(1280, 7)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)

        return l0, l1, l2


# In[ ]:


class BengaliDatasetTest:
    def __init__(self, df, img_height, img_width, mean, std):
        
        self.image_ids = df.image_id.values
        self.img_arr = df.iloc[:, 1:].values

        self.aug = albumentations.Compose([
            albumentations.Resize(img_height, img_width, always_apply=True),
            albumentations.Normalize(mean, std, always_apply=True)
#             albumentations.OneOf([
#                     GridMask(num_grid=3, mode=0, rotate=15),
#                     GridMask(num_grid=3, mode=2, rotate=15),
#                 ], p=0.75)
        ])


    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, item):
        image = self.img_arr[item, :]
        img_id = self.image_ids[item]
        
        image = image.reshape(137, 236).astype(float)
        image = Image.fromarray(image).convert("RGB")
        image = self.aug(image=np.array(image))["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        

        return {
            "image": torch.tensor(image, dtype=torch.float),
            "image_id": img_id
        }


# In[ ]:


def model_predict():
    g_pred, v_pred, c_pred = [], [], []
    img_ids_list = [] 
    
    for file_idx in range(4):
        df = pd.read_parquet(f"../input/bengaliai-cv19/test_image_data_{file_idx}.parquet")

        dataset = BengaliDatasetTest(df=df,
                                    img_height=IMG_HEIGHT,
                                    img_width=IMG_WIDTH,
                                    mean=MODEL_MEAN,
                                    std=MODEL_STD)

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size= TEST_BATCH_SIZE,
            shuffle=False,
            num_workers=4
        )

        for bi, d in enumerate(data_loader):
            image = d["image"]
            img_id = d["image_id"]
            image = image.to(DEVICE, dtype=torch.float)

            g, v, c = model(image)
            #g = np.argmax(g.cpu().detach().numpy(), axis=1)
            #v = np.argmax(v.cpu().detach().numpy(), axis=1)
            #c = np.argmax(c.cpu().detach().numpy(), axis=1)

            for ii, imid in enumerate(img_id):
                g_pred.append(g[ii].cpu().detach().numpy())
                v_pred.append(v[ii].cpu().detach().numpy())
                c_pred.append(c[ii].cpu().detach().numpy())
                img_ids_list.append(imid)
        
    return g_pred, v_pred, c_pred, img_ids_list


# In[ ]:


model = EfficientNetB1(pretrained=False)
TEST_BATCH_SIZE = 32
Start_fold=3
final_g_pred = []
final_v_pred = []
final_c_pred = []
final_img_ids = []

for i in range(Start_fold,4):
    model.load_state_dict(torch.load(f"../input/efficientnetweights/efficientNetb1_fold{i}.pth"),strict=False)
    model.to(DEVICE)
    model.eval()
    g_pred, v_pred, c_pred, img_ids_list = model_predict()
    final_g_pred.append(g_pred)
    final_v_pred.append(v_pred)
    final_c_pred.append(c_pred)
    if i == Start_fold:
        final_img_ids.extend(img_ids_list)


# In[ ]:


final_g = np.argmax(np.mean(np.array(final_g_pred), axis=0), axis=1)
final_v = np.argmax(np.mean(np.array(final_v_pred), axis=0), axis=1)
final_c = np.argmax(np.mean(np.array(final_c_pred), axis=0), axis=1)
# print(final_g)
# print(final_img_ids)
predictions = []
for ii, imid in enumerate(final_img_ids):

    predictions.append((f"{imid}_grapheme_root", final_g[ii]))
    predictions.append((f"{imid}_vowel_diacritic", final_v[ii]))
    predictions.append((f"{imid}_consonant_diacritic", final_c[ii]))


# In[ ]:


sub = pd.DataFrame(predictions,columns=["row_id","target"])
print(sub)


# In[ ]:


sub.to_csv("submission.csv",index=False)


# In[ ]:




