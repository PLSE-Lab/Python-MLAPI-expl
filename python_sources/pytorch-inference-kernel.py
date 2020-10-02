#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import torchvision
import torch.nn as nn
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import os

device = torch.device("cuda:0")
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[ ]:


class RetinopathyDatasetTest(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('../input/aptos2019-blindness-detection/test_images', self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        image = image.resize((256, 256), resample=Image.BILINEAR)
        return {'image': transforms.ToTensor()(image)}


# In[ ]:


model = torchvision.models.resnet101(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(2048, 1)
model.load_state_dict(torch.load("../input/dr-model/model.bin"))
model = model.to(device)


# In[ ]:


for param in model.parameters():
    param.requires_grad = False

model.eval()


# In[ ]:


test_dataset = RetinopathyDatasetTest(csv_file='../input/aptos2019-blindness-detection/sample_submission.csv')
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


# In[ ]:


test_preds = np.zeros((len(test_dataset), 1))
tk0 = tqdm(test_data_loader)
for i, x_batch in enumerate(tk0):
    x_batch = x_batch["image"]
    pred = model(x_batch.to(device))
    test_preds[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)


# In[ ]:


coef = [0.5, 1.5, 2.5, 3.5]

for i, pred in enumerate(test_preds):
    if pred < coef[0]:
        test_preds[i] = 0
    elif pred >= coef[0] and pred < coef[1]:
        test_preds[i] = 1
    elif pred >= coef[1] and pred < coef[2]:
        test_preds[i] = 2
    elif pred >= coef[2] and pred < coef[3]:
        test_preds[i] = 3
    else:
        test_preds[i] = 4


sample = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
sample.diagnosis = test_preds.astype(int)
sample.to_csv("submission.csv", index=False)


# In[ ]:


sample

