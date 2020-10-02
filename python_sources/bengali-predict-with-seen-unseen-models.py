#!/usr/bin/env python
# coding: utf-8

# Hi everyone, I'm making a minimal version of my pipeline on this competition here, and making it public.
# 
# My pipeline contains 2 models:
# 
# * Seen Model, which is trained here (https://www.kaggle.com/haqishen/bengali-train-seen-model)
# * Unseen Model, which is trained here (https://www.kaggle.com/haqishen/bengali-train-unseen-model)
# 
# For the details of Seen Model and Unseen Model, please refer to those kernels. 
# 
# I've already trained my models on the kernels showed above, and loaded the output files into this inference kernel. Now I'm going to show you how to combine Seen Model and Unseen Model to make the prediction.

# In[ ]:


DEBUG = False


# In[ ]:


import os
import sys
sys.path.append('../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master')


# In[ ]:


import os
import gc
import cv2
import math
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from efficientnet_pytorch import model as enet


# In[ ]:


data_dir = '../input/bengaliai-cv19'
device = torch.device('cuda')

HEIGHT = 137
WIDTH = 236

img_size = 128

seen_th = 0.825  # seen / unseen threshold

c0_dim = 1295
c1_dim = 168
c2_dim = 11
c3_dim = 7
out_dim = c0_dim + c1_dim + c2_dim + c3_dim

num_workers = 4
batch_size = 32

files_test = [f'test_image_data_{fid}.parquet' for fid in range(4)]
df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
df_sub = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))

id2grapheme = {i: grapheme for i, grapheme in enumerate(df_train.grapheme.unique())}
grapheme2id = {grapheme: i for i, grapheme in enumerate(df_train.grapheme.unique())}
df_train['grapheme_id'] = df_train['grapheme'].map(grapheme2id)

df_label_map = []
for i, df in tqdm(df_train.groupby('grapheme_id')):
    df_label_map.append(df.iloc[:, 1:6].drop_duplicates())
df_label_map = pd.concat(df_label_map).reset_index(drop=True)

if DEBUG:
    files_test = [f'train_image_data_{fid}.parquet' for fid in range(4)]  # train files
    df_test = pd.read_csv(os.path.join(data_dir, 'train.csv'))  # train files
    seen_th = 0.94
#     device = torch.device('cpu')


# In[ ]:


def read_data(f):
    f = os.path.join(data_dir, f)
    data = pd.read_parquet(f)
    data = data.iloc[:, 1:].values
    return data


# In[ ]:


class BengaliDataset(Dataset):
    def __init__(self, data, image_size=128):

        self.data = data
        self.image_size = image_size

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        image = 255 - self.data[index].reshape(HEIGHT, WIDTH)

        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32) / 255
        image = image[np.newaxis, :, :]
        image = np.repeat(image, 3, 0)  # 1ch to 3ch

        return torch.tensor(image)


# In[ ]:


sigmoid = torch.nn.Sigmoid()
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

swish = Swish.apply

class Swish_module(nn.Module):
    def forward(self, x):
        return swish(x)

swish_layer = Swish_module()

def relu_fn(x):
    """ Swish activation function """
    return swish_layer(x)


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target, reduction='mean'):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss


class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=30.0, m=0.5, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.s = s
        self.cos_m = math.cos(m)             #  0.87758
        self.sin_m = math.sin(m)             #  0.47943
        self.th = math.cos(math.pi - m)      # -0.87758
        self.mm = math.sin(math.pi - m) * m  #  0.23971

    def forward(self, logits, labels):
        logits = logits.float()  # float16 to float32 (if used float16)
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))  # equals to **2
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = DenseCrossEntropy()(output, labels, self.reduction)
        return loss / 2


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine
    
    
class enet_3cg(nn.Module):

    def __init__(self, backbone, out_dim_1, out_dim_2):
        super(enet_3cg, self).__init__()
        self.enet = enet.EfficientNet.from_name(backbone)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        self.myfc_1 = nn.Linear(self.enet._fc.in_features, out_dim_2)
        self.activate = Swish_module()
        self.myfc_2 = nn.Linear(out_dim_2, out_dim_1)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out_2 = self.myfc_1(dropout(x))
            else:
                out_2 += self.myfc_1(dropout(x))
        out_2 /= len(self.dropouts)
        out_1 = self.myfc_2(self.activate(out_2))
        return out_1, out_2


class enet_arcface_v2(nn.Module):

    def __init__(self, backbone, out_dim_1, out_dim_2):
        super(enet_arcface_v2, self).__init__()
        self.enet = enet.EfficientNet.from_name(backbone)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])

        self.gfc = nn.Linear(self.enet._fc.in_features, 4096)
        self.metric_classify = ArcMarginProduct(4096, out_dim_1)
        self.myfc_1 = nn.Linear(4096, out_dim_1)
        self.myfc_2_1 = nn.Linear(4096, 512)
        self.myfc_2_2 = nn.Linear(512, out_dim_2)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = Swish_module()(self.gfc(x))
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out_1 = self.myfc_1(dropout(x))
                out_2 = self.myfc_2_1(dropout(x))
            else:
                out_1 += self.myfc_1(dropout(x))
                out_2 += self.myfc_2_1(dropout(x))
        out_1 /= len(self.dropouts)
        out_2 /= len(self.dropouts)
        out_2 = self.myfc_2_2(Swish_module()(out_2))
        metric_output = self.metric_classify(x)
        return out_1, out_2, metric_output


# In[ ]:


def get_models(model_files, model_class):
    enet_type = 'efficientnet-b1'
    models = []
    for model_f in model_files:

        model = model_class(enet_type, out_dim_1=c0_dim, out_dim_2=c1_dim+c2_dim+c3_dim)
        model = model.to(device)
        model.load_state_dict(torch.load(model_f, map_location=lambda storage, loc: storage), strict=True)
        model.eval()
        models.append(model)
        print(model_f, 'loaded!')

    return models


# In[ ]:


model_files_unseen = [
    '../input/bengali-train-unseen-model/effnet-b1-unseen_model_fold0.pth',
]
model_files_arcface = [
    '../input/bengali-train-seen-model/effnet-b1-seen_model_fold0.pth',
]


# In[ ]:


print('loading unseen models...')
models_unseen = get_models(model_files_unseen, enet_3cg)
print('loading arcface models...')
models_arcface = get_models(model_files_arcface, enet_arcface_v2)
print(len(models_unseen), len(models_arcface))


# In[ ]:


FINAL_P = []
with torch.no_grad():
    for file in tqdm(files_test):

        data = read_data(file)
        dataset_test = BengaliDataset(data, img_size)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        for (image) in tqdm(test_loader):
            image = image.to(device)

            logits_1 = torch.zeros(image.shape[0], c0_dim).to(device)
            logits_metric = torch.zeros(image.shape[0], c0_dim).to(device)

            # predict by arcface models
            for mid, model in enumerate(models_arcface):
                l1, l2, l3 = model(image)
                logits_1 += l1.softmax(1)
                logits_metric += l3
            logits_metric /= len(models_arcface)

            # fill predictions with Seen Model prediction as first
            # I decode 3 components from predicted grapheme here
            pred = df_label_map.iloc[logits_metric.detach().cpu().numpy().argmax(1), :3].values


            # use Arcface prediction threshold to find out unseen samples
            max_p = logits_metric.cpu().numpy().max(1)
            unseen_idx = np.where(max_p <= seen_th)[0]
            # if unseen_idx id not empty, use Unseen Models to predict them
            if unseen_idx.shape[0] > 0:
                logits_2_unseen = torch.zeros(unseen_idx.shape[0], c1_dim+c2_dim+c3_dim).to(device)
                for mid, model in enumerate(models_unseen):
                    _, l2 = model(image[unseen_idx])
                    logits_2_unseen[:, :c1_dim] += l2[:, :c1_dim].softmax(1)
                    logits_2_unseen[:, c1_dim:c1_dim+c2_dim] += l2[:, c1_dim:c1_dim+c2_dim].softmax(1)
                    logits_2_unseen[:, c1_dim+c2_dim:] += l2[:, c1_dim+c2_dim:].softmax(1)
                # overwrite prediction for unseen samples
                pred[unseen_idx, 0] = logits_2_unseen[:, :c1_dim].detach().cpu().numpy().argmax(1)
                pred[unseen_idx, 1] = logits_2_unseen[:, c1_dim:c1_dim+c2_dim].detach().cpu().numpy().argmax(1)
                pred[unseen_idx, 2] = logits_2_unseen[:, c1_dim+c2_dim:].detach().cpu().numpy().argmax(1)

            FINAL_P += pred.reshape(-1).tolist()
        del data
        gc.collect()


# In[ ]:


df_sub = pd.DataFrame({
    'row_id': [f'Test_{i}_{p}' for i in range(len(FINAL_P) // 3) for p in ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']],
    'target': FINAL_P
})

df_sub.to_csv('submission.csv',index=False)


# In[ ]:


df_sub


# In[ ]:




