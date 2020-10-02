#!/usr/bin/env python
# coding: utf-8

# ## Configuration

# * Usual Conv Net
# * noisy all
# * noisy pitch

# ## imports

# In[1]:


import gc
import os
import sys
import time
import pickle
import random
import logging
import datetime as dt

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from psutil import cpu_count

from fastprogress import master_bar, progress_bar
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from sklearn.model_selection import train_test_split, KFold

from tqdm import tqdm_notebook


# ## utils

# In[2]:


def get_logger(name="Main", tag="exp", log_dir="log/"):
    log_path = Path(log_dir)
    path = log_path / tag
    path.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(
        path / (dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".log"))
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s")

    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# In[3]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 520
seed_everything(SEED)


# In[4]:


# from official code https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8#scrollTo=cRCaCIb9oguU
def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      truth: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.  Then the overall unbalanced lwlrap is
        simply np.sum(per_class_lwlrap * weight_per_class)
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    # Space to store a distinct precision value for each class on each sample.
    # Only the classes that are true for each sample will be filled in.
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = (
            _one_sample_positive_class_precisions(scores[sample_num, :],
                                                  truth[sample_num, :]))
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
            precision_at_hits)
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    # Form average of each column, i.e. all the precisions assigned to labels in
    # a particular class.
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
    #                = np.sum(per_class_lwlrap * weight_per_class)
    return per_class_lwlrap, weight_per_class


# ## Dataset

# In[5]:


dataset_dir = Path('../input/freesound-audio-tagging-2019')
preprocessed_dir = Path('../input/freesound-normal-log-mel-features-1-channel/')
augmented_dir = Path("../input/freesound-sox-noisy-pitch-dataset/")


# In[6]:


csvs = {
    'train_curated': dataset_dir / 'train_curated.csv',
    'train_noisy': dataset_dir / 'train_noisy.csv',
    'sample_submission': dataset_dir / 'sample_submission.csv',
    "pitch": augmented_dir / "pitch.csv"
}

dataset = {
    'train_curated': dataset_dir / 'train_curated',
    'train_noisy': dataset_dir / 'train_noisy',
    'test': dataset_dir / 'test',
}

mels = {
    'train_curated': preprocessed_dir / 'mels_train_curated.pkl',
    'test': preprocessed_dir / 'mels_test.pkl',  # NOTE: this data doesn't work at 2nd stage
    'train_noisy': preprocessed_dir / 'mels_train_noisy.pkl',
    "pitch": augmented_dir / "mel_pitch.pkl"
}


# In[9]:


train_noisy = pd.read_csv(csvs['train_noisy'])
train_noisy.head()


# ## Train Test Split

# In[10]:


sampled = train_noisy.sample(frac=0.2, replace=False, random_state=SEED)
val_idx = sampled.index.values

train_idx = []
for i in range(len(train_noisy)):
    if i not in val_idx:
        train_idx.append(i)
        
train_idx = np.array(train_idx)

len(train_idx), len(val_idx), len(train_noisy)


# In[12]:


train_df = train_noisy.loc[train_idx, :].reset_index(drop=True)
val_df = train_noisy.loc[val_idx, :].reset_index(drop=True)


# ## Get melspectrogram

# In[13]:


with open(mels["train_noisy"], "rb") as f:
    x_noisy = pickle.load(f)
    
with open(mels["pitch"], "rb") as f:
    x_pitch = pickle.load(f)
    
len(x_noisy), len(x_pitch)


# In[16]:


x_train = []
x_val = []

for i in val_idx:
    x_val.append(x_noisy[i])
    
for i in train_idx:
    x_train.append(x_noisy[i])
    x_train.append(x_pitch[i])
    
len(x_train), len(x_val)


# ## Create target

# In[17]:


test_df = pd.read_csv(csvs['sample_submission'])
labels = test_df.columns[1:].tolist()
num_classes = len(labels)
print(num_classes)


# In[22]:


y_train = np.zeros((len(train_idx) * 2, num_classes), dtype=np.int)
for i, row in train_df.iterrows():
    label_list = row.labels.split(",")
    for label in label_list:
        idx = labels.index(label)
        y_train[2 * i, idx] = 1
        y_train[2 * i + 1, idx] = 1


y_val = np.zeros((len(val_idx), num_classes)).astype(int)
for i, row in val_df.iterrows():
    label_list = row.labels.split(",")
    for label in label_list:
        idx = labels.index(label)
        y_val[i, idx] = 1
        
y_train.shape, y_val.shape


# In[23]:


gc.collect()


# ## Data Transformation

# In[24]:


transforms_dict = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ]),
}


# In[25]:


class FATTrainDataset(Dataset):
    def __init__(self, mels, labels, transforms):
        super().__init__()
        self.mels = mels
        self.labels = labels
        self.transforms = transforms
        
    def __len__(self):
        return len(self.mels)
    
    def __getitem__(self, idx):
        # crop 1sec
        image = Image.fromarray(self.mels[idx], mode='L')        
        time_dim, base_dim = image.size
        crop = random.randint(0, time_dim - base_dim)
        image = image.crop([crop, 0, crop + base_dim, base_dim])
        image = self.transforms(image).div_(255)[0, :, :]
        
        label = self.labels[idx]
        label = torch.from_numpy(label).float()
        
        return image, label


# ## model

# In[26]:


def init_layer(layer, nonlinearity="leaky_relu"):
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
    
    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)
            
            
def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.running_mean.data.fill_(0.0)
    bn.weight.data.fill_(1.0)
    bn.running_var.data.fill_(1.0)
    
    
class SpatialAttention2d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z


class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z

    
class SCse(nn.Module):
    def __init__(self, dim):
        super(SCse, self).__init__()
        self.satt = SpatialAttention2d(dim)
        self.catt = GAB(dim)

    def forward(self, x):
        return self.satt(x) + self.catt(x)
    
    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.sigmoid(
            self.linear2(
                self.relu(
                    self.linear1(y))))
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)
    
    
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(2, 2),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(2, 2),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.scse = SCse(out_channels)
        # self.se = SELayer(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, inp, pool_size=(2, 2), pool_type="avg"):
        x = inp
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.scse(self.bn2(self.conv2(x))))
        # x = F.relu_(self.se(self.bn2(self.conv2(x))))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "both":
            x1 = F.max_pool2d(x, kernel_size=pool_size)
            x2 = F.avg_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            import pdb
            pdb.set_trace()
        return x
    
    
class ConvNet(nn.Module):
    def __init__(self, n_classes=80):
        super(ConvNet, self).__init__()
        self.conv1 = ConvBlock(1, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)
        self.conv5 = ConvBlock(256, 512)
        
        self.bn1 = nn.BatchNorm1d((1 + 4 + 20) * 512)
        self.drop1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear((1 + 4 + 20) * 512, 512)
        self.prelu = nn.PReLU()
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, n_classes)
        
    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_bn(self.bn1)
        init_bn(self.bn2)
    
    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = self.conv1(x, pool_size=(1, 1), pool_type="both")
        x = self.conv2(x, pool_size=(4, 1), pool_type="both")
        x = self.conv3(x, pool_size=(1, 3), pool_type="both")
        x = self.conv4(x, pool_size=(4, 1), pool_type="both")
        x = self.conv5(x, pool_size=(1, 3), pool_type="both")
        
        x1_max = F.max_pool2d(x, (5, 8))
        x1_mean = F.avg_pool2d(x, (5, 8))
        x1 = (x1_max + x1_mean).reshape(x.size(0), -1)
        
        x2_max = F.max_pool2d(x, (2, 4))
        x2_mean = F.avg_pool2d(x, (2, 4))
        x2 = (x2_max + x2_mean).reshape(x.size(0), -1)
        
        x = torch.mean(x, dim=3)
        x, _ = torch.max(x, dim=2)
        
        x = torch.cat([x, x1, x2], dim=1)
        x = self.drop1(self.bn1(x))
        x = self.prelu(self.fc1(x))
        x = self.drop2(self.bn2(x))
        x = self.fc2(x)
         
        return x


# ## train

# In[27]:


def train_model(x_train, y_train, x_val, y_val, train_transforms):
    num_epochs = 60
    batch_size = 128
    test_batch_size = 128
    lr = 1e-3
    eta_min = 1e-5
    t_max = 10
    
    num_classes = y_train.shape[1]
    
    train_dataset = FATTrainDataset(x_train, y_train, train_transforms)
    valid_dataset = FATTrainDataset(x_val, y_val, train_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=test_batch_size, shuffle=False)

    model = ConvNet(n_classes=80).cuda()
    criterion1 = nn.BCEWithLogitsLoss().cuda()

    optimizer = Adam(params=model.parameters(), lr=lr, amsgrad=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    best_epoch = -1
    best_lwlrap = 0.
    mb = master_bar(range(num_epochs))
    torch.cuda.empty_cache()

    for epoch in mb:
        start_time = time.time()
        model.train()
        avg_loss = 0.

        for x_batch, y_batch in progress_bar(train_loader, parent=mb):
            preds = model(x_batch.cuda())
            loss = criterion1(preds, y_batch.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() / len(train_loader)

        model.eval()
        valid_preds = np.zeros((len(x_val), num_classes))
        avg_val_loss = 0.

        for i, (x_batch, y_batch) in enumerate(valid_loader):
            with torch.no_grad():
                preds = model(x_batch.cuda()).detach()
                loss = criterion1(preds, y_batch.cuda())
                preds = torch.sigmoid(preds)
                valid_preds[i * test_batch_size: (i+1) * test_batch_size] = preds.cpu().numpy()

                avg_val_loss += loss.item() / len(valid_loader)
            
        score, weight = calculate_per_class_lwlrap(y_val, valid_preds)
        lwlrap = (score * weight).sum()
        
        scheduler.step()

        if (epoch + 1) % 1 == 0:
            elapsed = time.time() - start_time
            mb.write(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  val_lwlrap: {lwlrap:.6f}  time: {elapsed:.0f}s')
    
        if lwlrap > best_lwlrap:
            best_epoch = epoch + 1
            best_lwlrap = lwlrap
            torch.save(model.state_dict(), 'weight_best.pt')
            
    return {
        'best_epoch': best_epoch,
        'best_lwlrap': best_lwlrap,
    }


# In[ ]:


result = train_model(x_train, y_train, x_val, y_val, transforms_dict["train"])


# In[ ]:


torch.cuda.empty_cache()
gc.collect()


# In[ ]:


result

