#!/usr/bin/env python
# coding: utf-8

# ## This notebook show you how to use PyTorch lightening on Tabular dataset with Titanic
# 
# #### references
# 
# + [Official GitHub](https://github.com/williamFalcon/pytorch-lightning)
# 
# + [Docs](https://williamfalcon.github.io/pytorch-lightning/Trainer/Checkpointing/#model-saving)
# 
# + [My last post showing how to use pytorch NN with tatanic](https://www.kaggle.com/kaerunantoka/titanic-pytorch-nn-tutorial)
# 
# + [yashuseth's blog post : 'A Neural Network in PyTorch for Tabular Data with Categorical Embeddings'](https://yashuseth.blog/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/)

# ### First install pytorch_lightning

# In[ ]:


get_ipython().system('pip install pytorch_lightning')


# ### Library

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import os
import gc
import random
import sys
import time
import feather
import numpy as np
import pandas as pd
import logging
from contextlib import contextmanager
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import tensorflow as tf
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models

import pytorch_lightning as pl


# In[ ]:


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

kaeru_seed = 1337
seed_everything(seed=kaeru_seed)

DATA_DIR = '../input/titanic/'


# In[ ]:


train = pd.read_csv(DATA_DIR  + 'train.csv')
test = pd.read_csv(DATA_DIR  + 'test.csv')
target = train["Survived"]

n_train = len(train)
train = pd.concat([train, test], sort=True)

print(train.shape)
# train.head()


# ### Preprocessing

# In[ ]:


categorical_features = []
for col in tqdm(train.columns):
    if train[col].dtype == 'object':
        categorical_features.append(col)
print(len(categorical_features))


# In[ ]:


from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for cat_col in categorical_features:
    label_encoders[cat_col] = LabelEncoder()
    # fill missing values with <Miising> tokens
    train[cat_col] = train[cat_col].fillna('<Missing>')
    train[cat_col] = label_encoders[cat_col].fit_transform(train[cat_col])


# In[ ]:


cat_dims = [int(train[col].nunique()) for col in categorical_features]
emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]


# In[ ]:


from sklearn.preprocessing import StandardScaler

numerical_features = list(set(train.columns) - set(categorical_features) - set(['Survived']))
print(len(numerical_features))
    
train[numerical_features] = train[numerical_features].fillna(-999)
print('scaling numerical columns')

scaler = StandardScaler()
train[numerical_features] = scaler.fit_transform(train[numerical_features]) 


# In[ ]:


data = train[:n_train]
test = train[n_train:]
del test['Survived']
gc.collect()

print(data.shape, test.shape)


# In[ ]:


from sklearn.model_selection import train_test_split

train_idx, valid_idx = train_test_split(list(data.index),test_size=0.2, random_state=kaeru_seed, stratify=target)
output_feature = "Survived"


# In[ ]:


class TitanicDataset(Dataset):
  def __init__(self, data, cat_cols=None, output_col=None):

    self.n = data.shape[0]

    if output_col:
      self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
    else:
      self.y =  np.zeros((self.n, 1))

    self.cat_cols = cat_cols if cat_cols else []
    self.cont_cols = [col for col in data.columns
                      if col not in self.cat_cols + [output_col]]

    if self.cont_cols:
      self.cont_X = data[self.cont_cols].astype(np.float32).values
    else:
      self.cont_X = np.zeros((self.n, 1))

    if self.cat_cols:
      self.cat_X = data[cat_cols].astype(np.int64).values
    else:
      self.cat_X =  np.zeros((self.n, 1))

  def __len__(self):

    return self.n

  def __getitem__(self, idx):

    return [self.y[idx], self.cont_X[idx], self.cat_X[idx]]


# In[ ]:


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss
      
criterion = RMSELoss()


# ### Define LightningModule which contains model, dataloader, and so on...

# In[ ]:


class CoolSystem(pl.LightningModule):

    def __init__(self, emb_dims, no_of_cont, lin_layer_sizes,
               output_size, emb_dropout, lin_layer_dropouts):
        super(CoolSystem, self).__init__()
        
        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y)
                                     for x, y in emb_dims])

        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont

        # Linear Layers
        first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont,
                                lin_layer_sizes[0])

        self.lin_layers =            nn.ModuleList([first_lin_layer] +            [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1])
            for i in range(len(lin_layer_sizes) - 1)])
    
        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1],
                                  output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size)
                                    for size in lin_layer_sizes])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size)
                                  for size in lin_layer_dropouts])

    def forward(self, cont_data, cat_data):
      
        if self.no_of_embs != 0:
              x = [emb_layer(cat_data[:, i])
                  for i,emb_layer in enumerate(self.emb_layers)]
              x = torch.cat(x, 1)
              x = self.emb_dropout_layer(x)

        if self.no_of_cont != 0:
            normalized_cont_data = self.first_bn_layer(cont_data)

        if self.no_of_embs != 0:
            x = torch.cat([x, normalized_cont_data], 1) 
        else:
            x = normalized_cont_data

        for lin_layer, dropout_layer, bn_layer in            zip(self.lin_layers, self.droput_layers, self.bn_layers):
      
            x = F.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)

        x = self.output_layer(x)

        return x

    def training_step(self, batch, batch_nb):
        # REQUIRED
        y, cont_x, cat_x = batch
        y_hat = self.forward(cont_x, cat_x)
        return {'loss': criterion(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        y, cont_x, cat_x = batch
        y_hat = self.forward(cont_x, cat_x)
        return {'val_loss': criterion(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), lr=0.02)

    @pl.data_loader
    def tng_dataloader(self):
        # REQUIRED
        return DataLoader(TitanicDataset(data=data.loc[train_idx], cat_cols=categorical_features, output_col=output_feature), batch_size=256)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(TitanicDataset(data=data.loc[valid_idx], cat_cols=categorical_features, output_col=output_feature), batch_size=256)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(TitanicDataset(data=test, cat_cols=categorical_features, output_col=None), batch_size=256)


# ### Save model settings

# In[ ]:


from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    filepath='../input/weights.ckpt',
    save_best_only=True,
    verbose=True,
    monitor='avg_val_loss',
    mode='min'
)


# ### Training

# In[ ]:


from pytorch_lightning import Trainer
from test_tube import Experiment

model = CoolSystem(emb_dims, no_of_cont=len(numerical_features), 
                          lin_layer_sizes=[50, 100],
                          output_size=1, emb_dropout=0.04,
                          lin_layer_dropouts=[0.001,0.01])

exp = Experiment(save_dir='../input/')

# most basic trainer, uses good defaults
trainer = Trainer(experiment=exp, max_nb_epochs=5, train_percent_check=0.1, checkpoint_callback=checkpoint_callback) 
trainer.fit(model)


# In[ ]:


# !tensorboard --logdir ../input/


# ### Check outputs

# In[ ]:


get_ipython().system('ls ../input/weights.ckpt/')


# In[ ]:


get_ipython().system('ls ../input/default/version_0/')


# In[ ]:


res = pd.read_csv('../input/default/version_0/metrics.csv')
res


# ### Predict

# In[ ]:


device = 'cpu'
checkpoint = torch.load('../input/weights.ckpt/_ckpt_epoch_5.ckpt')

def predict(fold, model, tk0, TEST_BATCH_SZ):    
    model.load_state_dict(checkpoint['state_dict'])
    for param in model.parameters():
        param.requires_grad = False    
    model.eval()
    for i, (y, cont_x, cat_x) in enumerate(tk0):
        with torch.no_grad():
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)
            preds = model(cont_x, cat_x)
            test_preds[i * TEST_BATCH_SZ:(i + 1) * TEST_BATCH_SZ] = preds.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)               
    return test_preds


# In[ ]:


test_dataset = TitanicDataset(data=test, cat_cols=categorical_features, output_col=None)
TEST_BATCH_SZ = 256
all_test_preds = []

test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BATCH_SZ, shuffle=False, num_workers=4, drop_last=True)
test_preds = np.zeros((len(test_dataset), 1))
tk0 = tqdm(test_data_loader)
test_preds = predict(0, model, tk0, TEST_BATCH_SZ)
all_test_preds.append(test_preds)

test_preds = np.mean(all_test_preds, 0) 
print(test_preds.shape)


# In[ ]:


# define threshold
delta = 0.5
sub = pd.read_csv('../input/titanic/gender_submission.csv')

sub['Survived'] = (test_preds > delta).astype(int)
sub.to_csv('submission.csv', index=False)


# In[ ]:


sub.head()


# ### Thank you for Reading ;)

# In[ ]:




