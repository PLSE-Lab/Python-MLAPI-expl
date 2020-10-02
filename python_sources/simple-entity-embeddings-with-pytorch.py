#!/usr/bin/env python
# coding: utf-8

# # Inspired by [Abhishek's Kernel](https://www.kaggle.com/abhishek/same-old-entity-embeddings), I just tried to retranslate handling categorical features in simple way using Pytorch

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import copy
import traceback
import datetime
import random


# In[ ]:


def reduce_mem_usage(df, verbose=True):
    '''
    Reduce file memory usage
    Source: https://www.kaggle.com/artgor
    
    Parameters:
    -----------
    df: DataFrame
        Dataset on which to perform transformation
    verbose: bool
        Print additional information
    Returns:
    --------
    DataFrame
        Dataset as pandas DataFrame
    '''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16)                                               .max and c_prec == np.finfo(np.float16).precision:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32)                                                .max and c_prec == np.finfo(np.float32).precision:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'                                               .format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    
    return (df)


# In[ ]:


class CatDataset(Dataset):
    def __init__(self, data, cat_cols=None, output_col=None, train=True):
        self.n = data.shape[0]

        if output_col:
            self.y = data[output_col].astype(np.float32).values
        else:
            self.y =  np.zeros((self.n, 1))

        self.cat_cols = cat_cols if cat_cols else []

        if self.cat_cols:
            self.cat_X = data[cat_cols].astype(np.int64).values
        else:
            self.cat_X =  np.zeros((self.n, 1))

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return [self.y[idx], self.cat_X[idx]]


# In[ ]:


class GottaTorch(nn.Module):

    def __init__(self, emb_dims, lin_layer_sizes,
               output_size, emb_dropout, lin_layer_dropouts):
                
        """
        emb_dims: List of two element tuples
        For each categorical feature the first element of a tuple will
        denote the number of unique values of the categorical
        feature. The second element will denote the embedding
        dimension to be used for that feature.
        """

        super().__init__()

        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])   
        self.no_of_embs = sum([y for x, y in emb_dims])

        # Linear Layers
        first_lin_layer = nn.Linear(in_features=self.no_of_embs, 
                                    out_features=lin_layer_sizes[0])

        self.lin_layers =         nn.ModuleList([first_lin_layer] +              [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1])
               for i in range(len(lin_layer_sizes) - 1)])

        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1],
                                      output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_embs)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size)
                                        for size in lin_layer_sizes])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout) 
        self.droput_layers = nn.ModuleList([nn.Dropout(size)
                                      for size in lin_layer_dropouts])
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, cat_data):

        if self.no_of_embs != 0:
            x = [emb_layer(cat_data[:, i])
               for i, emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, 1)
            x = self.first_bn_layer(x)
            x = self.emb_dropout_layer(x)       

        for lin_layer, dropout_layer, bn_layer in            zip(self.lin_layers, self.droput_layers, self.bn_layers):

            x = F.relu(lin_layer(x))
            x = dropout_layer(x)
            x = bn_layer(x)
            
        x = self.output_layer(x)
        x = self.softmax(x)
        
        return x


# In[ ]:


def predict_with_model(model, dataset, device=None, batch_size=32, num_workers=0):
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results_by_batch = []

    device = torch.device(device)
    model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    with torch.no_grad():
        import tqdm
        for _, batch_x in tqdm.tqdm(dataloader, total=len(dataset)/batch_size):
            batch_x = copy_data_to_device(batch_x, device)

            batch_pred = model(batch_x)
            results_by_batch.append(batch_pred.detach().cpu().numpy())
            
    return np.concatenate(results_by_batch, 0)


# In[ ]:


def train_eval_loop(model, train_dataset, val_dataset, criterion,
                    lr=1e-4, epoch_n=5, batch_size=1024,
                    device=None, early_stopping_patience=10, l2_reg_alpha=0,
                    max_batches_per_epoch_train=10000,
                    max_batches_per_epoch_val=1000,
                    data_loader_ctor=DataLoader,
                    optimizer_ctor=None,
                    lr_scheduler_ctor=None,
                    shuffle_train=True,
                    dataloader_workers_n=1):
    
    """
    Useful sourse: https://github.com/Samsung-IT-Academy/stepik-dl-nlp/blob/master/dlnlputils
    """
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model.to(device)

    if optimizer_ctor is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg_alpha)
    else:
        optimizer = optimizer_ctor(model.parameters(), lr=lr)

    if lr_scheduler_ctor is not None:
        lr_scheduler = lr_scheduler_ctor(optimizer)
    else:
        lr_scheduler = None

    train_dataloader = data_loader_ctor(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                        num_workers=dataloader_workers_n)
    val_dataloader = data_loader_ctor(val_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=dataloader_workers_n)

    best_val_loss = float('inf')
    auc_valid = 0
    best_epoch_i = 0
    best_model = copy.deepcopy(model)

    for epoch_i in range(epoch_n):
        try:
            epoch_start = datetime.datetime.now()
            print('epoch {}'.format(epoch_i + 1))

            model.train()
            mean_train_loss = 0
            train_batches_n = 0
            for batch_i, (batch_y, batch_x) in enumerate(train_dataloader):
                if batch_i > max_batches_per_epoch_train:
                    break

                batch_x = copy_data_to_device(batch_x, device)
                batch_y = copy_data_to_device(batch_y, device)

                pred = model(batch_x)[:, 1]
                loss = criterion(pred, batch_y)

                model.zero_grad()
                loss.backward()

                optimizer.step()

                mean_train_loss += float(loss)
                train_batches_n += 1

            mean_train_loss /= train_batches_n
            print('epoch: {} iterations, {:0.2f} sec'.format(train_batches_n,
                                                           (datetime.datetime.now() - epoch_start).total_seconds()))
            print('mean loss on train: ', mean_train_loss)

            model.eval()
            mean_val_loss = 0
            val_batches_n = 0

            with torch.no_grad():
                for batch_i, (batch_y, batch_x) in enumerate(val_dataloader):
                    if batch_i > max_batches_per_epoch_val:
                        break

                    batch_x = copy_data_to_device(batch_x, device)
                    batch_y = copy_data_to_device(batch_y, device)

                    pred = model(batch_x)[:, 1] 
                    loss = criterion(pred, batch_y)

                    mean_val_loss += float(loss)
                    auc = roc_auc_score(batch_y.data.cpu().numpy(), pred.data.cpu().numpy())
                    val_batches_n += 1

            mean_val_loss /= val_batches_n
            print('mean loss on validation', mean_val_loss)
            print('valid batch auc --> %.5f'% auc)

            if mean_val_loss < best_val_loss:
                best_epoch_i = epoch_i
                best_val_loss = mean_val_loss
                best_model = copy.deepcopy(model)
                print('New best model!')
            elif epoch_i - best_epoch_i > early_stopping_patience:
                print('The model has not improved over the past {} epochs, stop training'.format(
                    early_stopping_patience))
                break

            if lr_scheduler is not None:
                lr_scheduler.step(mean_val_loss)

            print()
        except KeyboardInterrupt:
            print('Stopped by user')
            break
        except Exception as ex:
            print('Training error: {}\n{}'.format(ex, traceback.format_exc()))
            break

    return {"loss": best_val_loss,
            "model": best_model}


def seed_everything(seed=100):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything()
    

def copy_data_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [copy_data_to_device(elem, device) for elem in data]
    raise ValueError('Invalid data type {}'.format(type(data)))


# In[ ]:


train = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")
test = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")
subm = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/sample_submission.csv")


# In[ ]:


test.loc[:, "target"] = -1
data = pd.concat([train, test]).reset_index(drop=True)
print("dim data: ", data.shape)

features = data.columns.difference(["id", "target"]).tolist()
target = "target"


# In[ ]:


label_encoders = {}
for cat_col in features:
    label_encoders[cat_col] = LabelEncoder()
    data[cat_col] = label_encoders[cat_col].fit_transform(data[cat_col]                                                          .astype('category').cat.codes                                                          .fillna(-1).values)
    
data = reduce_mem_usage(data)


# In[ ]:


cat_dims = [int(data[col].nunique()) for col in features]
emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

print("cat_dims: ", cat_dims, ',\n')
print("emb_dims: ", emb_dims, ',')


# In[ ]:


train = data[data.target != -1].reset_index(drop=True)
test = data[data.target == -1].reset_index(drop=True)
train.shape, test.shape


# In[ ]:


train_df, valid_df = train_test_split(train, test_size=0.3, stratify=train.target)
print(train_df.shape, valid_df.shape)

train_dataset = CatDataset(data=train_df, cat_cols=features, output_col=target)
valid_dataset = CatDataset(data=valid_df, cat_cols=features, output_col=target)


# In[ ]:


model = GottaTorch(emb_dims, lin_layer_sizes=[300, 300],
                   output_size=2, emb_dropout=0.3,
                   lin_layer_dropouts=[0.3, 0.3])


# In[ ]:


BATCH_SIZE = 1024*2
LR = 9e-3
LOSS_FN = nn.BCELoss()
LR_SCHEDULER = lambda optim: torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3, 
                                                                     factor=0.5, verbose=True)

d = train_eval_loop(model, train_dataset, valid_dataset, lr=LR, criterion=LOSS_FN, 
                    lr_scheduler_ctor=LR_SCHEDULER, batch_size=BATCH_SIZE)


# In[ ]:


test_dataset = CatDataset(data=test, cat_cols=features, output_col=target)
res = predict_with_model(d["model"], test_dataset, batch_size=64*4, num_workers=0)


# In[ ]:


subm.target = res[:, 1]
subm.to_csv("submission.csv", index=False)


# # Feel free to continue experimenting...
