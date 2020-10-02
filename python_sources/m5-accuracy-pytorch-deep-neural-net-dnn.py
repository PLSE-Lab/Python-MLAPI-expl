#!/usr/bin/env python
# coding: utf-8

# # Pytorch DNN
# - A simple, timeseries agnostic deep neural network using pytorch framework. 
# - This kernel can be considered a pytorch port of this amazing [kernel](https://www.kaggle.com/mayer79/m5-forecast-keras-with-categorical-embeddings-v2) by [MichaelMayer](https://www.kaggle.com/mayer79) with minor changes.
# - Embedding categorical features using `nn.Embedding`.
# - The Neural net here has no time series based component which is a very bad idea, this is just started code.

# # Imports

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import gc
import os
from tqdm.notebook import tqdm
from sklearn.preprocessing import OrdinalEncoder

import torch
import pandas as pd
import numpy as np
import gc
import math
import datetime
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error

import os
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader,Dataset
from tqdm import tqdm_notebook as tqdm

device = torch.device('cuda')
# device = torch.device('cpu')

NUM_ITEMS = 30490
DAYS_PRED = 28


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()    


# # Make data

# In[ ]:


path = "../input/m5-forecasting-accuracy"
calendar = pd.read_csv(os.path.join(path, "calendar.csv"))
selling_prices = pd.read_csv(os.path.join(path, "sell_prices.csv"))
sample_submission = pd.read_csv(os.path.join(path, "sample_submission.csv"))
sales = pd.read_csv(os.path.join(path, "sales_train_validation.csv"))


# In[ ]:


def reduce_mem_usage(df, verbose=True):
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
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


def prep_calendar(df):
    df = df.drop(["date", "weekday"], axis=1)
    df = df.assign(d = df.d.str[2:].astype(int))
    df = df.fillna("missing")
    cols = list(set(df.columns) - {"wm_yr_wk", "d"})
    df[cols] = OrdinalEncoder(dtype="int").fit_transform(df[cols])
    df = reduce_mem_usage(df)
    return df

def prep_selling_prices(df):
    gr = df.groupby(["store_id", "item_id"])["sell_price"]
    df["sell_price_rel_diff"] = gr.pct_change()
    df["sell_price_roll_sd7"] = gr.transform(lambda x: x.rolling(7).std())
    df["sell_price_cumrel"] = (gr.shift(0) - gr.cummin()) / (1 + gr.cummax() - gr.cummin())
    df["price_unique"] = gr.transform('nunique')
    df = reduce_mem_usage(df)
    return df

def reshape_sales(df, drop_d = None):
    if drop_d is not None:
        df = df.drop(["d_" + str(i + 1) for i in range(drop_d)], axis=1)
    df = df.assign(id=df.id.str.replace("_validation", ""))
    df = df.reindex(columns=df.columns.tolist() + ["d_" + str(1913 + i + 1) for i in range(2 * 28)])
    df = df.melt(id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
                 var_name='d', value_name='demand')
    df = df.assign(d=df.d.str[2:].astype("int16"))
    return df

def prep_sales(df):
    df['lag_t28'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28))
    df['rolling_mean_t7'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())
    df['rolling_mean_t28'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(28).mean())
    df['rolling_mean_t56'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(56).mean())
    df['rolling_mean_t84'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(84).mean())
    df['rolling_mean_t168'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(168).mean())
    df['rolling_std_t7'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).std())
    df['rolling_std_t28'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(28).std())

    # Remove rows with NAs except for submission rows. rolling_mean_t168 was selected as it produces most missings
    df = df[(df.d >= 1914) | (pd.notna(df.rolling_mean_t168))]
    df = reduce_mem_usage(df)

    return df


# In[ ]:


calendar = prep_calendar(calendar)
selling_prices = prep_selling_prices(selling_prices)
sales = reshape_sales(sales, 1000)
sales = prep_sales(sales)

sales = sales.merge(calendar, how="left", on="d")
gc.collect()

sales = sales.merge(selling_prices, how="left", on=["wm_yr_wk", "store_id", "item_id"])
sales.drop(["wm_yr_wk"], axis=1, inplace=True)
gc.collect()
sales.head()

del selling_prices


# In[ ]:


sales.info()


# In[ ]:


cat_id_cols = ["item_id", "dept_id", "store_id", "cat_id", "state_id"]
cat_cols = cat_id_cols + ["wday", "month", "year", "event_name_1", 
                          "event_type_1", "event_name_2", "event_type_2"]

for i, v in enumerate(tqdm(cat_id_cols)):
    sales[v] = OrdinalEncoder(dtype="int").fit_transform(sales[[v]])

sales = reduce_mem_usage(sales)
sales.head()
gc.collect()


# In[ ]:


num_cols = ["sell_price", "sell_price_rel_diff", "sell_price_roll_sd7", "sell_price_cumrel",
            "lag_t28", "rolling_mean_t7", "rolling_mean_t28", "rolling_mean_t56", 
            "rolling_mean_t84", "rolling_mean_t168", "rolling_std_t7", "rolling_std_t28"]
bool_cols = ["snap_CA", "snap_TX", "snap_WI"]

dense_cols = num_cols + bool_cols

# Need to do column by column due to memory constraints
for i, v in enumerate(tqdm(num_cols)):
    sales[v] = sales[v].fillna(sales[v].median())
    
sales.head()


# In[ ]:


test = sales[sales.d >= 1914]
test = test.assign(id=test.id + "_" + np.where(test.d <= 1941, "validation", "evaluation"),
                   F="F" + (test.d - 1913 - 28 * (test.d > 1941)).astype("str"))
test.head()
gc.collect()


# In[ ]:


def make_X(df):
    X = {"dense1": df[dense_cols].to_numpy()}
    for i, v in enumerate(cat_cols):
        X[v] = df[[v]].to_numpy()
    return X

# Submission data
X_test = make_X(test)


total_train = sales[sales.d < 1914]
train_idx, val_idx = train_test_split(total_train.index, test_size=0.05, 
                                      random_state=42, shuffle=True)



valid = (make_X(total_train.iloc[val_idx]),
         total_train["demand"].iloc[val_idx])

X_train = make_X(total_train.iloc[train_idx])
y_train = total_train["demand"].iloc[train_idx]
                             
del sales, total_train
gc.collect()


# # Fast DataLoader
# I am not using the pytorch dataset and dataloader classes because they are very slow for large batch sizes of tabular data. Therefore, I have created a custom dataloader of my own that creates and return batches of shuffled data. 

# In[ ]:


class M5Loader:
    def __init__(self, X, y, shuffle=True, batch_size=10000, cat_cols=[]):
        self.X_cont = X["dense1"]
        self.X_cat = np.concatenate([X[k] for k in cat_cols], axis=1)
        self.y = y

        self.shuffle = shuffle
        self.batch_size =batch_size
        self.n_conts = self.X_cont.shape[1]
        self.len = self.X_cont.shape[0]
        n_batches, remainder = divmod(self.len, self.batch_size)
        
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
        self.remainder = remainder #for debugging
        
        self.idxes = np.array([i for i in range(self.len)])

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            ridxes = self.idxes
            np.random.shuffle(ridxes)
            self.X_cat = self.X_cat[[ridxes]]
            self.X_cont = self.X_cont[[ridxes]]
            if self.y is not None:
                self.y = self.y[[ridxes]]
                
        return self

    def __next__(self):
        if self.i  >= self.len:
            raise StopIteration
            
        if self.y is not None:
            y = torch.FloatTensor(self.y[self.i:self.i+self.batch_size].astype(np.float32))
        else:
            y = None
            
        xcont = torch.FloatTensor(self.X_cont[self.i:self.i+self.batch_size])
        xcat = torch.LongTensor(self.X_cat[self.i:self.i+self.batch_size])
        
        batch = (xcont, xcat, y)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches           


# In[ ]:


bs = 10000
shuffle = True
train_loader = M5Loader(X_train, y_train.values, cat_cols=cat_cols, batch_size=bs, shuffle=shuffle)
val_loader = M5Loader(valid[0], valid[1].values, cat_cols=cat_cols, batch_size=bs, shuffle=shuffle)


# In[ ]:


# Sorry for harcoding these but this saves RAM
uniques = [3049, 7, 10, 3, 3, 7, 12, 6, 31, 5, 5, 5]
dims = [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
emb_dims = [(x, y) for x, y in zip(uniques, dims)]
print(emb_dims)
n_cont = train_loader.n_conts


# # Model
# ## Loss functions

# In[ ]:


class ZeroBalance_RMSE(nn.Module):
    def __init__(self, penalty=1.12):
        super().__init__()
        self.penalty = penalty
        
    def forward(self, y_pred, y_true):
        y_pred = y_pred.squeeze()
        y_true = torch.FloatTensor(y).to(device)
        sq_error = torch.where(y_true==0, (y_true-y_pred)**2, self.penalty*(y_true-y_pred)**2)
        return torch.sqrt(torch.mean(sq_error))

class RMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, y_pred, y_true):
        y_pred = y_pred.squeeze()
        y_true = torch.FloatTensor(y).to(device)
        return torch.sqrt(self.mse(y_pred, y_true))    
    
class MSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, y_pred, y_true):
        y_pred = y_pred.squeeze()
        y_true = torch.FloatTensor(y).to(device)
        return self.mse(y_pred, y_true)  

def rmse_metric(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return np.sqrt(np.mean((y_pred-y_true)**2))     


# ## Architecture

# In[ ]:


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
        
class LinearBlock(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.block = nn.Sequential(
                            nn.Linear(inp, out),
#                             nn.LeakyReLU(),
                            nn.ReLU(),
#                             nn.Tanh(),
        )
    
    def forward(self, x):
        return self.block(x)

class M5Net(nn.Module):
    def __init__(self, emb_dims, n_cont, device=device):
        super().__init__()
        self.device = device

        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        n_embs = sum([y for x, y in emb_dims])
        
        self.n_embs = n_embs
        self.n_cont = n_cont
        inp_dim = n_embs + n_cont
        self.inp_dim = inp_dim
        
        hidden_dim = 200

        self.fn = nn.Sequential(
                 LinearBlock(inp_dim, hidden_dim),
                 LinearBlock(hidden_dim, hidden_dim//2),
                 LinearBlock(hidden_dim//2, hidden_dim//4),
                 LinearBlock(hidden_dim//4, hidden_dim//8),
        )          
        
        self.out = nn.Linear(hidden_dim//8, 1)

        self.fn.apply(init_weights)
        self.out.apply(init_weights)
        

    def encode_and_combine_data(self, cont_data, cat_data):
        xcat = [el(cat_data[:, k]) for k, el in enumerate(self.emb_layers)]
        xcat = torch.cat(xcat, 1)
        x = torch.cat([xcat, cont_data], 1)
        return x   
    
    def forward(self, cont_data, cat_data):
        cont_data = cont_data.to(self.device)
        cat_data = cat_data.to(self.device)
        inputs = self.encode_and_combine_data(cont_data, cat_data)
        x = self.fn(inputs)
        x = self.out(x)
        return x


# In[ ]:


model = M5Net(emb_dims, n_cont).to(device)


# In[ ]:


model


# # Train

# In[ ]:


epochs = 45
lr = 0.0002
criterion = ZeroBalance_RMSE()

torch.manual_seed(42)
model = M5Net(emb_dims, n_cont).to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, [20, 35, 40], gamma=0.5, 
                    last_epoch=-1)


# In[ ]:


train_losses = []
val_losses = []

for epoch in tqdm(range(epochs)):
    train_loss, val_loss = 0, 0

    #Training phase
    model.train()
    bar = tqdm(train_loader)
    
    for i, (X_cont, X_cat, y) in enumerate(bar):
        optimizer.zero_grad()
        out = model(X_cont, X_cat)
        loss = criterion(out, y)   
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            train_loss += loss.item()/len(train_loader)
            bar.set_description(f"{loss.item():.3f}")
    
    print(f"Running Train loss: {train_loss}")
    
    #Validation phase      
    with torch.no_grad():
        model.eval()
        for phase in ["train", "valid"]:
            rloss = 0
            if phase == "train":
                loader = train_loader
            else:
                loader = val_loader
            
            y_true = []
            y_pred = []
            
            for i, (X_cont, X_cat, y) in enumerate(loader):
                out = model(X_cont, X_cat)
                loss = criterion(out, y)
                rloss += loss.item()/len(loader)
                y_pred += list(out.detach().cpu().numpy().flatten())
                y_true += list(y.cpu().numpy())
                
            rrmse = rmse_metric(y_pred, y_true)    
            print(f"[{phase}] Epoch: {epoch} | Loss: {rloss:.4f} | RMSE: {rrmse:.4f}")
      
    train_losses.append(train_loss)    
    val_losses.append(rloss)        
    scheduler.step()


# In[ ]:


import pickle
def save(x, fname):
    with open(fname, "wb") as handle:
        pickle.dump(x, handle)
        
save(model, "model.pth")


# In[ ]:


plt.plot(train_losses)
plt.plot(val_losses)
plt.show()


# # Make Predictions

# In[ ]:


test_loader = M5Loader(X_test, y=None, cat_cols=cat_cols, batch_size=bs, shuffle=False)
pred = []
with torch.no_grad():
        model.eval()
        for i, (X_cont, X_cat, y) in enumerate(tqdm(test_loader)):
            out = model(X_cont, X_cat)
            pred += list(out.cpu().numpy().flatten())    
pred = np.array(pred)            


# In[ ]:


test["demand"] = pred.clip(0)
submission = test.pivot(index="id", columns="F", values="demand").reset_index()[sample_submission.columns]
submission = sample_submission[["id"]].merge(submission, how="left", on="id")
submission.head()


# In[ ]:


submission.to_csv("submission.csv", index=False)

