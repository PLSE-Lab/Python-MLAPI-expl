#!/usr/bin/env python
# coding: utf-8

# # N-Beats basics (PyTorch)
# https://github.com/philipperemy/n-beats
# 
# https://arxiv.org/pdf/1905.10437.pdf (180 models ensembled in this paper with different backcast, loss and seed to beat M4 winner model)
# 
# Univariate Time Series (sales) predicted with N-Beats (forecast=28 days, backcast=3x28 days). Result is far away from simple LGB. Some model and hyperparameters tuning needs to be done. Feel free to fork and improve it.
# 
# - v1.0: 3 N-BEATS models with different backcast
# - v1.1: Add first sales and cleaning
# - v1.2: Adam/0.001, **LB=1.791 **,
#   stack_types=[NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK, NBeatsNet.GENERIC_BLOCK],
#   thetas_dims=[2, 8, 3],
#   nb_blocks_per_stack=3,
#   hidden_layer_units=1024,
#   share_weights_in_stack=False
# - v1.3: Adam/0.001, **LB=1.076 **,
#   stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
#   thetas_dims=[4, 4],
#   nb_blocks_per_stack=2,
#   hidden_layer_units=1024,
#   share_weights_in_stack=True
# - v1.4: Adam/0.001, **LB=1.119 **,
#   stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
#   thetas_dims=[4, 4, 4],
#   nb_blocks_per_stack=3,
#   hidden_layer_units=1024,
#   share_weights_in_stack=True  
# - v1.5: Adam/0.001, **LB= **,
#   stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
#   thetas_dims=[8, 8],
#   nb_blocks_per_stack=3,
#   hidden_layer_units=1024,
#   share_weights_in_stack=True    
#   
#   

# In[ ]:


import os, sys, random, gc, math, glob, time
import numpy as np
import pandas as pd
import io, timeit, os, gc, pickle, psutil
import joblib
from matplotlib import cm
from datetime import datetime, timedelta
import warnings
from tqdm.notebook import tqdm
from functools import partial
from collections import OrderedDict

# warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 4000)


# In[ ]:


seed = 2020
random.seed(seed)
np.random.seed(seed)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
DEFAULT_FIG_WIDTH = 20
sns.set_context("paper", font_scale=1.2) 


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit, KFold, GroupKFold, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict


print('Python    : ' + sys.version.split('\n')[0])
print('Numpy     : ' + np.__version__)
print('Pandas    : ' + pd.__version__)


# In[ ]:


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from torch import optim
from torch.optim import Adam, SGD
print('PyTorch        : ' + torch.__version__)


# In[ ]:


def seed_numpy_and_pytorch(s, sc):
    random.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s)
    # Torch
    torch.manual_seed(sc)
    torch.cuda.manual_seed(sc)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sc)

seed_numpy_and_pytorch(seed, seed)


# In[ ]:


HOME =  "./"
DATA_HOME = "/kaggle/input/m5-forecasting-accuracy/"
TRAIN_DATA_HOME = DATA_HOME

CALENDAR = DATA_HOME + "calendar.csv"
SALES = DATA_HOME + "sales_train_validation.csv"
PRICES = DATA_HOME + "sell_prices.csv"
SAMPLE_SUBMISSION = DATA_HOME + "sample_submission.csv"

MODELS_DIR = "models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
           


# Load data and remove days with no sales at the beginning.

# In[ ]:


train_sales_pd = pd.read_csv(SALES)
d_cols = [c for c in train_sales_pd.columns if 'd_' in c]
index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']

grid_df = pd.melt(frame = train_sales_pd, 
                 id_vars = index_columns,
                 var_name = 'd',
                 value_vars = d_cols,
                 value_name = "sales")

grid_df["sales"] = grid_df["sales"].astype(np.float32)


# In[ ]:


train_prices_pd = pd.read_csv(PRICES)
# No price = Nothing to sell, find initial release date. It allows removing real leading zeros.
release_df = train_prices_pd.groupby(['store_id','item_id'])['wm_yr_wk'].agg(['min']).reset_index()
release_df.columns = ['store_id','item_id','release']
# Now we can merge release_df
grid_df = pd.merge(left=grid_df, right=release_df, on=['store_id','item_id'], how='left')


# In[ ]:


train_calendar_pd = pd.read_csv(CALENDAR)
grid_df = pd.merge(left=grid_df, right=train_calendar_pd[['wm_yr_wk','d']], on=['d'], how="left")
# Now we can cutoff some rows and free memory 
grid_df = grid_df[grid_df['wm_yr_wk']>=grid_df['release']]
grid_df = grid_df.reset_index(drop=True)
grid_df.drop(columns=["release", "wm_yr_wk"], inplace=True)


# Go back to column per-day format.

# In[ ]:


grid_df["d"] = grid_df["d"].apply(lambda x: x[2:])
grid_df["d"] = grid_df["d"].astype(np.int16)
grid_df = grid_df.sort_values(["id", "d"])
full_train_pd = grid_df.pivot(index='id', columns='d', values='sales')


# In[ ]:


full_train_pd.head()


# Check how much missing data.

# In[ ]:


d = full_train_pd.isnull().sum().reset_index().plot(x="d")
d = plt.title("Nan by day")


# In[ ]:


del grid_df
gc.collect()


# In[ ]:


# Clean data: remove leading zeros and outliers
# def clean_data(df_train, day_cols, indx):
#     t = df_train.loc[indx].copy()
#     t.loc[day_cols[((t.loc[day_cols]>0).cumsum()==0).values]] = np.nan
#     q1 = t.loc[day_cols].quantile(0.25)
#     q3 = t.loc[day_cols].quantile(0.75)
#     iqr = q3-q1
#     qm = (q3+1.5*iqr)
#     t.loc[day_cols][t.loc[day_cols]>qm] = qm
#     return t


# For each time series, extract one day (backcast/forecast) randomly within full history. Fill in leading missing data (if any) with median. If no data available at all then assume it's zero.

# In[ ]:


def get_m5_data(backcast_length, forecast_length, df, is_training=True):
    x = np.array([]).reshape(0, backcast_length)
    y = np.array([]).reshape(0, forecast_length)
    x_tl_tl = df.values
    for i in range(x_tl_tl.shape[0]):
        if len(x_tl_tl[i]) < backcast_length + forecast_length:
            continue
        time_series = np.array(x_tl_tl[i])
        time_series_cleaned = time_series[~np.isnan(time_series)]
        if len(time_series_cleaned) < backcast_length + forecast_length:
            # Not enough data, fill missing value with median        
            median = np.nanmedian(time_series_cleaned) if len(time_series_cleaned) > 0 else 0.0
            missing_data = median * np.ones(((backcast_length + forecast_length)-time_series_cleaned.shape[0]))
            time_series_cleaned = np.concatenate([missing_data, time_series_cleaned])
        if is_training:
            time_series_cleaned_forlearning_x = np.zeros((1, backcast_length))
            time_series_cleaned_forlearning_y = np.zeros((1, forecast_length))
            # Random day within [140, 1886] i.e. 831 (if backcast = 140)
            j = np.random.randint(backcast_length, time_series_cleaned.shape[0] + 1 - forecast_length)
            # 140 backcast before random day
            time_series_cleaned_forlearning_x[0, :] = time_series_cleaned[j - backcast_length: j]
            # 28 forecast days after random day 691 <--> 831 <--> 859
            time_series_cleaned_forlearning_y[0, :] = time_series_cleaned[j:j + forecast_length]
        else:
            time_series_cleaned_forlearning_x = np.zeros(
                (time_series_cleaned.shape[0] + 1 - (backcast_length + forecast_length), backcast_length))
            time_series_cleaned_forlearning_y = np.zeros(
                (time_series_cleaned.shape[0] + 1 - (backcast_length + forecast_length), forecast_length))
            for j in range(backcast_length, time_series_cleaned.shape[0] + 1 - forecast_length):
                time_series_cleaned_forlearning_x[j - backcast_length, :] = time_series_cleaned[j - backcast_length:j]
                time_series_cleaned_forlearning_y[j - backcast_length, :] = time_series_cleaned[j: j + forecast_length]
        x = np.vstack((x, time_series_cleaned_forlearning_x))
        y = np.vstack((y, time_series_cleaned_forlearning_y))

    return x, y


# In[ ]:


class NBeatsNet(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'

    def __init__(self,
                 device,
                 stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
                 nb_blocks_per_stack=3,
                 forecast_length=5,
                 backcast_length=10,
                 thetas_dims=(4, 8),
                 share_weights_in_stack=False,
                 hidden_layer_units=256,
                 nb_harmonics=None):
        super(NBeatsNet, self).__init__()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics
        self.stack_types = stack_types
        self.stacks = []
        self.thetas_dim = thetas_dims
        self.parameters = []
        self.device = device
        print(f'| N-Beats')
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)
        self.to(self.device)

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        print(f'| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsNet.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                block = block_init(self.hidden_layer_units, self.thetas_dim[stack_id],
                                   self.device, self.backcast_length, self.forecast_length, self.nb_harmonics)
                self.parameters.extend(block.parameters())
            print(f'     | -- {block}')
            blocks.append(block)
        return blocks

    @staticmethod
    def select_block(block_type):
        if block_type == NBeatsNet.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == NBeatsNet.TREND_BLOCK:
            return TrendBlock
        else:
            return GenericBlock

    def forward(self, backcast):
        forecast = torch.zeros(size=(backcast.size()[0], self.forecast_length,))  # maybe batch size here.
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast.to(self.device) - b
                forecast = forecast.to(self.device) + f
        return backcast, forecast


def seasonality_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= thetas.shape[1], 'thetas_dim is too big.'
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor([np.cos(2 * np.pi * i * t) for i in range(p1)]).float()  # H/2-1
    s2 = torch.tensor([np.sin(2 * np.pi * i * t) for i in range(p2)]).float()
    S = torch.cat([s1, s2])
    return thetas.mm(S.to(device))


def trend_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= 4, 'thetas_dim is too big.'
    T = torch.tensor([t ** i for i in range(p)]).float()
    return thetas.mm(T.to(device))


def linspace(backcast_length, forecast_length):
    lin_space = np.linspace(-backcast_length, forecast_length, backcast_length + forecast_length)
    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]
    return b_ls, f_ls


class Block(nn.Module):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, share_thetas=False,
                 nb_harmonics=None):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        self.device = device
        self.backcast_linspace, self.forecast_linspace = linspace(backcast_length, forecast_length)
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x.to(self.device)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, '                f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, '                f'share_thetas={self.share_thetas}) at @{id(self)}'


class SeasonalityBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None):
        if nb_harmonics:
            super(SeasonalityBlock, self).__init__(units, nb_harmonics, device, backcast_length,
                                                   forecast_length, share_thetas=True)
        else:
            super(SeasonalityBlock, self).__init__(units, forecast_length, device, backcast_length,
                                                   forecast_length, share_thetas=True)

    def forward(self, x):
        x = super(SeasonalityBlock, self).forward(x)
        backcast = seasonality_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = seasonality_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        return backcast, forecast


class TrendBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None):
        super(TrendBlock, self).__init__(units, thetas_dim, device, backcast_length,
                                         forecast_length, share_thetas=True)

    def forward(self, x):
        x = super(TrendBlock, self).forward(x)
        backcast = trend_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = trend_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        return backcast, forecast


class GenericBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None):
        super(GenericBlock, self).__init__(units, thetas_dim, device, backcast_length, forecast_length)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        # no constraint for generic arch.
        x = super(GenericBlock, self).forward(x)

        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        return backcast, forecast


# In[ ]:


def split(arr, size):
    arrays = []
    while len(arr) > size:
        slice_ = arr[:size]
        arrays.append(slice_)
        arr = arr[size:]
    arrays.append(arr)
    return arrays


def batcher(dataset, batch_size, infinite=False):
    while True:
        x, y = dataset
        for x_, y_ in zip(split(x, batch_size), split(y, batch_size)):
            yield x_, y_
        if not infinite:
            break


# In[ ]:


def model_fit(net, optimiser, data_generator, on_save_callback, device, max_grad_steps=10000):
    train_loss = 0.0
    count = 0
    for grad_step, (x, target) in enumerate(data_generator):
        count = count + 1
        optimiser.zero_grad()
        net.train()
        backcast, forecast = net(torch.tensor(x, dtype=torch.float).to(device))
        loss = F.mse_loss(forecast, torch.tensor(target, dtype=torch.float).to(device))
        loss.backward()
        optimiser.step()
        train_loss = train_loss + loss.item()
        if grad_step > max_grad_steps:
            break
    return train_loss/count


def save(path, model, optimiser, grad_step):
    torch.save({
        #'grad_step': grad_step,
        'model_state_dict': model.state_dict(),
        #'optimizer_state_dict': optimiser.state_dict(),
    }, path)


def load(path, model, optimiser):
    if os.path.exists(path):
        grad_step = 0
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        #grad_step = checkpoint['grad_step']
        print(f'Restored checkpoint from {path}.')
        return grad_step
    return 0


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 24
batch_size = 3049 # 15245 # 3049 # 10 # 6098 # 15245


# In[ ]:


forecast_length = 28

MODEL_NAMES= {
    #"NNv5_cv1_tss_Adam_2011-2016-N-BEATS_Hx2": forecast_length*2,
    "NNv5_cv1_tss_Adam_2011-2016-N-BEATS_Hx3": forecast_length*3,
    #"NNv5_cv1_tss_Adam_2011-2016-N-BEATS_Hx4": forecast_length*4,
    #"NNv5_cv1_tss_Adam_2011-2016-N-BEATS_Hx5": forecast_length*5,
}
            
for name, backcast_length in MODEL_NAMES.items():
    
    MODEL_PATH = HOME + "models/" + name
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    
    BEST_NAME = MODEL_PATH + "/best.pth"
    
    print("Training:", MODEL_PATH)
    val_cols = [c for c in range(1914-(backcast_length+forecast_length), 1914)]
    val_pd = full_train_pd[[c for c in val_cols]]

    test_cols = [c for c in range(1914-(backcast_length), 1914)]
    test_pd = full_train_pd[[c for c in test_cols]]

    train_cols = [c for c in range(1, 1914-(backcast_length+forecast_length))]
    train_pd = full_train_pd[[c for c in train_cols]]     
    
    # Validation data
    X_val, y_val = get_m5_data(backcast_length, forecast_length, val_pd, is_training=False)

    net = NBeatsNet(device=device,
                    #stack_types=[NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK, NBeatsNet.GENERIC_BLOCK],
                    stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
                    forecast_length=forecast_length,
                    thetas_dims=[8, 8], # [2, 8, 3],
                    nb_blocks_per_stack=3, #3,
                    backcast_length=backcast_length,
                    hidden_layer_units=1024,
                    share_weights_in_stack=True, # False,
                    nb_harmonics=None)

    optimiser = optim.Adam(net.parameters(), lr=0.001)
    
    best_score = 99999
    total_steps = 0
    loops = 3
    history = []
    for epoch in range(1, EPOCHS + 1):
        print("Epoch:", epoch)
        # Pick random range for each time series within [1-1745] days available
        X_train, y_train = get_m5_data(backcast_length, forecast_length, train_pd, is_training=True)
        max_grad_steps = int(X_train.shape[0]/batch_size) * loops
        print("X_train:", X_train.shape, "y_train:", y_train.shape, "batches:", max_grad_steps)
        #print("NaN X_train:", len(X_train[np.isnan(X_train)]), "NaN y_train:", len(y_train[np.isnan(y_train)]))

        # Train all batches
        data_gen = batcher((X_train, y_train), batch_size=batch_size, infinite=True)    
        train_loss = model_fit(net, optimiser, data_gen, None, device, max_grad_steps)
        total_steps = total_steps + max_grad_steps

        # Evaluation
        net.eval()
        with torch.no_grad():
            _, f = net(torch.tensor(X_val, dtype=torch.float))
            predictions = f.cpu().numpy()
            if len(predictions[np.isnan(predictions)]) == 0:
                score = mean_squared_error(y_val, predictions) # np.sqrt
                print("Training loss: %.5f, Evaluation loss: %.5f" % (train_loss, score))
                history.append((epoch, train_loss, score))
                if score < best_score:
                    print("Saving model, score improvement from %.5f to %.5f" % (best_score, score))
                    best_score = score
                    save(BEST_NAME, net, optimiser, total_steps)
            else:
                print("Some NaN predictions:", len(predictions[np.isnan(predictions)]), "train_loss:", train_loss)
                
    history_pd = pd.DataFrame(history[1:], columns=["epoch", "train_loss", "val_loss"])
    fig, ax = plt.subplots(figsize=(24, 5))
    d = history_pd.plot(kind="line", x="epoch", ax=ax) 
    plt.show()
    
    # Load best and predict
    _ = load(BEST_NAME, net, optimiser)
    net.eval()
    with torch.no_grad():
        _, f = net(torch.tensor(test_pd.values, dtype=torch.float))
        predictions = f.cpu().numpy()
        
    # Clean and save
    predictions_pd = pd.DataFrame(predictions)
    predictions_pd.index = test_pd.index
    predictions_pd.columns = ["F%d" % c for c in range(1,29)]
    for c in predictions_pd.columns:
        predictions_pd[c] = np.where(predictions_pd[c] < 0, 0.0, predictions_pd[c])
    predictions_pd = predictions_pd.reset_index()
    # predictions_pd["id"] = predictions_pd["id"].astype(str) + "_validation"
    # display(predictions_pd.head())
    # print("NaN:", predictions_pd.isnull().sum().sum())
    predictions_pd.fillna(0, inplace=True)
    submission = pd.read_csv(SAMPLE_SUBMISSION)[['id']]
    submission = submission.merge(predictions_pd, on=['id'], how='left').fillna(0) # evaluation filled in with 0
    submission.to_csv(MODEL_PATH + "/submission.csv.gz", index=False, compression="gzip")


# In[ ]:


submission.head()

