#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import copy
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm


# In[ ]:


os.listdir("../input")


# In[ ]:


main_data_dir = "../input/covid19-global-forecasting-week-4"
metadata_dir = "../input/covid19-countrywise-metadata"


# In[ ]:


data_train = pd.read_csv(os.path.join(main_data_dir, "train.csv"))
data_test = pd.read_csv(os.path.join(main_data_dir, "test.csv"))
sample_sub = pd.read_csv(os.path.join(main_data_dir, "submission.csv"))
metadata = pd.read_csv(os.path.join(metadata_dir, "covid19_countrywise_metadata.csv"))


# # Preprocess original data

# In[ ]:


def generate_loc(x):
    if isinstance(x["Province_State"], float):
        return x["Country_Region"]
    else:
        return "_".join([x["Country_Region"], x["Province_State"]])


# In[ ]:


def generate_loc_date(x):
    return "_".join([x["location"], x["Date"]])


# In[ ]:


data_train["location"] = data_train.apply(generate_loc, axis=1)
data_test["location"] = data_test.apply(generate_loc, axis=1)


# In[ ]:


data_train["log_cfm"] = data_train["ConfirmedCases"].map(np.log1p)
data_train["log_ftl"] = data_train["Fatalities"].map(np.log1p)


# In[ ]:


data_train["loc_date"] = data_train.apply(generate_loc_date, axis=1)
data_test["loc_date"] = data_test.apply(generate_loc_date, axis=1)


# In[ ]:


data_train_by_loc = data_train.groupby("location")


# In[ ]:


data_train["Date"] = pd.to_datetime(data_train["Date"])
data_test["Date"] = pd.to_datetime(data_test["Date"])


# In[ ]:


data_train["rel_date"] = data_train.apply(lambda x: (x["Date"] - data_train["Date"].min()), axis=1).dt.days
data_test["rel_date"] = data_test.apply(lambda x: (x["Date"] - data_train["Date"].min()), axis=1).dt.days


# In[ ]:


data_train["rel_date_pct"] = data_train.rel_date / data_test.rel_date.max()
data_test["rel_date_pct"] = data_test.rel_date / data_test.rel_date.max()


# In[ ]:


def get_day_zero(df):
    progress = pd.pivot_table(data_train[["location", "ConfirmedCases", "rel_date"]],
        values="ConfirmedCases", index="location", columns="rel_date")
    day_zero = np.argmax((progress.values > 0).cumsum(axis=1) == 1, axis=1)
    day_zero_location = {progress.index[i]: day_zero[i] for i in range(len(progress))}
    
    return day_zero_location


# In[ ]:


def get_since_day_zero(row, day_zero):
    cur_date = row["rel_date"]
    cur_loc = row["location"]
    cur_day_zero = day_zero[cur_loc]
    return max(cur_date - cur_day_zero, -1)


# In[ ]:


day_zero_location = get_day_zero(data_train)
data_train["since_day_zero"] = data_train.apply(get_since_day_zero, axis=1, day_zero=day_zero_location)
data_test["since_day_zero"] = data_test.apply(get_since_day_zero, axis=1, day_zero=day_zero_location)


# In[ ]:


data_train["since_day_zero"] = data_train["since_day_zero"].map(lambda x: -0.1 if x < 0 else x / data_test.since_day_zero.max())
data_test["since_day_zero"] = data_test["since_day_zero"].map(lambda x: -0.1 if x < 0 else x / data_test.since_day_zero.max())


# # Add lockdown data

# In[ ]:


lockdown_cols = ["quarantine", "close_school", "close_public_place", "limit_gathering", "stay_home"]
metadata[lockdown_cols] = metadata[lockdown_cols].fillna("2099-12-31")
for col in lockdown_cols:
    metadata[col] = pd.to_datetime(metadata[col], format="%Y-%m-%d")


# In[ ]:


def add_lockdown_data(df, metadata):
    lockdown_cols = ["quarantine", "close_school", "close_public_place", "limit_gathering", "stay_home"]
    lockdown_df = pd.DataFrame(np.zeros((len(df), len(lockdown_cols))), columns=lockdown_cols)
    for i in range(len(df)):
        cur_date = df.Date[i]
        cur_loc = df.location[i]
        idx = metadata.loc[metadata.location == cur_loc, :].index.values[0]
        lockdown_df.loc[i, :] = (metadata.loc[idx, lockdown_cols] <= cur_date).astype(int)
    
    return pd.concat([df, lockdown_df], axis=1)


# In[ ]:


data_train = add_lockdown_data(data_train, metadata)
data_test = add_lockdown_data(data_test, metadata)


# # Metadata preprocessing

# In[ ]:


metadata.drop(lockdown_cols, axis=1, inplace=True)


# In[ ]:


def min_max_normalize(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)


# In[ ]:


num_cols = metadata.columns[3:]
metadata[num_cols] = min_max_normalize(metadata[num_cols])


# In[ ]:


metadata.drop(["region", "country"], axis=1, inplace=True)


# In[ ]:


metadata.set_index("location", inplace=True)


# # Prepare test data

# In[ ]:


data_test["log_cfm"] = 0
data_test["log_ftl"] = 0


# In[ ]:


data_test_known = data_test.loc[data_test.rel_date <= data_train.rel_date.max(), :].sort_values(["rel_date", "location"]).reset_index(drop=True)
data_test_unknown = data_test.loc[data_test.rel_date > data_train.rel_date.max(), :].sort_values(["rel_date", "location"]).reset_index(drop=True)
data_train.sort_values(["rel_date", "location"], inplace=True)


# In[ ]:


data_test_known[["log_cfm", "log_ftl"]] = data_train.loc[data_train.rel_date >= data_test.rel_date.min(), ["log_cfm", "log_ftl"]].values


# In[ ]:


data_test_input = pd.concat([data_test_known, data_test_unknown], axis=0)
input_len = 30
data_test_input = pd.concat([data_train.loc[(data_train.rel_date >= data_test_unknown.rel_date.min() - input_len)    & (data_train.rel_date < data_test_known.rel_date.min()), data_test_input.columns],
    data_test_input], axis=0).sort_values(["rel_date", "location"]).reset_index(drop=True)


# In[ ]:


class TrainDataset(Dataset):
    
    def __init__(self, df, metadata, x_cols, y_cols, input_len=30):
        self.df = df
        self.metadata = metadata
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.input_len = input_len
        
        self.loc_list = list(self.df.location.unique())
        self.num_pos_train_period = self.df.rel_date.max() - self.input_len + 1
    
    def __len__(self):
        return len(self.loc_list) * self.num_pos_train_period
    
    def __getitem__(self, idx):
        cur_loc = self.loc_list[idx // self.num_pos_train_period]
        input_beg = idx % self.num_pos_train_period
        input_end = input_beg + self.input_len
        in_input_period = (self.df.rel_date >= input_beg) & (self.df.rel_date < input_end)
        is_cur_loc = self.df.location == cur_loc
        inputs = self.df.loc[(is_cur_loc) & (in_input_period), self.x_cols].values
        meta_inputs = self.metadata.loc[cur_loc, :].values
        targets = self.df.loc[(is_cur_loc) & (self.df.rel_date == input_end), self.y_cols].values.reshape((-1))
        
        return torch.tensor(inputs, dtype=torch.float32),            torch.tensor(meta_inputs, dtype=torch.float32),            torch.tensor(targets, dtype=torch.float32)
    
    @staticmethod
    def get_dataloader(dataset, batch_size, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# In[ ]:


rand_seed = 42
loc_list = data_train.location.unique()
loc_train, loc_val = train_test_split(loc_list, test_size=0.2, random_state=rand_seed)
data_train.set_index("location", inplace=True, drop=True)
data_val = data_train.loc[loc_val, :]
data_train = data_train.loc[loc_train, :]
data_train.reset_index(drop=False, inplace=True)
data_val.reset_index(drop=False, inplace=True)


# In[ ]:


x_cols = ["log_cfm", "log_ftl", "rel_date_pct", "since_day_zero"] + lockdown_cols
y_cols = ["log_cfm", "log_ftl"]
batch_size = 512
d_in = len(x_cols)
d_meta = len(metadata.columns)
ds_train = TrainDataset(data_train, metadata, x_cols, y_cols, input_len)
dl_train = TrainDataset.get_dataloader(ds_train, batch_size)
ds_val = TrainDataset(data_val, metadata, x_cols, y_cols, input_len)
dl_val = TrainDataset.get_dataloader(ds_val, batch_size, shuffle=False)


# In[ ]:


for _, sample in enumerate(dl_train):
    for x in sample:
        print(x.size())
    break


# In[ ]:


class TestDataset(Dataset):
    
    def __init__(self, df, metadata, x_cols, test_period_len, input_len=30):
        self.df = df
        self.metadata = metadata
        self.x_cols = x_cols
        self.test_period_len = test_period_len
        self.input_len = input_len
        
        self.loc_list = list(self.df.location.unique())
        self.test_beg = self.df.rel_date.max() - test_period_len + 1
    
    def __len__(self):
        return len(self.df) - self.input_len * len(self.loc_list)
    
    def __getitem__(self, idx):
        df_idx = idx + self.input_len * len(self.loc_list)
        cur_loc = self.df.location[df_idx]
        cur_date = self.df.rel_date[df_idx]
        input_beg = cur_date - self.input_len
        input_end = cur_date
        in_input_period = (self.df.rel_date >= input_beg) & (self.df.rel_date < input_end)
        inputs = self.df.loc[(in_input_period) & (self.df.location == cur_loc), self.x_cols].values
        inputs = np.expand_dims(inputs, axis=0)
        meta_inputs = self.metadata.loc[cur_loc, :].values
        meta_inputs = np.expand_dims(meta_inputs, axis=0)
        
        return torch.tensor(inputs, dtype=torch.float32),            torch.tensor(meta_inputs, dtype=torch.float32)
    
    @staticmethod
    def get_dataloader(dataset, shuffle=False):
        return DataLoader(dataset, batch_size=None, shuffle=shuffle)


# In[ ]:


test_period_len = data_test.rel_date.max() - data_test.rel_date.min() + 1
ds_test = TestDataset(data_test_input, metadata, x_cols, test_period_len)
dl_test = TestDataset.get_dataloader(ds_test, shuffle=False)


# In[ ]:


for i, sample in enumerate(dl_test):
    for x in sample:
        print(x.size())
    if i == 10:
        break


# In[ ]:


class CovidTransformer(nn.Module):
    
    def __init__(self, d_in, d_out, d_meta, d_model=256, d_fwd=512, n_head=4,
            num_layers=6, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(d_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_fwd, dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.meta_linear = nn.Linear(d_meta, d_model)
        self.regressor = nn.Linear(2 * d_model, d_out)
    
    def forward(self, inputs, meta_inputs):
        x = self.linear(inputs)
        features = self.transformer(x)
        features = features[-1, :, :]
        meta_features = self.meta_linear(meta_inputs)
        features = torch.cat([features, meta_features], dim=-1)
        output = self.regressor(features)
        
        return output


# In[ ]:


class RMSLE(nn.Module):
    
    def forward(self, output, target):
        return torch.sqrt(F.mse_loss(output, target))


# In[ ]:


class Engine(object):
    
    def compile(self, model, criterion, optimizer, scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def _init_history(self):
        self.history = {}
        self.history["loss_train"] = []
        self.history["loss_val"] = []
    
    def _update_history(self, train_loss, val_loss):
        self.history["loss_train"].append(train_loss)
        self.history["loss_val"].append(val_loss)
    
    def plot_loss(self):
        fig, ax = plt.subplots()
        ax.plot(self.history["loss_train"], label="train")
        ax.plot(self.history["loss_val"], label="val")
        plt.legend()
        plt.show()
    
    def _fit_epoch(self, dl_train):
        train_loss = 0.0
        for _, sample in enumerate(dl_train):
            inputs, meta_inputs, target = sample
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                meta_inputs = meta_inputs.cuda()
                target = target.cuda()
            inputs = torch.transpose(inputs, 1, 0)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            output = self.model(inputs, meta_inputs)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.detach().cpu().item()
        
        train_loss /= len(dl_train)
        return train_loss
    
    def evaluate(self, dl_val):
        val_loss = 0.0
        self.model.eval()
        
        with torch.no_grad():
            for _, sample in enumerate(dl_val):
                inputs, meta_inputs, target = sample
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    meta_inputs = meta_inputs.cuda()
                    target = target.cuda()
                inputs = torch.transpose(inputs, 1, 0)

                output = self.model(inputs, meta_inputs)

                loss = self.criterion(output, target)
                val_loss += loss.detach().cpu().item()
        
        val_loss /= len(dl_val)
        return val_loss
    
    def fit(self, epochs, dl_train, dl_val=None):
        self._init_history()
        progress = tqdm(total=epochs)
        
        for i in range(epochs):
            progress.set_description_str("Epoch {}/{}"
                .format(i + 1, epochs))
            
            train_loss = self._fit_epoch(dl_train)
            val_loss = self.evaluate(dl_val)
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            self._update_history(train_loss, val_loss)
            
            postfix = "RMSLE - train: {:.4f}, val: {:.4f}"                .format(train_loss, val_loss)
            progress.set_postfix_str(postfix)
            progress.update(1)
    
    def predict_batch(self, sample):
        self.model.eval()
        
        with torch.no_grad():
            inputs, meta_inputs = sample
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                meta_inputs = meta_inputs.cuda()
            inputs = torch.transpose(inputs, 1, 0)
            output = self.model(inputs, meta_inputs)
        
        return output
    
    def predict(self, dl):
        self.model.eval()
        output = []
        
        with torch.no_grad():
            for _, sample in enumerate(dl):
                inputs, meta_inputs, target = sample
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    meta_inputs = meta_inputs.cuda()
                    target = target.cuda()
                inputs = torch.transpose(inputs, 1, 0)

                output_batch = self.model(inputs, meta_inputs)
                output.append(output_batch)
        
        output = torch.cat(output, dim=0).expm1()
        
        return output


# In[ ]:


model = CovidTransformer(d_in, 2, d_meta)
if torch.cuda.is_available():
    model = model.cuda()
criterion = RMSLE()
optimizer = AdamW(model.parameters())
epochs = 20
scheduler = ExponentialLR(optimizer, 0.707)


# In[ ]:


engine = Engine()
engine.compile(model, criterion, optimizer, scheduler)


# In[ ]:


engine.fit(epochs, dl_train, dl_val)


# In[ ]:


engine.plot_loss()


# In[ ]:


def make_predictions(dl_test, engine, y_cols):
    progress = tqdm(total=len(dl_test))
    for i, sample in enumerate(dl_test):
        idx = i + dl_test.dataset.input_len * len(dl_test.dataset.loc_list)
        cur_loc = dl_test.dataset.df.location[idx]
        cur_date = dl_test.dataset.df.rel_date[idx]
        output = engine.predict_batch(sample).detach().cpu().numpy()
        prev_date = cur_date - 1
        prev_output = dl_test.dataset.df.loc[(dl_test.dataset.df.rel_date == prev_date)
            & (dl_test.dataset.df.location == cur_loc), y_cols].values
        output = np.maximum(output, prev_output)
        dl_test.dataset.df.loc[(dl_test.dataset.df.rel_date == cur_date)
            & (dl_test.dataset.df.location == cur_loc), y_cols] = output
        progress.update(1)
        
    return dl_test.dataset.df


# In[ ]:


predictions = make_predictions(dl_test, engine, y_cols)


# In[ ]:


predictions["ConfirmedCases"] = predictions["log_cfm"].map(np.expm1)
predictions["Fatalities"] = predictions["log_ftl"].map(np.expm1)


# In[ ]:


predictions = predictions.loc[~predictions.ForecastId.isna(), :]


# In[ ]:


predictions = predictions.sort_values(["ForecastId"]).reset_index(drop=True)


# In[ ]:


predictions


# In[ ]:


submission = predictions[["ForecastId", "ConfirmedCases", "Fatalities"]]
submission["ForecastId"] = submission["ForecastId"].astype(int)


# In[ ]:


submission.to_csv("submission.csv", index=False)

