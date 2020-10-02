#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm


# # Load raw data

# In[ ]:


os.listdir("../input")


# In[ ]:


main_data_dir = "../input/covid19-global-forecasting-week-3"
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
data_train["cfm_growth"] = data_train["log_cfm"].diff()
data_train["ftl_growth"] = data_train["log_ftl"].diff()


# In[ ]:


data_train.fillna(0, inplace=True)


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


# In[ ]:


data_test["rel_date"] = data_test.apply(lambda x: (x["Date"] - data_train["Date"].min()), axis=1).dt.days


# # Add lockdown data

# In[ ]:


lockdown_cols = ["quarantine", "close_school", "close_public_place", "limit_gathering", "stay_home"]
lockdown_df = pd.DataFrame(np.zeros((len(data_train), len(lockdown_cols))), columns=lockdown_cols)
metadata[lockdown_cols] = metadata[lockdown_cols].fillna("2099-12-31")
for col in lockdown_cols:
    metadata[col] = pd.to_datetime(metadata[col], format="%Y-%m-%d")


# In[ ]:


set(data_train.location.unique()) - set(metadata.location.unique())


# In[ ]:


for i in range(len(data_train)):
    cur_date = data_train.Date[i]
    cur_loc = data_train.location[i]
    idx = metadata.loc[metadata.location == cur_loc, :].index.values[0]
    lockdown_df.loc[i, :] = (metadata.loc[idx, lockdown_cols] <= cur_date).astype(int)


# In[ ]:


lockdown_df


# In[ ]:


data_train = pd.concat([data_train, lockdown_df], axis=1)


# In[ ]:


data_test_known = data_test.loc[data_test.rel_date <= data_train.rel_date.max(), :]
data_test_unknown = data_test.loc[data_test.rel_date > data_train.rel_date.max(), :]


# In[ ]:


data_train.set_index("loc_date", inplace=True, drop=True)
pred_test_known = data_train.loc[data_test_known.loc_date, ["ConfirmedCases", "Fatalities"]].reset_index(drop=False)
data_test_known = data_test_known.merge(pred_test_known, on="loc_date")
data_train.reset_index(inplace=True, drop=False)


# In[ ]:


train_input_beg = data_train.rel_date.min()
output_len = data_test.rel_date.max() - data_train.rel_date.max()
train_input_end = data_train.rel_date.max() - output_len
input_len = train_input_end - train_input_beg
test_input_beg = data_test.rel_date.max() - output_len - input_len
test_output_beg = data_test_unknown.rel_date.min()


# In[ ]:


print("Number of days in the input: {}d.".format(input_len))
print("Number of days in the output: {}d.".format(output_len))
print("Training input ends on day {}.".format(train_input_end))
print("Testing input begins on day {}.".format(test_input_beg))


# In[ ]:


input_test = data_train.loc[data_train.rel_date >= test_input_beg, :]


# In[ ]:


rand_seed = 42
loc_list = data_train.location.unique()
loc_train, loc_val = train_test_split(loc_list, test_size=0.2, random_state=rand_seed)
data_train.set_index("location", inplace=True, drop=True)
data_val = data_train.loc[loc_val, :]
data_train = data_train.loc[loc_train, :]
data_train.reset_index(drop=False, inplace=True)
data_val.reset_index(drop=False, inplace=True)


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


# In[ ]:


class CovidDataset(Dataset):
    
    def __init__(self, df, metadata, x_cols, input_end, input_beg=0, is_predict=False, random_clip=False):
        self.is_predict = is_predict
        self.metadata = metadata
        self.x_cols = x_cols
        self.random_clip = random_clip
        if not self.is_predict:
            self.df_input = df.loc[df.rel_date <= input_end, :]
            self.df_target = df.loc[df.rel_date > input_end, :]
            self.loc_list = df.location.unique()
        else:
            self.df_input = df.loc[df.rel_date >= input_beg, :]
            self.loc_list = df.location.unique()
        
    def __len__(self):
        return len(self.loc_list)
    
    def __getitem__(self, idx):
        loc = self.loc_list[idx]
        inputs = self.df_input.loc[self.df_input.location == loc, self.x_cols].values
#         if self.random_clip:
#             start = self.df_input.loc[(self.df_input.location == loc) &
#                 (self.df_input.ConfirmedCases > 0), "rel_date"].min()
#         else:
#             clip_start = np.random.randint(0, len(inputs) // 2)
#             inputs[:clip_start] = -9e6

        meta_input = self.metadata.loc[loc, :].values
        if not self.is_predict:
            data_target = self.df_target.loc[self.df_target.location == loc, :]
            cfm_target = data_target.log_cfm.values
            ftl_target = data_target.log_ftl.values
            target = np.concatenate([cfm_target, ftl_target], axis=0)
        
        if not self.is_predict:
            return torch.tensor(inputs, dtype=torch.float),                torch.tensor(meta_input, dtype=torch.float),                torch.tensor(target, dtype=torch.float)
        else:
            return torch.tensor(inputs, dtype=torch.float),                torch.tensor(meta_input, dtype=torch.float)
    
    @staticmethod
    def get_dataloader(dataset, batch_size):
        return DataLoader(dataset, batch_size, shuffle=not dataset.is_predict)


# In[ ]:


batch_size = 64
x_cols = ["log_cfm", "log_ftl"]
dataset_train = CovidDataset(data_train, metadata, x_cols, train_input_end, random_clip=True)
dataset_val = CovidDataset(data_val, metadata, x_cols, train_input_end)
dataset_test = CovidDataset(input_test, metadata, x_cols, 0, test_input_beg, True)
dl_train = CovidDataset.get_dataloader(dataset_train, batch_size)
dl_val = CovidDataset.get_dataloader(dataset_val, batch_size)
dl_test = CovidDataset.get_dataloader(dataset_test, batch_size)


# In[ ]:


class CovidTransformer(nn.Module):
    
    def __init__(self, d_in, d_out, d_model=128, d_fwd=512, n_head=4,
            num_layers=6, dropout=0.1, d_meta=13, num_mlp_layers=3, d_mlp=512):
        super().__init__()
        self.linear = nn.Linear(d_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_fwd, dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.meta_transform = nn.Linear(d_meta, d_model)
        mlp_components = []
        for i in range(num_mlp_layers):
            mlp_components.append(nn.Dropout(dropout))
            if i == 0:
                mlp_components.append(nn.Linear(2 * d_model, d_mlp))
            elif i == num_mlp_layers - 1:
                mlp_components.append(nn.Linear(d_mlp, d_out))
            else:
                mlp_components.append(nn.Linear(d_mlp, d_mlp))
            mlp_components.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_components)
    
    def forward(self, inputs, meta_features):
        x = self.linear(inputs)
        features = self.transformer(x)
        meta_features = self.meta_transform(meta_features)
        features = torch.cat([features.mean(dim=0), meta_features], dim=-1)
        output = self.mlp(features)
        
        return output


# In[ ]:


class RMSLE(nn.Module):
    
    def forward(self, output, target):
        return torch.sqrt(F.mse_loss(output, target))


# In[ ]:


class Engine(object):
    
    def compile(self, model, criterion, optimizer, lr_scheduler):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
    
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
            inputs, meta_input, target = sample
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                meta_input = meta_input.cuda()
                target = target.cuda()
            inputs = torch.transpose(inputs, 1, 0)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            output = self.model(inputs, meta_input)
            
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.detach().cpu().item()
        
        train_loss /= len(dl_train)
        self.lr_scheduler.step()
        return train_loss
    
    def evaluate(self, dl_val):
        val_loss = 0.0
        self.model.eval()
        
        with torch.no_grad():
            for _, sample in enumerate(dl_val):
                inputs, meta_input, target = sample
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    meta_input = meta_input.cuda()
                    target = target.cuda()
                inputs = torch.transpose(inputs, 1, 0)

                output = self.model(inputs, meta_input)

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
            
            self._update_history(train_loss, val_loss)
            
            postfix = "RMSLE - train: {:.4f}, val: {:.4f}"                .format(train_loss, val_loss)
            progress.set_postfix_str(postfix)
            progress.update(1)
    
    def predict(self, dl):
        self.model.eval()
        cfm_output = []
        ftl_output = []
        
        with torch.no_grad():
            for _, sample in enumerate(dl):
                inputs, meta_input = sample
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    meta_input = meta_input.cuda()
                inputs = torch.transpose(inputs, 1, 0)

                output_batch = self.model(inputs, meta_input)
                cfm_output.append(output_batch[:, :output_batch.size(1) // 2])
                ftl_output.append(output_batch[:, output_batch.size(1) // 2:])
        
        cfm_output = torch.cat(cfm_output, dim=0).expm1()
        ftl_output = torch.cat(ftl_output, dim=0).expm1()
        
        return cfm_output, ftl_output


# In[ ]:


model = CovidTransformer(2, 2 * output_len, dropout=0.2)
if torch.cuda.is_available():
    model = model.cuda()
criterion = RMSLE()
optimizer = AdamW(model.parameters())
lr_scheduler = ExponentialLR(optimizer, 0.95)


# In[ ]:


engine = Engine()
engine.compile(model, criterion, optimizer, lr_scheduler)


# In[ ]:


engine.fit(100, dl_train, dl_val)


# In[ ]:


engine.plot_loss()


# In[ ]:


cfm_pred, ftl_pred = engine.predict(dl_test)


# In[ ]:


def make_predictions(loc_list, date_pred, cfm_pred, ftl_pred):
    num_loc = len(loc_list)
    num_days = cfm_pred.size(1)
    cfm_pred = cfm_pred.numpy().reshape(-1)
    ftl_pred = ftl_pred.numpy().reshape(-1)
    location = [""] * len(date_pred)
    for i in range(num_loc):
        for j in range(num_days):
            location[i * num_days + j] = loc_list[i]
    pred_unknown = pd.DataFrame({
        "location": location,
        "Date": date_pred,
        "ConfirmedCases": cfm_pred,
        "Fatalities": ftl_pred
    })
    pred_unknown["Date"] = pred_unknown.Date.map(lambda x: x.strftime("%Y-%m-%d"))
    pred_unknown["loc_date"] = pred_unknown.apply(lambda x: "_".join([x["location"], x["Date"]]), axis=1)
    
    return pred_unknown[["loc_date", "ConfirmedCases", "Fatalities"]]


# In[ ]:


pred_unknown = make_predictions(dataset_test.loc_list, data_test_unknown.Date, cfm_pred, ftl_pred)


# In[ ]:


pred_unknown


# In[ ]:


data_test_unknown = data_test_unknown.merge(pred_unknown, on="loc_date")


# In[ ]:


data_test_known


# In[ ]:


data_test_unknown


# In[ ]:


submission = pd.concat([data_test_known, data_test_unknown], axis=0)[["ForecastId", "ConfirmedCases", "Fatalities"]]


# In[ ]:


submission.to_csv("submission.csv", index=False)


# In[ ]:




