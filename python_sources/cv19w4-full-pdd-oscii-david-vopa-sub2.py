#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import datetime
import numpy as np
import pandas as pd
import time
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from tqdm.notebook import tqdm

path = '../input/covid19-global-forecasting-week-4/'
train = pd.read_csv(path + 'train.csv')
test  = pd.read_csv(path + 'test.csv')
sub   = pd.read_csv(path + 'submission.csv')

train['Date'] = train['Date'].apply(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d')))
test['Date'] = test['Date'].apply(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d')))
#path_ext = '../input/novel-corona-virus-2019-dataset/'
#ext_rec = pd.read_csv(path_ext + 'time_series_covid_19_recovered.csv').\
#        melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], 
#            var_name="Date", 
#            value_name="Recoveries")
#ext_rec['Date'] = ext_rec['Date'].apply(lambda x: (datetime.datetime.strptime(x+"20", '%m/%d/%Y')))
#train = train.merge(ext_rec[['Province/State', 'Country/Region', 'Date', 'Recoveries']], how='left',
#           left_on=['Province/State', 'Country/Region', 'Date'],
#           right_on=['Province/State', 'Country/Region', 'Date'])

train['days'] = (train['Date'].dt.date - train['Date'].dt.date.min()).dt.days
test['days'] = (test['Date'].dt.date - train['Date'].dt.date.min()).dt.days
#train['isTest'] = train['Date'].dt.date >= datetime.date(2020, 3, 12)
#train['isVal'] = np.logical_and(train['Date'].dt.date >= datetime.date(2020, 3, 11), train['Date'].dt.date <= datetime.date(9999, 3, 18))
train.loc[train['Province_State'].isnull(), 'Province_State'] = 'N/A'
test.loc[test['Province_State'].isnull(), 'Province_State'] = 'N/A'

train['Area'] = train['Country_Region'] + '_' + train['Province_State']
test['Area'] = test['Country_Region'] + '_' + test['Province_State']

print('train Date min',train['Date'].min())
print('train Date max',train['Date'].max())
print('test Date min',test['Date'].min())
print('train days max', train['days'].max())
N_AREAS = train['Area'].nunique()
AREAS = np.sort(train['Area'].unique())
START_PUBLIC = test['days'].min()

print('public LB start day', START_PUBLIC)
print(' ')


TRAIN_N = 84
VAL_DAYS = train['days'].max() - TRAIN_N + 1
VAL_START_DATE = train[train['days'] >= TRAIN_N]['Date'].min() # need for davids model
print(train[train['days'] < TRAIN_N]['Date'].max())
print(train[train['days'] >= TRAIN_N]['Date'].min())
print(train[train['days'] >= TRAIN_N]['Date'].max())
train.head()

test_orig = test.copy()


# In[ ]:


VAL_DAYS


# In[ ]:


print('test Date max',test['days'].max())


# In[ ]:


train_p_c_raw = train.pivot(index='Area', columns='days', values='ConfirmedCases').sort_index()
train_p_f_raw = train.pivot(index='Area', columns='days', values='Fatalities').sort_index()

train_p_c = np.maximum.accumulate(train_p_c_raw, axis=1)
train_p_f = np.maximum.accumulate(train_p_f_raw, axis=1)

f_rate = (train_p_f / train_p_c).fillna(0)

X_c = np.log(1+train_p_c.values)[:,:TRAIN_N]
X_f = train_p_f.values[:,:TRAIN_N]


# In[ ]:


from sklearn.metrics import mean_squared_error

def eval1(y, p):
    val_len = y.shape[1] - TRAIN_N
    return np.sqrt(mean_squared_error(y[:, TRAIN_N:TRAIN_N+val_len].flatten(), p[:, TRAIN_N:TRAIN_N+val_len].flatten()))

def run_c(params, X, test_size=50):
    
    gr_base = []
    gr_base_factor = []
    
    x_min = np.ma.MaskedArray(X, X<1)
    x_min = x_min.argmin(axis=1) 
    
    for i in range(X.shape[0]):
        temp = X[i,:]
        threshold = np.log(1+params['min cases for growth rate'])
        num_days = params['last N days']
        if (temp > threshold).sum() > num_days:
            d = np.diff(temp[temp > threshold])[-num_days:]
            w = np.arange(len(d))+1
            w = w**5
            w = w / np.sum(w)
            gr_base.append(np.clip(np.average(d, weights=w), 0, params['growth rate max']))
            d2 = np.diff(d)
            w = np.arange(len(d2))+1
            w = w**10
            w = w / np.sum(w)
            gr_base_factor.append(np.clip(np.average(d2, weights=w), -0.5, params["growth rate factor max"]))
        else:
            gr_base.append(params['growth rate default'])
            gr_base_factor.append(params['growth rate factor'])

    gr_base = np.array(gr_base)
    gr_base_factor = np.array(gr_base_factor)
    #print(gr_base_factor)
    #gr_base = np.clip(gr_base, 0.02, 0.8)
    preds = X.copy()

    for i in range(test_size):
        delta = np.clip(preds[:, -1], np.log(2), None) + gr_base * (1 + params['growth rate factor']*(1 + params['growth rate factor factor'])**(i))**(np.log1p(i))
        #delta = np.clip(preds[:, -1], np.log(2), None) + gr_base * (1 + gr_base_factor*(1 + params['growth rate factor factor'])**(i))**(i)
        #delta = np.clip(preds[:, -1], np.log(2), None) + gr_base * (1 + params['growth rate factor']*(1 + params['growth rate factor factor'])**(i+X.shape[1]-x_min))**(i+X.shape[1]-x_min) 
        preds = np.hstack((preds, delta.reshape(-1,1)))

    return preds

params = {
    "min cases for growth rate": 0,
    "last N days": 20,
    "growth rate default": 0.10,
    "growth rate max": 0.3,
    "growth rate factor max": -0.1,
    "growth rate factor": -0.3,
    "growth rate factor factor": 0.02,
}
#x = train_p_c[train_p_c.index=="Austria_N/A"]

x = train_p_c

preds_c = run_c(params, np.log(1+x.values)[:,:TRAIN_N])
# eval1(np.log(1+x).values, preds_c)


# In[ ]:


for i in range(N_AREAS):
    if 'China' in AREAS[i] and preds_c[i, TRAIN_N-1] < np.log(31):
        preds_c[i, TRAIN_N:] = preds_c[i, TRAIN_N-1]


# In[ ]:


def run_f2(params, X, test_size=50):
    
    gr_base = []
    for i in range(X.shape[0]):
        temp = X[i,:]
        threshold = np.log(1.1+params['min cases for growth rate'])
        num_days = params['last N days']
        if (temp > threshold).sum() > num_days:
            d = np.diff(temp[temp > threshold])[-num_days:]
            gr_base.append(np.clip(np.mean(d), 0, params['growth rate max']))
        else:
            gr_base.append(params['growth rate default'])

    gr_base = np.array(gr_base)
    preds = X.copy()

    for i in range(test_size):
        #delta = np.clip(preds[:, -1], np.log(2), None) + gr_base * (1 + params['growth rate factor'])**(i)
        delta = preds[:, -1] + gr_base * (1 + params['growth rate factor'])**(i)
        preds = np.hstack((preds, delta.reshape(-1,1)))

    return preds

params = {
    "min cases for growth rate": 2,
    "last N days": 7,
    "growth rate default": 0.05,
    "growth rate max": 0.17,
    "growth rate factor": -0.18,
}

x = train_p_f

preds_f_1 = run_f2(params, np.log(1+x.values)[:,:TRAIN_N])
# eval1(np.log(1+x).values, preds_f_1)


# In[ ]:


from torch.utils.data import Dataset
from torch.nn import Parameter
import torch.nn as nn
from torch.nn import init
import math 
import torch
import time

class ZDatasetF(Dataset):
    def __init__(self, X_c, X_f=None, hist_len=10):
        self.X_c = X_c
        self.X_f = X_f
        self.hist_len = hist_len
        self.is_test = X_f is None
    def __len__(self):
        return self.X_c.shape[1]
    def __getitem__(self, idx):
        if self.is_test:
            return {'x_c':self.X_c[:, idx-self.hist_len:idx]}
        else:
            return {'x_c':self.X_c[:, idx-self.hist_len:idx],
                    'x_f':self.X_f[:, idx-1],
                    'y':np.log(1+self.X_f[:, idx])}

class PrLayer2(nn.Module):
    def __init__(self, in_features1, in_features2):
        super(PrLayer2, self).__init__()
        self.weight0 = Parameter(torch.Tensor(1, 1, in_features2))
        self.weight1 = Parameter(torch.Tensor(1, in_features1, in_features2))
        self.reset_parameters()
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight0, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
    def forward(self, input):
        return input * torch.sigmoid(self.weight0 + self.weight1)



class ZModelF(nn.Module):

    def __init__(self, hist_len):
        super(ZModelF, self).__init__()
        self.l_conv = PrLayer2(len(X_c),hist_len-1)

    def forward(self, x_c, x_f):
        x = x_c[:,:,1:] - x_c[:,:,:-1]
        res = torch.sum(self.l_conv(x), 2)
        return {'preds': torch.log(1 + x_f + res)}        
        

class DummySampler(torch.utils.data.sampler.Sampler):
    def __init__(self, idx):
        self.idx = idx
    def __iter__(self):
        return iter(self.idx)
    def __len__(self):
        return len(self.idx)
    
    
def _smooth_l1_loss(target):
    t = torch.abs(target)
    t = torch.where(t < 1, 0.5 * t ** 2, t - 0.5)
    return torch.mean(t)


n_epochs = 5000
lr = 0.18
bag_size = 4
device = 'cpu'
hist_len = 14
loss_func = torch.nn.MSELoss()
reg_loss_func = _smooth_l1_loss
reg_factor = 0.035


train_dataset = ZDatasetF(np.exp(X_c)-1, X_f, hist_len=hist_len)
test_dataset = ZDatasetF(np.exp(preds_c)-1, hist_len=hist_len)

#trn_idx = np.arange(hist_len+1, len(train_dataset))
trn_idx = np.arange(hist_len+1, len(train_dataset))
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(trn_idx)
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=0, pin_memory=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(trn_idx), sampler=train_sampler, num_workers=0, pin_memory=True)

test_idx = np.arange(TRAIN_N, len(test_dataset))
test_sampler = DummySampler(test_idx)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, sampler=test_sampler, num_workers=0, pin_memory=True)


#gradient_accumulation = len(trn_idx)
gradient_accumulation = 1

preds_f = 0

for m_i in range(bag_size):
    model_f = ZModelF(hist_len=hist_len).to(device)
    optimizer_f = torch.optim.Adam(model_f.parameters(), lr=lr)
    model_f.train()

    start_time = time.time()
    for epoch in range(n_epochs):

        s = time.time()
        avg_train_loss = 0
        
        optimizer_f.zero_grad()
        for idx, data in enumerate(train_loader):

            X1 = data['x_c'].to(device).float()
            X2 = data['x_f'].to(device).float()
            y = data['y'].to(device).float()
            
            preds = model_f(X1, X2)['preds'].float()

            cond = X2 > np.log(10)
            preds = preds[cond]
            y = y[cond]
            
            loss = loss_func(preds, y)
            
            loss += reg_factor * reg_loss_func(model_f.l_conv.weight1)
            
            avg_train_loss += loss  / len(train_loader)
            
            loss.backward()
            if (idx+1) % gradient_accumulation == 0 or idx == len(train_loader) - 1: 
                optimizer_f.step()
                optimizer_f.zero_grad()
                
        if epoch % 1000 == 0:
        
            model_f.eval()
            preds_f_delta = train_p_f.values[:,:TRAIN_N]

            for idx, data in enumerate(test_loader):
                X1 = data['x_c'].to(device).float()
                temp = model_f(X1, torch.Tensor(preds_f_delta[:,-1]).unsqueeze(0))['preds']
                temp = np.exp(temp.detach().cpu().numpy().reshape(-1,1)) - 1
                preds_f_delta = np.hstack((preds_f_delta, temp))

            preds_f_delta = np.log(1 + preds_f_delta)
#             val_len = train_p_c.values.shape[1] - TRAIN_N

#             m2 = np.sqrt(mean_squared_error(np.log(1 + train_p_f_raw.values[:, TRAIN_N:TRAIN_N+val_len]).flatten(), \
#                                             preds_f_delta[:, TRAIN_N:TRAIN_N+val_len].flatten()))
#             print(f"{epoch:2} train_loss {avg_train_loss:<8.4f} val_loss {m2:8.5f} {time.time()-s:<2.2f}")
                
            model_f.train()
        
    model_f.eval()
    preds_f_delta = train_p_f.values[:,:TRAIN_N]
    
    for idx, data in enumerate(test_loader):
        X1 = data['x_c'].to(device).float()
        temp = model_f(X1, torch.Tensor(preds_f_delta[:,-1]).unsqueeze(0))['preds']
        temp = np.exp(temp.detach().cpu().numpy().reshape(-1,1)) - 1
        preds_f_delta = np.hstack((preds_f_delta, temp))
    preds_f += preds_f_delta / bag_size

preds_f_2 = np.log(1 + preds_f)

print("Done")
# eval1(np.log(1+train_p_f).values, preds_f_2)


# In[ ]:


preds_f_2.shape


# In[ ]:


preds_f = np.average([preds_f_1, preds_f_2], axis=0, weights=[2,1])


# In[ ]:


from sklearn.metrics import mean_squared_error

if True:
    #val_len = train_p_c.values.shape[1] - TRAIN_N
    val_len = TRAIN_N - START_PUBLIC
    for i in range(val_len):
        d = i + START_PUBLIC
        m1 = np.sqrt(mean_squared_error(np.log(1 + train_p_c_raw.values[:, d]), preds_c[:, d]))
        m2 = np.sqrt(mean_squared_error(np.log(1 + train_p_f_raw.values[:, d]), preds_f[:, d]))
        print(f"{d}: {(m1 + m2)/2:8.5f} [{m1:8.5f} {m2:8.5f}]")

    print()

    m1 = np.sqrt(mean_squared_error(np.log(1 + train_p_c_raw.values[:, START_PUBLIC:START_PUBLIC+val_len]).flatten(), preds_c[:, START_PUBLIC:START_PUBLIC+val_len].flatten()))
    m2 = np.sqrt(mean_squared_error(np.log(1 + train_p_f_raw.values[:, START_PUBLIC:START_PUBLIC+val_len]).flatten(), preds_f[:, START_PUBLIC:START_PUBLIC+val_len].flatten()))
    print(f"{(m1 + m2)/2:8.5f} [{m1:8.5f} {m2:8.5f}]")


# ## vopani model

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


## defining constants
PATH_TRAIN = "/kaggle/input/covid19-global-forecasting-week-4/train.csv"
PATH_TEST = "/kaggle/input/covid19-global-forecasting-week-4/test.csv"

PATH_SUBMISSION = "submission.csv"
PATH_OUTPUT = "output.csv"

PATH_REGION_METADATA = "/kaggle/input/covid19-forecasting-metadata/region_metadata.csv"
PATH_REGION_DATE_METADATA = "/kaggle/input/covid19-forecasting-metadata/region_date_metadata.csv"

# VAL_DAYS = 5 
MAD_FACTOR = 0.5
DAYS_SINCE_CASES = [1, 10, 50, 100, 500, 1000, 5000, 10000]

SEED = 2357

LGB_PARAMS = {"objective": "regression",
              "num_leaves": 5,
              "learning_rate": 0.013,
              "bagging_fraction": 0.91,
              "feature_fraction": 0.81,
              "reg_alpha": 0.13,
              "reg_lambda": 0.13,
              "metric": "rmse",
              "seed": SEED
             }

## reading data
train = pd.read_csv(PATH_TRAIN)
test = pd.read_csv(PATH_TEST)

region_metadata = pd.read_csv(PATH_REGION_METADATA)
region_date_metadata = pd.read_csv(PATH_REGION_DATE_METADATA)

## preparing data
train = train.merge(test[["ForecastId", "Province_State", "Country_Region", "Date"]], on = ["Province_State", "Country_Region", "Date"], how = "left")
test = test[~test.Date.isin(train.Date.unique())]

df_panel = pd.concat([train, test], sort = False)

# combining state and country into 'geography'
df_panel["geography"] = (df_panel.Country_Region.astype(str) + ": " + df_panel.Province_State.astype(str)).values
df_panel.loc[df_panel.Province_State.isna(), "geography"] = df_panel[df_panel.Province_State.isna()].Country_Region

# fixing data issues with cummax
df_panel.ConfirmedCases = df_panel.groupby("geography")["ConfirmedCases"].cummax()
df_panel.Fatalities = df_panel.groupby("geography")["Fatalities"].cummax()

# merging external metadata
#print(df_panel.geography)

df_panel = df_panel.merge(region_metadata, on = ["Country_Region", "Province_State"], how="left")

df_panel = df_panel.merge(region_date_metadata, on = ["Country_Region", "Province_State", "Date"], how = "left")
df_panel.loc[df_panel.continent.isna(), "continent"] = ""
#label encoding continent
df_panel.continent = LabelEncoder().fit_transform(df_panel.continent)
df_panel.Date = pd.to_datetime(df_panel.Date, format = "%Y-%m-%d")

df_panel.loc[df_panel['Province_State'].isnull(), 'Province_State'] = 'N/A'
df_panel["geography"] = (df_panel.Country_Region.astype(str) + ": " + df_panel.Province_State.astype(str)).values


df_panel.sort_values(["geography", "Date"], inplace = True)

## feature engineering
min_date_train = np.min(df_panel[~df_panel.Id.isna()].Date)
max_date_train = np.max(df_panel[~df_panel.Id.isna()].Date)

min_date_test = np.min(df_panel[~df_panel.ForecastId.isna()].Date)
max_date_test = np.max(df_panel[~df_panel.ForecastId.isna()].Date)

n_dates_test = len(df_panel[~df_panel.ForecastId.isna()].Date.unique())

print("Train date range:", str(min_date_train), " - ", str(max_date_train))
print("Test date range:", str(min_date_test), " - ", str(max_date_test))

# creating lag features
for lag in range(1, 41):
    df_panel[f"lag_{lag}_cc"] = df_panel.groupby("geography")["ConfirmedCases"].shift(lag)
    df_panel[f"lag_{lag}_ft"] = df_panel.groupby("geography")["Fatalities"].shift(lag)
    df_panel[f"lag_{lag}_rc"] = df_panel.groupby("geography")["Recoveries"].shift(lag)

for case in DAYS_SINCE_CASES:
    df_panel = df_panel.merge(df_panel[df_panel.ConfirmedCases >= case].groupby("geography")["Date"].min().reset_index().rename(columns = {"Date": f"case_{case}_date"}), on = "geography", how = "left")


# In[ ]:


def prepare_features(df, gap):
    
    df["perc_1_ac"] = (df[f"lag_{gap}_cc"] - df[f"lag_{gap}_ft"] - df[f"lag_{gap}_rc"]) / df[f"lag_{gap}_cc"]
    df["perc_1_cc"] = df[f"lag_{gap}_cc"] / df.population
    
    df["diff_1_cc"] = df[f"lag_{gap}_cc"] - df[f"lag_{gap + 1}_cc"]
    df["diff_2_cc"] = df[f"lag_{gap + 1}_cc"] - df[f"lag_{gap + 2}_cc"]
    df["diff_3_cc"] = df[f"lag_{gap + 2}_cc"] - df[f"lag_{gap + 3}_cc"]
    
    df["diff_1_ft"] = df[f"lag_{gap}_ft"] - df[f"lag_{gap + 1}_ft"]
    df["diff_2_ft"] = df[f"lag_{gap + 1}_ft"] - df[f"lag_{gap + 2}_ft"]
    df["diff_3_ft"] = df[f"lag_{gap + 2}_ft"] - df[f"lag_{gap + 3}_ft"]
    
    df["diff_123_cc"] = (df[f"lag_{gap}_cc"] - df[f"lag_{gap + 3}_cc"]) / 3
    df["diff_123_ft"] = (df[f"lag_{gap}_ft"] - df[f"lag_{gap + 3}_ft"]) / 3

    df["diff_change_1_cc"] = df.diff_1_cc / df.diff_2_cc
    df["diff_change_2_cc"] = df.diff_2_cc / df.diff_3_cc
    
    df["diff_change_1_ft"] = df.diff_1_ft / df.diff_2_ft
    df["diff_change_2_ft"] = df.diff_2_ft / df.diff_3_ft

    df["diff_change_12_cc"] = (df.diff_change_1_cc + df.diff_change_2_cc) / 2
    df["diff_change_12_ft"] = (df.diff_change_1_ft + df.diff_change_2_ft) / 2
    
    df["change_1_cc"] = df[f"lag_{gap}_cc"] / df[f"lag_{gap + 1}_cc"]
    df["change_2_cc"] = df[f"lag_{gap + 1}_cc"] / df[f"lag_{gap + 2}_cc"]
    df["change_3_cc"] = df[f"lag_{gap + 2}_cc"] / df[f"lag_{gap + 3}_cc"]

    df["change_1_ft"] = df[f"lag_{gap}_ft"] / df[f"lag_{gap + 1}_ft"]
    df["change_2_ft"] = df[f"lag_{gap + 1}_ft"] / df[f"lag_{gap + 2}_ft"]
    df["change_3_ft"] = df[f"lag_{gap + 2}_ft"] / df[f"lag_{gap + 3}_ft"]

    df["change_1_3_cc"] = df[f"lag_{gap}_cc"] / df[f"lag_{gap + 3}_cc"]
    df["change_1_3_ft"] = df[f"lag_{gap}_ft"] / df[f"lag_{gap + 3}_ft"]
    
    df["change_1_7_cc"] = df[f"lag_{gap}_cc"] / df[f"lag_{gap + 7}_cc"]
    df["change_1_7_ft"] = df[f"lag_{gap}_ft"] / df[f"lag_{gap + 7}_ft"]
    
    for case in DAYS_SINCE_CASES:
        df[f"days_since_{case}_case"] = (df[f"case_{case}_date"] - df.Date).astype("timedelta64[D]")
        df.loc[df[f"days_since_{case}_case"] < gap, f"days_since_{case}_case"] = np.nan

    df["country_flag"] = df.Province_State.isna().astype(int)

    # target variable is log of change from last known value
    df["target_cc"] = np.log1p(df.ConfirmedCases - df[f"lag_{gap}_cc"])
    df["target_ft"] = np.log1p(df.Fatalities - df[f"lag_{gap}_ft"])
    
    features = [
        f"lag_{gap}_cc",
        f"lag_{gap}_ft",
        f"lag_{gap}_rc",
        "perc_1_ac",
        "perc_1_cc",
        "diff_1_cc",
        "diff_2_cc",
        "diff_3_cc",
        "diff_1_ft",
        "diff_2_ft",
        "diff_3_ft",
        "diff_123_cc",
        "diff_123_ft",
        "diff_change_1_cc",
        "diff_change_2_cc",
        "diff_change_1_ft",
        "diff_change_2_ft",
        "diff_change_12_cc",
        "diff_change_12_ft",
        "change_1_cc",
        "change_2_cc",
        "change_3_cc",
        "change_1_ft",
        "change_2_ft",
        "change_3_ft",
        "change_1_3_cc",
        "change_1_3_ft",
        "change_1_7_cc",
        "change_1_7_ft",
        "days_since_1_case",
        "days_since_10_case",
        "days_since_50_case",
        "days_since_100_case",
        "days_since_500_case",
        "days_since_1000_case",
        "days_since_5000_case",
        "days_since_10000_case",
        "country_flag",
        "lat",
        "lon",
        "continent",
        "population",
        "area",
        "density",
        "target_cc",
        "target_ft"
    ]
    
    return df[features]


# In[ ]:


gap, VAL_DAYS


# In[ ]:


## function for building and predicting using LGBM model
def build_predict_lgbm(df_train, df_test, gap):
    
    df_train.dropna(subset = ["target_cc", "target_ft", f"lag_{gap}_cc", f"lag_{gap}_ft"], inplace = True)
    
    target_cc = df_train.target_cc
    target_ft = df_train.target_ft
    
    test_lag_cc = df_test[f"lag_{gap}_cc"].values
    test_lag_ft = df_test[f"lag_{gap}_ft"].values
    
    df_train.drop(["target_cc", "target_ft"], axis = 1, inplace = True)
    df_test.drop(["target_cc", "target_ft"], axis = 1, inplace = True)
    
    categorical_features = ["continent"]
    
    dtrain_cc = lgb.Dataset(df_train, label = target_cc, categorical_feature = categorical_features)
    dtrain_ft = lgb.Dataset(df_train, label = target_ft, categorical_feature = categorical_features)

    model_cc = lgb.train(LGB_PARAMS, train_set = dtrain_cc, num_boost_round = 200)
    model_ft = lgb.train(LGB_PARAMS, train_set = dtrain_ft, num_boost_round = 200)
    
    # inverse transform from log of change from last known value
    y_pred_cc = np.expm1(model_cc.predict(df_test, num_boost_round = 200)) + test_lag_cc
    y_pred_ft = np.expm1(model_ft.predict(df_test, num_boost_round = 200)) + test_lag_ft
    
    return y_pred_cc, y_pred_ft, model_cc, model_ft



## function for predicting moving average decay model
def predict_mad(df_test, gap, val = False):
    
    df_test["avg_diff_cc"] = (df_test[f"lag_{gap}_cc"] - df_test[f"lag_{gap + 3}_cc"]) / 3
    df_test["avg_diff_ft"] = (df_test[f"lag_{gap}_ft"] - df_test[f"lag_{gap + 3}_ft"]) / 3

    if val:
        y_pred_cc = df_test[f"lag_{gap}_cc"] + gap * df_test.avg_diff_cc - (1 - MAD_FACTOR) * df_test.avg_diff_cc * np.sum([x for x in range(gap)]) / VAL_DAYS
        y_pred_ft = df_test[f"lag_{gap}_ft"] + gap * df_test.avg_diff_ft - (1 - MAD_FACTOR) * df_test.avg_diff_ft * np.sum([x for x in range(gap)]) / VAL_DAYS
    else:
        y_pred_cc = df_test[f"lag_{gap}_cc"] + gap * df_test.avg_diff_cc - (1 - MAD_FACTOR) * df_test.avg_diff_cc * np.sum([x for x in range(gap)]) / n_dates_test
        y_pred_ft = df_test[f"lag_{gap}_ft"] + gap * df_test.avg_diff_ft - (1 - MAD_FACTOR) * df_test.avg_diff_ft * np.sum([x for x in range(gap)]) / n_dates_test

    return y_pred_cc, y_pred_ft

## building lag x-days models
df_train = df_panel[~df_panel.Id.isna()]
df_test_full = df_panel[~df_panel.ForecastId.isna()]

df_preds_val = []
df_preds_test = []

for date in df_test_full.Date.unique():
    
    print("Processing date:", date)
    
    # ignore date already present in train data
    if date in df_train.Date.values:
        df_pred_test = df_test_full.loc[df_test_full.Date == date, ["ForecastId", "ConfirmedCases", "Fatalities"]].rename(columns = {"ConfirmedCases": "ConfirmedCases_test", "Fatalities": "Fatalities_test"})
        
        # multiplying predictions by 41 to not look cool on public LB
        #df_pred_test.ConfirmedCases_test = df_pred_test.ConfirmedCases_test * 41
        #df_pred_test.Fatalities_test = df_pred_test.Fatalities_test * 41
    else:
        df_test = df_test_full[df_test_full.Date == date]
        
        gap = (pd.Timestamp(date) - max_date_train).days
        
        if gap <= VAL_DAYS:
            val_date = max_date_train - pd.Timedelta(VAL_DAYS, "D") + pd.Timedelta(gap, "D")

            df_build = df_train[df_train.Date < val_date]
            df_val = df_train[df_train.Date == val_date]
            
            X_build = prepare_features(df_build, gap)
            X_val = prepare_features(df_val, gap)
            
            y_val_cc_lgb, y_val_ft_lgb, _, _ = build_predict_lgbm(X_build, X_val, gap)
            y_val_cc_mad, y_val_ft_mad = predict_mad(df_val, gap, val = True)
            
            df_pred_val = pd.DataFrame({"Id": df_val.Id.values,
                                        "ConfirmedCases_val_lgb": y_val_cc_lgb,
                                        "Fatalities_val_lgb": y_val_ft_lgb,
                                        "ConfirmedCases_val_mad": y_val_cc_mad,
                                        "Fatalities_val_mad": y_val_ft_mad,
                                       })

            df_preds_val.append(df_pred_val)

        X_train = prepare_features(df_train, gap)
        X_test = prepare_features(df_test, gap)

        y_test_cc_lgb, y_test_ft_lgb, model_cc, model_ft = build_predict_lgbm(X_train, X_test, gap)
        y_test_cc_mad, y_test_ft_mad = predict_mad(df_test, gap)
        
        if gap == 1:
            model_1_cc = model_cc
            model_1_ft = model_ft
            features_1 = X_train.columns.values
        elif gap == 14:
            model_14_cc = model_cc
            model_14_ft = model_ft
            features_14 = X_train.columns.values
        elif gap == 28:
            model_28_cc = model_cc
            model_28_ft = model_ft
            features_28 = X_train.columns.values

        df_pred_test = pd.DataFrame({"ForecastId": df_test.ForecastId.values,
                                     "ConfirmedCases_test_lgb": y_test_cc_lgb,
                                     "Fatalities_test_lgb": y_test_ft_lgb,
                                     "ConfirmedCases_test_mad": y_test_cc_mad,
                                     "Fatalities_test_mad": y_test_ft_mad,
                                    })
    
    df_preds_test.append(df_pred_test)


# In[ ]:


## validation score
# df_panel = df_panel.merge(pd.concat(df_preds_val, sort = False), on = "Id", how = "left")
df_panel = df_panel.merge(pd.concat(df_preds_test, sort = False), on = "ForecastId", how = "left")

# rmsle_cc_lgb = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel.ConfirmedCases_val_lgb.isna()].ConfirmedCases), np.log1p(df_panel[~df_panel.ConfirmedCases_val_lgb.isna()].ConfirmedCases_val_lgb)))
# rmsle_ft_lgb = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel.Fatalities_val_lgb.isna()].Fatalities), np.log1p(df_panel[~df_panel.Fatalities_val_lgb.isna()].Fatalities_val_lgb)))

# rmsle_cc_mad = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel.ConfirmedCases_val_mad.isna()].ConfirmedCases), np.log1p(df_panel[~df_panel.ConfirmedCases_val_mad.isna()].ConfirmedCases_val_mad)))
# rmsle_ft_mad = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel.Fatalities_val_mad.isna()].Fatalities), np.log1p(df_panel[~df_panel.Fatalities_val_mad.isna()].Fatalities_val_mad)))

# print("LGB CC RMSLE Val of", VAL_DAYS, "days for CC:", round(rmsle_cc_lgb, 2))
# print("LGB FT RMSLE Val of", VAL_DAYS, "days for FT:", round(rmsle_ft_lgb, 2))
# print("LGB Overall RMSLE Val of", VAL_DAYS, "days:", round((rmsle_cc_lgb + rmsle_ft_lgb) / 2, 2))
# print("\n")
# print("MAD CC RMSLE Val of", VAL_DAYS, "days for CC:", round(rmsle_cc_mad, 2))
# print("MAD FT RMSLE Val of", VAL_DAYS, "days for FT:", round(rmsle_ft_mad, 2))
# print("MAD Overall RMSLE Val of", VAL_DAYS, "days:", round((rmsle_cc_mad + rmsle_ft_mad) / 2, 2))

# df_panel.loc[~df_panel.ConfirmedCases_val_lgb.isna(), "ConfirmedCasesPredLGB"] = df_panel[~df_panel.ConfirmedCases_val_lgb.isna()].ConfirmedCases_val_lgb
# df_panel.loc[~df_panel.ConfirmedCases_val_mad.isna(), "ConfirmedCasesPredMAD"] = df_panel[~df_panel.ConfirmedCases_val_mad.isna()].ConfirmedCases_val_mad

df_panel.loc[~df_panel.ConfirmedCases_test_lgb.isna(), "ConfirmedCasesPredLGB"] = df_panel[~df_panel.ConfirmedCases_test_lgb.isna()].ConfirmedCases_test_lgb
df_panel.loc[~df_panel.ConfirmedCases_test_mad.isna(), "ConfirmedCasesPredMAD"] = df_panel[~df_panel.ConfirmedCases_test_mad.isna()].ConfirmedCases_test_mad

# df_panel.loc[~df_panel.Fatalities_val_lgb.isna(), "FatalitiesPredLGB"] = df_panel[~df_panel.Fatalities_val_lgb.isna()].Fatalities_val_lgb
# df_panel.loc[~df_panel.Fatalities_val_mad.isna(), "FatalitiesPredMAD"] = df_panel[~df_panel.Fatalities_val_mad.isna()].Fatalities_val_mad

df_panel.loc[~df_panel.Fatalities_test_lgb.isna(), "FatalitiesPredLGB"] = df_panel[~df_panel.Fatalities_test_lgb.isna()].Fatalities_test_lgb
df_panel.loc[~df_panel.Fatalities_test_mad.isna(), "FatalitiesPredMAD"] = df_panel[~df_panel.Fatalities_test_mad.isna()].Fatalities_test_mad


# In[ ]:


p_f_vopani_lgb = np.log1p(df_panel.pivot(index='geography', columns='Date', values='FatalitiesPredLGB').sort_index().fillna(0).values)
p_c_vopani_lgb = np.log1p(df_panel.pivot(index='geography', columns='Date', values='ConfirmedCasesPredLGB').sort_index().fillna(0).values)

p_f_vopani_mad = np.log1p(df_panel.pivot(index='geography', columns='Date', values='FatalitiesPredMAD').sort_index().fillna(0).values)
p_c_vopani_mad = np.log1p(df_panel.pivot(index='geography', columns='Date', values='ConfirmedCasesPredMAD').sort_index().fillna(0).values)

p_f_vopani_lgb = np.maximum.accumulate(p_f_vopani_lgb, axis=1)
p_c_vopani_lgb = np.maximum.accumulate(p_c_vopani_lgb, axis=1)

p_f_vopani = 0.2*p_f_vopani_lgb + 0.8*p_f_vopani_mad
p_c_vopani = 0.2*p_c_vopani_lgb + 0.8*p_c_vopani_mad


# In[ ]:


p_c_vopani.shape, p_f_vopani.shape


# In[ ]:


if True:
    #val_len = train_p_c.values.shape[1] - TRAIN_N
    val_len = TRAIN_N - START_PUBLIC
    for i in range(val_len):
        d = i + START_PUBLIC
        m1 = np.sqrt(mean_squared_error(np.log(1 + train_p_c_raw.values[:, d]), p_c_vopani[:, d]))
        m2 = np.sqrt(mean_squared_error(np.log(1 + train_p_f_raw.values[:, d]), p_f_vopani[:, d]))
        print(f"{d}: {(m1 + m2)/2:8.5f} [{m1:8.5f} {m2:8.5f}]")

    print()

    m1 = np.sqrt(mean_squared_error(np.log(1 + train_p_c_raw.values[:, START_PUBLIC:START_PUBLIC+val_len]).flatten(), p_c_vopani[:, START_PUBLIC:START_PUBLIC+val_len].flatten()))
    m2 = np.sqrt(mean_squared_error(np.log(1 + train_p_f_raw.values[:, START_PUBLIC:START_PUBLIC+val_len]).flatten(), p_f_vopani[:, START_PUBLIC:START_PUBLIC+val_len].flatten()))
    print(f"{(m1 + m2)/2:8.5f} [{m1:8.5f} {m2:8.5f}]")


# ## David model 

# In[ ]:



VAL_START_DATE


# In[ ]:


import pandas as pd
import numpy as np
import os
from collections import Counter
from random import shuffle
import math
from scipy.stats.mstats import gmean
import datetime
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import seaborn as sns


# In[ ]:


pd.options.display.float_format = '{:.8}'.format
plt.rcParams["figure.figsize"] = (12, 4.75)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.options.display.max_rows = 999


# In[ ]:


# Parameters - can be changed
BAGS = 3
SEED = 123
SET_FRAC = 0.01
TRUNCATED = False
DROPS = True
PRIVATE = True

USE_PRIORS = False
SUP_DROP = 0.0
ACTIONS_DROP = 0.0
PLACE_FRACTION = 1.0  # 0.4 
#** FEATURE_DROP = 0.4 # drop random % of features (HIGH!!!, speeds it up)
#** COUNTRY_DROP = 0.35 # drop random % of countries (20-30pct)
#** FIRST_DATE_DROP = 0.5 # Date_f must be after a certain date, randomly applied
# FEATURE_DROP_MAX = 0.3
LT_DECAY_MAX = 0.3
LT_DECAY_MIN = -0.4
SINGLE_MODEL = False
MODEL_Y = 'agg_dff' # 'slope'  # 'slope' or anything else for difference/aggregate log gain


# In[ ]:


path = '/kaggle/input/c19week3v2/'
input_path = '/kaggle/input/covid19-global-forecasting-week-4/'
train = pd.read_csv(input_path + 'train.csv')
test = pd.read_csv(input_path  + 'test.csv')
sub = pd.read_csv(input_path + 'submission.csv')
tt = pd.merge(train, test, on=['Country_Region', 'Province_State', 'Date'], how='right', validate="1:1").fillna(method = 'ffill')
public = tt[['ForecastId', 'ConfirmedCases', 'Fatalities']]
test_dates = test.Date.unique()
pp = 'public'
if PRIVATE:
    test = test[ pd.to_datetime(test.Date) >  train.Date.max()]
    pp = 'private'
train.Date = pd.to_datetime(train.Date)


# In[ ]:


revised = pd.read_csv(path + 'outside_data/covid19_train_data_us_states_before_march_09_new.csv')
revised = revised[['Province_State', 'Country_Region', 'Date', 'ConfirmedCases', 'Fatalities']]
revised.Date = pd.to_datetime(revised.Date)
rev_train = pd.merge(train, revised, on=['Province_State', 'Country_Region', 'Date'],suffixes = ('', '_r'), how='left')
rev_train[~rev_train.ConfirmedCases_r.isnull()].head()
rev_train.ConfirmedCases = np.where( (rev_train.ConfirmedCases == 0) & ((rev_train.ConfirmedCases_r > 0 )) & (rev_train.Country_Region == 'US'), rev_train.ConfirmedCases_r,rev_train.ConfirmedCases)
rev_train.Fatalities =np.where( ~rev_train.Fatalities_r.isnull() & (rev_train.Fatalities == 0) & ((rev_train.Fatalities_r > 0 )) & (rev_train.Country_Region == 'US') ,rev_train.Fatalities_r,rev_train.Fatalities)
rev_train.drop(columns = ['ConfirmedCases_r', 'Fatalities_r'], inplace=True)
train = rev_train


# In[ ]:


contain_data = pd.read_csv(path + 'outside_data/OxCGRT_Download_070420_160027_Full.csv')
delcols = ['_Notes','Unnamed', 'Confirmed','CountryCode','S8', 'S9', 'S10','S11', 'StringencyIndexForDisplay']
contain_data = contain_data[[c for c in contain_data.columns if not any(z in c for z in delcols)] ]
contain_data.rename(columns = {'CountryName': "Country"}, inplace=True)
contain_data.Date = contain_data.Date.astype(str).apply(datetime.datetime.strptime, args=('%Y%m%d', ))
contain_data_orig = contain_data.copy()
cds = []
for country in contain_data.Country.unique():
    cd = contain_data[contain_data.Country==country]
    cd = cd.fillna(method = 'ffill').fillna(0)
    cd.StringencyIndex = cd.StringencyIndex.cummax()  # for now
    col_count = cd.shape[1]
    
    # now do a diff columns
    # and ewms of it
    for col in [c for c in contain_data.columns if 'S' in c]:
        col_diff = cd[col].diff()
        cd[col+"_chg_5d_ewm"] = col_diff.ewm(span = 5).mean()
        cd[col+"_chg_20_ewm"] = col_diff.ewm(span = 20).mean()
        
    # stringency
    cd['StringencyIndex_5d_ewm'] = cd.StringencyIndex.ewm(span = 5).mean()
    cd['StringencyIndex_20d_ewm'] = cd.StringencyIndex.ewm(span = 20).mean()
    
    cd['S_data_days'] =  (cd.Date - cd.Date.min()).dt.days
    for s in [1, 10, 20, 30, 50, ]:
        cd['days_since_Stringency_{}'.format(s)] =                 np.clip((cd.Date - cd[(cd.StringencyIndex > s)].Date.min()).dt.days, 0, None)
    
    
    cds.append(cd.fillna(0)[['Country', 'Date'] + cd.columns.to_list()[col_count:]])
contain_data = pd.concat(cds)
contain_data.Country.replace({ 'United States': "US",'South Korea': "Korea, South",'Taiwan': "Taiwan*",'Myanmar': "Burma", 'Slovak Republic': "Slovakia",'Czech Republic': 'Czechia'}, inplace=True)

sup_data = pd.read_excel(path + 'outside_data/Data Join - Copy1.xlsx')
sup_data.columns = [c.replace(' ', '_') for c in sup_data.columns.to_list()]
sup_data.drop(columns = [c for c in sup_data.columns.to_list() if 'Unnamed:' in c], inplace=True)
sup_data.drop(columns = [ 'Date', 'ConfirmedCases','Fatalities', 'log-cases', 'log-fatalities', 'continent'], inplace=True)
sup_data['Migrants_in'] = np.clip(sup_data.Migrants, 0, None)
sup_data['Migrants_out'] = -np.clip(sup_data.Migrants, None, 0)
sup_data.drop(columns = 'Migrants', inplace=True)


# In[ ]:


train.Date = pd.to_datetime(train.Date)
test.Date = pd.to_datetime(test.Date)
train.rename(columns={'Country_Region': 'Country'}, inplace=True)
test.rename(columns={'Country_Region': 'Country'}, inplace=True)
sup_data.rename(columns={'Country_Region': 'Country'}, inplace=True)

# train['Place'] = train.Country + train.Province_State.fillna("")
# test['Place'] = test.Country +  test.Province_State.fillna("")
# sup_data['Place'] = sup_data.Country +  sup_data.Province_State.fillna("")

train.loc[train['Province_State'].isnull(), 'Province_State'] = 'N/A'
test.loc[test['Province_State'].isnull(), 'Province_State'] = 'N/A'
sup_data.loc[sup_data['Province_State'].isnull(), 'Province_State'] = 'N/A'

train['Place'] = train['Country'] + '_' + train['Province_State']
test['Place'] = test['Country'] + '_' + test['Province_State']
sup_data['Place'] = sup_data['Country'] + '_' + sup_data['Province_State']



sup_data = sup_data[sup_data.columns.to_list()[2:]]
sup_data = sup_data.replace('N.A.', np.nan).fillna(-0.5)
for c in sup_data.columns[:-1]:
    m = sup_data[c].max() #- sup_data 
    
    if m > 300 and c!='TRUE_POPULATION':
        print(c)
        sup_data[c] = np.log(sup_data[c] + 1)
        assert sup_data[c].min() > -1

for c in sup_data.columns[:-1]:
    m = sup_data[c].max() #- sup_data 
    
    if m > 300:
        print(c)


# In[ ]:


DEATHS = 'Fatalities'
train.ConfirmedCases =     np.where(
        (train.ConfirmedCases.shift(1) > train.ConfirmedCases) & 
        (train.ConfirmedCases.shift(1) > 0) & (train.ConfirmedCases.shift(-1) > 0) &
         (train.Place == train.Place.shift(1)) & (train.Place == train.Place.shift(-1)) & 
        ~train.ConfirmedCases.shift(-1).isnull(),
        
        np.sqrt(train.ConfirmedCases.shift(1) * train.ConfirmedCases.shift(-1)),
        
        train.ConfirmedCases)

train.Fatalities =     np.where(
        (train.Fatalities.shift(1) > train.Fatalities) & 
        (train.Fatalities.shift(1) > 0) & (train.Fatalities.shift(-1) > 0) &
         (train.Place == train.Place.shift(1)) & (train.Place == train.Place.shift(-1)) & 
        ~train.Fatalities.shift(-1).isnull(),
        
        np.sqrt(train.Fatalities.shift(1) * train.Fatalities.shift(-1)),
        
        train.Fatalities)

for i in [0, -1]:
    train.ConfirmedCases =         np.where(
            (train.ConfirmedCases.shift(2+ i ) > train.ConfirmedCases) & 
            (train.ConfirmedCases.shift(2+ i) > 0) & (train.ConfirmedCases.shift(-1+ i) > 0) &
         (train.Place == train.Place.shift(2+ i)) & (train.Place == train.Place.shift(-1+ i)) & 
            ~train.ConfirmedCases.shift(-1+ i).isnull(),

            np.sqrt(train.ConfirmedCases.shift(2+ i) * train.ConfirmedCases.shift(-1+ i)),train.ConfirmedCases)

train_bk = train.copy()
full_train = train.copy()
train_c = train[train.Country == 'China']
train_nc = train[train.Country != 'China']
train_us = train[train.Country == 'US']


# In[ ]:


def lplot(data, minDate = datetime.datetime(2000, 1, 1), columns = ['ConfirmedCases', 'Fatalities']):
    return
        
REAL = datetime.datetime(2020, 2, 10)
dataset = train.copy()

if TRUNCATED:
    dataset = dataset[dataset.Country.isin(['Italy', 'Spain', 'Germany', 'Portugal', 'Belgium', 'Austria', 'Switzerland' ])]

dataset.head()


# In[ ]:


def rollDates(df, i, preserve=False):
    df = df.copy()
    if preserve:
        df['Date_i'] = df.Date
    df.Date = df.Date + datetime.timedelta(i)
    return df

WINDOWS = [1, 2,  4, 7, 12, 20, 30]
for window in WINDOWS:
    csuffix = '_{}d_prior_value'.format(window)
    
    base = rollDates(dataset, window)
    dataset = pd.merge(dataset, base[['Date', 'Place',
                'ConfirmedCases', 'Fatalities']], on = ['Date', 'Place'],
            suffixes = ('', csuffix), how='left')
    for c in ['ConfirmedCases', 'Fatalities']:
        dataset[c+ csuffix].fillna(0, inplace=True)
        dataset[c+ csuffix] = np.log(dataset[c + csuffix] + 1)
        dataset[c+ '_{}d_prior_slope'.format(window)] =                     (np.log(dataset[c] + 1)                          - dataset[c+ csuffix]) / window
        dataset[c+ '_{}d_ago_zero'.format(window)] = 1.0*(dataset[c+ csuffix] == 0)     
    
for window1 in WINDOWS:
    for window2 in WINDOWS:
        for c in ['ConfirmedCases', 'Fatalities']:
            if window1 * 1.3 < window2 and window1 * 5 > window2:
                dataset[ c +'_{}d_{}d_prior_slope_chg'.format(window1, window2) ] =                         dataset[c+ '_{}d_prior_slope'.format(window1)]                                 - dataset[c+ '_{}d_prior_slope'.format(window2)]
                
                


# In[ ]:


print('start features')

first_case = dataset[dataset.ConfirmedCases >= 1].groupby('Place').min() 
tenth_case = dataset[dataset.ConfirmedCases >= 10].groupby('Place').min()
hundredth_case = dataset[dataset.ConfirmedCases >= 100].groupby('Place').min()
thousandth_case = dataset[dataset.ConfirmedCases >= 1000].groupby('Place').min()

# %% [code]
first_fatality = dataset[dataset.Fatalities >= 1].groupby('Place').min()
tenth_fatality = dataset[dataset.Fatalities >= 10].groupby('Place').min()
hundredth_fatality = dataset[dataset.Fatalities >= 100].groupby('Place').min()
thousandth_fatality = dataset[dataset.Fatalities >= 1000].groupby('Place').min()


dataset['days_since_first_case'] = np.clip((dataset.Date - first_case.loc[dataset.Place].Date.values).dt.days.fillna(-1), -1, None)
dataset['days_since_tenth_case'] = np.clip((dataset.Date - tenth_case.loc[dataset.Place].Date.values).dt.days.fillna(-1), -1, None)
dataset['days_since_hundredth_case'] = np.clip((dataset.Date - hundredth_case.loc[dataset.Place].Date.values).dt.days.fillna(-1), -1, None)
dataset['days_since_thousandth_case'] = np.clip((dataset.Date - thousandth_case.loc[dataset.Place].Date.values).dt.days.fillna(-1), -1, None)

dataset['days_since_first_fatality'] = np.clip((dataset.Date - first_fatality.loc[dataset.Place].Date.values).dt.days.fillna(-1), -1, None)
dataset['days_since_tenth_fatality'] = np.clip((dataset.Date - tenth_fatality.loc[dataset.Place].Date.values).dt.days.fillna(-1), -1, None)
dataset['days_since_hundredth_fatality'] = np.clip((dataset.Date - hundredth_fatality.loc[dataset.Place].Date.values).dt.days.fillna(-1), -1, None)
dataset['days_since_thousandth_fatality'] = np.clip((dataset.Date - thousandth_fatality.loc[dataset.Place].Date.values).dt.days.fillna(-1), -1, None)

dataset['case_rate_since_first_case'] = np.clip((np.log(dataset.ConfirmedCases + 1) - np.log(first_case.loc[dataset.Place].ConfirmedCases.fillna(0).values + 1))/ (dataset.days_since_first_case+0.01), 0, 1)
dataset['case_rate_since_tenth_case'] = np.clip((np.log(dataset.ConfirmedCases + 1) - np.log(tenth_case.loc[dataset.Place].ConfirmedCases.fillna(0).values + 1))/ (dataset.days_since_tenth_case+0.01), 0, 1)
dataset['case_rate_since_hundredth_case'] = np.clip((np.log(dataset.ConfirmedCases + 1) - np.log(hundredth_case.loc[dataset.Place].ConfirmedCases.fillna(0).values + 1)) / (dataset.days_since_first_case+0.01), 0, 1)
dataset['case_rate_since_thousandth_case'] = np.clip((np.log(dataset.ConfirmedCases + 1) - np.log(thousandth_case.loc[dataset.Place].ConfirmedCases.fillna(0).values + 1)) / (dataset.days_since_first_case+0.01), 0, 1)

dataset['fatality_rate_since_first_case'] =np.clip((np.log(dataset.Fatalities + 1) - np.log(first_case.loc[dataset.Place].Fatalities.fillna(0).values + 1)) / (dataset.days_since_first_case+0.01), 0, 1)
dataset['fatality_rate_since_tenth_case'] = np.clip((np.log(dataset.Fatalities + 1) - np.log(tenth_case.loc[dataset.Place].Fatalities.fillna(0).values + 1)) / (dataset.days_since_first_case+0.01), 0, 1)
dataset['fatality_rate_since_hundredth_case'] = np.clip((np.log(dataset.Fatalities + 1) - np.log(hundredth_case.loc[dataset.Place].Fatalities.fillna(0).values + 1)) / (dataset.days_since_first_case+0.01), 0, 1)
dataset['fatality_rate_since_thousandth_case'] = np.clip((np.log(dataset.Fatalities + 1) - np.log(thousandth_case.loc[dataset.Place].Fatalities.fillna(0).values + 1)) / (dataset.days_since_first_case+0.01), 0, 1)

dataset['fatality_rate_since_first_fatality'] = np.clip((np.log(dataset.Fatalities + 1) - np.log(first_fatality.loc[dataset.Place].Fatalities.fillna(0).values + 1)) / (dataset.days_since_first_fatality+0.01), 0, 1)
dataset['fatality_rate_since_tenth_fatality'] = np.clip((np.log(dataset.Fatalities + 1) - np.log(tenth_fatality.loc[dataset.Place].Fatalities.fillna(0).values + 1)) / (dataset.days_since_tenth_fatality+0.01), 0, 1)
dataset['fatality_rate_since_hundredth_fatality'] = np.clip((np.log(dataset.Fatalities + 1) - np.log(hundredth_fatality.loc[dataset.Place].Fatalities.fillna(0).values + 1)) / (dataset.days_since_hundredth_fatality+0.01), 0, 1)
dataset['fatality_rate_since_thousandth_fatality'] = np.clip((np.log(dataset.Fatalities + 1) - np.log(thousandth_fatality.loc[dataset.Place].Fatalities.fillna(0).values + 1)) / (dataset.days_since_thousandth_fatality+0.01), 0, 1)
 
dataset['first_case_ConfirmedCases'] = np.log(first_case.loc[dataset.Place].ConfirmedCases.values + 1)
dataset['first_case_Fatalities'] = np.log(first_case.loc[dataset.Place].Fatalities.values + 1)

dataset['first_fatality_ConfirmedCases'] = np.log(first_fatality.loc[dataset.Place].ConfirmedCases.fillna(0).values + 1) * (dataset.days_since_first_fatality >= 0 )
dataset['first_fatality_Fatalities'] =np.log(first_fatality.loc[dataset.Place].Fatalities.fillna(0).values + 1) * (dataset.days_since_first_fatality >= 0 )

dataset['first_fatality_cfr'] =np.where(dataset.days_since_first_fatality < 0,-8,(dataset.first_fatality_Fatalities) -(dataset.first_fatality_ConfirmedCases ))
dataset['first_fatality_lag_vs_first_case'] = np.where(dataset.days_since_first_fatality >= 0,dataset.days_since_first_case - dataset.days_since_first_fatality , -1)

dataset['case_chg'] = np.clip(np.log(dataset.ConfirmedCases + 1 )- np.log(dataset.ConfirmedCases.shift(1) +1), 0, None).fillna(0)
dataset['case_chg_ema_3d'] = dataset.case_chg.ewm(span = 3).mean() * np.clip( (dataset.Date - dataset.Date.min() ).dt.days/3, 0, 1)
dataset['case_chg_ema_10d'] = dataset.case_chg.ewm(span = 10).mean() * np.clip( (dataset.Date - dataset.Date.min() ).dt.days/10, 0, 1)
dataset['case_chg_stdev_5d'] = dataset.case_chg.rolling(5).std() * np.clip( (dataset.Date - dataset.Date.min() ).dt.days/5, 0, 1)
dataset['case_chg_stdev_15d'] = dataset.case_chg.rolling(15).std() * np.clip( (dataset.Date - dataset.Date.min() ).dt.days/15, 0, 1)
dataset['case_update_pct_3d_ewm'] = (dataset.case_chg > 0).ewm(span = 3).mean() * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/3, 0, 1), 2)
dataset['case_update_pct_10d_ewm'] = (dataset.case_chg > 0).ewm(span = 10).mean() * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/10, 0, 1), 2)
dataset['case_update_pct_30d_ewm'] = (dataset.case_chg > 0).ewm(span = 30).mean() * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/30, 0, 1), 2)
dataset['fatality_chg'] = np.clip(np.log(dataset.Fatalities + 1 )- np.log(dataset.Fatalities.shift(1) +1), 0, None).fillna(0)
dataset['fatality_chg_ema_3d'] = dataset.fatality_chg.ewm(span = 3).mean() * np.clip( (dataset.Date - dataset.Date.min() ).dt.days/33, 0, 1)
dataset['fatality_chg_ema_10d'] = dataset.fatality_chg.ewm(span = 10).mean() * np.clip( (dataset.Date - dataset.Date.min() ).dt.days/10, 0, 1)
dataset['fatality_chg_stdev_5d'] = dataset.fatality_chg.rolling(5).std() * np.clip( (dataset.Date - dataset.Date.min() ).dt.days/5, 0, 1)
dataset['fatality_chg_stdev_15d'] = dataset.fatality_chg.rolling(15).std() * np.clip( (dataset.Date - dataset.Date.min() ).dt.days/15, 0, 1)
dataset['fatality_update_pct_3d_ewm'] = (dataset.fatality_chg > 0).ewm(span = 3).mean() * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/3, 0, 1), 2)
dataset['fatality_update_pct_10d_ewm'] = (dataset.fatality_chg > 0).ewm(span = 10).mean() * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/10, 0, 1), 2)
dataset['fatality_update_pct_30d_ewm'] = (dataset.fatality_chg > 0).ewm(span = 30).mean() * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/30, 0, 1), 2)
dataset.tail()


# In[ ]:


contain_data.Date = contain_data.Date + datetime.timedelta(7)
contain_data.Date.max()
dataset = pd.merge(dataset, sup_data, on='Place', how='left', validate='m:1')
dataset = pd.merge(dataset, contain_data, on = ['Country', 'Date'], how='left', validate='m:1')
dataset['log_true_population'] =   np.log(dataset.TRUE_POPULATION + 1)
dataset['ConfirmedCases_percapita'] = np.log(dataset.ConfirmedCases + 1)- np.log(dataset.TRUE_POPULATION + 1)
dataset['Fatalities_percapita'] = np.log(dataset.Fatalities + 1)- np.log(dataset.TRUE_POPULATION + 1)
dataset['log_cfr'] = np.log(    (dataset.Fatalities + np.clip(0.015 * dataset.ConfirmedCases, 0, 0.3)) / ( dataset.ConfirmedCases + 0.1) )


# In[ ]:


def cfr(case, fatality):
    cfr_calc = np.log((fatality + np.clip(0.015 * case, 0, 0.3)) / ( case + 0.1) )
    return np.where(np.isnan(cfr_calc) | np.isinf(cfr_calc), BLCFR, cfr_calc)

BLCFR = np.median(dataset[dataset.ConfirmedCases==1].log_cfr[::10])
dataset.log_cfr.fillna(BLCFR, inplace=True)
dataset.log_cfr = np.where(dataset.log_cfr.isnull() | np.isinf(dataset.log_cfr),BLCFR, dataset.log_cfr)
BLCFR


# In[ ]:


print('rates and cross effects')
dataset['log_cfr_3d_ewm'] = BLCFR + (dataset.log_cfr - BLCFR).ewm(span = 3).mean() * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/3, 0, 1), 2)
dataset['log_cfr_8d_ewm'] = BLCFR + (dataset.log_cfr - BLCFR).ewm(span = 8).mean() * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/8, 0, 1), 2)
dataset['log_cfr_20d_ewm'] = BLCFR + (dataset.log_cfr - BLCFR).ewm(span = 20).mean()  * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/20, 0, 1), 2)
dataset['log_cfr_3d_20d_ewm_crossover'] = dataset.log_cfr_3d_ewm - dataset.log_cfr_20d_ewm
dataset.drop(columns = 'log_cfr', inplace=True)
date_totals = dataset.groupby('Date').sum()
mean_7d_c_slope = dataset.groupby('Date')[['ConfirmedCases_7d_prior_slope']].apply(lambda x: np.mean(x[x > 0]) ).ewm(span = 3).mean() 
mean_7d_f_slope = dataset.groupby('Date')[['Fatalities_7d_prior_slope']].apply(lambda x: np.mean(x[x > 0]) ).ewm(span = 7).mean()
dataset['ConfirmedCases_percapita_vs_world'] = np.log(dataset.ConfirmedCases + 1)- np.log(dataset.TRUE_POPULATION + 1) -  (np.log(date_totals.loc[dataset.Date].ConfirmedCases + 1)-np.log(date_totals.loc[dataset.Date].TRUE_POPULATION + 1)).values
dataset['Fatalities_percapita_vs_world'] = np.log(dataset.Fatalities + 1)- np.log(dataset.TRUE_POPULATION + 1)-(np.log(date_totals.loc[dataset.Date].Fatalities + 1) -np.log(date_totals.loc[dataset.Date].TRUE_POPULATION + 1)).values
dataset['cfr_vs_world'] = dataset.log_cfr_3d_ewm - np.log(date_totals.loc[dataset.Date].Fatalities / date_totals.loc[dataset.Date].ConfirmedCases ).values

cont_date_totals = dataset.groupby(['Date', 'continent_generosity']).sum()

dataset['ConfirmedCases_percapita_vs_continent_mean'] = 0
dataset['Fatalities_percapita_vs_continent_mean'] = 0
dataset['ConfirmedCases_percapita_vs_continent_median'] = 0
dataset['Fatalities_percapita_vs_continent_median'] = 0

for cg in dataset.continent_generosity.unique():
    ps = dataset.groupby("Place").last()
    tp = ps[ps.continent_generosity==cg].TRUE_POPULATION.sum()
    print(tp / 1e9)
    for Date in dataset.Date.unique():
        cd =  dataset[(dataset.Date == Date) & (dataset.continent_generosity == cg)][['ConfirmedCases', 'Fatalities', 'TRUE_POPULATION']]
#         print(cd)
        cmedian = np.median(np.log(cd.ConfirmedCases + 1)- np.log(cd.TRUE_POPULATION+1))
        cmean = np.log(cd.ConfirmedCases.sum() + 1) - np.log(tp + 1)
        fmedian = np.median(np.log(cd.Fatalities + 1)- np.log(cd.TRUE_POPULATION+1))
        fmean = np.log(cd.Fatalities.sum() + 1) - np.log(tp + 1)
        cfrmean = cfr( cd.ConfirmedCases.sum(),  cd.Fatalities.sum()   ) 
        dataset.loc[(dataset.Date == Date) & (dataset.continent_generosity == cg),'ConfirmedCases_percapita_vs_continent_mean'] = dataset['ConfirmedCases_percapita'] - (cmean)
        dataset.loc[(dataset.Date == Date) & (dataset.continent_generosity == cg),'ConfirmedCases_percapita_vs_continent_median'] = dataset['ConfirmedCases_percapita'] - (cmedian)
        dataset.loc[(dataset.Date == Date) & (dataset.continent_generosity == cg),'Fatalities_percapita_vs_continent_mean'] = dataset['Fatalities_percapita'] - (fmean)
        dataset.loc[(dataset.Date == Date) & (dataset.continent_generosity == cg),'Fatalities_percapita_vs_continent_median'] = dataset['Fatalities_percapita']- (fmedian)
        dataset.loc[(dataset.Date == Date) &(dataset.continent_generosity == cg),'cfr_vs_continent'] = dataset.log_cfr_3d_ewm - cfrmean
        
print('handling location')
all_places = dataset[['Place', 'latitude', 'longitude']].drop_duplicates().set_index('Place',drop=True)

def surroundingPlaces(place, d = 10):
    dist = (all_places.latitude - all_places.loc[place].latitude)**2 + (all_places.longitude - all_places.loc[place].longitude) ** 2 
    return all_places[dist < d**2][1:n+1]

def nearestPlaces(place, n = 10):
    dist = (all_places.latitude - all_places.loc[place].latitude)**2 + (all_places.longitude - all_places.loc[place].longitude) ** 2
    ranked = np.argsort(dist) 
    return all_places.iloc[ranked][1:n+1]


dgp = dataset.groupby('Place').last()
for n in [5, 10, 20]:

    for place in dataset.Place.unique():
        nps = nearestPlaces(place, n)
        tp = dgp.loc[nps.index].TRUE_POPULATION.sum()

        dataset.loc[dataset.Place==place,'ratio_population_vs_nearest{}'.format(n)] = np.log(dataset.loc[dataset.Place==place].TRUE_POPULATION.mean() + 1)- np.log(tp+1)
        nbps =  dataset[(dataset.Place.isin(nps.index))].groupby('Date')[['ConfirmedCases', 'Fatalities']].sum()

        nppc = (np.log( nbps.loc[dataset[dataset.Place==place].Date].fillna(0).ConfirmedCases + 1) - np.log(tp + 1))
        nppf = (np.log( nbps.loc[dataset[dataset.Place==place].Date].fillna(0).Fatalities + 1) - np.log(tp + 1))
        npp_cfr = cfr( nbps.loc[dataset[dataset.Place==place].Date].fillna(0).ConfirmedCases,nbps.loc[dataset[dataset.Place==place].Date].fillna(0).Fatalities)
        dataset.loc[(dataset.Place == place),'ConfirmedCases_percapita_vs_nearest{}'.format(n)] = dataset[(dataset.Place == place)].ConfirmedCases_percapita - nppc.values
        dataset.loc[(dataset.Place == place),'Fatalities_percapita_vs_nearest{}'.format(n)] = dataset[(dataset.Place == place)].Fatalities_percapita - nppf.values
        dataset.loc[(dataset.Place == place),'cfr_vs_nearest{}'.format(n)] = dataset[(dataset.Place == place)].log_cfr_3d_ewm - npp_cfr   
        dataset.loc[(dataset.Place == place),'ConfirmedCases_nearest{}_percapita'.format(n)] = nppc.values
        dataset.loc[(dataset.Place == place),'Fatalities_nearest{}_percapita'.format(n)] = nppf.values
        dataset.loc[(dataset.Place == place),'cfr_nearest{}'.format(n)] = npp_cfr
        dataset.loc[(dataset.Place == place),'ConfirmedCases_nearest{}_10d_slope'.format(n)] = ( nppc.ewm(span = 1).mean() - nppc.ewm(span = 10).mean() ).values
        dataset.loc[(dataset.Place == place),'Fatalities_nearest{}_10d_slope'.format(n)] =  ( nppf.ewm(span = 1).mean() - nppf.ewm(span = 10).mean() ).values
        
        npp_cfr_s = pd.Series(npp_cfr)
        dataset.loc[(dataset.Place == place),'cfr_nearest{}_10d_slope'.format(n)] = ( npp_cfr_s.ewm(span = 1).mean()- npp_cfr_s.ewm(span = 10).mean() ) .values

dgp = dataset.groupby('Place').last()
for d in [5, 10, 20]:

    for place in dataset.Place.unique():
        nps = surroundingPlaces(place, d)
        dataset.loc[dataset.Place==place, 'num_surrounding_places_{}_degrees'.format(d)] = len(nps)
        tp = dgp.loc[nps.index].TRUE_POPULATION.sum()
        dataset.loc[dataset.Place==place,'ratio_population_vs_surrounding_places_{}_degrees'.format(d)] = np.log(dataset.loc[dataset.Place==place].TRUE_POPULATION.mean() + 1)- np.log(tp+1)
        
        if len(nps)==0:
            continue;
        nbps =  dataset[(dataset.Place.isin(nps.index))].groupby('Date')[['ConfirmedCases', 'Fatalities']].sum()
        nppc = (np.log( nbps.loc[dataset[dataset.Place==place].Date].fillna(0).ConfirmedCases + 1) - np.log(tp + 1))
        nppf = (np.log( nbps.loc[dataset[dataset.Place==place].Date].fillna(0).Fatalities + 1) - np.log(tp + 1))
        npp_cfr = cfr( nbps.loc[dataset[dataset.Place==place].Date].fillna(0).ConfirmedCases,nbps.loc[dataset[dataset.Place==place].Date].fillna(0).Fatalities)
        dataset.loc[(dataset.Place == place),'ConfirmedCases_percapita_vs_surrounding_places_{}_degrees'.format(d)] = dataset[(dataset.Place == place)].ConfirmedCases_percapita - nppc.values
        dataset.loc[(dataset.Place == place),'Fatalities_percapita_vs_surrounding_places_{}_degrees'.format(d)] = dataset[(dataset.Place == place)].Fatalities_percapita - nppf.values
        dataset.loc[(dataset.Place == place),'cfr_vs_surrounding_places_{}_degrees'.format(d)] = dataset[(dataset.Place == place)].log_cfr_3d_ewm - npp_cfr   
        dataset.loc[(dataset.Place == place),'ConfirmedCases_surrounding_places_{}_degrees_percapita'.format(d)] = nppc.values
        dataset.loc[(dataset.Place == place),'Fatalities_surrounding_places_{}_degrees_percapita'.format(d)] = nppf.values
        dataset.loc[(dataset.Place == place),'cfr_surrounding_places_{}_degrees'.format(d)] = npp_cfr
        
        dataset.loc[(dataset.Place == place),'ConfirmedCases_surrounding_places_{}_degrees_10d_slope'.format(d)] = ( nppc.ewm(span = 1).mean() - nppc.ewm(span = 10).mean() ).values
        dataset.loc[(dataset.Place == place),'Fatalities_surrounding_places_{}_degrees_10d_slope'.format(d)] = ( nppf.ewm(span = 1).mean() - nppf.ewm(span = 10).mean() ).values
        npp_cfr_s = pd.Series(npp_cfr)
        dataset.loc[(dataset.Place == place),'cfr_surrounding_places_{}_degrees_10d_slope'.format(d)] = ( npp_cfr_s.ewm(span = 1).mean()- npp_cfr_s.ewm(span = 10).mean() ) .values
        
for col in [c for c in dataset.columns if 'surrounding_places' in c and 'num_sur' not in c]:
    dataset[col] = dataset[col].fillna(0)
    n_col = 'num_surrounding_places_{}_degrees'.format(col.split('degrees')[0].split('_')[-2])

    print(col)
    dataset[col + "_times_num_places"] = dataset[col] * np.sqrt(dataset[n_col])
dataset[dataset.Country=='US'][['Place', 'Date'] + [c for c in dataset.columns if 'ratio_p' in c]][::50]

dataset['first_case_ConfirmedCases_percapita'] = np.log(dataset.first_case_ConfirmedCases + 1) - np.log(dataset.TRUE_POPULATION + 1)
dataset['first_case_Fatalities_percapita'] = np.log(dataset.first_case_Fatalities + 1) - np.log(dataset.TRUE_POPULATION + 1)
dataset['first_fatality_Fatalities_percapita'] = np.log(dataset.first_fatality_Fatalities + 1) - np.log(dataset.TRUE_POPULATION + 1)
dataset['first_fatality_ConfirmedCases_percapita'] = np.log(dataset.first_fatality_ConfirmedCases + 1)- np.log(dataset.TRUE_POPULATION + 1)
dataset['days_to_saturation_ConfirmedCases_4d'] = ( - np.log(dataset.ConfirmedCases + 1)+ np.log(dataset.TRUE_POPULATION + 1)) / dataset.ConfirmedCases_4d_prior_slope         
dataset['days_to_saturation_ConfirmedCases_7d'] = ( - np.log(dataset.ConfirmedCases + 1)+ np.log(dataset.TRUE_POPULATION + 1)) / dataset.ConfirmedCases_7d_prior_slope         
dataset['days_to_saturation_Fatalities_20d_cases'] = ( - np.log(dataset.Fatalities + 1)+ np.log(dataset.TRUE_POPULATION + 1)) / dataset.ConfirmedCases_20d_prior_slope         
dataset['days_to_saturation_Fatalities_12d_cases'] = ( - np.log(dataset.Fatalities + 1)+ np.log(dataset.TRUE_POPULATION + 1)) / dataset.ConfirmedCases_12d_prior_slope         
dataset['days_to_3pct_ConfirmedCases_4d'] = ( - np.log(dataset.ConfirmedCases + 1)+ np.log(dataset.TRUE_POPULATION + 1) - 3.5) / dataset.ConfirmedCases_4d_prior_slope         
dataset['days_to_3pct_ConfirmedCases_7d'] = ( - np.log(dataset.ConfirmedCases + 1)+ np.log(dataset.TRUE_POPULATION + 1) - 3.5) / dataset.ConfirmedCases_7d_prior_slope         
dataset['days_to_0.3pct_Fatalities_20d_cases'] = ( - np.log(dataset.Fatalities + 1)+ np.log(dataset.TRUE_POPULATION + 1) - 5.8)/ dataset.ConfirmedCases_20d_prior_slope         
dataset['days_to_0.3pct_Fatalities_12d_cases'] = ( - np.log(dataset.Fatalities + 1)+ np.log(dataset.TRUE_POPULATION + 1) - 5.8) / dataset.ConfirmedCases_12d_prior_slope         
dataset.tail()


# In[ ]:





# In[ ]:





# In[ ]:


print('rolling dates')
dataset = dataset[dataset.ConfirmedCases > 0]
datas = []
for window in range(1, 35):
    base = rollDates(dataset, window, True)
    datas.append(pd.merge(dataset[['Date', 'Place','ConfirmedCases', 'Fatalities']], base, on = ['Date', 'Place'],how = 'right',suffixes = ('_f', '')))
data = pd.concat(datas, axis =0).astype(np.float32, errors ='ignore')

data['Date_f'] = data.Date
data.Date = data.Date_i

data['elapsed'] = (data.Date_f - data.Date_i).dt.days
data['CaseChgRate'] = (np.log(data.ConfirmedCases_f + 1) - np.log(data.ConfirmedCases + 1))/ data.elapsed
data['FatalityChgRate'] = (np.log(data.Fatalities_f + 1) - np.log(data.Fatalities + 1))/ data.elapsed


# In[ ]:


# find HK and MACAU Area
#data['is_China'] = (data.Country=='China') & (~data.Place.isin(['Hong Kong', 'Macau']))


# In[ ]:


print('true aggregation')
falloff_hash = {}

def true_agg(rate_i, elapsed, bend_rate):
    elapsed = int(elapsed)
    if (bend_rate, elapsed) not in falloff_hash:
        falloff_hash[(bend_rate, elapsed)] = np.sum( [  np.power(bend_rate, e) for e in range(1, elapsed+1)] )
    return falloff_hash[(bend_rate, elapsed)] * rate_i
     

#true_agg(0.3, 30, 0.9)

slope_cols = [c for c in data.columns if any(z in c for z in ['prior_slope', 'chg', 'rate']) and not any(z in c for z in ['bend', 'prior_slope_chg', 'Country', 'ewm',]) ] # ** bid change; since rate too stationary
# print(slope_cols)
bend_rates = [1, 0.95, 0.90]
for bend_rate in bend_rates:
    bend_agg = data[['elapsed']].apply(lambda x: true_agg(1, *x, bend_rate), axis=1)
     
    for sc in slope_cols:
        if bend_rate < 1:
            data[sc+"_slope_bend_{}".format(bend_rate)] =  data[sc]  * np.power((bend_rate + 1)/2, data.elapsed)
            data[sc+"_true_slope_bend_{}".format(bend_rate)] = bend_agg *  data[sc] / data.elapsed
            
        data[sc+"_agg_bend_{}".format(bend_rate)] =  data[sc] * data.elapsed * np.power((bend_rate + 1)/2, data.elapsed)
        data[sc+"_true_agg_bend_{}".format(bend_rate)] = bend_agg *  data[sc]

for col in [c for c in data.columns if any(z in c for z in['vs_continent', 'nearest', 'vs_world', 'surrounding_places'])]:
    data[col + '_times_days'] = data[col] * data.elapsed


data['saturation_slope_ConfirmedCases'] = (- np.log(data.ConfirmedCases + 1)+ np.log(data.TRUE_POPULATION + 1)) / data.elapsed
data['saturation_slope_Fatalities'] = (- np.log(data.Fatalities + 1)+ np.log(data.TRUE_POPULATION + 1)) / data.elapsed
data['dist_to_ConfirmedCases_saturation_times_days'] = (- np.log(data.ConfirmedCases + 1)+ np.log(data.TRUE_POPULATION + 1)) * data.elapsed
data['dist_to_Fatalities_saturation_times_days'] = (- np.log(data.Fatalities + 1)+ np.log(data.TRUE_POPULATION + 1)) * data.elapsed
data['slope_to_1pct_ConfirmedCases'] = (- np.log(data.ConfirmedCases + 1)+ np.log(data.TRUE_POPULATION + 1) - 4.6) / data.elapsed
data['slope_to_0.1pct_Fatalities'] = (- np.log(data.Fatalities + 1)+ np.log(data.TRUE_POPULATION + 1) - 6.9) / data.elapsed
data['dist_to_1pct_ConfirmedCases_times_days'] = (- np.log(data.ConfirmedCases + 1)+ np.log(data.TRUE_POPULATION + 1) - 4.6) * data.elapsed
data['dist_to_0.1pct_Fatalities_times_days'] = (- np.log(data.Fatalities + 1)+ np.log(data.TRUE_POPULATION + 1) - 6.9) * data.elapsed
data['trendline_per_capita_ConfirmedCases_4d_slope'] = ( np.log(data.ConfirmedCases + 1)- np.log(data.TRUE_POPULATION + 1)) + (data.ConfirmedCases_4d_prior_slope * data.elapsed)
data['trendline_per_capita_ConfirmedCases_7d_slope'] = ( np.log(data.ConfirmedCases + 1)- np.log(data.TRUE_POPULATION + 1)) + (data.ConfirmedCases_7d_prior_slope * data.elapsed)
data['trendline_per_capita_Fatalities_12d_slope'] = ( np.log(data.Fatalities + 1)- np.log(data.TRUE_POPULATION + 1)) + (data.ConfirmedCases_12d_prior_slope * data.elapsed)
data['trendline_per_capita_Fatalities_20d_slope'] = ( np.log(data.Fatalities + 1)- np.log(data.TRUE_POPULATION + 1)) + (data.ConfirmedCases_20d_prior_slope * data.elapsed)

 

# def logHist(x, b = 150):
#     return

data['log_fatalities'] = np.log(data.Fatalities + 1) #  + 0.4 * np.random.normal(0, 1, len(data))
data['log_cases'] = np.log(data.ConfirmedCases + 1) # + 0.2 *np.random.normal(0, 1, len(data))
# find HK and MACAU Area
#data['is_China'] = (data.Country=='China') & (~data.Place.isin(['Hong Kong', 'Macau']))
data['is_China'] = (data.Country=='China') & (~data.Place.isin(['China_Hong Kong', 'China_Macau']))

for col in [c for c in data.columns if 'd_ewm' in c]:
    data[col] += np.random.normal(0, 1, len(data)) * np.std(data[col]) * 0.2
    
data['is_province'] = 1.0* (~data.Province_State.isnull() )
data['log_elapsed'] = np.log(data.elapsed + 1)

data.drop(columns = ['TRUE_POPULATION'], inplace=True)
data['final_day_of_week'] = data.Date_f.apply(datetime.datetime.weekday)
data['base_date_day_of_week'] = data.Date.apply(datetime.datetime.weekday)
data['date_difference_modulo_7_days'] = (data.Date_f - data.Date).dt.days % 7
for c in data.columns.to_list():
    if 'days_to' in c:
        data[c] = data[c].where(~np.isinf(data[c]), 1e3)
        data[c] = np.clip(data[c], 0, 365)
        data[c] = np.sqrt(data[c])


# In[ ]:


data.shape


# In[ ]:


new_places = train[(train.Date == test.Date.min() - datetime.timedelta(1)) &(train.ConfirmedCases == 0)].Place


# In[ ]:


test.Date.min(), VAL_START_DATE


# In[ ]:





# In[ ]:


model_data = data[ (( len(test) ==0 ) | (data.Date_f < test.Date.min()))& (data.ConfirmedCases > 0) & (~data.ConfirmedCases_f.isnull())].copy()
# model_data = data[ (( len(test) ==0 ) | (data.Date_f < VAL_START_DATE))& (data.ConfirmedCases > 0) & (~data.ConfirmedCases_f.isnull())].copy()


# print(test.Date.min())


model_data.Date_f.max()
model_data.Date.max()
model_data.Date_f.min()

model_data = model_data[~(( np.random.rand(len(model_data)) < 0.8 )  &( model_data.Country == 'China') &(model_data.Date < datetime.datetime(2020, 2, 15)) )]
x_dates = model_data[['Date_i', 'Date_f', 'Place']]
x = model_data[model_data.columns.to_list()[model_data.columns.to_list().index('ConfirmedCases_1d_prior_value'):]].drop(columns = ['Date_i', 'Date_f', 'CaseChgRate', 'FatalityChgRate'])

if PRIVATE:
    data_test = data[ (data.Date_i == train.Date.max() ) & (data.Date_f.isin(test.Date.unique() ) ) ].copy()
else:
    # data_test = data[ (data.Date_i == test.Date.min() - datetime.timedelta(1) ) & (data.Date_f.isin(test.Date.unique() ) ) ].copy()
    data_test = data[ (data.Date_i == VAL_START_DATE - datetime.timedelta(1) ) & (data.Date_f.isin(test.Date.unique() ) ) ].copy()
    


# data_test.Date.unique()
# test.Date.unique()

x_test =  data_test[x.columns].copy()

# train.Date.max()
# test.Date.max()

if MODEL_Y is 'slope':
    y_cases = model_data.CaseChgRate 
    y_fatalities = model_data.FatalityChgRate 
else:
    y_cases = model_data.CaseChgRate * model_data.elapsed
    y_fatalities = model_data.FatalityChgRate * model_data.elapsed
    
y_cfr = np.log(    (model_data.Fatalities_f + np.clip(0.015 * model_data.ConfirmedCases_f, 0, 0.3)) / ( model_data.ConfirmedCases_f + 0.1) )


groups = model_data.Country
places = model_data.Place
x.shape, x_test.shape


# In[ ]:





# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, PredefinedSplit
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import make_scorer
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import HuberRegressor, ElasticNet
import lightgbm as lgb


np.random.seed(SEED)

enet_params = { 'alpha': [   3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3,  ],
                'l1_ratio': [  0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.97, 0.99 ]}

et_params = {        'n_estimators': [50, 70, 100, 140],
                    'max_depth': [3, 5, 7, 8, 9, 10],
                      'min_samples_leaf': [30, 50, 70, 100, 130, 165, 200, 300, 600],
                     'max_features': [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
                    'min_impurity_decrease': [0, 1e-5 ], #1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
                    'bootstrap': [ True, False], # False is clearly worse          
                 #   'criterion': ['mae'],
                   }


lgb_params = {
                'max_depth': [5, 12],
                'n_estimators': [ 100, 200, 300, 500],   # continuous
                'min_split_gain': [0, 0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
                'min_child_samples': [ 7, 10, 14, 20, 30, 40, 70, 100, 200, 400, 700, 1000, 2000],
                'min_child_weight': [0], #, 1e-3],
                'num_leaves': [5, 10, 20, 30],
                'learning_rate': [0.05, 0.07, 0.1],   #, 0.1],       
                'colsample_bytree': [0.1, 0.2, 0.33, 0.5, 0.65, 0.8, 0.9], 
                'colsample_bynode':[0.1, 0.2, 0.33, 0.5, 0.65, 0.81],
                'reg_lambda': [1e-5, 3e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000,   ],
                'reg_alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 30, 1000,], # 1, 10, 100, 1000, 10000],
                'subsample': [  0.8, 0.9, 1],
                'subsample_freq': [1],
                'max_bin': [ 7, 15, 31, 63, 127, 255],
  #               'extra_trees': [True, False],
#                 'boosting': ['gbdt', 'dart'],
    #     'subsample_for_bin': [200000, 500000],
               }    


MSE = 'neg_mean_squared_error'
MAE = 'neg_mean_absolute_error'


def trainENet(x, y, groups, cv = 0, **kwargs):
    return trainModel(x, y, groups, 
                      clf = ElasticNet(normalize = True, selection = 'random', 
                                       max_iter = 3000),
                      params = enet_params, 
                      cv = cv, **kwargs)

# %% [code]
def trainETR(x, y, groups, cv = 0, n_jobs = 5,  **kwargs):
    clf = ExtraTreesRegressor(n_jobs = 1)
    params = et_params
    return trainModel(x, y, groups, clf, params, cv, n_jobs, **kwargs)

# %% [code]
def trainLGB(x, y, groups, cv = 0, n_jobs = 4, **kwargs):
    clf = lgb.LGBMRegressor(verbosity=-1, hist_pool_size = 1000,  
                      )
    params = lgb_params
    
    return trainModel(x, y, groups, clf, params, cv, n_jobs,  **kwargs)

# %% [code]
def trainModel(x, y, groups, clf, params, cv = 0, n_jobs = None, 
                   verbose=0, splits=None, **kwargs):

        if n_jobs is None:
            n_jobs = 4
        if np.random.rand() < 0.8: # all shuffle, don't want overfit models, just reasonable
            folds = GroupShuffleSplit(n_splits=4, 
                                                   test_size= 0.2 + 0.10 * np.random.rand())
        else:
            folds = GroupKFold(4)
        clf = RandomizedSearchCV(clf, params, 
                            cv=  folds, 
#                                  cv = GroupKFold(4),
                                 n_iter=12, 
                                verbose = 0, n_jobs = n_jobs, scoring = MSE)
        f = clf.fit(x, y, groups)
        #if verbose > 0:
        print(pd.DataFrame(clf.cv_results_['mean_test_score'])); print();  
        
        
        best = clf.best_estimator_;  print(best)
        print("Best Score: {}".format(np.round(clf.best_score_,4)))
        
        return best

np.mean(y_cases)


def getSparseColumns(x, verbose = 0):
    sc = []
    for c in x.columns.to_list():
        u = len(x[c].unique())
        if u > 10 and u < 0.01*len(x) :
            sc.append(c)
            if verbose > 0:
                print("{}: {}".format(c, u))

    return sc


def noisify(x, noise = 0.1):
    x = x.copy()
   # cols = x.columns.to_list()
    cols = getSparseColumns(x)
    for c in cols:
        u = len(x[c].unique())
        if u > 50:
            x[c].values[:] = x[c].values + np.random.normal(0, noise, len(x)) * np.std(x[c])
    return x;


def getMaxOverlap(row, df):
#     max_overlap_frac = 0

    df_place = df[df.Place == row.Place]
    if len(df_place)==0:
        return 0
#     print(df_place)
    overlap = (np.clip( df_place.Date_f, None, row.Date_f) - np.clip( df_place.Date_i, row.Date_i, None) ).dt.days
    overlap = np.clip(overlap, 0, None)
    length = np.clip(  (df_place.Date_f - df_place.Date_i).dt.days, 
                        (row.Date_f - row.Date_i).days,  None)

    return np.amax(overlap / length) 


def getSampleWeight(x, groups):
 
    
    counter = Counter(groups)
    median_count = np.median( [counter[group] for group in groups.unique()])
#     print(median_count)
    c_count = [counter[group] for group in groups]
    
    e_decay = np.round(LT_DECAY_MIN + np.random.rand() * ( LT_DECAY_MAX - LT_DECAY_MIN), 1) 
    print("LT weight decay: {:.2f}".format(e_decay));
    ssr =  np.power(  1 / np.clip( c_count / median_count , 0.1,  30) ,0.1 + np.random.rand() * 0.6) /   np.power(x.elapsed / 3, e_decay) *  SET_FRAC * np.exp(  -    np.random.rand()  )
    
    # drop % of groups at random
    group_drop = dict([(group, np.random.rand() < 0.15) for group in groups.unique()])
    ssr = ssr * (  [ 1 -group_drop[group] for group in groups])
#     print(ssr[::171])
#     print(np.array([ 1 -group_drop[group] for group in groups]).sum() / len(groups))

#     pd.Series(ssr).plot(kind='hist', bins = 100)
    return ssr

def runBags(x, y, groups, cv, bags = 3, model_type = trainLGB, 
            noise = 0.1, splits = None, weights = None, **kwargs):
    models = []
    for bag in range(bags):
        print("\nBAG {}".format(bag+1))
        
        x = x.copy()  # copy X to modify it with noise
        
        if DROPS:
            # drop 0-70% of the bend/slope/prior features, just for speed and model diversity
            for col in [c for c in x.columns if any(z in c for z in ['bend', 'slope', 'prior'])]:
                if np.random.rand() < np.sqrt(np.random.rand()) * 0.7:
                    x[col].values[:] = 0
            

        if DROPS and (np.random.rand() < 0.30):
            print('dropping nearest features')
            for col in [c for c in x.columns if 'nearest' in c]:    
                x[col].values[:] = 0
        
        #  % of the time drop all 'surrounding_places' features 
        if DROPS and (np.random.rand() < 0.25):
            print('dropping \'surrounding places\' features')
            for col in [c for c in x.columns if 'surrounding_places' in c]:    
                x[col].values[:] = 0
        
        

        col_drop_frac = np.sqrt(np.random.rand()) * 0.5
        for col in [c for c in x.columns if 'elapsed' not in c ]:
            if np.random.rand() < col_drop_frac:
                x[col].values[:] = 0

        
        x = noisify(x, noise)
        
        
        if DROPS and (np.random.rand() < SUP_DROP):
            print("Dropping supplemental country data")
            for col in x[[c for c in x.columns if c in sup_data.columns]]:  
                x[col].values[:] = 0
                
        if DROPS and (np.random.rand() < ACTIONS_DROP): 
            for col in x[[c for c in x.columns if c in contain_data.columns]]:  
                x[col].values[:] = 0
#             print(x.StringencyIndex_20d_ewm[::157])
        else:
            print("*using containment data")
            
        if np.random.rand() < 0.6: 
            x.S_data_days = 0
            
        ssr = getSampleWeight(x, groups)
        
        date_falloff = 0 + (1/30) * np.random.rand()
        if weights is not None:
            ssr = ssr * np.exp(-weights * date_falloff)
        
        ss = ( np.random.rand(len(y)) < ssr  )
        print("n={}".format(len(x[ss])))
        
        p1 =x.elapsed[ss].plot(kind='hist', bins = int(x.elapsed.max() - x.elapsed.min() + 1))
        p1 = plt.figure();
#         break
#        print(Counter(groups[ss]))
        print((ss).sum())
        models.append(model_type(x[ss], y[ss], groups[ss], cv,   **kwargs))
    return models


# In[ ]:



x = x.astype(np.float32)
BAG_MULT = 1

print(x.shape)

lgb_c_clfs = []
lgb_c_noise = []

date_weights =  np.abs((model_data.Date_f - test.Date.min()).dt.days) 
print(date_weights)
#david original
C_NOISES = [ 0.05, 0.1, 0.2, 0.3, 0.4  ]
F_NOISES = [  0.5,  1, 2, 3,  ]
CFR_NOISES = [    0.4, 1, 2, 3]

#debug
# C_NOISES = [ 0.05]
# F_NOISES = [  0.5]
# CFR_NOISES = [ 0.4]


for iteration in range(0, int(math.ceil(1.1 * BAGS))):
    for noise in C_NOISES:
        print("\n---\n\nNoise of {}".format(noise));
        num_bags = 1 * BAG_MULT;
        if np.random.rand() < PLACE_FRACTION:
            cv_group = places
            print("CV by Place")
        else:
            cv_group = groups
            print("CV by Country")
             
        
        lgb_c_clfs.extend(runBags(x, y_cases,cv_group,MSE, num_bags, trainLGB, verbose = 0,noise = noise, weights = date_weights))
        lgb_c_noise.extend([noise] * num_bags)
        if SINGLE_MODEL:
            break;

lgb_f_clfs = [] 
lgb_f_noise = []

for iteration in range(0, int(np.ceil(np.sqrt(BAGS)))):
    for noise in F_NOISES:
        print("\n---\n\nNoise of {}".format(noise));
        num_bags = 1 * int(np.ceil(np.sqrt(BAG_MULT)))
        if np.random.rand() < PLACE_FRACTION  :
            cv_group = places
            print("CV by Place")
        else:
            cv_group = groups
            print("CV by Country")
            
   
        lgb_f_clfs.extend(runBags(x, y_fatalities, 
                                  cv_group, #places, # groups, 
                                  MSE, num_bags, trainLGB, 
                                  verbose = 0, noise = noise,
                                  weights = date_weights
                                 ))
        lgb_f_noise.extend([noise] * num_bags)
        if SINGLE_MODEL:
            break;

lgb_cfr_clfs = []
lgb_cfr_noise = []

for iteration in range(0, int(np.ceil(np.sqrt(BAGS)))):
    for noise in CFR_NOISES:
        print("\n---\n\nNoise of {}".format(noise));
        num_bags = 1 * BAG_MULT;
        if np.random.rand() < 0.5 * PLACE_FRACTION :
            cv_group = places
            print("CV by Place")
        else:
            cv_group = groups
            print("CV by Country")
 
        lgb_cfr_clfs.extend(runBags(x, y_cfr, 
                          cv_group, #groups
                          MSE, num_bags, trainLGB, verbose = 0, 
                                          noise = noise, 
                                          weights = date_weights

                                 ))
        lgb_cfr_noise.extend([noise] * num_bags)
        if SINGLE_MODEL:
            break;

lgb_cfr_clfs[0].predict(x_test)
                                                  


# In[ ]:


def show_FI(model, featNames, featCount):
   # show_FI_plot(model.feature_importances_, featNames, featCount)
    fis = model.feature_importances_
    fig, ax = plt.subplots(figsize=(6, 5))
    indices = np.argsort(fis)[::-1][:featCount]
    g = sns.barplot(y=featNames[indices][:featCount],
                    x = fis[indices][:featCount] , orient='h' )
    g.set_xlabel("Relative importance")
    g.set_ylabel("Features")
    g.tick_params(labelsize=12)
    g.set_title( " feature importance")
    

def avg_FI(all_clfs, featNames, featCount):
    # 1. Sum
    clfs = []
    for clf_set in all_clfs:
        for clf in clf_set:
            clfs.append(clf);
    print("{} classifiers".format(len(clfs)))
    fi = np.zeros( (len(clfs), len(clfs[0].feature_importances_)) )
    for idx, clf in enumerate(clfs):
        fi[idx, :] = clf.feature_importances_
    avg_fi = np.mean(fi, axis = 0)

    # 2. Plot
    fis = avg_fi
    fig, ax = plt.subplots(figsize=(6, 5))
    indices = np.argsort(fis)[::-1]#[:featCount]
    #print(indices)
    g = sns.barplot(y=featNames[indices][:featCount],
                    x = fis[indices][:featCount] , orient='h' )
    g.set_xlabel("Relative importance")
    g.set_ylabel("Features")
    g.tick_params(labelsize=12)
    g.set_title( " feature importance")
    
    return pd.Series(fis[indices], featNames[indices])

def linear_FI_plot(fi, featNames, featCount):
   # show_FI_plot(model.feature_importances_, featNames, featCount)
    fig, ax = plt.subplots(figsize=(6, 5))
    indices = np.argsort(np.absolute(fi))[::-1]#[:featCount]
    g = sns.barplot(y=featNames[indices][:featCount],
                    x = fi[indices][:featCount] , orient='h' )
    g.set_xlabel("Relative importance")
    g.set_ylabel("Features")
    g.tick_params(labelsize=12)
    g.set_title( " feature importance")
    return pd.Series(fi[indices], featNames[indices])

f = avg_FI([lgb_c_clfs], x.columns, 25)


for feat in ['bend', 'capita', 'cfr', 'slope', 'since', 'chg', 'ersonal','world', 'continent', 'nearest', 'surrounding']:
    print("{}: {:.2f}".format(feat, f.filter(like=feat).sum() / f.sum()))

print("{}: {:.2f}".format('sup_data',f[[c for c in f.index if c in sup_data.columns]].sum() / f.sum()))
print("{}: {:.2f}".format('contain_data',f[[c for c in f.index if c in contain_data.columns]].sum() / f.sum()))

f = avg_FI([lgb_f_clfs], x.columns, 25)


for feat in ['bend', 'capita', 'cfr', 'slope', 'since', 'chg', 'ersonal','world', 'continent', 'nearest', 'surrounding']:
    print("{}: {:.2f}".format(feat, f.filter(like=feat).sum() / f.sum()))


print("{}: {:.2f}".format('sup_data',f[[c for c in f.index if c in sup_data.columns]].sum() / f.sum()))
print("{}: {:.2f}".format('contain_data',f[[c for c in f.index if c in contain_data.columns]].sum() / f.sum()))



f = avg_FI([lgb_cfr_clfs], x.columns, 25)


for feat in ['bend', 'capita', 'cfr', 'slope', 'since', 'chg', 'ersonal','world', 'continent', 'nearest', 'surrounding']:
    print("{}: {:.2f}".format(feat, f.filter(like=feat).sum() / f.sum()))


print("{}: {:.2f}".format('sup_data',f[[c for c in f.index if c in sup_data.columns]].sum() / f.sum()))
print("{}: {:.2f}".format('contain_data',f[[c for c in f.index if c in contain_data.columns]].sum() / f.sum()))


# In[ ]:


print('prediction')

all_c_clfs = [lgb_c_clfs]#  enet_c_clfs]
all_f_clfs = [lgb_f_clfs] #, enet_f_clfs]
all_cfr_clfs = [lgb_cfr_clfs]

all_c_noise = [lgb_c_noise]
all_f_noise = [lgb_f_noise]
all_cfr_noise = [lgb_cfr_noise]

NUM_TEST_RUNS = 1

c_preds = np.zeros((NUM_TEST_RUNS * sum([len(x) for x in all_c_clfs]), len(x_test)))
f_preds = np.zeros((NUM_TEST_RUNS * sum([len(x) for x in all_f_clfs]), len(x_test)))
cfr_preds = np.zeros((NUM_TEST_RUNS * sum([len(x) for x in all_cfr_clfs]), len(x_test)))


def avg(x):
    return (np.mean(x, axis=0) + np.median(x, axis=0))/2


count = 0
for idx, clf in enumerate(lgb_c_clfs):
    for i in range(0, NUM_TEST_RUNS):
        noise = lgb_c_noise[idx]
        c_preds[count,:] = np.clip(clf.predict(noisify(x_test, noise)), -1 , 10)
        count += 1

count = 0
for idx, clf in enumerate(lgb_f_clfs):
    for i in range(0, NUM_TEST_RUNS):
        noise = lgb_f_noise[idx]
        f_preds[count,:] = np.clip(clf.predict(noisify(x_test, noise)), -1 , 10)
        count += 1

count = 0
for idx, clf in enumerate(lgb_cfr_clfs):
    for i in range(0, NUM_TEST_RUNS):
        noise = lgb_cfr_noise[idx]
        cfr_preds[count,:] = np.clip(clf.predict(noisify(x_test, noise)), -10 , 10)
        count += 1

def qPred(preds, pctile, simple=False):
    q = np.percentile(preds, pctile, axis = 0)
    if simple:
        return q;
    resid = preds - q
    resid_wtg = 2/100/len(preds)* ( np.clip(resid, 0, None) * (pctile) + np.clip(resid, None, 0) * (100- pctile) )
    adj = np.sum(resid_wtg, axis = 0)
    return q + adj


q = 50
y_cases_pred_blended_full = qPred(c_preds, q) #avg(c_preds)
y_fatalities_pred_blended_full = qPred(f_preds, q) # avg(f_preds)
y_cfr_pred_blended_full = qPred(cfr_preds, q) #avg(cfr_preds)


print(np.mean(np.corrcoef(c_preds[::NUM_TEST_RUNS]),axis=0))
print(np.mean(np.corrcoef(f_preds[::NUM_TEST_RUNS]), axis=0))
print(np.mean(np.corrcoef(cfr_preds[::NUM_TEST_RUNS]), axis = 0))

pd.Series(np.std(c_preds, axis = 0)).plot(kind='hist', bins = 50)
pd.Series(np.std(f_preds, axis = 0)).plot(kind='hist', bins = 50)
pd.Series(np.std(cfr_preds, axis = 0)).plot(kind='hist', bins = 50)
y_cfr

pred = pd.DataFrame(np.hstack((np.transpose(c_preds),np.transpose(f_preds))), index=x_test.index)
pred['Place'] = data_test.Place
pred['Date'] = data_test.Date
pred['Date_f'] = data_test.Date_f

# pred[(pred.Date == pred.Date.max()) & (pred.Date_f == pred.Date_f.max())][30: 60]

# np.round(pred[(pred.Date == pred.Date.max()) & (pred.Date_f == pred.Date_f.max())], 2)[190:220:]
# np.round(pred[(pred.Date == pred.Date.max()) & (pred.Date_f == pred.Date_f.max())][220:-20],2)


c_preds.shape, x_test.shape


# In[ ]:


c_preds.shape[1] / N_AREAS


# In[ ]:


# first_c_slope


# In[ ]:


data_wp = data_test.copy()

if MODEL_Y is 'slope':
    data_wp['case_slope'] = y_cases_pred_blended_full 
    data_wp['fatality_slope'] = y_fatalities_pred_blended_full 
else:
    data_wp['case_slope'] = y_cases_pred_blended_full / x_test.elapsed
    data_wp['fatality_slope'] = y_fatalities_pred_blended_full / x_test.elapsed

data_wp['cfr_pred'] = y_cfr_pred_blended_full

train.Date.max()
test.Date.min()

if len(test) > 0:
    base_date = test.Date.min() - datetime.timedelta(1)
    #base_date = VAL_START_DATE - datetime.timedelta(1)
else:
    base_date = train.Date.max()

base_date

data_wp_ss = data_wp[data_wp.Date == base_date]
data_wp_ss = data_wp_ss.drop(columns='Date').rename(columns = {'Date_f': 'Date'})
test_wp = pd.merge(test, data_wp_ss[['Date', 'Place', 'case_slope', 'fatality_slope', 'cfr_pred','elapsed']],how='left', on = ['Date', 'Place'])

first_c_slope = test_wp[~test_wp.case_slope.isnull()].groupby('Place').first()
last_c_slope = test_wp[~test_wp.case_slope.isnull()].groupby('Place').last()

first_f_slope = test_wp[~test_wp.fatality_slope.isnull()].groupby('Place').first()
last_f_slope = test_wp[~test_wp.fatality_slope.isnull()].groupby('Place').last()

first_cfr_pred = test_wp[~test_wp.cfr_pred.isnull()].groupby('Place').first()
last_cfr_pred = test_wp[~test_wp.cfr_pred.isnull()].groupby('Place').last()

test_wp.case_slope = np.where(test_wp.case_slope.isnull() & (test_wp.Date < first_c_slope.loc[test_wp.Place].Date.values),first_c_slope.loc[test_wp.Place].case_slope.values,test_wp.case_slope)
test_wp.case_slope = np.where(test_wp.case_slope.isnull() & (test_wp.Date > last_c_slope.loc[test_wp.Place].Date.values),last_c_slope.loc[test_wp.Place].case_slope.values,test_wp.case_slope)


test_wp.fatality_slope = np.where(  test_wp.fatality_slope.isnull() & 
                     (test_wp.Date < first_f_slope.loc[test_wp.Place].Date.values),first_f_slope.loc[test_wp.Place].fatality_slope.values,test_wp.fatality_slope)

test_wp.fatality_slope = np.where(  test_wp.fatality_slope.isnull() & 
                     (test_wp.Date > last_f_slope.loc[test_wp.Place].Date.values),last_f_slope.loc[test_wp.Place].fatality_slope.values,test_wp.fatality_slope)

test_wp.cfr_pred = np.where(  test_wp.cfr_pred.isnull() & 
                     (test_wp.Date < first_cfr_pred.loc[test_wp.Place].Date.values),first_cfr_pred.loc[test_wp.Place].cfr_pred.values,test_wp.cfr_pred)

test_wp.cfr_pred = np.where(  test_wp.cfr_pred.isnull() & 
                     (test_wp.Date > last_cfr_pred.loc[test_wp.Place].Date.values),last_cfr_pred.loc[test_wp.Place].cfr_pred.values,test_wp.cfr_pred)

test_wp.case_slope = test_wp.case_slope.interpolate('linear')
test_wp.fatality_slope = test_wp.fatality_slope.interpolate('linear')
test_wp.cfr_pred = test_wp.cfr_pred.interpolate('linear')
test_wp.case_slope = test_wp.case_slope.fillna(0)
test_wp.fatality_slope = test_wp.fatality_slope.fillna(0)

LAST_DATE = test.Date.min() - datetime.timedelta(1)
#LAST_DATE = VAL_START_DATE - datetime.timedelta(1)
print(LAST_DATE)

final = train_bk[train_bk.Date == LAST_DATE  ]
test_wp = pd.merge(test_wp, final[['Place', 'ConfirmedCases', 'Fatalities']], on='Place',how ='left', validate='m:1')
test_wp.ConfirmedCases = np.exp(np.log(test_wp.ConfirmedCases + 1) + test_wp.case_slope * (test_wp.Date - LAST_DATE).dt.days )- 1
test_wp.Fatalities = np.exp(np.log(test_wp.Fatalities + 1) + test_wp.fatality_slope *(test_wp.Date - LAST_DATE).dt.days )  -1

final = train_bk[train_bk.Date == test.Date.min() - datetime.timedelta(1) ]
#final = train_bk[train_bk.Date == VAL_START_DATE - datetime.timedelta(1) ]
final.head()

test['elapsed'] = (test.Date - final.Date.max()).dt.days 
full_bk = test_wp.copy()
full = test_wp.copy()

BASE_RATE = 0.01
CFR_CAP = 0.13
lplot(full_bk)
lplot(full_bk, columns = ['case_slope', 'fatality_slope'])

full['cfr_imputed_fatalities_low'] = full.ConfirmedCases * np.exp(full.cfr_pred) / np.exp(0.5)
full['cfr_imputed_fatalities_high'] = full.ConfirmedCases * np.exp(full.cfr_pred) * np.exp(0.5)
full['cfr_imputed_fatalities'] = full.ConfirmedCases * np.exp(full.cfr_pred)  

# full[(full.case_slope > 0.02) &(full.Fatalities < full.cfr_imputed_fatalities_low    ) &(full.cfr_imputed_fatalities_low > 0.3) &
#                 ( full.Fatalities < 100000 ) &(full.Country!='China') &(full.Date == datetime.datetime(2020, 4,15))].groupby('Place').last().sort_values('Fatalities', ascending=False).iloc[:, 9:]


# (np.log(full.Fatalities + 1) -np.log(full.cfr_imputed_fatalities) ).plot(kind='hist', bins = 250)


# full[(full.case_slope > 0.02) & 
#                    (full.Fatalities < full.cfr_imputed_fatalities_low    ) &
#                 (full.cfr_imputed_fatalities_low > 0.3) &
#                 ( full.Fatalities < 100000 ) &
#     (~full.Country.isin(['China', 'Korea, South']))][full.Date==train.Date.max()]\
#      .groupby('Place').first()\
#     .sort_values('cfr_imputed_fatalities', ascending=False).iloc[:, 9:]

# # %% [code]
# full.Fatalities = np.where(   
#     (full.case_slope > 0.02) & 
#                    (full.Fatalities <= full.cfr_imputed_fatalities_low    ) &
#                 (full.cfr_imputed_fatalities_low > 0.3) &
#                 ( full.Fatalities < 100000 ) &
#     (~full.Country.isin(['China', 'Korea, South'])) ,
                        
#                         (full.cfr_imputed_fatalities_high + full.cfr_imputed_fatalities)/2,
#                                     full.Fatalities)
    


full['elapsed'] = (test_wp.Date - LAST_DATE).dt.days

# full[ (full.case_slope > 0.02) & 
#           (np.log(full.Fatalities + 1) < np.log(full.ConfirmedCases * BASE_RATE + 1) - 0.5) &
#                            (full.Country != 'China')]\
#             [full.Date == datetime.datetime(2020, 4, 5)] \
#             .groupby('Place').last().sort_values('ConfirmedCases', ascending=False).iloc[:,8:]


# full.Fatalities = np.where((full.case_slope > 0.02) & 
#                       (full.Fatalities < full.ConfirmedCases * BASE_RATE) &
#                            (full.Country != 'China'),np.exp(np.log( full.ConfirmedCases * BASE_RATE + 1) \
#                            * np.clip(   0.5* (full.elapsed - 1) / 30, 0, 1)+  np.log(full.Fatalities +1 ) * np.clip(1 - 0.5* (full.elapsed - 1) / 30, 0, 1)) -1,full.Fatalities)  


# full[(full.case_slope > 0.02) & 
#                    (full.Fatalities > full.cfr_imputed_fatalities_high   ) &
#                 (full.cfr_imputed_fatalities_low > 0.4) &
#     (full.Country!='China')]\
#      .groupby('Place').count()\
#     .sort_values('ConfirmedCases', ascending=False).iloc[:, 8:]


# full[(full.case_slope > 0.02) & 
#                    (full.Fatalities > full.cfr_imputed_fatalities_high * 2   ) &
#                 (full.cfr_imputed_fatalities_low > 0.4) &
#     (full.Country!='China')  ]\
#      .groupby('Place').last()\
#     .sort_values('ConfirmedCases', ascending=False).iloc[:, 8:]


# full[(full.case_slope > 0.02) & 
#                    (full.Fatalities > full.cfr_imputed_fatalities_high * 1.5   ) &
#                 (full.cfr_imputed_fatalities_low > 0.4) &
#     (full.Country!='China')][full.Date==train.Date.max()]\
#      .groupby('Place').first()\
#     .sort_values('ConfirmedCases', ascending=False).iloc[:, 8:]


full.Fatalities =  np.where(  (full.case_slope > 0.02) & 
                   (full.Fatalities > full.cfr_imputed_fatalities_high      * 2   ) &
                (full.cfr_imputed_fatalities_low > 0.4) &
                (full.Country!='China') ,full.cfr_imputed_fatalities,full.Fatalities)

full.Fatalities =  np.where(  (full.case_slope > 0.02) & (full.Fatalities > full.cfr_imputed_fatalities_high) & (full.cfr_imputed_fatalities_low > 0.4) &
                (full.Country!='China') ,np.exp(0.6667 * np.log(full.Fatalities + 1) + 0.3333 * np.log(full.cfr_imputed_fatalities + 1)) - 1,full.Fatalities)

full[(full.Fatalities > full.ConfirmedCases * CFR_CAP) & (full.ConfirmedCases > 1000)].groupby('Place').last().sort_values('Fatalities', ascending=False)


# (np.log(full.Fatalities + 1) -np.log(full.cfr_imputed_fatalities) ).plot(kind='hist', bins = 250)


assert len(pd.merge(full, final, on='Place', suffixes = ('', '_i'), validate='m:1')) == len(full)

# %% [code]
ffm = pd.merge(full, final, on='Place', suffixes = ('', '_i'), validate='m:1')
ffm['fatality_slope'] = (np.log(ffm.Fatalities + 1 )- np.log(ffm.Fatalities_i + 1 ) ) / ffm.elapsed
ffm['case_slope'] = (np.log(ffm.ConfirmedCases + 1 ) - np.log(ffm.ConfirmedCases_i + 1 ) ) / ffm.elapsed
ffm[np.log(ffm.Fatalities+1) < np.log(ffm.Fatalities_i+1) - 0.2][['Place', 'Date', 'elapsed', 'Fatalities', 'Fatalities_i']]
ffm[np.log(ffm.ConfirmedCases + 1) < np.log(ffm.ConfirmedCases_i+1) - 0.2][['Place', 'elapsed', 'ConfirmedCases', 'ConfirmedCases_i']]

AGG_DICT = {'ForecastId': 'count','case_slope': 'last','fatality_slope': 'last','ConfirmedCases': 'sum','Fatalities': 'sum'}
full_bk[(full_bk.Date == test.Date.max() ) & 
   (~full_bk.Place.isin(new_places))].groupby('Country').agg(AGG_DICT).sort_values('ConfirmedCases', ascending=False)


lplot(ffm[~ffm.Place.isin(new_places)])


# In[ ]:





# In[ ]:


# lplot(ffm[~ffm.Place.isin(new_places)], columns = ['case_slope', 'fatality_slope'])


ffm.fatality_slope = np.clip(ffm.fatality_slope, None, 0.5)

for lr in [0.2, 0.14, 0.1, 0.07, 0.05, 0.03, 0.01 ]:

    ffm.loc[ (ffm.Place==ffm.Place.shift(4) )
         & (ffm.Place==ffm.Place.shift(-4) ), 'fatality_slope'] = \
         ( ffm.fatality_slope.shift(-2) * 0.25 \
              + ffm.fatality_slope.shift(-1) * 0.5 \
                + ffm.fatality_slope \
                  + ffm.fatality_slope.shift(1) * 0.5 \
                    + ffm.fatality_slope.shift(2) * 0.25 ) / 2.5


ffm.ConfirmedCases = np.exp(np.log(ffm.ConfirmedCases_i + 1) + ffm.case_slope *ffm.elapsed ) - 1
ffm.Fatalities = np.exp(np.log(ffm.Fatalities_i + 1) + ffm.fatality_slope *ffm.elapsed ) - 1

# lplot(ffm[~ffm.Place.isin(new_places)])
# lplot(ffm[~ffm.Place.isin(new_places)], columns = ['case_slope', 'fatality_slope'])


ffm[(ffm.Date == test.Date.max() ) & (~ffm.Place.isin(new_places))].groupby('Country').agg(AGG_DICT).sort_values('ConfirmedCases', ascending=False)
ffm_bk = ffm.copy()

ffm = ffm_bk.copy()

counter = Counter(data.Place)
median_count = np.median([ counter[group] for group in ffm.Place])
c_count = [ np.clip(np.power(counter[group] / median_count, -1.5), None, 2.5) for group in ffm.Place]

RATE_MULT = 0.00
RATE_ADD = 0.003
LAG_FALLOFF = 15

ma_factor = np.clip( ( ffm.elapsed - 14) / 14 , 0, 1)

ffm.case_slope = np.where(ffm.elapsed > 0,
    0.7 * ffm.case_slope * (1+ ma_factor * RATE_MULT) \
         + 0.3 * (  ffm.case_slope.ewm(span=LAG_FALLOFF).mean()* np.clip(ma_factor, 0, 1) 
                  + ffm.case_slope    * np.clip( 1 - ma_factor, 0, 1))+ RATE_ADD * ma_factor * c_count,ffm.case_slope)


RATE_MULT = 0
RATE_ADD = 0.015
LAG_FALLOFF = 15

ma_factor = np.clip( ( ffm.elapsed - 10) / 14 , 0, 1)


ffm.fatality_slope = np.where(ffm.elapsed > 0,
    0.3 * ffm.fatality_slope * (1+ ma_factor * RATE_MULT) \
         + 0.7* (  ffm.fatality_slope.ewm(span=LAG_FALLOFF).mean()* np.clip( ma_factor, 0, 1)
                      + ffm.fatality_slope * np.clip( 1 - ma_factor, 0, 1)   ) + RATE_ADD * ma_factor * c_count * (ffm.Country != 'China'),ffm.case_slope)

ffm.ConfirmedCases = np.exp(np.log(ffm.ConfirmedCases_i + 1) + ffm.case_slope * ffm.elapsed ) - 1
ffm.Fatalities = np.exp(np.log(ffm.Fatalities_i + 1)+ ffm.fatality_slope * ffm.elapsed ) - 1

# lplot(ffm[~ffm.Place.isin(new_places)])

# lplot(ffm[~ffm.Place.isin(new_places)], columns = ['case_slope', 'fatality_slope'])

# ffm_bk[(ffm_bk.Date == test.Date.max() ) & 
#    (~ffm_bk.Place.isin(new_places))].groupby('Country').agg(
#     {'ForecastId': 'count',
#      'case_slope': 'last',
#         'fatality_slope': 'last',
#             'ConfirmedCases': 'sum',
#                 'Fatalities': 'sum',
#                     }
# ).sort_values('ConfirmedCases', ascending=False)[:15]


# ffm[(ffm.Date == test.Date.max() ) & 
#    (~ffm.Place.isin(new_places))].groupby('Country').agg(
#     {'ForecastId': 'count',
#      'case_slope': 'last',
#         'fatality_slope': 'last',
#             'ConfirmedCases': 'sum',
#                 'Fatalities': 'sum',
#                     }
# ).sort_values('ConfirmedCases', ascending=False)[:15]

# ffm_bk[(ffm_bk.Date == test.Date.max() ) & 
#    (~ffm_bk.Place.isin(new_places))].groupby('Country').agg(
#     {'ForecastId': 'count',
#      'case_slope': 'last',
#         'fatality_slope': 'last',
#             'ConfirmedCases': 'sum',
#                 'Fatalities': 'sum',
#                     }
# ).sort_values('ConfirmedCases', ascending=False)[-50:]

# ffm[(ffm.Date == test.Date.max() ) & 
#    (~ffm.Place.isin(new_places))].groupby('Country').agg(
#     {'ForecastId': 'count',
#      'case_slope': 'last',
#         'fatality_slope': 'last',
#             'ConfirmedCases': 'sum',
#                 'Fatalities': 'sum',
#                     }
# ).loc[ffm_bk[(ffm_bk.Date == test.Date.max() ) & 
#    (~ffm_bk.Place.isin(new_places))].groupby('Country').agg(
#     {'ForecastId': 'count',
#      'case_slope': 'last',
#         'fatality_slope': 'last',
#             'ConfirmedCases': 'sum',
#                 'Fatalities': 'sum',
#                     }
# ).sort_values('ConfirmedCases', ascending=False)[-50:].index]

NUM_TEST_DATES = len(test.Date.unique())

base = np.zeros((2, NUM_TEST_DATES))
base2 = np.zeros((2, NUM_TEST_DATES))

for idx, c in enumerate(['ConfirmedCases', 'Fatalities']):
    for n in range(0, NUM_TEST_DATES):
        #base[idx,n] = np.mean(np.log(  train[((train.Date < test.Date.min())) & (train.ConfirmedCases > 0)].groupby('Country').nth(n)[c]+1))
        base[idx,n] = np.mean(np.log(  train[((train.Date < VAL_START_DATE)) & (train.ConfirmedCases > 0)].groupby('Country').nth(n)[c]+1))

base = np.pad( base, ((0,0), (6,0)), mode='constant', constant_values = 0)

for n in range(0, base2.shape[1]):
    base2[:, n] = np.mean(base[:, n+0: n+7], axis = 1)

# new_places = train[(train.Date == test.Date.min() - datetime.timedelta(1)) &(train.ConfirmedCases == 0)].Place
new_places = train[(train.Date == VAL_START_DATE - datetime.timedelta(1)) &(train.ConfirmedCases == 0)].Place


ffm.ConfirmedCases = np.where(   ffm.Place.isin(new_places),base2[ 0, (ffm.Date - test.Date.min()).dt.days],ffm.ConfirmedCases)
ffm.Fatalities = np.where(   ffm.Place.isin(new_places),base2[ 1, (ffm.Date - test.Date.min()).dt.days],ffm.Fatalities)


ffm[ffm.Country=='US'].groupby('Date').agg(
    {'ForecastId': 'count',
     'case_slope': 'mean',
        'fatality_slope': 'mean',
            'ConfirmedCases': 'sum',
                'Fatalities': 'sum',
                    })


# In[ ]:


sub = pd.read_csv(input_path + 'submission.csv')

scl = sub.columns.to_list()

print(full_bk.groupby('Place').last()[['Date', 'ConfirmedCases', 'Fatalities']])
print(ffm.groupby('Place').last()[['Date', 'ConfirmedCases', 'Fatalities']])


# %% [code]
if ffm[scl].isnull().sum().sum() == 0:
    out = full_bk[scl] * 0.7 + ffm[scl] * 0.3
else:
    print('using full-bk')
    out = full_bk[scl]


out = out[sub.columns.to_list()]
out.ForecastId = np.round(out.ForecastId, 0).astype(int) 
out = np.round(out, 2)
private = out
  


full_pred = pd.concat((private, public[~public.ForecastId.isin(private.ForecastId)]),ignore_index=True).sort_values('ForecastId')

# full_pred.to_csv('submission.csv', index=False)


# In[ ]:


full_pred['ForecastId'] = full_pred['ForecastId'].astype('int')
david_preds = test_orig.merge(full_pred, on='ForecastId')
david_p_c = david_preds.pivot(index='Area', columns='days', values='ConfirmedCases').sort_index()
david_p_f = david_preds.pivot(index='Area', columns='days', values='Fatalities').sort_index()
preds_c_david = np.log1p(david_p_c.values)
preds_f_david = np.log1p(david_p_f.values)
preds_c_david.shape, preds_f_david.shape


# In[ ]:




import matplotlib.pyplot as plt

#for _ in range(5):
plt.style.use(['default'])
fig = plt.figure(figsize = (15, 5))

idx = np.random.choice(N_AREAS)
print(AREAS[idx])

plt.plot(np.log(1+train_p_c.values[idx]), label=AREAS[idx], color='darkblue')
plt.plot(preds_c[idx], linestyle='--', color='darkblue', label = 'pdd pred cases')
plt.plot(np.pad(preds_c_david[idx],(START_PUBLIC,0)),label = 'david pred cases', linestyle='-.', color='darkred')
plt.legend()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

#for _ in range(5):
plt.style.use(['default'])
fig = plt.figure(figsize = (15, 5))

idx = np.random.choice(N_AREAS)
print(AREAS[idx])

plt.plot(np.log(1+train_p_f.values[idx]), label=AREAS[idx], color='darkblue')
plt.plot(preds_f[idx], linestyle='--', color='darkblue', label = 'pdd pred fat')
plt.plot(np.pad(preds_f_david[idx],(START_PUBLIC,0)),label = 'david pred fat', linestyle='-.', color='darkred')
plt.legend()
plt.show()


# In[ ]:




from sklearn.metrics import mean_squared_error

if True:
    val_len = TRAIN_N - START_PUBLIC
    m1s = []
    m2s = []
    for i in range(val_len):
        d = i + START_PUBLIC
        m1 = np.sqrt(mean_squared_error(np.log(1 + train_p_c_raw.values[:, d]), preds_c_david[:, i]))
        m2 = np.sqrt(mean_squared_error(np.log(1 + train_p_f_raw.values[:, d]), preds_f_david[:, i]))
        print(f"{d}: {(m1 + m2)/2:8.5f} [{m1:8.5f} {m2:8.5f}]")
        m1s += [m1]
        m2s += [m2]
    print()

    
    m1 = np.sqrt(mean_squared_error(np.log(1 + train_p_c_raw.values[:, START_PUBLIC:START_PUBLIC+val_len]).flatten(), preds_c_david[:, :val_len].flatten()))
    m2 = np.sqrt(mean_squared_error(np.log(1 + train_p_f_raw.values[:, START_PUBLIC:START_PUBLIC+val_len]).flatten(), preds_f_david[:, :val_len].flatten()))
    print(f"{(m1 + m2)/2:8.5f} [{m1:8.5f} {m2:8.5f}]")


# ## oscii model

# In[ ]:


import copy
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


FIRST_TEST = test_orig['Date'].apply(lambda x: x.dayofyear).min()


# In[ ]:


# day_before_valid = FIRST_TEST-1 # 3-11 day  before of validation
# day_before_public = FIRST_TEST-1 # 3-18 last day of train
# day_before_private = df_traintest['day'][pd.isna(df_traintest['ForecastId'])].max() # last day of train


# In[ ]:


def do_aggregation(df, col, mean_range):
    df_new = copy.deepcopy(df)
    col_new = '{}_({}-{})'.format(col, mean_range[0], mean_range[1])
    df_new[col_new] = 0
    tmp = df_new[col].rolling(mean_range[1]-mean_range[0]+1).mean()
    df_new[col_new][mean_range[0]:] = tmp[:-(mean_range[0])]
    df_new[col_new][pd.isna(df_new[col_new])] = 0
    return df_new[[col_new]].reset_index(drop=True)

def do_aggregations(df):
    df = pd.concat([df, do_aggregation(df, 'cases/day', [1,1]).reset_index(drop=True)], axis=1)
    df = pd.concat([df, do_aggregation(df, 'cases/day', [1,7]).reset_index(drop=True)], axis=1)
    df = pd.concat([df, do_aggregation(df, 'cases/day', [8,14]).reset_index(drop=True)], axis=1)
    df = pd.concat([df, do_aggregation(df, 'cases/day', [15,21]).reset_index(drop=True)], axis=1)
    df = pd.concat([df, do_aggregation(df, 'fatal/day', [1,1]).reset_index(drop=True)], axis=1)
    df = pd.concat([df, do_aggregation(df, 'fatal/day', [1,7]).reset_index(drop=True)], axis=1)
    df = pd.concat([df, do_aggregation(df, 'fatal/day', [8,14]).reset_index(drop=True)], axis=1)
    df = pd.concat([df, do_aggregation(df, 'fatal/day', [15,21]).reset_index(drop=True)], axis=1)
    for threshold in [1, 10, 100]:
        days_under_threshold = (df['ConfirmedCases']<threshold).sum()
        tmp = df['day'].values - 22 - days_under_threshold
        tmp[tmp<=0] = 0
        df['days_since_{}cases'.format(threshold)] = tmp

    for threshold in [1, 10, 100]:
        days_under_threshold = (df['Fatalities']<threshold).sum()
        tmp = df['day'].values - 22 - days_under_threshold
        tmp[tmp<=0] = 0
        df['days_since_{}fatal'.format(threshold)] = tmp

    # process China/Hubei
    if df['place_id'][0]=='China/Hubei':
        df['days_since_1cases'] += 35 # 2019/12/8
        df['days_since_10cases'] += 35-13 # 2019/12/8-2020/1/2 assume 2019/12/8+13
        df['days_since_100cases'] += 4 # 2020/1/18
        df['days_since_1fatal'] += 13 # 2020/1/9
    return df


# In[ ]:


def feature_engineering_oscii():

    # helper fucntions
    
    def fix_area(x):
        try:
            x_new = x['Country_Region'] + "/" + x['Province_State']
        except:
            x_new = x['Country_Region']
        return x_new
    
    def fix_area2(x):
        try:
            x_new = x['Country/Region'] + "/" + x['Province/State']
        except:
            x_new = x['Country/Region']
        return x_new


    
    def encode_label(df, col, freq_limit=0):
        df[col][pd.isna(df[col])] = 'nan'
        tmp = df[col].value_counts()
        cols = tmp.index.values
        freq = tmp.values
        num_cols = (freq>=freq_limit).sum()
        print("col: {}, num_cat: {}, num_reduced: {}".format(col, len(cols), num_cols))

        col_new = '{}_le'.format(col)
        df_new = pd.DataFrame(np.ones(len(df), np.int16)*(num_cols-1), columns=[col_new])
        for i, item in enumerate(cols[:num_cols]):
            df_new[col_new][df[col]==item] = i

        return df_new

    def get_df_le(df, col_index, col_cat):
        df_new = df[[col_index]]
        for col in col_cat:
            df_tmp = encode_label(df, col)
            df_new = pd.concat([df_new, df_tmp], axis=1)
        return df_new
    
    def to_float(x):
        x_new = 0
        try:
            x_new = float(x.replace(",", ""))
        except:
            x_new = np.nan
        return x_new
    
    df_train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
    df_test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
    df_traintest = pd.concat([df_train, df_test])
    
    print('process_date')
    df_traintest['Date'] = pd.to_datetime(df_traintest['Date'])
    df_traintest['day'] = df_traintest['Date'].apply(lambda x: x.dayofyear).astype(np.int16)
    day_min = df_traintest['day'].min()
    df_traintest['days'] = df_traintest['day'] - day_min
    
    df_traintest.loc[df_traintest['Province_State'].isnull(), 'Province_State'] = 'N/A'
    df_traintest['place_id'] = df_traintest['Country_Region'] + '_' + df_traintest['Province_State']
    
    #df_traintest['place_id'] = df_traintest.apply(lambda x: fix_area(x), axis=1)
    
    print('add lat and long')
    df_latlong = pd.read_csv("../input/smokingstats/df_Latlong.csv")
    
    df_latlong.loc[df_latlong['Province/State'].isnull(), 'Province_State'] = 'N/A'
    df_latlong['place_id'] = df_latlong['Country/Region'] + '_' + df_latlong['Province/State']
    
    # df_latlong['place_id'] = df_latlong.apply(lambda x: fix_area2(x), axis=1)
    df_latlong = df_latlong[df_latlong['place_id'].duplicated()==False]
    df_traintest = pd.merge(df_traintest, df_latlong[['place_id', 'Lat', 'Long']], on='place_id', how='left')
    
    places = np.sort(df_traintest['place_id'].unique())
    
    print('calc cases, fatalities per day')
    df_traintest2 = copy.deepcopy(df_traintest)
    df_traintest2['cases/day'] = 0
    df_traintest2['fatal/day'] = 0
    tmp_list = np.zeros(len(df_traintest2))
    for place in places:
        tmp = df_traintest2['ConfirmedCases'][df_traintest2['place_id']==place].values
        tmp[1:] -= tmp[:-1]
        df_traintest2['cases/day'][df_traintest2['place_id']==place] = tmp
        tmp = df_traintest2['Fatalities'][df_traintest2['place_id']==place].values
        tmp[1:] -= tmp[:-1]
        df_traintest2['fatal/day'][df_traintest2['place_id']==place] = tmp

    print('do agregation')
    df_traintest3 = []
    for place in places[:]:
        df_tmp = df_traintest2[df_traintest2['place_id']==place].reset_index(drop=True)
        df_tmp = do_aggregations(df_tmp)
        df_traintest3.append(df_tmp)
    df_traintest3 = pd.concat(df_traintest3).reset_index(drop=True)
    
    
    print('add smoking')
    df_smoking = pd.read_csv("../input/smokingstats/share-of-adults-who-smoke.csv")
    df_smoking_recent = df_smoking.sort_values('Year', ascending=False).reset_index(drop=True)
    df_smoking_recent = df_smoking_recent[df_smoking_recent['Entity'].duplicated()==False]
    df_smoking_recent['Country_Region'] = df_smoking_recent['Entity']
    df_smoking_recent['SmokingRate'] = df_smoking_recent['Smoking prevalence, total (ages 15+) (% of adults)']
    df_traintest4 = pd.merge(df_traintest3, df_smoking_recent[['Country_Region', 'SmokingRate']], on='Country_Region', how='left')
    SmokingRate = df_smoking_recent['SmokingRate'][df_smoking_recent['Entity']=='World'].values[0]
    # print("Smoking rate of the world: {:.6f}".format(SmokingRate))
    df_traintest4['SmokingRate'][pd.isna(df_traintest4['SmokingRate'])] = SmokingRate
    
    print('add data from World Economic Outlook Database')
    # https://www.imf.org/external/pubs/ft/weo/2017/01/weodata/index.aspx
    df_weo = pd.read_csv("../input/smokingstats/WEO.csv")
    subs  = df_weo['Subject Descriptor'].unique()[:-1]
    df_weo_agg = df_weo[['Country']][df_weo['Country'].duplicated()==False].reset_index(drop=True)
    for sub in subs[:]:
        df_tmp = df_weo[['Country', '2019']][df_weo['Subject Descriptor']==sub].reset_index(drop=True)
        df_tmp = df_tmp[df_tmp['Country'].duplicated()==False].reset_index(drop=True)
        df_tmp.columns = ['Country', sub]
        df_weo_agg = df_weo_agg.merge(df_tmp, on='Country', how='left')
    df_weo_agg.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df_weo_agg.columns]
    df_weo_agg.columns
    df_weo_agg['Country_Region'] = df_weo_agg['Country']
    df_traintest5 = pd.merge(df_traintest4, df_weo_agg, on='Country_Region', how='left')
    
    print('add Life expectancy')
    # Life expectancy at birth obtained from http://hdr.undp.org/en/data
    df_life = pd.read_csv("../input/smokingstats/Life expectancy at birth.csv")
    tmp = df_life.iloc[:,1].values.tolist()
    df_life = df_life[['Country', '2018']]
    def func(x):
        x_new = 0
        try:
            x_new = float(x.replace(",", ""))
        except:
    #         print(x)
            x_new = np.nan
        return x_new

    df_life['2018'] = df_life['2018'].apply(lambda x: func(x))
    df_life = df_life[['Country', '2018']]
    df_life.columns = ['Country_Region', 'LifeExpectancy']
    df_traintest6 = pd.merge(df_traintest5, df_life, on='Country_Region', how='left')
    
    print("add additional info from countryinfo dataset")
    df_country = pd.read_csv("../input/countryinfo/covid19countryinfo.csv")
    df_country['Country_Region'] = df_country['country']
    df_country = df_country[df_country['country'].duplicated()==False]
    df_traintest7 = pd.merge(df_traintest6, 
                             df_country.drop(['tests', 'testpop', 'country'], axis=1), 
                             on=['Country_Region',], how='left')
    


    df_traintest7['id'] = np.arange(len(df_traintest7))
    df_le = get_df_le(df_traintest7, 'id', ['Country_Region', 'Province_State'])
    df_traintest8 = pd.merge(df_traintest7, df_le, on='id', how='left')

    
    
    print('covert object type to float')

    cols = [
        'Gross_domestic_product__constant_prices', 
        'Gross_domestic_product__current_prices', 
        'Gross_domestic_product__deflator', 
        'Gross_domestic_product_per_capita__constant_prices', 
        'Gross_domestic_product_per_capita__current_prices', 
        'Output_gap_in_percent_of_potential_GDP', 
        'Gross_domestic_product_based_on_purchasing_power_parity__PPP__valuation_of_country_GDP', 
        'Gross_domestic_product_based_on_purchasing_power_parity__PPP__per_capita_GDP', 
        'Gross_domestic_product_based_on_purchasing_power_parity__PPP__share_of_world_total', 
        'Implied_PPP_conversion_rate', 'Total_investment', 
        'Gross_national_savings', 'Inflation__average_consumer_prices', 
        'Inflation__end_of_period_consumer_prices', 
        'Six_month_London_interbank_offered_rate__LIBOR_', 
        'Volume_of_imports_of_goods_and_services', 
        'Volume_of_Imports_of_goods', 
        'Volume_of_exports_of_goods_and_services', 
        'Volume_of_exports_of_goods', 'Unemployment_rate', 'Employment', 'Population', 
        'General_government_revenue', 'General_government_total_expenditure', 
        'General_government_net_lending_borrowing', 'General_government_structural_balance', 
        'General_government_primary_net_lending_borrowing', 'General_government_net_debt', 
        'General_government_gross_debt', 'Gross_domestic_product_corresponding_to_fiscal_year__current_prices', 
        'Current_account_balance', 'pop'
    ]
    df_traintest8['cases/day'] = df_traintest8['cases/day'].astype(np.float)
    df_traintest8['fatal/day'] = df_traintest8['fatal/day'].astype(np.float)   
    for col in cols:
        df_traintest8[col] = df_traintest8[col].apply(lambda x: to_float(x))  
    # print(df_traintest8['pop'].dtype)
    
    return df_traintest8


# In[ ]:


df_traintest = feature_engineering_oscii()
df_traintest.shape


# In[ ]:


# day_before_valid = FIRST_TEST -1 # 3-11 day  before of validation
# day_before_public = FIRST_TEST -1 # 3-18 last day of train
# day_before_launch = 85 # 4-1 last day before launch


# In[ ]:


def calc_score(y_true, y_pred):
    y_true[y_true<0] = 0
    score = mean_squared_error(np.log(y_true.clip(0, 1e10)+1), np.log(y_pred[:]+1))**0.5
    return score


# In[ ]:


# train model to predict fatalities/day
# params
SEED = 42
params = {'num_leaves': 8,
          'min_data_in_leaf': 5,  # 42,
          'objective': 'regression',
          'max_depth': 8,
          'learning_rate': 0.02,
          'boosting': 'gbdt',
          'bagging_freq': 5,  # 5
          'bagging_fraction': 0.8,  # 0.5,
          'feature_fraction': 0.8201,
          'bagging_seed': SEED,
          'reg_alpha': 1,  # 1.728910519108444,
          'reg_lambda': 4.9847051755586085,
          'random_state': SEED,
          'metric': 'mse',
          'verbosity': 100,
          'min_gain_to_split': 0.02,  # 0.01077313523861969,
          'min_child_weight': 5,  # 19.428902804238373,
          'num_threads': 6,
          }


# In[ ]:


# train model to predict fatalities/day
# features are selected manually based on valid score
col_target = 'fatal/day'
col_var = [
    'Lat', 'Long',
#     'days_since_1cases', 
#     'days_since_10cases', 
#     'days_since_100cases',
#     'days_since_1fatal', 
#     'days_since_10fatal', 'days_since_100fatal',
#     'days_since_1recov',
#     'days_since_10recov', 'days_since_100recov', 
    'cases/day_(1-1)', 
    'cases/day_(1-7)', 
#     'cases/day_(8-14)',  
#     'cases/day_(15-21)', 
    
#     'fatal/day_(1-1)', 
    'fatal/day_(1-7)', 
    'fatal/day_(8-14)', 
    'fatal/day_(15-21)', 
    'SmokingRate',
#     'Gross_domestic_product__constant_prices',
#     'Gross_domestic_product__current_prices',
#     'Gross_domestic_product__deflator',
#     'Gross_domestic_product_per_capita__constant_prices',
#     'Gross_domestic_product_per_capita__current_prices',
#     'Output_gap_in_percent_of_potential_GDP',
#     'Gross_domestic_product_based_on_purchasing_power_parity__PPP__valuation_of_country_GDP',
#     'Gross_domestic_product_based_on_purchasing_power_parity__PPP__per_capita_GDP',
#     'Gross_domestic_product_based_on_purchasing_power_parity__PPP__share_of_world_total',
#     'Implied_PPP_conversion_rate', 'Total_investment',
#     'Gross_national_savings', 'Inflation__average_consumer_prices',
#     'Inflation__end_of_period_consumer_prices',
#     'Six_month_London_interbank_offered_rate__LIBOR_',
#     'Volume_of_imports_of_goods_and_services', 'Volume_of_Imports_of_goods',
#     'Volume_of_exports_of_goods_and_services', 'Volume_of_exports_of_goods',
#     'Unemployment_rate', 
#     'Employment', 'Population',
#     'General_government_revenue', 'General_government_total_expenditure',
#     'General_government_net_lending_borrowing',
#     'General_government_structural_balance',
#     'General_government_primary_net_lending_borrowing',
#     'General_government_net_debt', 'General_government_gross_debt',
#     'Gross_domestic_product_corresponding_to_fiscal_year__current_prices',
#     'Current_account_balance', 
#     'LifeExpectancy',
#     'pop',
    'density', 
#     'medianage', 
#     'urbanpop', 
#     'hospibed', 'smokers', 
]
col_cat = []
df_train = df_traintest[(pd.isna(df_traintest['ForecastId'])) & (df_traintest['days']<TRAIN_N)]
df_valid = df_traintest[(pd.isna(df_traintest['ForecastId'])) & (df_traintest['days']<TRAIN_N)]
# df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]
X_train = df_train[col_var]
X_valid = df_valid[col_var]
y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)
y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)
valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)
num_round = 340 
model = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],
                  verbose_eval=100,
                  early_stopping_rounds=150,)

best_itr = model.best_iteration


# In[ ]:


# y_true = df_valid['fatal/day'].values
# y_pred = np.exp(model.predict(X_valid))-1
# score = calc_score(y_true, y_pred)
# print("{:.6f}".format(score))


# In[ ]:


# train model to predict fatalities/day
col_target2 = 'cases/day'
col_var2 = [
    'Lat', 'Long',
#     'days_since_1cases', 
    'days_since_10cases', #selected
#     'days_since_100cases',
#     'days_since_1fatal', 
#     'days_since_10fatal',
#     'days_since_100fatal',
#     'days_since_1recov',
#     'days_since_10recov', 'days_since_100recov', 
    'cases/day_(1-1)', 
    'cases/day_(1-7)', 
    'cases/day_(8-14)',  
    'cases/day_(15-21)', 
    
#     'fatal/day_(1-1)', 
#     'fatal/day_(1-7)', 
#     'fatal/day_(8-14)', 
#     'fatal/day_(15-21)', 
#     'recov/day_(1-1)', 'recov/day_(1-7)', 
#     'recov/day_(8-14)',  'recov/day_(15-21)',
#     'active_(1-1)', 
#     'active_(1-7)', 
#     'active_(8-14)',  'active_(15-21)', 
#     'SmokingRate',
#     'Gross_domestic_product__constant_prices',
#     'Gross_domestic_product__current_prices',
#     'Gross_domestic_product__deflator',
#     'Gross_domestic_product_per_capita__constant_prices',
#     'Gross_domestic_product_per_capita__current_prices',
#     'Output_gap_in_percent_of_potential_GDP',
#     'Gross_domestic_product_based_on_purchasing_power_parity__PPP__valuation_of_country_GDP',
#     'Gross_domestic_product_based_on_purchasing_power_parity__PPP__per_capita_GDP',
#     'Gross_domestic_product_based_on_purchasing_power_parity__PPP__share_of_world_total',
#     'Implied_PPP_conversion_rate', 'Total_investment',
#     'Gross_national_savings', 'Inflation__average_consumer_prices',
#     'Inflation__end_of_period_consumer_prices',
#     'Six_month_London_interbank_offered_rate__LIBOR_',
#     'Volume_of_imports_of_goods_and_services', 'Volume_of_Imports_of_goods',
#     'Volume_of_exports_of_goods_and_services', 'Volume_of_exports_of_goods',
#     'Unemployment_rate', 
#     'Employment', 
#     'Population',
#     'General_government_revenue', 'General_government_total_expenditure',
#     'General_government_net_lending_borrowing',
#     'General_government_structural_balance',
#     'General_government_primary_net_lending_borrowing',
#     'General_government_net_debt', 'General_government_gross_debt',
#     'Gross_domestic_product_corresponding_to_fiscal_year__current_prices',
#     'Current_account_balance', 
#     'LifeExpectancy',
#     'pop',
#     'density', 
#     'medianage', 
#     'urbanpop', 
#     'hospibed', 'smokers', 
]


# In[ ]:


df_traintest[df_traintest['days']<TRAIN_N].Date.max()


# In[ ]:


# train model to predict cases/day
df_train = df_traintest[(pd.isna(df_traintest['ForecastId'])) & (df_traintest['days']<TRAIN_N)]
df_valid = df_traintest[(pd.isna(df_traintest['ForecastId'])) & (df_traintest['days']<TRAIN_N)]
X_train = df_train[col_var2]
X_valid = df_valid[col_var2]
y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)
y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)
valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)
model2 = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],
                  verbose_eval=100,
                  early_stopping_rounds=150,)
best_itr2 = model2.best_iteration


# In[ ]:


# y_true = df_valid['cases/day'].values
# y_pred = np.exp(model2.predict(X_valid))-1
# score = calc_score(y_true, y_pred)
# print("{:.6f}".format(score))


# In[ ]:


places = AREAS.copy()


# In[ ]:


# remove overlap for public LB prediction

df_tmp = df_traintest[((df_traintest['days']<TRAIN_N)  & (pd.isna(df_traintest['ForecastId'])))  | ((TRAIN_N<=df_traintest['days']) & (pd.isna(df_traintest['ForecastId'])==False))].reset_index(drop=True)
df_tmp = df_tmp.drop([
    'cases/day_(1-1)', 'cases/day_(1-7)', 'cases/day_(8-14)', 'cases/day_(15-21)', 
    'fatal/day_(1-1)', 'fatal/day_(1-7)', 'fatal/day_(8-14)', 'fatal/day_(15-21)',
    'days_since_1cases', 'days_since_10cases', 'days_since_100cases',
    'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',
                               ],  axis=1)
df_traintest9 = []
for i, place in tqdm(enumerate(places[:])):
    df_tmp2 = df_tmp[df_tmp['place_id']==place].reset_index(drop=True)
    df_tmp2 = do_aggregations(df_tmp2)
    df_traintest9.append(df_tmp2)
df_traintest9 = pd.concat(df_traintest9).reset_index(drop=True)
#df_traintest9[df_traintest9['days']>TRAIN_N-2].head()


# In[ ]:


# predict test data in public
# predict the cases and fatatilites one day at a time and use the predicts as next day's feature recursively.
df_preds = []
for i, place in tqdm(enumerate(places[:])):
    df_interest = copy.deepcopy(df_traintest9[df_traintest9['place_id']==place].reset_index(drop=True))
    df_interest['cases/day'][(pd.isna(df_interest['ForecastId']))==False] = -1
    df_interest['fatal/day'][(pd.isna(df_interest['ForecastId']))==False] = -1
    len_known = (df_interest['days']<TRAIN_N).sum()
    #len_unknown = (TRAIN_N<=df_interest['day']).sum()
    len_unknown = df_interest.day.nunique() - len_known
    for j in range(len_unknown): # use predicted cases and fatal for next days' prediction
        X_valid = df_interest[col_var].iloc[j+len_known]
        X_valid2 = df_interest[col_var2].iloc[j+len_known]
        pred_f = model.predict(X_valid)
        pred_c = model2.predict(X_valid2)
        pred_c = (np.exp(pred_c)-1).clip(0, 1e10)
        pred_f = (np.exp(pred_f)-1).clip(0, 1e10)
        df_interest['fatal/day'][j+len_known] = pred_f
        df_interest['cases/day'][j+len_known] = pred_c
        df_interest['Fatalities'][j+len_known] = df_interest['Fatalities'][j+len_known-1] + pred_f
        df_interest['ConfirmedCases'][j+len_known] = df_interest['ConfirmedCases'][j+len_known-1] + pred_c
#         print(df_interest['ConfirmedCases'][j+len_known-1], df_interest['ConfirmedCases'][j+len_known], pred_c)
        df_interest = df_interest.drop([
            'cases/day_(1-1)', 'cases/day_(1-7)', 'cases/day_(8-14)', 'cases/day_(15-21)', 
            'fatal/day_(1-1)', 'fatal/day_(1-7)', 'fatal/day_(8-14)', 'fatal/day_(15-21)',
            'days_since_1cases', 'days_since_10cases', 'days_since_100cases',
            'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',

                                       ],  axis=1)
        df_interest = do_aggregations(df_interest)
    if (i+1)%10==0:
        print("{:3d}/{}  {}, len known: {}, len unknown: {}".format(i+1, len(places), place, len_known, len_unknown), df_interest.shape)
    df_interest['fatal_pred'] = np.cumsum(df_interest['fatal/day'].values)
    df_interest['cases_pred'] = np.cumsum(df_interest['cases/day'].values)
    df_preds.append(df_interest)
df_preds = pd.concat(df_preds)


# In[ ]:


df_preds.head()


# In[ ]:


df_preds.shape


# In[ ]:


# df_preds['cases/day']


# In[ ]:


p_f_oscii = df_preds.pivot(index='place_id', columns='days', values='fatal_pred').sort_index()
p_c_oscii = df_preds.pivot(index='place_id', columns='days', values='cases_pred').sort_index()
p_c_oscii


# In[ ]:


preds_f_oscii = np.log1p(p_f_oscii.values[:].copy())
preds_c_oscii = np.log1p(p_c_oscii.values[:].copy())
preds_f_oscii.shape, preds_c_oscii.shape


# In[ ]:


START_PUBLIC, TRAIN_N


# In[ ]:


if True:
    val_len = TRAIN_N - START_PUBLIC
    #val_len = 12
    m1s = []
    m2s = []
    for i in range(val_len):
        d = i + START_PUBLIC
        m1 = np.sqrt(mean_squared_error(np.log(1 + train_p_c_raw.values[:, d]), preds_c_oscii[:, d]))
        m2 = np.sqrt(mean_squared_error(np.log(1 + train_p_f_raw.values[:, d]), preds_f_oscii[:, d]))
        print(f"{d}: {(m1 + m2)/2:8.5f} [{m1:8.5f} {m2:8.5f}]")
        m1s += [m1]
        m2s += [m2]
    print()

    
    m1 = np.sqrt(mean_squared_error(np.log(1 + train_p_c_raw.values[:, START_PUBLIC:START_PUBLIC+val_len]).flatten(), preds_c_oscii[:, START_PUBLIC:START_PUBLIC+val_len].flatten()))
    m2 = np.sqrt(mean_squared_error(np.log(1 + train_p_f_raw.values[:, START_PUBLIC:START_PUBLIC+val_len]).flatten(), preds_f_oscii[:, START_PUBLIC:START_PUBLIC+val_len].flatten()))
    print(f"{(m1 + m2)/2:8.5f} [{m1:8.5f} {m2:8.5f}]")


# In[ ]:


preds_c_oscii.shape, preds_c.shape, p_c_vopani.shape, preds_c_david.shape


# ## BLEND

# In[ ]:


# preds_c_blend = np.log1p(np.average([np.expm1(p_c_beluga[:,64:107]),np.expm1(preds_c_cpmp[:]),np.expm1(preds_c[:,64:107])],axis=0, weights=[2,1,2]))
# preds_f_blend = np.log1p(np.average([np.expm1(p_f_beluga[:,64:107]),np.expm1(preds_f_cpmp[:]),np.expm1(preds_f[:,64:107])],axis=0, weights=[2,1,2]))


# In[ ]:


preds_c_david2 = np.pad(preds_c_david,[(0,0),(START_PUBLIC,0)])
preds_f_david2 = np.pad(preds_f_david,[(0,0),(START_PUBLIC,0)])


# In[ ]:


preds_c_oscii.shape, preds_c.shape, p_c_vopani.shape, preds_c_david2.shape


# In[ ]:


# p_c_vopani
# p_f_vopani

preds_c_blend = np.log1p(np.average([np.expm1(preds_c_oscii[:,:]),np.expm1(preds_c[:,:preds_c_oscii.shape[1]]),np.expm1(preds_c_david2),np.expm1(p_c_vopani_mad)],axis=0, weights=[0,4,2,1]))
preds_f_blend = np.log1p(np.average([np.expm1(preds_f_oscii[:,:]),np.expm1(preds_f[:,:preds_c_oscii.shape[1]]),np.expm1(preds_f_david2),np.expm1(p_f_vopani_mad)],axis=0, weights=[0,8,1,1]))

#preds_f_blend = np.log1p(np.average([np.expm1(preds_f_oscii[:,:]),np.expm1(preds_f[:,:preds_c_oscii.shape[1]])],axis=0, weights=[1,1]))


# In[ ]:


if True:
    val_len = TRAIN_N - START_PUBLIC
    #val_len = 12
    m1s = []
    m2s = []
    for i in range(val_len):
        d = i + START_PUBLIC
        m1 = np.sqrt(mean_squared_error(np.log(1 + train_p_c_raw.values[:, d]), preds_c_blend[:, d]))
        m2 = np.sqrt(mean_squared_error(np.log(1 + train_p_f_raw.values[:, d]), preds_f_blend[:, d]))
        print(f"{d}: {(m1 + m2)/2:8.5f} [{m1:8.5f} {m2:8.5f}]")
        m1s += [m1]
        m2s += [m2]
    print()

    
    m1 = np.sqrt(mean_squared_error(np.log(1 + train_p_c_raw.values[:, START_PUBLIC:START_PUBLIC+val_len]).flatten(), preds_c_blend[:, START_PUBLIC:START_PUBLIC+val_len].flatten()))
    m2 = np.sqrt(mean_squared_error(np.log(1 + train_p_f_raw.values[:, START_PUBLIC:START_PUBLIC+val_len]).flatten(), preds_f_blend[:, START_PUBLIC:START_PUBLIC+val_len].flatten()))
    print(f"{(m1 + m2)/2:8.5f} [{m1:8.5f} {m2:8.5f}]")


# In[ ]:


import matplotlib.pyplot as plt

for _ in range(5):
    plt.style.use(['default'])
    fig = plt.figure(figsize = (15, 5))

    idx = np.random.choice(N_AREAS)
    print(AREAS[idx])#, places[idx])

    plt.plot(np.log(1+train_p_f.values[idx]), label=AREAS[idx], color='darkblue')
    plt.plot(preds_f[idx], linestyle='--', color='darkblue', label = 'pdd fat ')
    plt.plot(p_f_vopani[idx],label = 'vopani fat ', linestyle='-.', color='purple')
    plt.plot(preds_f_oscii[idx],label = 'oscii fat ', linestyle='-.', color='red')
    plt.plot(preds_f_david2[idx],label = 'david fat ', linestyle='-.', color='orange')
    plt.plot(preds_f_blend[idx],label = 'blend fat ', linestyle='-.', color='darkgreen')
    plt.legend()
    plt.show()


# In[ ]:


import matplotlib.pyplot as plt

for _ in range(5):
    plt.style.use(['default'])
    fig = plt.figure(figsize = (15, 5))

    idx = np.random.choice(N_AREAS)
    print(AREAS[idx], places[idx])

    plt.plot(np.log(1+train_p_c.values[idx]), label=AREAS[idx], color='darkblue')
    plt.plot(preds_c[idx], linestyle='--', color='darkblue', label = 'pdd cases ')
    plt.plot(p_c_vopani[idx],label = 'vopani cases ', linestyle='-.', color='purple')
    plt.plot(preds_c_oscii[idx],label = 'oscii cases ', linestyle='-.', color='red')
    plt.plot(preds_c_david2[idx],label = 'david cases ', linestyle='-.', color='orange')
    plt.plot(preds_c_blend[idx],label = 'blend cases ', linestyle='-.', color='darkgreen')
    plt.legend()
    plt.show()


# In[ ]:


EU_COUNTRIES = ['Austria', 'Italy', 'Belgium', 'Latvia', 'Bulgaria', 'Lithuania', 'Croatia', 'Luxembourg', 'Cyprus', 'Malta', 'Czechia', 
                'Netherlands', 'Denmark', 'Poland', 'Estonia', 'Portugal', 'Finland', 'Romania', 'France', 'Slovakia', 'Germany', 'Slovenia', 
                'Greece', 'Spain', 'Hungary', 'Sweden', 'Ireland']
EUROPE_OTHER = ['Albania', 'Andorra', 'Bosnia and Herzegovina', 'Liechtenstein', 'Monaco', 'Montenegro', 'North Macedonia',
                'Norway', 'San Marino', 'Serbia', 'Switzerland', 'Turkey', 'United Kingdom']
AFRICA = ['Algeria', 'Burkina Faso', 'Cameroon', 'Congo (Kinshasa)', "Cote d'Ivoire", 'Egypt', 'Ghana', 'Kenya', 'Madagascar',
                'Morocco', 'Nigeria', 'Rwanda', 'Senegal', 'South Africa', 'Togo', 'Tunisia', 'Uganda', 'Zambia']
NORTH_AMERICA = ['US', 'Canada', 'Mexico']
SOUTH_AMERICA = ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Paraguay', 'Peru', 'Uruguay', 'Venezuela']
MIDDLE_EAST = ['Afghanistan', 'Bahrain', 'Iran', 'Iraq', 'Israel', 'Jordan', 'Kuwait', 'Lebanon', 'Oman', 'Qatar', 'Saudi Arabia', 'United Arab Emirates']
ASIA = ['Bangladesh', 'Brunei', 'Cambodia', 'India', 'Indonesia', 'Japan', 'Kazakhstan', 'Korea, South', 'Kyrgyzstan', 'Malaysia',
                'Pakistan', 'Singapore', 'Sri Lanka', 'Taiwan*', 'Thailand', 'Uzbekistan', 'Vietnam']


# In[ ]:


non_china_mask = np.array(['China' not in a for a in AREAS]).astype(bool)
non_china_mask.shape


# In[ ]:


preds_c2 = preds_c.copy()
preds_f2 = preds_f.copy()
preds_c2[non_china_mask,:114] = preds_c_blend[non_china_mask]
preds_f2[non_china_mask,:114] = preds_f_blend[non_china_mask]


# In[ ]:


import matplotlib.pyplot as plt

def plt1(ar, ar2, ax, col='darkblue', linew=0.2):
    ax.plot(ar2, linestyle='--', linewidth=linew/2, color=col)
    ax.plot(np.log(1+ar), linewidth=linew, color=col)

plt.style.use(['default'])
fig, axs = plt.subplots(3, 2, figsize=(18, 15), sharey=True)

X = train_p_c.values
#X = train_p_f.values

for ar in range(X.shape[0]):
    
    temp = X[ar]
    temp2 = preds_c2[ar]
    if 'China' in AREAS[ar]:
        plt1(temp, temp2, axs[0,0])
    elif AREAS[ar].split('_')[0] in NORTH_AMERICA:
        plt1(temp, temp2, axs[0,1])
    elif AREAS[ar].split('_')[0] in EU_COUNTRIES + EUROPE_OTHER:
        plt1(temp, temp2, axs[1,0])
    elif AREAS[ar].split('_')[0] in SOUTH_AMERICA + AFRICA:
        plt1(temp, temp2, axs[1,1])
    elif AREAS[ar].split('_')[0] in MIDDLE_EAST + ASIA:
        plt1(temp, temp2, axs[2,0])
    else:
        plt1(temp, temp2, axs[2,1])

print("Confirmed Cases")
axs[0,0].set_title('China')
axs[0,1].set_title('North America')
axs[1,0].set_title('Europe')
axs[1,1].set_title('Africa + South America')
axs[2,0].set_title('Asia + Middle East')
axs[2,1].set_title('Other')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

def plt1(ar, ar2, ax, col='darkblue', linew=0.2):
    ax.plot(ar2, linestyle='--', linewidth=linew/2, color=col)
    ax.plot(np.log(1+ar), linewidth=linew, color=col)

plt.style.use(['default'])
fig, axs = plt.subplots(3, 2, figsize=(18, 15), sharey=True)

#X = train_p_c.values
X = train_p_f.values

for ar in range(X.shape[0]):
    
    temp = X[ar]
    temp2 = preds_f2[ar]
    if 'China' in AREAS[ar]:
        plt1(temp, temp2, axs[0,0])
    elif AREAS[ar].split('_')[0] in NORTH_AMERICA:
        plt1(temp, temp2, axs[0,1])
    elif AREAS[ar].split('_')[0] in EU_COUNTRIES + EUROPE_OTHER:
        plt1(temp, temp2, axs[1,0])
    elif AREAS[ar].split('_')[0] in SOUTH_AMERICA + AFRICA:
        plt1(temp, temp2, axs[1,1])
    elif AREAS[ar].split('_')[0] in MIDDLE_EAST + ASIA:
        plt1(temp, temp2, axs[2,0])
    else:
        plt1(temp, temp2, axs[2,1])

print("Fatalities")
axs[0,0].set_title('China')
axs[0,1].set_title('North America')
axs[1,0].set_title('Europe')
axs[1,1].set_title('Africa + South America')
axs[2,0].set_title('Asia + Middle East')
axs[2,1].set_title('Other')
plt.show()


# In[ ]:


preds_c.shape, preds_c_blend.shape


# In[ ]:


# preds_c2 = preds_c.copy()
# preds_f2 = preds_f.copy()
# preds_c2[:,64:107] = preds_c_blend
# preds_f2[:,64:107] = preds_f_blend


# In[ ]:



temp = pd.DataFrame(np.clip(np.exp(preds_c2) - 1, 0, None))
temp['Area'] = AREAS
temp = temp.melt(id_vars='Area', var_name='days', value_name="ConfirmedCases")

test = test_orig.merge(temp, how='left', left_on=['Area', 'days'], right_on=['Area', 'days'])

temp = pd.DataFrame(np.clip(np.exp(preds_f2) - 1, 0, None))
temp['Area'] = AREAS
temp = temp.melt(id_vars='Area', var_name='days', value_name="Fatalities")

test = test.merge(temp, how='left', left_on=['Area', 'days'], right_on=['Area', 'days'])
test.head()


# In[ ]:


test.to_csv("submission.csv", index=False, columns=["ForecastId", "ConfirmedCases", "Fatalities"])


# In[ ]:


test.days.nunique()


# In[ ]:


for i, rec in test.groupby('Area').last().sort_values("ConfirmedCases", ascending=False).iterrows():
    print(f"{rec['ConfirmedCases']:10.1f} {rec['Fatalities']:10.1f}  {rec['Country_Region']}, {rec['Province_State']}")


# In[ ]:


print(f"{test.groupby('Area')['ConfirmedCases'].last().sum():10.1f}")
print(f"{test.groupby('Area')['Fatalities'].last().sum():10.1f}")


# In[ ]:


test_p_c = test.pivot(index='Area', columns='days', values='ConfirmedCases').sort_index().values
test_p_f = test.pivot(index='Area', columns='days', values='Fatalities').sort_index().values
dates = test.Date.dt.strftime('%d.%m.%Y').unique()


# In[ ]:


print("Confirmed Cases")
for i in [7,14,21,28,35,42]:
    print(f'week{i//7-1}  ', dates[i],  f'   {round(test_p_c[:,i].sum(),0):,}')


# In[ ]:


print("Fatalities")
for i in [7,14,21,28,35,42]:
    print(f'week{i//7-1}  ', dates[i],  f'   {round(test_p_f[:,i].sum(),0):,}', )


# In[ ]:





# In[ ]:




