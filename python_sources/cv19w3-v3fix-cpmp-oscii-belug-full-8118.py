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

path = '../input/covid19-global-forecasting-week-3/'
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

print('train Date max',train['Date'].max())
print('test Date min',test['Date'].min())
print('train days max', train['days'].max())
N_AREAS = train['Area'].nunique()
AREAS = np.sort(train['Area'].unique())
START_PUBLIC = test['days'].min()
print('public LB start day', START_PUBLIC)
print(' ')


TRAIN_N = 77
print(train[train['days'] < TRAIN_N]['Date'].max())
print(train[train['days'] >= TRAIN_N]['Date'].min())
print(train[train['days'] >= TRAIN_N]['Date'].max())
train.head()

test_orig = test.copy()


# In[ ]:


print('test Date max',test['days'].max())


# In[ ]:


106-77


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
    "last N days": 15,
    "growth rate default": 0.2,
    "growth rate max": 0.3,
    "growth rate factor max": -0.1,
    "growth rate factor": -0.3,
    "growth rate factor factor": 0.01,
}
#x = train_p_c[train_p_c.index=="Austria_N/A"]

x = train_p_c

preds_c = run_c(params, np.log(1+x.values)[:,:TRAIN_N])
#eval1(np.log(1+x).values, preds_c)


# In[ ]:




for i in range(N_AREAS):
    if 'China' in AREAS[i] and preds_c[i, TRAIN_N-1] < np.log(31):
        preds_c[i, TRAIN_N:] = preds_c[i, TRAIN_N-1]



# In[ ]:


def lin_w(sz):
    res = np.linspace(0, 1, sz+1, endpoint=False)[1:]
    return np.append(res, np.append([1], res[::-1]))


def run_f(params, X_c, X_f, X_f_r, test_size=50):
  
    X_f_r = np.array(np.ma.mean(np.ma.masked_outside(X_f_r, 0.03, 0.5)[:,:], axis=1))
    X_f_r = np.clip(X_f_r, params['fatality_rate_lower'], params['fatality_rate_upper'])
    #print(X_f_r)
    
    X_c = np.clip(np.exp(X_c)-1, 0, None)
    preds = X_f.copy()
    #print(preds.shape)
    
    train_size = X_f.shape[1] - 1
    for i in range(test_size):
        
        t_lag = train_size+i-params['length']
        t_wsize = 5
        d = np.diff(X_c, axis=1)[:, t_lag-t_wsize:t_lag+1+t_wsize]
#         w = np.arange(d.shape[1])[::-1]+1
#         w = w**1
#         w = w / np.sum(w)
        delta = np.average(d, axis=1)
        #delta = np.average(np.diff(X_c, axis=1)[:, t_lag-t_wsize:t_lag+1+t_wsize], axis=1, weights=lin_w(t_wsize))
        
        delta = params['absolute growth'] + delta * X_f_r
        
        preds = np.hstack((preds, preds[:, -1].reshape(-1,1) + delta.reshape(-1,1)))

    return preds

params = {
    "length": 7,
    "absolute growth": 0.02,
    "fatality_rate_lower": 0.02,
    "fatality_rate_upper": 0.3,
}

preds_f_1 = run_f(params, preds_c, X_f, f_rate.values[:,:TRAIN_N])
preds_f_1 = np.log(1+preds_f_1)


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


# In[ ]:


preds_f_2.shape


# In[ ]:


preds_f = np.mean([preds_f_1, preds_f_2], axis=0)


# In[ ]:


from sklearn.metrics import mean_squared_error

if False:
    val_len = train_p_c.values.shape[1] - TRAIN_N
    
    for i in range(val_len):
        d = i + TRAIN_N
        m1 = np.sqrt(mean_squared_error(np.log(1 + train_p_c_raw.values[:, d]), preds_c[:, d]))
        m2 = np.sqrt(mean_squared_error(np.log(1 + train_p_f_raw.values[:, d]), preds_f[:, d]))
        print(f"{d}: {(m1 + m2)/2:8.5f} [{m1:8.5f} {m2:8.5f}]")

    print()

    m1 = np.sqrt(mean_squared_error(np.log(1 + train_p_c_raw.values[:, TRAIN_N:TRAIN_N+val_len]).flatten(), preds_c[:, TRAIN_N:TRAIN_N+val_len].flatten()))
    m2 = np.sqrt(mean_squared_error(np.log(1 + train_p_f_raw.values[:, TRAIN_N:TRAIN_N+val_len]).flatten(), preds_f[:, TRAIN_N:TRAIN_N+val_len].flatten()))
    print(f"{(m1 + m2)/2:8.5f} [{m1:8.5f} {m2:8.5f}]")


# In[ ]:





# ## cpmp model

# In[ ]:


import gc
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge

def get_cpmp_sub(save_oof=False, save_public_test=False):
    train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')
    train['Province_State'].fillna('', inplace=True)
    train['Date'] = pd.to_datetime(train['Date'])
    train['day'] = train.Date.dt.dayofyear
#     train = train[train.day < 86]
    train['geo'] = ['_'.join(x) for x in zip(train['Country_Region'], train['Province_State'])]
    train

    test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')
    test['Province_State'].fillna('', inplace=True)
    test['Date'] = pd.to_datetime(test['Date'])
    test['day'] = test.Date.dt.dayofyear
    test['geo'] = ['_'.join(x) for x in zip(test['Country_Region'], test['Province_State'])]
    test

    day_min = train['day'].min()
    train['day'] -= day_min
    test['day'] -= day_min
    train = train[train.day < TRAIN_N]
    
    min_test_val_day = test.day.min()
    max_test_val_day = train.day.max()
    max_test_day = test.day.max()
    num_days = max_test_day + 1

    min_test_val_day, max_test_val_day, num_days

    train['ForecastId'] = -1
    test['Id'] = -1
    test['ConfirmedCases'] = 0
    test['Fatalities'] = 0

    debug = False

    data = pd.concat([train,
                      test[test.day > max_test_val_day][train.columns]
                     ]).reset_index(drop=True)
    if debug:
        data = data[data['geo'] >= 'France_'].reset_index(drop=True)
    #del train, test
    gc.collect()

    dates = data[data['geo'] == 'France_'].Date.values

    if 0:
        gr = data.groupby('geo')
        data['ConfirmedCases'] = gr.ConfirmedCases.transform('cummax')
        data['Fatalities'] = gr.Fatalities.transform('cummax')

    geo_data = data.pivot(index='geo', columns='day', values='ForecastId')
    num_geo = geo_data.shape[0]
    geo_data

    geo_id = {}
    for i,g in enumerate(geo_data.index):
        geo_id[g] = i


    ConfirmedCases = data.pivot(index='geo', columns='day', values='ConfirmedCases')
    Fatalities = data.pivot(index='geo', columns='day', values='Fatalities')

    if debug:
        cases = ConfirmedCases.values
        deaths = Fatalities.values
    else:
        cases = np.log1p(ConfirmedCases.values)
        deaths = np.log1p(Fatalities.values)


    def get_dataset(start_pred, num_train, lag_period):
        days = np.arange( start_pred - num_train + 1, start_pred + 1)
        lag_cases = np.vstack([cases[:, d - lag_period : d] for d in days])
        lag_deaths = np.vstack([deaths[:, d - lag_period : d] for d in days])
        target_cases = np.vstack([cases[:, d : d + 1] for d in days])
        target_deaths = np.vstack([deaths[:, d : d + 1] for d in days])
        geo_ids = np.vstack([geo_ids_base for d in days])
        country_ids = np.vstack([country_ids_base for d in days])
        return lag_cases, lag_deaths, target_cases, target_deaths, geo_ids, country_ids, days

    def update_valid_dataset(data, pred_death, pred_case):
        lag_cases, lag_deaths, target_cases, target_deaths, geo_ids, country_ids, days = data
        day = days[-1] + 1
        new_lag_cases = np.hstack([lag_cases[:, 1:], pred_case])
        new_lag_deaths = np.hstack([lag_deaths[:, 1:], pred_death]) 
        new_target_cases = cases[:, day:day+1]
        new_target_deaths = deaths[:, day:day+1] 
        new_geo_ids = geo_ids  
        new_country_ids = country_ids  
        new_days = 1 + days
        return new_lag_cases, new_lag_deaths, new_target_cases, new_target_deaths, new_geo_ids, new_country_ids, new_days

    def fit_eval(lr_death, lr_case, data, start_lag_death, end_lag_death, num_lag_case, fit, score):
        lag_cases, lag_deaths, target_cases, target_deaths, geo_ids, country_ids, days = data

        X_death = np.hstack([lag_cases[:, -start_lag_death:-end_lag_death], country_ids])
        X_death = np.hstack([lag_deaths[:, -num_lag_case:], country_ids])
        X_death = np.hstack([lag_cases[:, -start_lag_death:-end_lag_death], lag_deaths[:, -num_lag_case:], country_ids])
        y_death = target_deaths
        y_death_prev = lag_deaths[:, -1:]
        if fit:
            if 0:
                keep = (y_death > 0).ravel()
                X_death = X_death[keep]
                y_death = y_death[keep]
                y_death_prev = y_death_prev[keep]
            lr_death.fit(X_death, y_death)
        y_pred_death = lr_death.predict(X_death)
        y_pred_death = np.maximum(y_pred_death, y_death_prev)

        X_case = np.hstack([lag_cases[:, -num_lag_case:], geo_ids])
        X_case = lag_cases[:, -num_lag_case:]
        y_case = target_cases
        y_case_prev = lag_cases[:, -1:]
        if fit:
            lr_case.fit(X_case, y_case)
        y_pred_case = lr_case.predict(X_case)
        y_pred_case = np.maximum(y_pred_case, y_case_prev)

        if score:
            death_score = val_score(y_death, y_pred_death)
            case_score = val_score(y_case, y_pred_case)
        else:
            death_score = 0
            case_score = 0

        return death_score, case_score, y_pred_death, y_pred_case

    def train_model(train, valid, start_lag_death, end_lag_death, num_lag_case, num_val, score=True):
        alpha = 3
        lr_death = Ridge(alpha=alpha, fit_intercept=False)
        lr_case = Ridge(alpha=alpha, fit_intercept=True)

        (train_death_score, train_case_score, train_pred_death, train_pred_case,
        ) = fit_eval(lr_death, lr_case, train, start_lag_death, end_lag_death, num_lag_case, fit=True, score=score)

        death_scores = []
        case_scores = []

        death_pred = []
        case_pred = []

        for i in range(num_val):

            (valid_death_score, valid_case_score, valid_pred_death, valid_pred_case,
            ) = fit_eval(lr_death, lr_case, valid, start_lag_death, end_lag_death, num_lag_case, fit=False, score=score)

            death_scores.append(valid_death_score)
            case_scores.append(valid_case_score)
            death_pred.append(valid_pred_death)
            case_pred.append(valid_pred_case)

            if 0:
                print('val death: %0.3f' %  valid_death_score,
                      'val case: %0.3f' %  valid_case_score,
                      'val : %0.3f' %  np.mean([valid_death_score, valid_case_score]),
                      flush=True)
            valid = update_valid_dataset(valid, valid_pred_death, valid_pred_case)

        if score:
            death_scores = np.sqrt(np.mean([s**2 for s in death_scores]))
            case_scores = np.sqrt(np.mean([s**2 for s in case_scores]))
            if 0:
                print('train death: %0.3f' %  train_death_score,
                      'train case: %0.3f' %  train_case_score,
                      'val death: %0.3f' %  death_scores,
                      'val case: %0.3f' %  case_scores,
                      'val : %0.3f' % ( (death_scores + case_scores) / 2),
                      flush=True)
            else:
                print('%0.4f' %  case_scores,
                      ', %0.4f' %  death_scores,
                      '= %0.4f' % ( (death_scores + case_scores) / 2),
                      flush=True)
        death_pred = np.hstack(death_pred)
        case_pred = np.hstack(case_pred)
        return death_scores, case_scores, death_pred, case_pred

    countries = [g.split('_')[0] for g in geo_data.index]
    countries = pd.factorize(countries)[0]

    country_ids_base = countries.reshape((-1, 1))
    ohe = OneHotEncoder(sparse=False)
    country_ids_base = 0.2 * ohe.fit_transform(country_ids_base)
    country_ids_base.shape

    geo_ids_base = np.arange(num_geo).reshape((-1, 1))
    ohe = OneHotEncoder(sparse=False)
    geo_ids_base = 0.1 * ohe.fit_transform(geo_ids_base)
    geo_ids_base.shape

    def val_score(true, pred):
        pred = np.log1p(np.round(np.expm1(pred) - 0.2))
        return np.sqrt(mean_squared_error(true.ravel(), pred.ravel()))

    def val_score(true, pred):
        return np.sqrt(mean_squared_error(true.ravel(), pred.ravel()))



    start_lag_death, end_lag_death = 14, 6,
    num_train = 5
    num_lag_case = 14
    lag_period = max(start_lag_death, num_lag_case)

    def get_oof(start_val_delta=0):   
        start_val = min_test_val_day + start_val_delta
        last_train = start_val - 1
        num_val = max_test_val_day - start_val + 1
        print(dates[start_val], start_val, num_val)
        train_data = get_dataset(last_train, num_train, lag_period)
        valid_data = get_dataset(start_val, 1, lag_period)
        _, _, val_death_preds, val_case_preds = train_model(train_data, valid_data, 
                                                            start_lag_death, end_lag_death, num_lag_case, num_val)

        pred_deaths = Fatalities.iloc[:, start_val:start_val+num_val].copy()
        pred_deaths.iloc[:, :] = np.expm1(val_death_preds)
        pred_deaths = pred_deaths.stack().reset_index()
        pred_deaths.columns = ['geo', 'day', 'Fatalities']
        pred_deaths

        pred_cases = ConfirmedCases.iloc[:, start_val:start_val+num_val].copy()
        pred_cases.iloc[:, :] = np.expm1(val_case_preds)
        pred_cases = pred_cases.stack().reset_index()
        pred_cases.columns = ['geo', 'day', 'ConfirmedCases']
        pred_cases

        sub = train[['Date', 'Id', 'geo', 'day']]
        sub = sub.merge(pred_cases, how='left', on=['geo', 'day'])
        sub = sub.merge(pred_deaths, how='left', on=['geo', 'day'])
        #sub = sub.fillna(0)
        sub = sub[sub.day >= start_val]
        sub = sub[['Id', 'ConfirmedCases', 'Fatalities']].copy()
        return sub


    if save_oof:
        for start_val_delta, date in zip(range(3, -8, -3),
                                  ['2020-03-22', '2020-03-19', '2020-03-16', '2020-03-13']):
            print(date, end=' ')
            oof = get_oof(start_val_delta)
            oof.to_csv('../submissions/cpmp-%s.csv' % date, index=None)

    def get_sub(start_val_delta=0):   
        start_val = min_test_val_day + start_val_delta
        last_train = start_val - 1
        num_val = max_test_val_day - start_val + 1
        print(dates[last_train], start_val, num_val)
        num_lag_case = 14
        train_data = get_dataset(last_train, num_train, lag_period)
        valid_data = get_dataset(start_val, 1, lag_period)
        _, _, val_death_preds, val_case_preds = train_model(train_data, valid_data, 
                                                            start_lag_death, end_lag_death, num_lag_case, num_val)

        pred_deaths = Fatalities.iloc[:, start_val:start_val+num_val].copy()
        pred_deaths.iloc[:, :] = np.expm1(val_death_preds)
        pred_deaths = pred_deaths.stack().reset_index()
        pred_deaths.columns = ['geo', 'day', 'Fatalities']
        pred_deaths

        pred_cases = ConfirmedCases.iloc[:, start_val:start_val+num_val].copy()
        pred_cases.iloc[:, :] = np.expm1(val_case_preds)
        pred_cases = pred_cases.stack().reset_index()
        pred_cases.columns = ['geo', 'day', 'ConfirmedCases']
        pred_cases

        sub = test[['Date', 'ForecastId', 'geo', 'day']]
        sub = sub.merge(pred_cases, how='left', on=['geo', 'day'])
        sub = sub.merge(pred_deaths, how='left', on=['geo', 'day'])
        sub = sub.fillna(0)
        sub = sub[['ForecastId', 'ConfirmedCases', 'Fatalities']]
        return sub
        return sub


    known_test = train[['geo', 'day', 'ConfirmedCases', 'Fatalities']
              ].merge(test[['geo', 'day', 'ForecastId']], how='left', on=['geo', 'day'])
    known_test = known_test[['ForecastId', 'ConfirmedCases', 'Fatalities']][known_test.ForecastId.notnull()].copy()
    known_test

    unknow_test = test[test.day > max_test_val_day]
    unknow_test

    def get_final_sub():   
        start_val = max_test_val_day + 1
        last_train = start_val - 1
        num_val = max_test_day - start_val + 1
        print(dates[last_train], start_val, num_val)
        num_lag_case = num_val + 3
        train_data = get_dataset(last_train, num_train, lag_period)
        valid_data = get_dataset(start_val, 1, lag_period)
        (_, _, val_death_preds, val_case_preds
        ) = train_model(train_data, valid_data, start_lag_death, end_lag_death, num_lag_case, num_val, score=False)

        pred_deaths = Fatalities.iloc[:, start_val:start_val+num_val].copy()
        pred_deaths.iloc[:, :] = np.expm1(val_death_preds)
        pred_deaths = pred_deaths.stack().reset_index()
        pred_deaths.columns = ['geo', 'day', 'Fatalities']
        pred_deaths

        pred_cases = ConfirmedCases.iloc[:, start_val:start_val+num_val].copy()
        pred_cases.iloc[:, :] = np.expm1(val_case_preds)
        pred_cases = pred_cases.stack().reset_index()
        pred_cases.columns = ['geo', 'day', 'ConfirmedCases']
        pred_cases
        print(unknow_test.shape, pred_deaths.shape, pred_cases.shape)

        sub = unknow_test[['Date', 'ForecastId', 'geo', 'day']]
        sub = sub.merge(pred_cases, how='left', on=['geo', 'day'])
        sub = sub.merge(pred_deaths, how='left', on=['geo', 'day'])
        #sub = sub.fillna(0)
        sub = sub[['ForecastId', 'ConfirmedCases', 'Fatalities']]
        sub = pd.concat([known_test, sub])
        return sub

    if save_public_test:
        sub = get_sub()
    else:
        sub = get_final_sub()
    
    return sub


# In[ ]:


cpmp_sub = get_cpmp_sub()
cpmp_sub['ForecastId'] = cpmp_sub['ForecastId'].astype('int')
cpmp_preds = test_orig.merge(cpmp_sub, on='ForecastId')
cpmp_p_c = cpmp_preds.pivot(index='Area', columns='days', values='ConfirmedCases').sort_index()
cpmp_p_f = cpmp_preds.pivot(index='Area', columns='days', values='Fatalities').sort_index()
preds_c_cpmp = np.log1p(cpmp_p_c.values)
preds_f_cpmp = np.log1p(cpmp_p_f.values)

'shape cpmp preds', preds_c_cpmp.shape, preds_f_cpmp.shape


# In[ ]:



from sklearn.metrics import mean_squared_error

if False:
    val_len = train_p_c.values.shape[1] - TRAIN_N
    m1s = []
    m2s = []
    for i in range(val_len):
        d = i + TRAIN_N
        m1 = np.sqrt(mean_squared_error(np.log(1 + train_p_c_raw.values[:, d]), preds_c_cpmp[:, d-START_PUBLIC]))
        m2 = np.sqrt(mean_squared_error(np.log(1 + train_p_f_raw.values[:, d]), preds_f_cpmp[:, d-START_PUBLIC]))
        print(f"{d}: {(m1 + m2)/2:8.5f} [{m1:8.5f} {m2:8.5f}]")
        m1s += [m1]
        m2s += [m2]
    print()

    
    m1 = np.sqrt(mean_squared_error(np.log(1 + train_p_c_raw.values[:, TRAIN_N:TRAIN_N+val_len]).flatten(), preds_c_cpmp[:, TRAIN_N-START_PUBLIC:TRAIN_N-START_PUBLIC+val_len].flatten()))
    m2 = np.sqrt(mean_squared_error(np.log(1 + train_p_f_raw.values[:, TRAIN_N:TRAIN_N+val_len]).flatten(), preds_f_cpmp[:, TRAIN_N-START_PUBLIC:TRAIN_N-START_PUBLIC+val_len].flatten()))
    print(f"{(m1 + m2)/2:8.5f} [{m1:8.5f} {m2:8.5f}]")


# In[ ]:


import matplotlib.pyplot as plt

for _ in range(5):
    plt.style.use(['default'])
    fig = plt.figure(figsize = (15, 5))

    idx = np.random.choice(N_AREAS)
    print(AREAS[idx])

    plt.plot(np.log(1+train_p_c.values[idx]), label=AREAS[idx], color='darkblue')
    plt.plot(preds_c[idx], linestyle='--', color='darkblue', label = 'pdd pred cases')
    plt.plot(np.pad(preds_c_cpmp[idx],(START_PUBLIC,0)),label = 'cpmp pred cases', linestyle='-.', color='darkblue')
    plt.legend()
    plt.show()


# In[ ]:


import matplotlib.pyplot as plt

for _ in range(5):
    plt.style.use(['default'])
    fig = plt.figure(figsize = (15, 5))

    idx = np.random.choice(N_AREAS)
    print(AREAS[idx])

    plt.plot(np.log(1+train_p_f.values[idx]), label=AREAS[idx], color='darkblue')
    plt.plot(preds_f[idx], linestyle='--', color='darkblue', label = 'pdd pred fatalities')
    plt.plot(np.pad(preds_f_cpmp[idx],(START_PUBLIC,0)),label = 'cpmp pred fatalities', linestyle='-.', color='darkblue')
    plt.legend()
    plt.show()


# ## beluga model

# In[ ]:


import datetime as dt

COMP = '../input/covid19-global-forecasting-week-3'
DATEFORMAT = '%Y-%m-%d'

def get_comp_data(COMP):
    train = pd.read_csv(f'{COMP}/train.csv')
    test = pd.read_csv(f'{COMP}/test.csv')
    submission = pd.read_csv(f'{COMP}/submission.csv')
    print(train.shape, test.shape, submission.shape)
    
    
    
    train['Country_Region'] = train['Country_Region'].str.replace(',', '').fillna('N/A')
    test['Country_Region'] = test['Country_Region'].str.replace(',', '').fillna('N/A')

    train['Location'] = train['Country_Region'].fillna('') + '-' + train['Province_State'].fillna('N/A')

    test['Location'] = test['Country_Region'].fillna('') + '-' + test['Province_State'].fillna('N/A')

    train['LogConfirmed'] = to_log(train.ConfirmedCases)
    train['LogFatalities'] = to_log(train.Fatalities)
    train = train.drop(columns=['Province_State'])
    test = test.drop(columns=['Province_State'])

    country_codes = pd.read_csv('../input/covid19-metadata/country_codes.csv', keep_default_na=False)
    train = train.merge(country_codes, on='Country_Region', how='left')
    test = test.merge(country_codes, on='Country_Region', how='left')
    
    train['continent'] = train['continent'].fillna('')
    test['continent'] = test['continent'].fillna('')

    train['DateTime'] = pd.to_datetime(train['Date'])
    test['DateTime'] = pd.to_datetime(test['Date'])
    
    return train, test, submission


def process_each_location(df):
    dfs = []
    for loc, df in tqdm(df.groupby('Location')):
        df = df.sort_values(by='Date')
        df['Fatalities'] = df['Fatalities'].cummax()
        df['ConfirmedCases'] = df['ConfirmedCases'].cummax()
        df['LogFatalities'] = df['LogFatalities'].cummax()
        df['LogConfirmed'] = df['LogConfirmed'].cummax()
        df['LogConfirmedNextDay'] = df['LogConfirmed'].shift(-1)
        df['ConfirmedNextDay'] = df['ConfirmedCases'].shift(-1)
        df['DateNextDay'] = df['Date'].shift(-1)
        df['LogFatalitiesNextDay'] = df['LogFatalities'].shift(-1)
        df['FatalitiesNextDay'] = df['Fatalities'].shift(-1)
        df['LogConfirmedDelta'] = df['LogConfirmedNextDay'] - df['LogConfirmed']
        df['ConfirmedDelta'] = df['ConfirmedNextDay'] - df['ConfirmedCases']
        df['LogFatalitiesDelta'] = df['LogFatalitiesNextDay'] - df['LogFatalities']
        df['FatalitiesDelta'] = df['FatalitiesNextDay'] - df['Fatalities']
        dfs.append(df)
    return pd.concat(dfs)


def add_days(d, k):
    return dt.datetime.strptime(d, DATEFORMAT) + dt.timedelta(days=k)


def to_log(x):
    return np.log(x + 1)


def to_exp(x):
    return np.exp(x) - 1

def get_beluga_sub():

    
    train, test, submission = get_comp_data(COMP)
    train.shape, test.shape, submission.shape
    
    TRAIN_START = train.Date.min()
    TEST_START = test.Date.min()
    TRAIN_END = train.Date.max()
    # TRAIN_END = "2020-03-21"
    TEST_END = test.Date.max()
    TRAIN_START, TRAIN_END, TEST_START, TEST_END
    
    train_clean = process_each_location(train)
    
    train_clean['Geo#Country#Contintent'] = train_clean.Location + '#' + train_clean.Country_Region + '#' + train_clean.continent
    
    DECAY = 0.93
    DECAY ** 7, DECAY ** 14, DECAY ** 21, DECAY ** 28

    confirmed_deltas = train.groupby(['Location', 'Country_Region', 'continent'])[[
        'Id']].count().reset_index()

    GLOBAL_DELTA = 0.11
    confirmed_deltas['DELTA'] = GLOBAL_DELTA

    confirmed_deltas.loc[confirmed_deltas.continent=='Africa', 'DELTA'] = 0.14
    confirmed_deltas.loc[confirmed_deltas.continent=='Oceania', 'DELTA'] = 0.06
    confirmed_deltas.loc[confirmed_deltas.Country_Region=='Korea South', 'DELTA'] = 0.011
    confirmed_deltas.loc[confirmed_deltas.Country_Region=='US', 'DELTA'] = 0.15
    confirmed_deltas.loc[confirmed_deltas.Country_Region=='China', 'DELTA'] = 0.01
    confirmed_deltas.loc[confirmed_deltas.Country_Region=='Japan', 'DELTA'] = 0.05
    confirmed_deltas.loc[confirmed_deltas.Country_Region=='Singapore', 'DELTA'] = 0.05
    confirmed_deltas.loc[confirmed_deltas.Country_Region=='Taiwan*', 'DELTA'] = 0.05
    confirmed_deltas.loc[confirmed_deltas.Country_Region=='Switzerland', 'DELTA'] = 0.05
    confirmed_deltas.loc[confirmed_deltas.Country_Region=='Norway', 'DELTA'] = 0.05
    confirmed_deltas.loc[confirmed_deltas.Country_Region=='Iceland', 'DELTA'] = 0.05
    confirmed_deltas.loc[confirmed_deltas.Country_Region=='Austria', 'DELTA'] = 0.06
    confirmed_deltas.loc[confirmed_deltas.Country_Region=='Italy', 'DELTA'] = 0.04
    confirmed_deltas.loc[confirmed_deltas.Country_Region=='Spain', 'DELTA'] = 0.08
    confirmed_deltas.loc[confirmed_deltas.Country_Region=='Portugal', 'DELTA'] = 0.12
    confirmed_deltas.loc[confirmed_deltas.Country_Region=='Israel', 'DELTA'] = 0.12
    confirmed_deltas.loc[confirmed_deltas.Country_Region=='Iran', 'DELTA'] = 0.08
    confirmed_deltas.loc[confirmed_deltas.Country_Region=='Germany', 'DELTA'] = 0.07
    confirmed_deltas.loc[confirmed_deltas.Country_Region=='Malaysia', 'DELTA'] = 0.06
    confirmed_deltas.loc[confirmed_deltas.Country_Region=='Russia', 'DELTA'] = 0.18
    confirmed_deltas.loc[confirmed_deltas.Country_Region=='Ukraine', 'DELTA'] = 0.18
    confirmed_deltas.loc[confirmed_deltas.Country_Region=='Brazil', 'DELTA'] = 0.12
    confirmed_deltas.loc[confirmed_deltas.Country_Region=='Turkey', 'DELTA'] = 0.18
    confirmed_deltas.loc[confirmed_deltas.Country_Region=='Philippines', 'DELTA'] = 0.18
    confirmed_deltas.loc[confirmed_deltas.Location=='France-N/A', 'DELTA'] = 0.1
    confirmed_deltas.loc[confirmed_deltas.Location=='United Kingdom-N/A', 'DELTA'] = 0.12
    confirmed_deltas.loc[confirmed_deltas.Location=='Diamond Princess-N/A', 'DELTA'] = 0.00
    confirmed_deltas.loc[confirmed_deltas.Location=='China-Hong Kong', 'DELTA'] = 0.08
    confirmed_deltas.loc[confirmed_deltas.Location=='San Marino-N/A', 'DELTA'] = 0.03


    confirmed_deltas.shape, confirmed_deltas.DELTA.mean()

    confirmed_deltas[confirmed_deltas.DELTA != GLOBAL_DELTA].shape, confirmed_deltas[confirmed_deltas.DELTA != GLOBAL_DELTA].DELTA.mean()
    confirmed_deltas[confirmed_deltas.DELTA != GLOBAL_DELTA]
    confirmed_deltas.describe()
    
    daily_log_confirmed = train_clean.pivot('Location', 'Date', 'LogConfirmed').reset_index()
    daily_log_confirmed = daily_log_confirmed.sort_values(TRAIN_END, ascending=False)
    # daily_log_confirmed.to_csv('daily_log_confirmed.csv', index=False)

    for i, d in tqdm(enumerate(pd.date_range(add_days(TRAIN_END, 1), add_days(TEST_END, 1)))):
        new_day = str(d).split(' ')[0]
        last_day = dt.datetime.strptime(new_day, DATEFORMAT) - dt.timedelta(days=1)
        last_day = last_day.strftime(DATEFORMAT)
        for loc in confirmed_deltas.Location.values:
            confirmed_delta = confirmed_deltas.loc[confirmed_deltas.Location == loc, 'DELTA'].values[0]
            daily_log_confirmed.loc[daily_log_confirmed.Location == loc, new_day] = daily_log_confirmed.loc[daily_log_confirmed.Location == loc, last_day] +                 confirmed_delta * DECAY ** i
            
    confirmed_prediciton = pd.melt(daily_log_confirmed, id_vars='Location')
    confirmed_prediciton['ConfirmedCases'] = to_exp(confirmed_prediciton['value'])
    
    confirmed_prediciton_cases = confirmed_prediciton.copy()
    
    latest = train_clean[train_clean.Date == TRAIN_END][[
    'Geo#Country#Contintent', 'ConfirmedCases', 'Fatalities', 'LogConfirmed', 'LogFatalities']]
    daily_death_deltas = train_clean[train_clean.Date >= '2020-03-17'].pivot(
        'Geo#Country#Contintent', 'Date', 'LogFatalitiesDelta').round(3).reset_index()
    daily_death_deltas = latest.merge(daily_death_deltas, on='Geo#Country#Contintent')
    
    death_deltas = train.groupby(['Location', 'Country_Region', 'continent'])[[
    'Id']].count().reset_index()

    GLOBAL_DELTA = 0.11
    death_deltas['DELTA'] = GLOBAL_DELTA

    death_deltas.loc[death_deltas.Country_Region=='China', 'DELTA'] = 0.005
    death_deltas.loc[death_deltas.continent=='Oceania', 'DELTA'] = 0.08
    death_deltas.loc[death_deltas.Country_Region=='Korea South', 'DELTA'] = 0.04
    death_deltas.loc[death_deltas.Country_Region=='Japan', 'DELTA'] = 0.04
    death_deltas.loc[death_deltas.Country_Region=='Singapore', 'DELTA'] = 0.05
    death_deltas.loc[death_deltas.Country_Region=='Taiwan*', 'DELTA'] = 0.06



    death_deltas.loc[death_deltas.Country_Region=='US', 'DELTA'] = 0.17

    death_deltas.loc[death_deltas.Country_Region=='Switzerland', 'DELTA'] = 0.15
    death_deltas.loc[death_deltas.Country_Region=='Norway', 'DELTA'] = 0.15
    death_deltas.loc[death_deltas.Country_Region=='Iceland', 'DELTA'] = 0.01
    death_deltas.loc[death_deltas.Country_Region=='Austria', 'DELTA'] = 0.14
    death_deltas.loc[death_deltas.Country_Region=='Italy', 'DELTA'] = 0.07
    death_deltas.loc[death_deltas.Country_Region=='Spain', 'DELTA'] = 0.1
    death_deltas.loc[death_deltas.Country_Region=='Portugal', 'DELTA'] = 0.13
    death_deltas.loc[death_deltas.Country_Region=='Israel', 'DELTA'] = 0.16
    death_deltas.loc[death_deltas.Country_Region=='Iran', 'DELTA'] = 0.06
    death_deltas.loc[death_deltas.Country_Region=='Germany', 'DELTA'] = 0.14
    death_deltas.loc[death_deltas.Country_Region=='Malaysia', 'DELTA'] = 0.14
    death_deltas.loc[death_deltas.Country_Region=='Russia', 'DELTA'] = 0.2
    death_deltas.loc[death_deltas.Country_Region=='Ukraine', 'DELTA'] = 0.2
    death_deltas.loc[death_deltas.Country_Region=='Brazil', 'DELTA'] = 0.2
    death_deltas.loc[death_deltas.Country_Region=='Turkey', 'DELTA'] = 0.22
    death_deltas.loc[death_deltas.Country_Region=='Philippines', 'DELTA'] = 0.12
    death_deltas.loc[death_deltas.Location=='France-N/A', 'DELTA'] = 0.14
    death_deltas.loc[death_deltas.Location=='United Kingdom-N/A', 'DELTA'] = 0.14
    death_deltas.loc[death_deltas.Location=='Diamond Princess-N/A', 'DELTA'] = 0.00

    death_deltas.loc[death_deltas.Location=='China-Hong Kong', 'DELTA'] = 0.01
    death_deltas.loc[death_deltas.Location=='San Marino-N/A', 'DELTA'] = 0.05


    death_deltas.shape
    death_deltas.DELTA.mean()

    death_deltas[death_deltas.DELTA != GLOBAL_DELTA].shape
    death_deltas[death_deltas.DELTA != GLOBAL_DELTA].DELTA.mean()
    death_deltas[death_deltas.DELTA != GLOBAL_DELTA]
    death_deltas.describe()
    
    daily_log_deaths = train_clean.pivot('Location', 'Date', 'LogFatalities').reset_index()
    daily_log_deaths = daily_log_deaths.sort_values(TRAIN_END, ascending=False)
    # daily_log_deaths.to_csv('daily_log_deaths.csv', index=False)

    for i, d in tqdm(enumerate(pd.date_range(add_days(TRAIN_END, 1), add_days(TEST_END, 1)))):
        new_day = str(d).split(' ')[0]
        last_day = dt.datetime.strptime(new_day, DATEFORMAT) - dt.timedelta(days=1)
        last_day = last_day.strftime(DATEFORMAT)
        for loc in death_deltas.Location:
            death_delta = death_deltas.loc[death_deltas.Location == loc, 'DELTA'].values[0]
            daily_log_deaths.loc[daily_log_deaths.Location == loc, new_day] = daily_log_deaths.loc[daily_log_deaths.Location == loc, last_day] +                 death_delta * DECAY ** i#
            
    confirmed_prediciton = pd.melt(daily_log_deaths, id_vars='Location')
    confirmed_prediciton['Fatalities'] = to_exp(confirmed_prediciton['value'])
    
    confirmed_prediciton_fatalities = confirmed_prediciton.copy()
    
    return confirmed_prediciton_cases, confirmed_prediciton_fatalities
    
preds_beluga_cases, preds_beluga_fatalities = get_beluga_sub()


# In[ ]:


p_f_beluga = preds_beluga_fatalities.pivot(index='Location', columns='Date', values='Fatalities').sort_values("Location")
p_c_beluga = preds_beluga_cases.pivot(index='Location', columns='Date', values='ConfirmedCases').sort_values("Location")


# In[ ]:


locs = p_f_beluga.index.values


# In[ ]:


p_f_beluga = np.log1p(p_f_beluga.values[:].copy())
p_c_beluga = np.log1p(p_c_beluga.values[:].copy())
p_f_beluga.shape, p_c_beluga.shape


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
    
    df_train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")
    df_test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")
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
    len_unknown = 30
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


df_preds.shape


# In[ ]:


# df_preds['cases/day']


# In[ ]:


# df_preds['Area'] = df_preds['Country_Region'] + '_' + df_preds['Province_State']


# In[ ]:


p_f_oscii = df_preds.pivot(index='place_id', columns='days', values='fatal_pred').sort_index()
p_c_oscii = df_preds.pivot(index='place_id', columns='days', values='cases_pred').sort_index()
p_f_oscii


# In[ ]:


preds_f_oscii = np.log1p(p_f_oscii.values[:].copy())
preds_c_oscii = np.log1p(p_c_oscii.values[:].copy())
preds_f_oscii.shape, preds_c_oscii.shape


# In[ ]:


START_PUBLIC, TRAIN_N


# In[ ]:


if False:
    val_len = train_p_c.values.shape[1] - TRAIN_N
    #val_len = 12
    m1s = []
    m2s = []
    for i in range(val_len):
        d = i + TRAIN_N
        m1 = np.sqrt(mean_squared_error(np.log(1 + train_p_c_raw.values[:, d]), preds_c_oscii[:, d-START_PUBLIC]))
        m2 = np.sqrt(mean_squared_error(np.log(1 + train_p_f_raw.values[:, d]), preds_f_oscii[:, d-START_PUBLIC]))
        print(f"{d}: {(m1 + m2)/2:8.5f} [{m1:8.5f} {m2:8.5f}]")
        m1s += [m1]
        m2s += [m2]
    print()

    
    m1 = np.sqrt(mean_squared_error(np.log(1 + train_p_c_raw.values[:, TRAIN_N:TRAIN_N+val_len]).flatten(), preds_c_oscii[:, TRAIN_N-START_PUBLIC:TRAIN_N-START_PUBLIC+val_len].flatten()))
    m2 = np.sqrt(mean_squared_error(np.log(1 + train_p_f_raw.values[:, TRAIN_N:TRAIN_N+val_len]).flatten(), preds_f_oscii[:, TRAIN_N-START_PUBLIC:TRAIN_N-START_PUBLIC+val_len].flatten()))
    print(f"{(m1 + m2)/2:8.5f} [{m1:8.5f} {m2:8.5f}]")


# In[ ]:


preds_c_oscii.shape, preds_c_cpmp.shape, preds_c.shape


# ## BLEND

# In[ ]:


# preds_c_blend = np.log1p(np.average([np.expm1(p_c_beluga[:,64:107]),np.expm1(preds_c_cpmp[:]),np.expm1(preds_c[:,64:107])],axis=0, weights=[2,1,2]))
# preds_f_blend = np.log1p(np.average([np.expm1(p_f_beluga[:,64:107]),np.expm1(preds_f_cpmp[:]),np.expm1(preds_f[:,64:107])],axis=0, weights=[2,1,2]))


# In[ ]:




preds_c_blend = np.log1p(np.average([np.expm1(preds_c_oscii[:,64:107]),np.expm1(preds_c_cpmp[:]),np.expm1(p_c_beluga[:,64:107]),np.expm1(preds_c[:,64:107])],axis=0, weights=[8,1,1,8]))
preds_f_blend = np.log1p(np.average([np.expm1(preds_f_oscii[:,64:107]),np.expm1(preds_f_cpmp[:]),np.expm1(p_f_beluga[:,64:107]),np.expm1(preds_f[:,64:107])],axis=0, weights=[8,1,1,8]))


# In[ ]:


preds_c_cpmp.shape, preds_c[:,64:107].shape


# In[ ]:


val_len = 13
#val_len = 12
m1s = []
m2s = []
for i in range(val_len):
    d = i + 64
    m1 = np.sqrt(mean_squared_error(np.log(1 + train_p_c_raw.values[:, d]), preds_c_blend[:, i]))
    m2 = np.sqrt(mean_squared_error(np.log(1 + train_p_f_raw.values[:, d]), preds_f_blend[:, i]))
    print(f"{d}: {(m1 + m2)/2:8.5f} [{m1:8.5f} {m2:8.5f}]")
    m1s += [m1]
    m2s += [m2]
print()


m1 = np.sqrt(mean_squared_error(np.log(1 + train_p_c_raw.values[:, 64:64+val_len]).flatten(), preds_c_blend[:, :val_len].flatten()))
m2 = np.sqrt(mean_squared_error(np.log(1 + train_p_f_raw.values[:, 64:64+val_len]).flatten(), preds_f_blend[:, :val_len].flatten()))
print(f"PUBLIC LB {(m1 + m2)/2:8.5f} [{m1:8.5f} {m2:8.5f}]")


# In[ ]:


import matplotlib.pyplot as plt

for _ in range(5):
    plt.style.use(['default'])
    fig = plt.figure(figsize = (15, 5))

    idx = np.random.choice(N_AREAS)
    print(AREAS[idx], places[idx])

    plt.plot(np.log(1+train_p_f.values[idx]), label=AREAS[idx], color='darkblue')
    plt.plot(preds_f[idx], linestyle='--', color='darkblue', label = 'pdd fat ')
    plt.plot(np.pad(preds_f_cpmp[idx],(START_PUBLIC,0)),label = 'cpmp fat ', linestyle='-.', color='darkblue')
    plt.plot(np.pad(preds_f_oscii[idx],(0,0)),label = 'oscii fat ', linestyle='-.', color='red')
    plt.plot(np.pad(p_f_beluga[idx],(0,0)),label = 'beluga fat ', linestyle='-.', color='orange')
    plt.plot(np.pad(preds_f_blend[idx],(START_PUBLIC,0)),label = 'blend fat ', linestyle='-.', color='darkgreen')
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
    plt.plot(preds_c[idx], linestyle='--', color='darkblue', label = 'pdd  cases')
    plt.plot(np.pad(preds_c_cpmp[idx],(START_PUBLIC,0)),label = 'cpmp  cases', linestyle='-.', color='darkblue')
    plt.plot(np.pad(preds_c_oscii[idx],(0,0)),label = 'oscii  cases', linestyle='-.', color='orange')
    plt.plot(np.pad(p_c_beluga[idx],(0,0)),label = 'beluga fat ', linestyle='-.', color='red')
    plt.plot(np.pad(preds_c_blend[idx],(START_PUBLIC,0)),label = 'blend  cases', linestyle='-.', color='darkgreen')
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
preds_c2[non_china_mask,64:107] = preds_c_blend[non_china_mask]
preds_f2[non_china_mask,64:107] = preds_f_blend[non_china_mask]


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




