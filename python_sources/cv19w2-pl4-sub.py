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

path = '../input/covid19-global-forecasting-week-2/'
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

print(train['Date'].max())
print(test['Date'].min())
print(train['days'].max())
N_AREAS = train['Area'].nunique()
AREAS = np.sort(train['Area'].unique())
#TRAIN_N = 50 + 7
TRAIN_N = 70

print(train[train['days'] < TRAIN_N]['Date'].max())
train.head()


# In[ ]:


train_p_c = train.pivot(index='Area', columns='days', values='ConfirmedCases').sort_index()
train_p_f = train.pivot(index='Area', columns='days', values='Fatalities').sort_index()

train_p_c = np.maximum.accumulate(train_p_c, axis=1)
train_p_f = np.maximum.accumulate(train_p_f, axis=1)

train_p_c_change = train_p_c.diff(axis=1).fillna(0)
train_p_f_change = train_p_f.diff(axis=1).fillna(0)

X_c = np.log(1+train_p_c.values)[:,:TRAIN_N]
X_f = train_p_f.values[:,:TRAIN_N]


# In[ ]:


train_p_c_change.head()


# In[ ]:


from sklearn.metrics import mean_squared_error
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
            gr_base.append(np.clip(np.diff(temp[temp > threshold])[-num_days:].mean(), 0, params['growth rate max']))
            gr_base_factor.append(np.clip(np.diff(np.diff(temp[temp > threshold]))[-num_days:].mean(), -0.2, params["growth rate factor max"]))
        else:
            gr_base.append(params['growth rate default'])
            gr_base_factor.append(params['growth rate factor'])

    gr_base = np.array(gr_base)
    gr_base_factor = np.array(gr_base_factor)
    #print(gr_base_factor)
    #gr_base = np.clip(gr_base, 0.02, 0.8)
    preds = X.copy()

    for i in range(test_size):
        delta = np.clip(preds[:, -1], np.log(2), None) + gr_base * (1 + params['growth rate factor']*(1 + params['growth rate factor factor'])**(i))**(i)
        #delta = np.clip(preds[:, -1], np.log(2), None) + gr_base * (1 + params['growth rate factor']*(1 + params['growth rate factor factor'])**(i+X.shape[1]-x_min))**(i+X.shape[1]-x_min) 
        preds = np.hstack((preds, delta.reshape(-1,1)))

    return preds

params = {
    "min cases for growth rate": 30,
    "last N days": 5,
    "growth rate default": 0.25,
    "growth rate max": 0.3,
    "growth rate factor max": -0.01,
    "growth rate factor": -0.05,
    "growth rate factor factor": 0.005,
}

x = train_p_c[train_p_c.index=="China_Qinghai"]

x = train_p_c

preds_c_1 = run_c(params, np.log(1+x.values)[:,:TRAIN_N])
#eval1(np.log(1+x).values, preds_c_1)


# In[ ]:


#plt.plot(preds_c[0,:])


# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error

from scipy.optimize import curve_fit
from scipy.special import gamma, gammainc

import warnings
warnings.filterwarnings("ignore")

def f(x, K, a, x0):
    return K * x ** a * np.exp(-x/x0)

def eval1(y, p):

    val_len = y.shape[1] - TRAIN_N

    return np.sqrt(mean_squared_error(y[:, TRAIN_N:TRAIN_N+val_len].flatten(), p[:, TRAIN_N:TRAIN_N+val_len].flatten()))

def run(params, X, X_change, test_size=50):
    
    print(X)
    
    x_mins = np.ma.MaskedArray(X, X<10)
    x_mins = x_mins.argmin(axis=1) 
    
    
    print(x_mins)
    
    popts = []
    scores = []
    
    for i in tqdm(range(X_change.shape[0])):
        
        if X[i,:].sum() == 0:
            x_mins[i] = X.shape[1] - 1
        
        best_score = 100
        best_popt = [1,1,1]
        #for x_min in np.arange(x_mins[i], X_change.shape[1]):
        x_min_max = X_change.shape[1]
        x_min_max = np.minimum(x_mins[i] + 10, X_change.shape[1])
        early_stopping = 5
        early_stopping_count = 0
        for x_min in np.arange(x_mins[i], x_min_max):

            
            x = np.arange(x_min, X_change.shape[1]) - x_min

            y = X_change[i, x_min:]
            #print(x)
            #print(y)
            try:
                popt, pcov = curve_fit(f, x,y, bounds=(0, [10, 5, np.max(x)]))
            except:
                popt = [1,1,1]
  
            p = np.zeros(X_change.shape[1])
            p[x_min:] = f(x, *popt)
            p = np.cumsum(p, axis=0)

            score = np.sqrt(mean_squared_error(np.log1p(X[i,:]), np.log1p(p)))

            if score < best_score:
                best_score = score
                best_popt = popt
                x_mins[i] = x_min
            else:
                early_stopping_count += 1
                
            if early_stopping_count >= early_stopping:
                continue
        #break
        #print(best_popt)
        
        popts.append(best_popt)
        scores.append(best_score)
        
    #print(x_mins)
    
    preds = X_change.copy()
    preds_new = np.zeros((X.shape[0], test_size))
    for i in range(X.shape[0]):
        x = np.arange(X.shape[1], test_size+X.shape[1]) - x_mins[i]
        y = f(x, *popts[i])
        #print(y)
        y = y[-test_size-X.shape[1]:]
        preds_new[i,:] = y
        
    preds = np.hstack((preds, preds_new))
    preds = np.cumsum(preds, axis=1)
    
#     gr_base = []
    
#     for i in range(X_c.shape[0]):
#         temp = X[i,:]
#         threshold = np.log(1+params['min cases for growth rate'])
#         num_days = params['last N days']
#         if (temp > threshold).sum() > num_days:
#             gr_base.append(np.clip(np.diff(temp[temp > threshold])[-num_days:].mean(), 0, params['growth rate max']))
#         else:
#             gr_base.append(params['growth rate default'])

#     gr_base = np.array(gr_base)
#     #gr_base = np.clip(gr_base, 0.02, 0.8)
#     preds = X.copy()

#     for i in range(test_size):
#         delta = np.clip(preds[:, -1], np.log(2), None) + gr_base * (1 + params['growth rate factor'])**i
#         preds = np.hstack((preds, delta.reshape(-1,1)))

    return preds, x_mins, scores

# x1 = train_p_c[train_p_c.index=="Angola_N/A"]
# x = train_p_c[train_p_c.index=="Angola_N/A"].values[:,:TRAIN_N]
# x_c = train_p_c_change[train_p_c_change.index=="Angola_N/A"].values[:,:TRAIN_N]

x1 = train_p_c
x = train_p_c.values[:,:TRAIN_N]
x_c = train_p_c_change.values[:,:TRAIN_N]

params = {}
preds_c_pl, x_mins, scores = run(params, x, x_c)
preds_c_pl = np.log1p(preds_c_pl)
#eval1(np.log(1+x1).values, preds_c_pl)

# params = {
#     "min cases for growth rate": 30,
#     "last N days": 8,
#     "growth rate default": 0.20,
#     "growth rate max": 0.3,
#     "growth rate factor": -0.09,
# }

# preds_c = run_c(params, np.log(1+train_p_c.values)[:,:TRAIN_N])
# eval1(np.log(1+train_p_c).values, preds_c)


# In[ ]:


# preds_c = np.zeros(preds_c_pl.shape)
# for area in AREAS:
#     idx = np.where(AREAS == area)[0][0]
#     score_pl = eval1(np.log(1+x1).values[idx,:].reshape(1,-1), preds_c_pl[idx,:].reshape(1,-1))
#     score_1 = eval1(np.log(1+x1).values[idx,:].reshape(1,-1), preds_c_1[idx,:].reshape(1,-1))
    
#     if score_pl < score_1:
#         preds_c[idx,:] = preds_c_pl[idx,:]
#     else:
#         preds_c[idx,:] = preds_c_1[idx,:]
                    
#     print(area, np.round(score_pl,2), np.round(score_1,2), np.round(scores[idx],2), x_mins[idx])


# In[ ]:


preds_c = preds_c_1.copy()
idx = np.where(x_mins<100)
#preds_c[idx] = preds_c_1[idx]
preds_c[idx] = 0.8 * preds_c_1[idx] + 0.2 * preds_c_pl[idx]

for i in range(N_AREAS):
    if 'China' in AREAS[i] and preds_c[i, TRAIN_N-1] < np.log(31):
        preds_c[i, TRAIN_N:] = preds_c[i, TRAIN_N-1]

#preds_c[idx] = np.max([preds_c_1[idx], preds_c_pl[idx]], axis=0)
#eval1(np.log(1+x1).values, preds_c)


# In[ ]:


f_rate = (train_p_f / train_p_c).fillna(0)

X_c = np.log(1+train_p_c.values)[:,:TRAIN_N]
X_f = train_p_f.values[:,:TRAIN_N]


# In[ ]:


def lin_w(sz):
    res = np.linspace(0, 1, sz+1, endpoint=False)[1:]
    return np.append(res, np.append([1], res[::-1]))


def run_f(params, X_c, X_f, X_f_r, test_size=50):


    
    X_f_r = np.array(np.ma.mean(np.ma.masked_outside(X_f_r, 0.06, 0.4)[:,:], axis=1))
    X_f_r = np.clip(X_f_r, params['fatality_rate_lower'], params['fatality_rate_upper'])
    #print(X_f_r)
    
    X_c = np.clip(np.exp(X_c)-1, 0, None)
    preds = X_f.copy()
    #print(preds.shape)
    
    train_size = X_f.shape[1] - 1
    for i in range(test_size):
        
        t_lag = train_size+i-params['length']
        t_wsize = 3
        delta = np.average(np.diff(X_c, axis=1)[:, t_lag-t_wsize:t_lag+1+t_wsize], axis=1)
        #delta = np.average(np.diff(X_c, axis=1)[:, t_lag-t_wsize:t_lag+1+t_wsize], axis=1, weights=lin_w(t_wsize))
        
        delta = params['absolute growth'] + delta * X_f_r
        
        preds = np.hstack((preds, preds[:, -1].reshape(-1,1) + delta.reshape(-1,1)))

    return preds

params = {
    "length": 6,
    "absolute growth": 0.02,
    "fatality_rate_lower": 0.035,
    "fatality_rate_upper": 0.40,
}

preds_f_1 = run_f(params, preds_c, X_f, f_rate.values[:,:TRAIN_N])
preds_f_1 = np.log(1+preds_f_1)

preds_f_1 = np.log1p(0.9*(np.exp(preds_f_1)-1))

#eval1(np.log(1+train_p_f).values, preds_f_1)


# In[ ]:


preds_f = preds_f_1


# In[ ]:


# x1 = train_p_f
# x = train_p_f.values[:,:TRAIN_N]
# x_c = train_p_f_change.values[:,:TRAIN_N]

# params = {}
# preds_f_pl, x_mins, scores = run(params, x, x_c)
# preds_f_pl = np.log1p(preds_f_pl)
# eval1(np.log(1+x1).values, preds_f_pl)


# In[ ]:


# preds_f = np.zeros(preds_f_pl.shape)
# for area in AREAS:
#     idx = np.where(AREAS == area)[0][0]
#     score_pl = eval1(np.log(1+x1).values[idx,:].reshape(1,-1), preds_f_pl[idx,:].reshape(1,-1))
#     score_1 = eval1(np.log(1+x1).values[idx,:].reshape(1,-1), preds_f_1[idx,:].reshape(1,-1))
    
#     if score_pl < score_1:
#         preds_f[idx,:] = preds_f_pl[idx,:]
#     else:
#         preds_f[idx,:] = preds_f_1[idx,:]
                    
#     print(area, np.round(score_pl,2), np.round(score_1,2))


# In[ ]:


# preds_f = preds_f_1.copy()
# idx = np.where(x_mins<2)
# #preds_c[idx] = preds_c_1[idx]
# preds_f[idx] = 0.9 * preds_f_1[idx] + 0.1 * preds_f_pl[idx]

# eval1(np.log(1+x1).values, preds_f)


# In[ ]:


#eval1(np.log(1+x1).values, preds_f)


# In[ ]:


from sklearn.metrics import mean_squared_error

if False:
    val_len = train_p_c.values.shape[1] - TRAIN_N

    for i in range(val_len):
        d = i + TRAIN_N
        m1 = np.sqrt(mean_squared_error(np.log(1 + train_p_c.values[:, d]), preds_c[:, d]))
        m2 = np.sqrt(mean_squared_error(np.log(1 + train_p_f.values[:, d]), preds_f[:, d]))
        print(f"{d}: {(m1 + m2)/2:8.5f} [{m1:8.5f} {m2:8.5f}]")

    print()

    m1 = np.sqrt(mean_squared_error(np.log(1 + train_p_c.values[:, TRAIN_N:TRAIN_N+val_len]).flatten(), preds_c[:, TRAIN_N:TRAIN_N+val_len].flatten()))
    m2 = np.sqrt(mean_squared_error(np.log(1 + train_p_f.values[:, TRAIN_N:TRAIN_N+val_len]).flatten(), preds_f[:, TRAIN_N:TRAIN_N+val_len].flatten()))
    print(f"{(m1 + m2)/2:8.5f} [{m1:8.5f} {m2:8.5f}]")


# In[ ]:


import matplotlib.pyplot as plt

plt.style.use(['default'])
fig = plt.figure(figsize = (15, 5))

#idx = worst_idx
#print(AREAS[idx])

idx = np.where(AREAS == 'Austria_N/A')[0][0]
plt.plot(np.log(1+train_p_c.values[idx]), label=AREAS[idx], color='darkblue')
plt.plot(preds_c[idx], linestyle='--', color='darkblue')

idx = np.where(AREAS == 'Germany_N/A')[0][0]
plt.plot(np.log(1+train_p_c.values[idx]), label=AREAS[idx], color='red')
plt.plot(preds_c[idx], linestyle='--', color='red')


idx = np.where(AREAS == 'China_Hubei')[0][0]
plt.plot(np.log(1+train_p_c.values[idx]), label=AREAS[idx], color='grey')
plt.plot(preds_c[idx], linestyle='--', color='grey')


idx = np.where(AREAS == 'Iran_N/A')[0][0]
plt.plot(np.log(1+train_p_c.values[idx]), label=AREAS[idx], color='green')
plt.plot(preds_c[idx], linestyle='--', color='green')


idx = np.where(AREAS == 'Japan_N/A')[0][0]
plt.plot(np.log(1+train_p_c.values[idx]), label=AREAS[idx], color='purple')
plt.plot(preds_c[idx], linestyle='--', color='purple')


idx = np.where(AREAS == 'Brazil_N/A')[0][0]
plt.plot(np.log(1+train_p_c.values[idx]), label=AREAS[idx], color='black')
plt.plot(preds_c[idx], linestyle='--', color='black')


idx = np.where(AREAS == 'Denmark_N/A')[0][0]
plt.plot(np.log(1+train_p_c.values[idx]), label=AREAS[idx], color='yellow')
plt.plot(preds_c[idx], linestyle='--', color='yellow')

idx = np.where(AREAS == 'Italy_N/A')[0][0]
plt.plot(np.log(1+train_p_c.values[idx]), label=AREAS[idx], color='blue')
plt.plot(preds_c[idx], linestyle='--', color='blue')

idx = np.where(AREAS == 'Canada_British Columbia')[0][0]
plt.plot(np.log(1+train_p_c.values[idx]), label=AREAS[idx], color='darkgreen')
plt.plot(preds_c[idx], linestyle='--', color='darkgreen')

idx = np.where(AREAS == 'Turkey_N/A')[0][0]
plt.plot(np.log(1+train_p_c.values[idx]), label=AREAS[idx], color='darkred')
plt.plot(preds_c[idx], linestyle='--', color='darkred')

plt.legend()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

plt.style.use(['default'])
fig = plt.figure(figsize = (15, 5))

#idx = worst_idx
#print(AREAS[idx])

idx = np.where(AREAS == 'Austria_N/A')[0][0]
plt.plot(np.log(1+train_p_f.values[idx]), label=AREAS[idx], color='darkblue')
plt.plot(preds_f[idx], linestyle='--', color='darkblue')

idx = np.where(AREAS == 'Germany_N/A')[0][0]
plt.plot(np.log(1+train_p_f.values[idx]), label=AREAS[idx], color='red')
plt.plot(preds_f[idx], linestyle='--', color='red')


idx = np.where(AREAS == 'China_Hubei')[0][0]
plt.plot(np.log(1+train_p_f.values[idx]), label=AREAS[idx], color='grey')
plt.plot(preds_f[idx], linestyle='--', color='grey')


idx = np.where(AREAS == 'Iran_N/A')[0][0]
plt.plot(np.log(1+train_p_f.values[idx]), label=AREAS[idx], color='green')
plt.plot(preds_f[idx], linestyle='--', color='green')


idx = np.where(AREAS == 'Japan_N/A')[0][0]
plt.plot(np.log(1+train_p_f.values[idx]), label=AREAS[idx], color='purple')
plt.plot(preds_f[idx], linestyle='--', color='purple')


idx = np.where(AREAS == 'Brazil_N/A')[0][0]
plt.plot(np.log(1+train_p_f.values[idx]), label=AREAS[idx], color='black')
plt.plot(preds_f[idx], linestyle='--', color='black')


idx = np.where(AREAS == 'Denmark_N/A')[0][0]
plt.plot(np.log(1+train_p_f.values[idx]), label=AREAS[idx], color='yellow')
plt.plot(preds_f[idx], linestyle='--', color='yellow')

idx = np.where(AREAS == 'Italy_N/A')[0][0]
plt.plot(np.log(1+train_p_f.values[idx]), label=AREAS[idx], color='blue')
plt.plot(preds_f[idx], linestyle='--', color='blue')

idx = np.where(AREAS == 'Canada_British Columbia')[0][0]
plt.plot(np.log(1+train_p_f.values[idx]), label=AREAS[idx], color='darkgreen')
plt.plot(preds_f[idx], linestyle='--', color='darkgreen')

plt.legend()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

plt.style.use(['default'])
fig = plt.figure(figsize = (15, 5))

idx = np.random.choice(N_AREAS)
print(AREAS[idx])

plt.plot(np.log(1+train_p_c.values[idx]), label=AREAS[idx], color='darkblue')
plt.plot(preds_c[idx], linestyle='--', color='darkblue')

plt.show()


# In[ ]:


#US_Arizona


# In[ ]:





# In[ ]:


temp = pd.DataFrame(np.clip(np.exp(preds_c) - 1, 0, None))
temp['Area'] = AREAS
temp = temp.melt(id_vars='Area', var_name='days', value_name="ConfirmedCases")

test = test.merge(temp, how='left', left_on=['Area', 'days'], right_on=['Area', 'days'])

temp = pd.DataFrame(np.clip(np.exp(preds_f) - 1, 0, None))
temp['Area'] = AREAS
temp = temp.melt(id_vars='Area', var_name='days', value_name="Fatalities")

test = test.merge(temp, how='left', left_on=['Area', 'days'], right_on=['Area', 'days'])
test.head()


# In[ ]:


test.to_csv("submission.csv", index=False, columns=["ForecastId", "ConfirmedCases", "Fatalities"])


# In[ ]:





# In[ ]:


for i, rec in test.groupby('Area').last().sort_values("ConfirmedCases", ascending=False).iterrows():
    print(f"{rec['ConfirmedCases']:10.1f} {rec['Fatalities']:10.1f}  {rec['Country_Region']}, {rec['Province_State']}")


# In[ ]:


print(f"{test.groupby('Area')['ConfirmedCases'].last().sum():10.1f}")
print(f"{test.groupby('Area')['Fatalities'].last().sum():10.1f}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




