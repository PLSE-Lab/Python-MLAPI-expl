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


train_p_c_raw = train.pivot(index='Area', columns='days', values='ConfirmedCases').sort_index()
train_p_f_raw = train.pivot(index='Area', columns='days', values='Fatalities').sort_index()

train_p_c = np.maximum.accumulate(train_p_c_raw, axis=1)
train_p_f = np.maximum.accumulate(train_p_f_raw, axis=1)

f_rate = (train_p_f / train_p_c).fillna(0)

X_c = np.log(1+train_p_c.values)[:,:TRAIN_N]
X_f = train_p_f.values[:,:TRAIN_N]


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

preds_f = run_f(params, preds_c, X_f, f_rate.values[:,:TRAIN_N])
preds_f = np.log(1+preds_f)
#eval1(np.log(1+train_p_f).values, preds_f)


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
    temp2 = preds_c[ar]
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
    temp2 = preds_f[ar]
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




