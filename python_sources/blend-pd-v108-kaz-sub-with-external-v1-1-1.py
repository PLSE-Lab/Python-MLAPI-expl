#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import datetime

path = '../input/covid19-global-forecasting-week-5/'
train = pd.read_csv(path + 'train.csv')
test  = pd.read_csv(path + 'test.csv')
sub   = pd.read_csv(path + 'submission.csv')

train['Date'] = train['Date'].apply(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d')))
test['Date'] = test['Date'].apply(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d')))

train['days'] = (train['Date'].dt.date - train['Date'].dt.date.min()).dt.days
test['days'] = (test['Date'].dt.date - train['Date'].dt.date.min()).dt.days

train.loc[train['Province_State'].isnull(), 'Province_State'] = 'N/A'
test.loc[test['Province_State'].isnull(), 'Province_State'] = 'N/A'

train.loc[train['County'].isnull(), 'County'] = 'N/A'
test.loc[test['County'].isnull(), 'County'] = 'N/A'

train['Area'] = train['Country_Region'] + '_' + train['Province_State'] + '_' + train['County']
test['Area'] = test['Country_Region'] + '_' + test['Province_State'] + '_' + test['County']

train_c_piv = train[train['Target'] == 'ConfirmedCases'].pivot(index='Area', columns='days', values='TargetValue').sort_index()
train_f_piv = train[train['Target'] == 'Fatalities'].pivot(index='Area', columns='days', values='TargetValue').sort_index()

train_c_raw = train_c_piv.values
train_f_raw = train_f_piv.values

#train_c = np.clip(train_c_raw, , None)
#train_f = np.clip(train_f_raw, 0, None)

train_c = train_c_raw
train_f = train_f_raw

# X_c = train_c[:,:TRAIN_N]
# X_f = train_f[:,:TRAIN_N]

weights_c = train[train['Target'] == 'ConfirmedCases'].groupby('Area')['Weight'].mean().sort_index().values.reshape(-1,1)
weights_f = train[train['Target'] == 'Fatalities'].groupby('Area')['Weight'].mean().sort_index().values.reshape(-1,1)


public_days = list(range(test['days'].min(),train['days'].max()+1))

AREAS = np.sort(train['Area'].unique())


# In[ ]:


def pinball_loss_single(ytrue, pred, weight, tau=0.5):
    cond = (ytrue >= pred).astype(int)
    error = np.sum(weight * (ytrue - pred) * cond * tau) -             np.sum(weight * (ytrue - pred) * (1-cond) * (1-tau))
    return error / ytrue.shape[0] / ytrue.shape[1]

def pinball_loss_many(ytrue, preds, weight, tau=[0.05, 0.50, 0.95]):
    return np.mean([pinball_loss_single(ytrue, preds[i], weight, t) for i,t in enumerate(tau)])


# In[ ]:


def convert2quantilepivot(full_sub, quantile):

    if quantile == 0.05:
        sub = full_sub[::3].copy()
    elif quantile == 0.5:
        sub = full_sub[1::3].copy()
    elif quantile == 0.95:
        sub = full_sub[2::3].copy()
    else:
        pass
        
    sub['ForecastId'] = sub['ForecastId_Quantile'].apply(lambda x: x.split('_')[0]).astype(int)
    x = pd.merge(test[['ForecastId','Area','days','Weight','Target']], sub[['ForecastId','TargetValue']], on = 'ForecastId')
    s_c_piv = x[x['Target'] == 'ConfirmedCases'].pivot(index='Area', columns='days', values='TargetValue').sort_index()
    s_f_piv = x[x['Target'] == 'Fatalities'].pivot(index='Area', columns='days', values='TargetValue').sort_index()
    return s_c_piv,s_f_piv


# In[ ]:


test.head()


# In[ ]:


sub_kaz = pd.read_csv('../input/sub-with-external-v1/submission.csv')
sub_pd = pd.read_csv('../input/cvw5-custom-hopt-v108-sub/submission.csv')


# In[ ]:


sub_kaz_c_piv_005 ,sub_kaz_f_piv_005 = convert2quantilepivot(sub_kaz, 0.05)
sub_kaz_c_piv_05 ,sub_kaz_f_piv_05 = convert2quantilepivot(sub_kaz, 0.5)
sub_kaz_c_piv_095 ,sub_kaz_f_piv_095 = convert2quantilepivot(sub_kaz, 0.95)

sub_pd_c_piv_005 ,sub_pd_f_piv_005 = convert2quantilepivot(sub_pd, 0.05)
sub_pd_c_piv_05 ,sub_pd_f_piv_05 = convert2quantilepivot(sub_pd, 0.5)
sub_pd_c_piv_095 ,sub_pd_f_piv_095 = convert2quantilepivot(sub_pd, 0.95)


# In[ ]:


pinball_loss_single(train_c_piv[public_days].values, sub_kaz_c_piv_005[public_days].values, weights_c, tau=0.05)


# In[ ]:


# individual scores
eval_days = public_days

pl_c_kaz = pinball_loss_many(train_c_piv[eval_days].values, [s[eval_days].values for s in [sub_kaz_c_piv_005,sub_kaz_c_piv_05,sub_kaz_c_piv_095]], weights_c)
pl_f_kaz = pinball_loss_many(train_f_piv[eval_days].values, [s[eval_days].values for s in [sub_kaz_f_piv_005,sub_kaz_f_piv_05,sub_kaz_f_piv_095]], weights_f)
pl_kaz = np.mean([pl_c_kaz,pl_f_kaz])

pl_c_pd = pinball_loss_many(train_c_piv[eval_days].values, [s[eval_days].values for s in [sub_pd_c_piv_005,sub_pd_c_piv_05,sub_pd_c_piv_095]], weights_c)
pl_f_pd = pinball_loss_many(train_f_piv[eval_days].values, [s[eval_days].values for s in [sub_pd_f_piv_005,sub_pd_f_piv_05,sub_pd_f_piv_095]], weights_f)
pl_pd = np.mean([pl_c_pd,pl_f_pd])

print('kaz',pl_kaz, pl_c_kaz, pl_f_kaz)
print('pd',pl_pd, pl_c_pd, pl_f_pd)
# simple mean scores

#simple blend scores

# per quantile scores


# In[ ]:


#simple blend scores
eval_days = public_days

c_weights = [1,1]
f_weights = [1,1]

blend_c_005 = np.average([s[eval_days].values for s in [sub_kaz_c_piv_005, sub_pd_c_piv_005]],axis=0, weights = c_weights)
blend_c_05 = np.average([s[eval_days].values for s in [sub_kaz_c_piv_05, sub_pd_c_piv_05]],axis=0, weights = c_weights)
blend_c_095 = np.average([s[eval_days].values for s in [sub_kaz_c_piv_095, sub_pd_c_piv_095]],axis=0, weights = c_weights)

blend_f_005 = np.average([s[eval_days].values for s in [sub_kaz_f_piv_005, sub_pd_f_piv_005]],axis=0, weights = f_weights)
blend_f_05 = np.average([s[eval_days].values for s in [sub_kaz_f_piv_05, sub_pd_f_piv_05]],axis=0, weights = f_weights)
blend_f_095 = np.average([s[eval_days].values for s in [sub_kaz_f_piv_095, sub_pd_f_piv_095]],axis=0, weights = f_weights)

pl_c = pinball_loss_many(train_c_piv[eval_days].values, [blend_c_005,blend_c_05,blend_c_095], weights_c)
pl_f = pinball_loss_many(train_f_piv[eval_days].values, [blend_f_005,blend_f_05,blend_f_095], weights_f)
print(np.mean([pl_c,pl_f]),pl_c,pl_f)


# In[ ]:


sub_kaz_f_piv_095.shape


# In[ ]:





# In[ ]:


# doing blend
blend_weights_c = [[0, 1],[1, 1],[1,1]]
blend_weights_f = [[0, 1],[1, 1],[1,1]]

blend_c_005 = np.average([s.values for s in [sub_kaz_c_piv_005, sub_pd_c_piv_005]],axis=0, weights = blend_weights_c[0])
blend_c_05 = np.average([s.values for s in [sub_kaz_c_piv_05, sub_pd_c_piv_05]],axis=0, weights = blend_weights_c[1])
blend_c_095 = np.average([s.values for s in [sub_kaz_c_piv_095, sub_pd_c_piv_095]],axis=0, weights = blend_weights_c[2])

blend_f_005 = np.average([s.values for s in [sub_kaz_f_piv_005, sub_pd_f_piv_005]],axis=0, weights = blend_weights_f[0])
blend_f_05 = np.average([s.values for s in [sub_kaz_f_piv_05, sub_pd_f_piv_05]],axis=0, weights = blend_weights_f[1])
blend_f_095 = np.average([s.values for s in [sub_kaz_f_piv_095, sub_pd_f_piv_095]],axis=0, weights = blend_weights_f[2])


# In[ ]:


import matplotlib.pyplot as plt

plt.style.use(['default'])
fig = plt.figure(figsize = (20, 8))

#for col in ['red', 'grey', 'green', 'purple', 'black', 'yellow', 'blue']:
for col in ['red', 'grey', 'green', 'purple']:
    #idx = np.random.choice(range(len(AREAS)), 1)[0]
    #idx = np.random.choice(np.where([x for x in AREAS if not 'US' in x])[0])
    idx = np.random.choice(np.where(train_c[:,-1] > 100)[0])
    plt.plot(train_c_piv.values[idx], label=AREAS[idx], color=col)
    plt.plot(np.pad(sub_kaz_c_piv_095.values[idx],(95,0)), linestyle='--', color=col)
    plt.plot(np.pad(sub_pd_c_piv_095.values[idx],(95,0)), linestyle='-.', color=col)
    plt.plot(np.pad(blend_c_095[idx],(95,0)), linestyle=':', color=col)

plt.title("Cases")
plt.legend()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

plt.style.use(['default'])
fig = plt.figure(figsize = (20, 8))

#for col in ['red', 'grey', 'green', 'purple', 'black', 'yellow', 'blue']:
for col in ['red', 'grey', 'green', 'purple']:
    #idx = np.random.choice(range(len(AREAS)), 1)[0]
    #idx = np.random.choice(np.where([x for x in AREAS if not 'US' in x])[0])
    idx = np.random.choice(np.where(train_c[:,-1] > 100)[0])
    plt.plot(train_c_piv.values[idx], label=AREAS[idx], color=col)
    plt.plot(np.pad(sub_kaz_c_piv_005.values[idx],(95,0)), linestyle='--', color=col)
    plt.plot(np.pad(sub_pd_c_piv_005.values[idx],(95,0)), linestyle='-.', color=col)
    plt.plot(np.pad(blend_c_005[idx],(95,0)), linestyle=':', color=col)

plt.title("Cases")
plt.legend()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

plt.style.use(['default'])
fig = plt.figure(figsize = (20, 8))

#for col in ['red', 'grey', 'green', 'purple', 'black', 'yellow', 'blue']:
for col in ['red', 'grey', 'green', 'purple']:
    #idx = np.random.choice(range(len(AREAS)), 1)[0]
    #idx = np.random.choice(np.where([x for x in AREAS if not 'US' in x])[0])
    idx = np.random.choice(np.where(train_c[:,-1] > 100)[0])
    plt.plot(train_c_piv.values[idx], label=AREAS[idx], color=col)
    plt.plot(np.pad(sub_kaz_c_piv_05.values[idx],(95,0)), linestyle='--', color=col)
    plt.plot(np.pad(sub_pd_c_piv_05.values[idx],(95,0)), linestyle='-.', color=col)
    plt.plot(np.pad(blend_c_05[idx],(95,0)), linestyle=':', color=col)

plt.title("Cases")
plt.legend()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

plt.style.use(['default'])
fig = plt.figure(figsize = (20, 8))

#for col in ['red', 'grey', 'green', 'purple', 'black', 'yellow', 'blue']:
for col in ['red', 'grey', 'green', 'purple']:
    #idx = np.random.choice(range(len(AREAS)), 1)[0]
    #idx = np.random.choice(np.where([x for x in AREAS if not 'US' in x])[0])
    idx = np.random.choice(np.where(train_f[:,-1] > 100)[0])
    plt.plot(train_f_piv.values[idx], label=AREAS[idx], color=col)
    plt.plot(np.pad(sub_kaz_f_piv_095.values[idx],(95,0)), linestyle='--', color=col)
    plt.plot(np.pad(sub_pd_f_piv_095.values[idx],(95,0)), linestyle='-.', color=col)
    plt.plot(np.pad(blend_f_095[idx],(95,0)), linestyle=':', color=col)

plt.title("Cases")
plt.legend()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

plt.style.use(['default'])
fig = plt.figure(figsize = (20, 8))

#for col in ['red', 'grey', 'green', 'purple', 'black', 'yellow', 'blue']:
for col in ['red', 'grey', 'green', 'purple']:
    #idx = np.random.choice(range(len(AREAS)), 1)[0]
    #idx = np.random.choice(np.where([x for x in AREAS if not 'US' in x])[0])
    idx = np.random.choice(np.where(train_f[:,-1] > 100)[0])
    plt.plot(train_f_piv.values[idx], label=AREAS[idx], color=col)
    plt.plot(np.pad(sub_kaz_f_piv_05.values[idx],(95,0)), linestyle='--', color=col)
    plt.plot(np.pad(sub_pd_f_piv_05.values[idx],(95,0)), linestyle='-.', color=col)
    plt.plot(np.pad(blend_f_05[idx],(95,0)), linestyle=':', color=col)

plt.title("Cases")
plt.legend()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

plt.style.use(['default'])
fig = plt.figure(figsize = (20, 8))

#for col in ['red', 'grey', 'green', 'purple', 'black', 'yellow', 'blue']:
for col in ['red', 'grey', 'green', 'purple']:
    #idx = np.random.choice(range(len(AREAS)), 1)[0]
    #idx = np.random.choice(np.where([x for x in AREAS if not 'US' in x])[0])
    idx = np.random.choice(np.where(train_f[:,-1] > 100)[0])
    plt.plot(train_f_piv.values[idx], label=AREAS[idx], color=col)
    plt.plot(np.pad(sub_kaz_f_piv_005.values[idx],(95,0)), linestyle='--', color=col)
    plt.plot(np.pad(sub_pd_f_piv_005.values[idx],(95,0)), linestyle='-.', color=col)
    plt.plot(np.pad(blend_f_005[idx],(95,0)), linestyle=':', color=col)

plt.title("Cases")
plt.legend()
plt.show()


# In[ ]:


c_piv_005 = sub_kaz_c_piv_005.copy()
c_piv_05 = sub_kaz_c_piv_05.copy()
c_piv_095 = sub_kaz_c_piv_095.copy()
f_piv_005 = sub_kaz_f_piv_005.copy()
f_piv_05 = sub_kaz_f_piv_05.copy()
f_piv_095 = sub_kaz_f_piv_095.copy()

c_piv_005[:] = blend_c_005
c_piv_05[:] = blend_c_05
c_piv_095[:] = blend_c_095
f_piv_005[:] = blend_f_005
f_piv_05[:] = blend_f_05
f_piv_095[:] = blend_f_095

for piv in [c_piv_005,c_piv_05,c_piv_095,f_piv_005,f_piv_05,f_piv_095]:
    piv['Area'] = AREAS

c_piv_005 = c_piv_005.melt(id_vars='Area', var_name='days', value_name="TargetValue")
c_piv_05 = c_piv_05.melt(id_vars='Area', var_name='days', value_name="TargetValue")
c_piv_095 = c_piv_095.melt(id_vars='Area', var_name='days', value_name="TargetValue")
f_piv_005 = f_piv_005.melt(id_vars='Area', var_name='days', value_name="TargetValue")
f_piv_05 = f_piv_05.melt(id_vars='Area', var_name='days', value_name="TargetValue")
f_piv_095 = f_piv_095.melt(id_vars='Area', var_name='days', value_name="TargetValue")


# In[ ]:


quantile_dfs = []
for item in [[c_piv_005,f_piv_005],[c_piv_05,f_piv_05],[c_piv_095,f_piv_095]]:
    item[0]['Target'] = 'ConfirmedCases'
    #item[0].rename(columns={'ConfirmedCases':'TargetValue'},inplace=True)
    item[1]['Target'] = 'Fatalities'
    #item[1].rename(columns={'Fatalities':'TargetValue'},inplace=True)
    df_tmp = pd.concat(item)
    df_tmp = pd.merge(test, df_tmp,  how='left', on=['Area','days','Target']).fillna(0)
    quantile_dfs += [df_tmp[['ForecastId', 'TargetValue']]]


# In[ ]:


taus = [0.05,0.5,0.95]
for i,qdf in enumerate(quantile_dfs):
    qdf['ForecastId_Quantile'] = qdf['ForecastId'].astype(str) + f'_{taus[i]}'
    qdf.drop('ForecastId',axis=1,inplace = True)
    
submission = pd.concat(quantile_dfs)


# In[ ]:


# fix index
sample_sub = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv').set_index('ForecastId_Quantile')
submission = submission.set_index('ForecastId_Quantile')
submission = submission.loc[sample_sub.index]
submission.to_csv('submission.csv')


# In[ ]:


submission.head(20)


# In[ ]:


submission.tail(20)


# In[ ]:





# In[ ]:




