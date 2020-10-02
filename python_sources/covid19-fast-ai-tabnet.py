#!/usr/bin/env python
# coding: utf-8

# # [#masks4all](https://masks4all.co/why-we-need-mandatory-mask-laws-masks4all/)
# 
# # Introduction
# 
# The goal of this notebook is to provide some basic [fast.ai](https://www.fast.ai/), [TabNet (paper here)](https://arxiv.org/abs/1908.07442) model for COVID-19 dataset.
# 
# Although it is not the best approach here (doesn't have an RNN part, for instance), it requires reasonably small amount of code and obviously no feature engineering.
# 
# The solution utilizes mostly [fast.ai](https://www.fast.ai/) library with [fast_tabnet](https://github.com/mgrankin/fast_tabnet) and stuff included in [this course](https://course.fast.ai/)

# In[ ]:


get_ipython().system('pip install fastai2')
get_ipython().system('pip install fast_tabnet')


# In[ ]:


from fastai2.basics import *
from fastai2.tabular.all import *
from fast_tabnet.core import *

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Load input data

# In[ ]:


PATH = '/kaggle/input/covid19-global-forecasting-week-4/'
train_df = pd.read_csv(PATH + 'train.csv', parse_dates=['Date'])
test_df = pd.read_csv(PATH + 'test.csv', parse_dates=['Date'])

add_datepart(train_df, 'Date', drop=False)
add_datepart(test_df, 'Date', drop=False)
train_df.shape


# # Metadata (continent, population, tests per million etc.)

# In[ ]:


PATH1 = '/kaggle/input/covid19-country-data-wk3-release/'

meta_convert_fun = lambda x: np.float32(x) if x not in ['N.A.', '#N/A', '#NULL!'] else np.nan

meta_df = pd.read_csv(PATH1 + 'Data Join - RELEASE.csv', thousands=",",
                     converters={
                         ' TFR ': meta_convert_fun,
                         'Personality_uai': meta_convert_fun,
                     }).rename(columns=lambda x: x.strip())
# meta_df.rename(columns{' TFR '}: 'TFR')

PATH2 = '/kaggle/input/countryinfo/'
countryinfo = pd.read_csv(PATH2 + 'covid19countryinfo.csv', thousands=",", parse_dates=['quarantine', 'schools', 'publicplace', 'gathering', 'nonessential'])
testinfo = pd.read_csv(PATH2 + 'covid19tests.csv', thousands=",")

countryinfo.rename(columns={'region': 'Province_State', 'country': 'Country_Region'}, inplace=True)
testinfo.rename(columns={'region': 'Province_State', 'country': 'Country_Region'}, inplace=True)
testinfo = testinfo.drop(['alpha3code', 'alpha2code', 'date'], axis=1)

PATH3 = '/kaggle/input/covid19-forecasting-metadata/'
continent_meta = pd.read_csv(PATH3 + 'region_metadata.csv').rename(columns={'density': 'pop_density'})
continent_meta = continent_meta[['Country_Region' ,'Province_State', 'continent', 'lat', 'lon', 'pop_density']]

recoveries_meta = pd.read_csv(PATH3 + 'region_date_metadata.csv', parse_dates=['Date'])

def fill_unknown_state(df):
    df.fillna({'Province_State': 'Unknown'}, inplace=True)
    
for d in [train_df, test_df, meta_df, countryinfo, testinfo, continent_meta, recoveries_meta]:
    fill_unknown_state(d)


# # Remove outliers - we don't trust China

# In[ ]:


outliars = ['China']
out_inputs = {}

for out in outliars:
    out_inputs[out] = train_df[train_df['Country_Region'] == out]
    train_df.drop(train_df.index[train_df['Country_Region'] == out], inplace=True)


# In[ ]:


# Save for later before submission.
test_ori = test_df.copy()


merge_cols = ['Province_State', 'Country_Region', 'Date']
test_hlp = test_df[merge_cols + ['ForecastId']]
fst_date = test_hlp.Date.min()
outlier_dfs = []

for out, in_df in out_inputs.items():
    last_date = in_df.Date.max()
    merged = in_df[in_df['Date'] >= fst_date].merge(test_hlp, on=merge_cols, how='left')[['ForecastId', 'ConfirmedCases', 'Fatalities']]
    future_test = test_hlp[(test_hlp['Country_Region'] == out) & (test_hlp['Date'] > last_date)]
    to_add = in_df.groupby(['Province_State', 'Country_Region']).last().reset_index()[['Province_State', 'Country_Region', 'ConfirmedCases', 'Fatalities']]
    merged_future = future_test.merge(to_add, on=['Province_State', 'Country_Region'], how='left')[['ForecastId', 'ConfirmedCases', 'Fatalities']]
    merged = pd.concat([merged, merged_future], sort=True)
    test_df.drop(test_df[test_df['ForecastId'].isin(merged.ForecastId)].index, inplace=True)
    outlier_dfs.append(merged)
    
outlier_df = pd.concat(outlier_dfs, sort=True)


# In[ ]:


outlier_df.index = outlier_df['ForecastId']
outlier_df.drop('ForecastId', axis=1, inplace=True)


# In[ ]:


train_max_date = train_df.Date.max()
outlier_all = test_df[test_df['Date'] <= train_max_date].merge(train_df, on=merge_cols, how='left')[['ForecastId', 'ConfirmedCases', 'Fatalities']]
outlier_all.index = outlier_all.ForecastId
test_df.drop(test_df[test_df['ForecastId'].isin(outlier_all.ForecastId)].index, inplace=True)
outlier_all.drop('ForecastId', axis=1, inplace=True)


# In[ ]:


outlier_df = pd.concat([outlier_df, outlier_all], sort=True)
outlier_df


# # Add external features to input dataframes
# For each country, we want to extract the first day when it reached at least 1 and 50 cases.
# 
# Average fatality rate will be calculated from last data available, simply taking deaths / cases.

# In[ ]:


idx_group = ['Country_Region', 'Province_State']

def day_reached_cases(df, name, no_cases=1):
    """For each country/province get first day of year with at least given number of cases."""
    gb = df[df['ConfirmedCases'] >= no_cases].groupby(idx_group)
    return gb.Dayofyear.first().reset_index().rename(columns={'Dayofyear': name})

def area_fatality_rate(df):
    """Get average fatality rate for last known entry, for each country/province."""
    gb = df[df['Fatalities'] >= 22].groupby(idx_group)
    res_df = (gb.Fatalities.last() / gb.ConfirmedCases.last()).reset_index()
    return res_df.rename(columns={0 : 'FatalityRate'})


# In[ ]:


def joined_data(df):
    res = df.copy()
    
    fatality = area_fatality_rate(train_df)
    first_nonzero = day_reached_cases(train_df, 'FirstCaseDay', 1)
    first_fifty = day_reached_cases(train_df, 'First50CasesDay', 50)
    
    # Add external features
    res = pd.merge(res, continent_meta, how='left')
    res = pd.merge(res, recoveries_meta, how='left')
    res = pd.merge(res, meta_df, how='left')
    res = pd.merge(res, countryinfo, how='left')
    res = pd.merge(res, testinfo, how='left', left_on=idx_group, right_on=idx_group)
    
    # Add calculated features
    res = pd.merge(res, fatality, how='left')
    res = pd.merge(res, first_nonzero, how='left')
    res = pd.merge(res, first_fifty, how='left')
    return res

train_df = joined_data(train_df)
test_df = joined_data(test_df)


# In[ ]:


# It turns out any country in train dataset has at least one case.
train_df.FirstCaseDay.isna().sum()


# # Add temporal features
# 
# Some basic features like number of days since the first case in each country/province with analogous feature for 50 days may be worth adding,
# including containment measures such as closed schools and quarantine.

# In[ ]:


def with_new_features(df):
    res = df.copy()
    add_datepart(res, 'quarantine', prefix='qua')
    add_datepart(res, 'schools', prefix='sql')
    
    res['DaysSinceFirst'] = res['Dayofyear'] - res['FirstCaseDay']
    res['DaysSince50'] = res['Dayofyear'] - res['First50CasesDay']
    res['DaysQua'] = res['Dayofyear'] - res['quaDayofyear']
    res['DaysSql'] = res['Dayofyear'] - res['sqlDayofyear']

    return res
    
train_df = with_new_features(train_df)
test_df = with_new_features(test_df)


# In[ ]:


train_df.shape


# In[ ]:


PATH4 = '/kaggle/input/covid19forecastforvalidset/'
valid_preds = pd.read_csv(PATH4 + 'forecast.csv')

test_with_preds = test_df.merge(valid_preds, how='left')
test_with_preds = test_with_preds[(test_with_preds['Date'] > train_df.Date.max()) & (test_with_preds['Date'] <= '2020-04-28')]
test_with_preds.index += train_df.shape[0]

train_new = pd.concat([train_df, test_with_preds], sort=True).sort_values(by='Date')
train_new


# In[ ]:


# def with_new_objectives(df):
#     res = df.sort_values(['Country_Region', 'Province_State', 'Date'])
#     res['dCases'] = np.clip(res.ConfirmedCases.diff(), a_min=0, a_max=None)
#     res['dFatalities'] = np.clip(res.Fatalities.diff(), a_min=0, a_max=None)
#     return res

# train_new = with_new_objectives(train_new)
train_new = train_new[train_new['Date'] >= '2020-02-20']
train_len = train_new[train_new['Date'] <= train_df['Date'].max()].shape[0]


# # Feature selection
# In fast.ai we can easily select categorical and continuous variables for training.
# 
# I decided not to choose any external data in baseline model. Adding numerical values from country data provided in this notebook doesn't seem to improve the validation score much.

# In[ ]:


# Categorical variables - only basic identifiers, some features like continent will be worth adding.
cat_vars = ['Country_Region', 'Province_State',
            'continent',
#             'publicplace', 'gathering', 'nonessential'
           ]

# Continuous variables - just ones directly connected with time.
cont_vars = ['DaysSinceFirst', 'DaysSince50', 'Dayofyear',
            'DaysQua', 'DaysSql', 'Recoveries',
            'recovered', 'active3', 'newcases3', 'critical3',
            'TRUE POPULATION', 'TFR', 'Personality_uai', 'murder',
            'testper1m', 'positiveper1m',
            'casediv1m', 'deathdiv1m', 
            'FatalityRate',
            'lat', 'lon', 'pop_density', 'urbanpop', 'medianage', 'hospibed','healthperpop',
            'smokers', 'lung', 
#             'continent_gdp_pc', 'continent_happiness', 'continent_Life_expectancy','GDP_region', 
#             'abs_latitude', 'temperature', 'humidity',
            ]

# dep_var = ['dCases', 'dFatalities']
dep_var = ['ConfirmedCases', 'Fatalities']

# TODO: change to train_df when avoiding leakage
# df = train_df[cont_vars + cat_vars + dep_var +['Date']].copy().sort_values('Date')
df = train_new[cont_vars + cat_vars + dep_var +['Date']].copy().sort_values('Date')


# In[ ]:


df


# # Avoid leakage - take only non-overlapping values for training
# 
# For now, the only available data to validate our model is in training set. 
# 
# As our test set starts on **02.04.2020**, we should take only rows before that date for training to avoid leakage.

# In[ ]:


# print(test_df.Date.min())
# MAX_TRAIN_IDX = df[df['Date'] < test_df.Date.min()].shape[0]
MAX_TRAIN_IDX = train_len


# # Preparing training set
# 
# In fast.ai v2, we have to manually take log of dependent variables. Preprocessing setup is vanilla fast.ai stuff,
# we utilize our training index to make a unleaky train/valid split. 
# 
# An important thing is also picking the right batch size in our DataLoader, I've chosen value 512 which is pretty big but doesn't seem to affect the training negatively.

# In[ ]:


df1 = df.copy()
# df1['dCases'] = np.log1p(df1['dCases'])
# df1['dFatalities'] = np.log1p(df1['dFatalities'])
df1['ConfirmedCases'] = np.log1p(df1['ConfirmedCases'])
df1['Fatalities'] = np.log1p(df1['Fatalities'])


# In[ ]:


path = '/kaggle/working/'

procs=[FillMissing, Categorify, Normalize]

splits = list(range(MAX_TRAIN_IDX)), (list(range(MAX_TRAIN_IDX, len(df))))

get_ipython().run_line_magic('time', 'to = TabularPandas(df1, procs, cat_vars.copy(), cont_vars.copy(), dep_var, y_block=TransformBlock(), splits=splits)')


# In[ ]:


dls = to.dataloaders(bs=512, path=path)
dls.show_batch()


# # Processed test set

# # Model
# 
# The model provided here is a simple baseline from fast_tabnet documentation, without any fine tuning.
# 
# It is capable of predicting both confirmed cases and fatalities at once, which is worth noting.

# In[ ]:


emb_szs = get_emb_sz(to); print(emb_szs)


# In[ ]:


dls.c = 2 # Number of outputs we expect from our network - in this case 2.
model = TabNetModel(emb_szs, len(to.cont_names), dls.c, n_d=64, n_a=32, n_steps=3)
# opt_func = partial(Adam, wd=0.01, eps=1e-5)
learn = Learner(dls, model, MSELossFlat(), opt_func=ranger, lr=3e-2, metrics=[rmse])


# In[ ]:


learn.lr_find()


# In[ ]:


learn.fit_one_cycle(10, lr_max=0.33)


# In[ ]:


learn.fit_one_cycle(50, lr_max=0.091)


# In[ ]:


learn.recorder.plot_sched()


# In[ ]:


learn.fit_one_cycle(80, lr_max=5e-2)


# In[ ]:


# cb = SaveModelCallback()
learn.fit_one_cycle(300, lr_max=1e-2)


# In[ ]:


learn.show_results()


# # Preparing submission

# In[ ]:


# learn.load('model')


# In[ ]:


to_tst = to.new(test_df)
to_tst.process()
to_tst.all_cols.head()


# In[ ]:


tst_dl = dls.valid.new(to_tst)
tst_dl.show_batch()


# In[ ]:


learn.metrics = []
tst_preds,_ = learn.get_preds(dl=tst_dl)


# In[ ]:


res1 = np.expm1(tst_preds)
res2 = list(map(lambda x: x[0], res1.numpy()))
res3 = list(map(lambda x: x[1], res1.numpy()))
submit = pd.DataFrame({'ConfirmedCases': res2, 'Fatalities': res3})
submit.index = test_df.ForecastId


# In[ ]:


# test = submit.merge(test_df[['ForecastId', 'Date', 'Country_Region', 'Province_State']], on='ForecastId', how='left')
# test.index = test_df.ForecastId
# begin_date = test.Date.min()
# hlp_train = train_df[train_df['Date'] == begin_date - np.timedelta64(1, 'D')]

# def get_cases(row):
#     if row['Date'] == begin_date:
#         starting = hlp_train[(hlp_train['Country_Region'] == row['Country_Region']) & (hlp_train['Province_State'] == row['Province_State'])].iloc[0]
#         return starting.ConfirmedCases, starting.Fatalities
#     return 0., 0.

# cases = test.apply(get_cases, axis=1)
# test.ConfirmedCases += list(map(lambda x: x[0], cases))
# test.Fatalities += list(map(lambda x: x[1], cases))


# In[ ]:


# submit.ConfirmedCases = test.groupby(['Country_Region', 'Province_State']).ConfirmedCases.cumsum()
# submit.Fatalities = test.groupby(['Country_Region', 'Province_State']).Fatalities.cumsum()


# In[ ]:


submit = pd.concat([submit, outlier_df], sort=True).sort_index()


# In[ ]:


test = submit.merge(test_ori[['ForecastId', 'Date', 'Country_Region', 'Province_State']], on='ForecastId', how='left')
test.index = test_ori.ForecastId
submit.ConfirmedCases = test.groupby(['Country_Region', 'Province_State']).ConfirmedCases.cummax()
submit.Fatalities = test.groupby(['Country_Region', 'Province_State']).Fatalities.cummax()


# In[ ]:


submit.to_csv('submission.csv')
submit


# # Example predictions on our validation set.
# 
# Display some interesting countries to see if our model performs any good.

# In[ ]:


import seaborn as sns

min_date = test_df.Date.min()
max_date = train_df.Date.max()

f, axes = plt.subplots(10, 1, figsize=(16, 60))

# def plot_preds(country, ax):
#     targets = train_df[(train_df['Country_Region'] == country) & (train_df['Date'] >= min_date)].ConfirmedCases
#     subset = test_hlp[(test_hlp['Country_Region'] == country) & (test_hlp['Date'] <= max_date)]
    
#     idx = subset.index
#     dates = subset.Date
#     predicted = submit.iloc[idx].ConfirmedCases
    
#     targets.index = dates
#     predicted.index = dates
    
#     combined = pd.DataFrame({'real' : targets, 'pred': predicted})
    
#     sns.lineplot(data=combined, ax=axes[ax]).set_title(country)

def plot_preds(country, ax):
    subset = test_hlp[(test_hlp['Country_Region'] == country)]
    
    idx = subset.index
    dates = subset.Date
    predicted = submit.iloc[idx].ConfirmedCases
    predicted.index = dates
    
    combined = pd.DataFrame({'pred': predicted})
    
    sns.lineplot(data=combined, ax=axes[ax]).set_title(country)

plot_preds('Italy', 0)
plot_preds('Spain', 1)
plot_preds('Germany', 2)
plot_preds('Poland', 3)
plot_preds('Czechia', 4)
plot_preds('Russia', 5)
plot_preds('Iran', 6)
plot_preds('Sweden', 7)
plot_preds('Japan', 8)
plot_preds('Belgium', 9)


# In[ ]:


submit.iloc[test_hlp[test_hlp['Country_Region'] == 'Poland'].index]


# # Conclusion
# 
# As neural networks are thought not to perform well on tabular data, which is true especially for some trivial architectures like MLP used in fast.ai tabular v1.
# 
# There is quite a lot to explore in this TabNet, in terms of hyperparameters and feature selection, so I would expect a much better performance with tuning.
# 
# # [#masks4all](https://masks4all.co/why-we-need-mandatory-mask-laws-masks4all/)
