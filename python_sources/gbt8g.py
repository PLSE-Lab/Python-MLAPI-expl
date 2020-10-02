#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os, gc
import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import date, datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.linear_model import Ridge
from scipy.optimize import nnls
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
np.set_printoptions(precision=6, suppress=True)


# In[ ]:


# note: clip, dow, lat lon, drop fb and extra, knn on scaled y, reduced cats, mad
mname = 'gbt8g'
path = '/kaggle/input/gbt8gx/'
pathk = '/kaggle/input/covid19-global-forecasting-week-5/'
nhorizon = 31
nhorizon = 31
win = 1
skip = 0
kv = [6,11]
val_scheme = 'forward'
prev_test = False
blend = False
fit_cum = False
train_full = True
save_data = False

booster = ['lgb']
# booster = ['lgb','xgb','ctb','rdg']
# booster = ['cas']

blender = []
# blender = ['nq0j_updated','kaz0z']

quant = [0.05, 0.5, 0.95]
qlab = ['q05','q50','q95']
nq = len(quant)

# for nq final day adjustment
# when validating make this the first validation day
# for final fitting with nhorizon = 30, make it today
TODAY = '2020-05-11'

teams = []

# if using updated daily data, also update time-varying external data
# in COVID-19, covid-19-data, covid-tracking-data, git pull origin master 
# ecdc wget https://opendata.ecdc.europa.eu/covid19/casedistribution/csv
# weather: https://www.kaggle.com/davidbnn92/weather-data/output?scriptVersionId=31103959
# google trends: pytrends0d.ipynb
# data scraped from https://www.worldometers.info/coronavirus/, including past daily snapshots
# download html for final day (country and us states) at 22:00 UTC and run wm0d.ipynb first


# In[ ]:


train = pd.read_csv(pathk+'train.csv')

# helper lists
ynames = ['ConfirmedCases', 'Fatalities']
yv = ['y0','y1']
yvs = ['y0_scaled','y1_scaled']
ny = len(ynames)
cp = ['Country_Region','Province_State','County']
cpd = cp + ['Date']

# from kaz
# train["key"] = train[["Province_State","Country_Region"]].apply(lambda row: \
#                                                 str(row[0]) + "_" + str(row[1]),axis=1)

# fill missing provinces with blanks, must also do this with external data before merging
# need to fillna so groupby works
train[cp] = train[cp].fillna('')
train = train.sort_values(cpd).reset_index(drop=True)

train


# In[ ]:


# having trouble with pivot so doing this
t0 = train.loc[train.Target=='ConfirmedCases'].reset_index(drop=True)
t0 = t0.rename(mapper={'TargetValue':'ConfirmedCases'}, axis=1)
t0['Fatalities'] = train.loc[train.Target=='Fatalities','TargetValue'].values
t0.drop('Target',axis=1,inplace=True)
train = t0
train


# In[ ]:


# pivot to create two target columns as in past weeks, allows feature engineering to stay the same
# remember to multiply weight by 10 when computing pinball loss for fatalities
# train.loc[train.Target=='Fatalities','Weight'] = train.loc[train.Target=='ConfirmedCases','Weight'].values 
# train


# In[ ]:


# t = train.set_index(cpd+['Population'])
# t.drop('Id',axis=1,inplace=True)
# t


# In[ ]:


# # t = pd.pivot(train, index = cpd+['Population'], columns = 'Target', \
# #                            values = 'TargetValue').reset_index()
# t = pd.pivot(t, columns = 'Target', \
#                            values = 'TargetValue')
# t


# In[ ]:


# use previous week test set in order to compare with previous week leaderboard
if prev_test:
    test = pd.read_csv('../'+pw+'/test.csv')
    ss = pd.read_csv('../'+pw+'/submission.csv')
else:
    test = pd.read_csv(pathk+'test.csv')
    ss = pd.read_csv(pathk+'submission.csv')

# from kaz
# test["key"] = test[["Province_State","Country_Region"]].apply(lambda row: \
#                                             str(row[0]) + "_" + str(row[1]),axis=1)

test[cp] = test[cp].fillna('')
test


# In[ ]:


# having trouble with pivot so doing this
t1 = test.loc[test.Target=='ConfirmedCases'].reset_index(drop=True)
t1.drop('Target',axis=1,inplace=True)
test = t1
test


# In[ ]:


# tmax and dmax are the last day of training
tmax = train.Date.max()
dmax = datetime.strptime(tmax,'%Y-%m-%d').date()
print(tmax, dmax)


# In[ ]:


fmax = test.Date.max()
fdate = datetime.strptime(fmax,'%Y-%m-%d').date()
fdate


# In[ ]:


tmin = train.Date.min()
fmin = test.Date.min()
tmin, fmin


# In[ ]:


dmin = datetime.strptime(tmin,'%Y-%m-%d').date()
print(dmin)


# In[ ]:


# prepare for concat
train = train.merge(test[cpd+['ForecastId']], how='left', on=cpd)
train['ForecastId'] = train['ForecastId'].fillna(0).astype(int)

test['Id'] = test.ForecastId + train.Id.max()
test['ConfirmedCases'] = np.nan
test['Fatalities'] = np.nan

# storage for predictions
for i in range(ny):
    for q in range(nq):
        train[yv[i]+'_pred_'+qlab[q]] = np.nan
        # use zeros here instead of nans so monotonic adjustment fills final dates if necessary
        test[yv[i]+'_pred_'+qlab[q]] = 0.0


# In[ ]:


# concat non-overlapping part of test to train for feature engineering
d = pd.concat([train,test[test.Date > train.Date.max()]],sort=True).reset_index(drop=True)
d


# In[ ]:


# day counter
d['day'] = pd.to_datetime(d['Date']).dt.dayofyear
d['day'] -= d['day'].min()


# In[ ]:


# override weights, adjust US due to roll-ups
# d['Weight'] = 0.1
# d.loc[d.Country_Region=='US','Weight'] = 0.1/3.0


# In[ ]:


# reduce data to US county only
# d = d.loc[(d.Country_Region=='US') & (d.County!='')]
# d = d.loc[(d.Country_Region!='US') | (d.County=='')]
# d


# In[ ]:


d['Date'].value_counts().std()


# In[ ]:


# fill missing province with blank, must also do this with external data before merging
d[cp] = d[cp].fillna('')

# create single location variable
d['Loc'] = d['Country_Region'] + ' ' + d['Province_State'] + ' ' + d['County']
d.loc[:,'Loc'] = d['Loc'].str.strip()
d['Loc'].value_counts()


# In[ ]:


# previous test set, drop new regions in order to compare with previous week leaderboard
if prev_test:
    test2 = pd.read_csv('../'+pw+'/test.csv')
    test2[cp] = test2[cp].fillna('')
    test2 = test2.drop(['ForecastId','Date'], axis=1).drop_duplicates()
    test2
    d = d.merge(test2, how='inner', on=cp)
    d.shape


# In[ ]:


# sort by location, date
d = d.sort_values(['Loc','Date']).reset_index(drop=True)


# In[ ]:


ynames


# In[ ]:


# target creation and cleaning
yvc = []
# qs = [0.95]
for i in range(ny):
    v = yv[i]
    d[v] = d[ynames[i]]
    # crude fix:  replace negative targets with zeros
    # should prolly use negatives to reduce previous positives, see Spain and France
    d.loc[d[v] < 0, v] = 0
    # create log1p cum counts as in weeks 1-4
    vc = v+'cum'
    d[vc] = d.groupby('Loc')[v].cumsum()
    # d[vc] = np.log1p(d[vc])
    yvc.append(vc)
#     da = d.groupby(['Loc']).agg({v:{ "min" , "max", "median"} } )
#     da.columns = ['_'.join(c) for c in da.columns]
#     da = da.reset_index()
#     d = d.merge(da, how='left', on='Loc')
#     d[v+'_min'] = d.groupby(['Loc'])[v].apply(lambda x: x.min())
#     d[v+'_med'] = d.groupby(['Loc'])[v].apply(lambda x: x.median())
#     d[v+'_max'] = d.groupby(['Loc'])[v].apply(lambda x: x.max())
#     d[v+'_range'] = d[v+'_max'] - d[v+'_min'] 
    # robust center
    # d[v] = d[v] - d[v+'_median']
    # robust center and scale 
    # d[v] = (d[v] - d[v+'_median'])/(1e-8 + d[v+'_range']) 
            
# enforce monotonicity, roughly cleans some data errors
# d[yv] = d.groupby(cp)[yv].cummax()

print(d[yv+yvc].describe())


# In[ ]:


# find max of smooth and compute scale factors
def ewma(x, com):
    return pd.Series.ewm(x, com=com).mean()

w = 7
ws = '_w'+str(w)

for i in range(ny):
    yw = yv[i]+ws
    d[yw] = d.groupby('Loc')[yv[i]].transform(lambda x: ewma(x,w))
    # print(d[yw].describe())

    d1 = d.sort_values(['Loc', yw,'Date'], ascending=[True,False,True])
    # print(d1.head())

    d2 = d1.groupby('Loc').first()
    # print(d2.head())

    d2 = d2[['day',yw]].reset_index()
    d2.columns = ['Loc','day_of_max'+str(i),  yw+'_max']
    d2

    d = d.merge(d2, how='left', on='Loc')
    # print(d.shape)

    d[yw+'_d1'] = d.groupby('Loc')[yw].transform(lambda x: x.diff(1))
    d1 = d[d.day==d['day_of_max'+str(i)]]
    d1 = d1.sort_values([yw+'_d1'], ascending=False)
    # print(d1.head(n=10))

    domax = d1['day'].max()
    print(domax)

    d['scale_factor'+str(i)] = 1.0 - np.maximum(10.0 - (domax - d['day_of_max'+str(i)]), 0.0)/20.0
    print(d['scale_factor'+str(i)].describe())
    print(d.shape)


# In[ ]:


# d.groupby('Loc').agg({})
d.shape


# In[ ]:


d.columns


# In[ ]:


# d.groupby('Loc')['y0'].quantile(0.95)


# In[ ]:


# d.loc[d.Loc=='US',['Date','y0','y0cum','y1','y1cum']][95:105]


# In[ ]:


# d.loc[d.Loc=='Spain',['Date','y0','y0cum','y1','y1cum']][95:105]


# In[ ]:


d['dow'] = pd.to_datetime(d.Date).dt.dayofweek


# In[ ]:


d['dow'].value_counts()


# In[ ]:


# d.shape


# In[ ]:


# merge fbprophet features
fbf = []
# gd = ['Loc','Date']
# for i in range(ny):
#     fb = pd.read_parquet(path+'fbprophet0d'+str(i)+'.pq')
#     fb.drop([f for f in fb.columns if f.startswith('mult')], axis=1, inplace=True)
#     fbf = fbf + [f for f in fb.columns if f not in gd]
#     fb.loc[:,'Date'] = fb.Date.astype(str)
#     fb = fb.sort_values(gd).reset_index(drop=True)
#     d = d.merge(fb, how='left', on=gd)
#     print(d.shape)
# print(fbf)


# In[ ]:


# # data scraped from https://www.worldometers.info/coronavirus/, including past daily snapshots
# # download html for final day (country and us states) at 22:00 UTC and run wm0d.ipynb first
# wmf = []
# wm = pd.read_csv('wmc.csv')
# wm[cp] = wm[cp].fillna('')
# # 12 new features, all log1p transformed, must be lagged
# wmf = [c for c in wm.columns if c not in cpd]

# # since wm leads by a day, shift the date to make it contemporaneous
# wmax = wm.Date.max()
# wmax = datetime.strptime(wmax,'%Y-%m-%d').date()
# woff = (dmax - wmax).days
# print(dmax, wmax, woff)
# wm1 = wm.copy()
# wm1['Date'] = (pd.to_datetime(wm1.Date) + timedelta(woff)).dt.strftime('%Y-%m-%d')

# wm1.Date.value_counts()[:10]


# In[ ]:


# wm1['Date'].max()


# In[ ]:


# d = d.merge(wm1, how='left', on=cpd)
# print(d.shape)
# d[wmf].describe()


# In[ ]:


# # google trends
# gt = pd.read_csv(path+'google_trends.csv')
# gt[cp] = gt[cp].fillna('')
# gt


# In[ ]:


# # since trends data lags behind a day or two, shift the date to make it contemporaneous
# gmax = gt.Date.max()
# gmax = datetime.strptime(gmax,'%Y-%m-%d').date()
# goff = (dmax - gmax).days
# print(dmax, gmax, goff)
# gt['Date'] = (pd.to_datetime(gt.Date) + timedelta(goff)).dt.strftime('%Y-%m-%d')
# gt['google_covid'] = gt['coronavirus'] + gt['covid-19'] + gt['covid19']
# gt.drop(['coronavirus','covid-19','covid19'], axis=1, inplace=True)
# google = ['google_covid']
# gt


# In[ ]:


# d = d.merge(gt, how='left', on=['Country_Region','Province_State','Date'])
# d


# In[ ]:


# d['google_covid'].describe()


# In[ ]:


# # merge country info
# country = pd.read_csv(path+'covid19countryinfo2.csv')
# # country["pop"] = country["pop"].str.replace(",","").astype(float)
# country


# In[ ]:


# country.columns


# In[ ]:


# # first merge by country
# d = d.merge(country.loc[country.medianage.notnull(),['country','pop','testpop','medianage']],
#             how='left', left_on='Country_Region', right_on='country')
# d


# In[ ]:


# # then merge by province
# c1 = country.loc[country.medianage.isnull(),['country','pop','testpop']]
# print(c1.shape)
# c1.columns = ['Province_State','pop1','testpop1']
# # d.update(c1)
# d = d.merge(c1,how='left',on='Province_State')
# d.loc[d.pop1.notnull(),'pop'] = d.loc[d.pop1.notnull(),'pop1']
# d.loc[d.testpop1.notnull(),'testpop'] = d.loc[d.testpop1.notnull(),'testpop1']
# d.drop(['pop1','testpop1'], axis=1, inplace=True)
# print(d.shape)
# print(d.loc[(d.Date=='2020-03-25') & (d['Province_State']=='New York')])


# In[ ]:


# covid tracking data
# testing data time series, us states only, would love to have this for all countries
ct = pd.read_csv(path+'states_daily_4pm_et.csv')
si = pd.read_csv(path+'states_info.csv')
si = si.rename(columns={'name':'Province_State'})
ct = ct.merge(si[['state','Province_State']], how='left', on='state')
ct['Date'] = ct['date'].apply(str).transform(lambda x: '-'.join([x[:4], x[4:6], x[6:]]))
ct.loc[ct.Province_State=='US Virgin Islands','Province_State'] = 'Virgin Islands'
ct.loc[ct.Province_State=='District Of Columbia','Province_State'] = 'District of Columbia'
pd.set_option('display.max_rows', 20)
ct
# ct = ct['Date','state','total']


# In[ ]:


ckeep = ['positiveIncrease','negativeIncrease','totalTestResultsIncrease']
# for c in ckeep: ct[c] = np.log1p(ct[c])


# In[ ]:


d = d.merge(ct[['Province_State','Date']+ckeep], how='left',
            on=['Province_State','Date'])
d


# In[ ]:


# covid tracking data
# testing data time series, us rollup
ct = pd.read_csv(path+'us_daily.csv')
ct['Date'] = ct['date'].apply(str).transform(lambda x: '-'.join([x[:4], x[4:6], x[6:]]))
ct = ct[['Date']+ckeep]
ckeep1 = [c+'1' for c in ckeep]
ct.columns = ['Date'] + ckeep1
ct['Country_Region'] = 'US'
ct['Province_State'] = ''
ct['County'] = ''
ct


# In[ ]:


# update ckeep for us rollup
d = d.merge(ct, how='left', on=cpd)
d.shape


# In[ ]:


# update
d.loc[d.Loc=='US',ckeep] = d.loc[d.Loc=='US',ckeep1]
d.drop(ckeep1, axis=1, inplace=True)
d.shape


# In[ ]:


# # approximate county tracking values using ratios from training
# cs = ['Country_Region','Province_State']
# d['csum'] = d.groupby(cs)['ConfirmedCases'].transform(lambda x:x.sum())
# d['fsum'] = d.groupby(cs)['Fatalities'].transform(lambda x:x.sum())
# d['cfrac'] = d.ConfirmedCases / d.csum
# d['ffrac'] = d.Fatalities / d.fsum 
# for c in ckeep: d[c] *= d.cfrac
# ckeep = ckeep + ['cfrac','ffrac']


# In[ ]:


# # weather data from from davide bonine
# w = pd.read_csv(path+'training_data_with_weather_info_week_4.csv')
# w.drop(['Id','ConfirmedCases','Fatalities','country+province','day_from_jan_first'], axis=1, inplace=True)
# w[cp] = w[cp].fillna('')
# wf = list(w.columns[5:])
# w


# In[ ]:


# w.describe()


# In[ ]:


# # replace values
# w['ah'] = w['ah'].replace(to_replace={np.inf:np.nan})
# w['wdsp'] = w['wdsp'].replace(to_replace={999.9:np.nan})
# w['prcp'] = w['prcp'].replace(to_replace={99.99:np.nan})
# w.describe()


# In[ ]:


# w[['Country_Region','Province_State']].nunique()


# In[ ]:


# w[['Country_Region','Province_State']].drop_duplicates().shape


# In[ ]:


# # since weather data may lag behind a day or two, adjust the date to make it contemporaneous
# wmax = w.Date.max()
# wmax = datetime.strptime(wmax,'%Y-%m-%d').date()
# woff = (dmax - wmax).days
# print(dmax, wmax, woff)
# w['Date'] = (pd.to_datetime(w.Date) + timedelta(woff)).dt.strftime('%Y-%m-%d')
# w


# In[ ]:


# merge Lat and Long for all times and the time-varying weather data based on date
geo = pd.read_csv(path+'geo_all.csv')
geo.loc[:,cp] = geo[cp].fillna('') 
geo


# In[ ]:


d = d.merge(geo, how='left', on=cp)
d.shape


# In[ ]:


# # combine ecdc and nytimes data as extra y0 and y1
# ecdc = pd.read_csv(path+'ecdc.csv', encoding = 'latin')
# ecdc


# In[ ]:


# # https://opendata.ecdc.europa.eu/covid19/casedistribution/csv
# ecdc['Date'] = pd.to_datetime(ecdc[['year','month','day']]).dt.strftime('%Y-%m-%d')
# ecdc = ecdc.rename(mapper={'countriesAndTerritories':'Country_Region'}, axis=1)
# ecdc['Country_Region'] = ecdc['Country_Region'].replace('_',' ',regex=True)
# ecdc['Province_State'] = ''
# ecdc['County'] = ''

# ecdc['extra_y0'] = ecdc.cases
# ecdc['extra_y1'] = ecdc.deaths

# # ecdc['cc'] = ecdc.groupby(cp)['cases'].cummax()
# # ecdc['extra_y0'] = np.log1p(ecdc.cc)
# # ecdc['cd'] = ecdc.groupby(cp)['deaths'].cummax()
# # ecdc['extra_y1'] = np.log1p(ecdc.cd)

# ecdc = ecdc[cpd + ['extra_y0','extra_y1']]
# ecdc[::63]


# In[ ]:


# ecdc = ecdc[(ecdc.Date >= '2020-01-22')]
# ecdc


# In[ ]:


# # new york times data
# # https://github.com/nytimes/covid-19-data
# # full us roll-up
# n0 = pd.read_csv(path+'covid-19-data/us.csv')
# n0['extra_y0'] = n0.cases.diff(1)
# n0['extra_y1'] = n0.deaths.diff(1)
# n0['Country_Region'] = 'US'
# n0['Province_State'] = ''
# n0['County'] = ''
# n0 = n0.rename(mapper={'date':'Date'},axis=1)
# n0.drop(['cases','deaths'],axis=1,inplace=True)
# n0


# In[ ]:


# # new york times data
# # https://github.com/nytimes/covid-19-data
# # us state-level
# n1 = pd.read_csv(path+'covid-19-data/us-states.csv')
# n1 = n1.sort_values(['state','date']).reset_index(drop=True)
# n1['extra_y0'] = n1.groupby('state')['cases'].diff(1)
# n1['extra_y1'] = n1.groupby('state')['deaths'].diff(1)
# n1['Country_Region'] = 'US'
# n1['County'] = ''
# n1 = n1.rename(mapper={'date':'Date','state':'Province_State'},axis=1)
# n1.drop(['fips','cases','deaths'],axis=1,inplace=True)
# n1


# In[ ]:


# # new york times data
# # https://github.com/nytimes/covid-19-data
# # us county-level
# n2 = pd.read_csv(path+'covid-19-data/us-counties.csv')
# n2 = n2.sort_values(['state','county','date']).reset_index(drop=True)
# n2['extra_y0'] = n2.groupby(['state','county'])['cases'].diff(1)
# n2['extra_y1'] = n2.groupby(['state','county'])['deaths'].diff(1)
# n2['Country_Region'] = 'US'
# n2 = n2.rename(mapper={'date':'Date','state':'Province_State','county':'County'},axis=1)
# n2.drop(['fips','cases','deaths'],axis=1,inplace=True)
# # fix for new york city, 4 out of 5 boroughs are left missing
# n2.loc[n2.County=='New York City','County'] = 'New York'
# n2.loc[n2.County=='New York']


# In[ ]:


# n2.sort_values('extra_y0', ascending=False)


# In[ ]:


# extra = pd.concat([ecdc,n0,n1,n2], sort=True)
# extra


# In[ ]:


# d = d.merge(extra, how='left', on=cpd)
# d


# In[ ]:


# # enforce monotonicity
# d = d.sort_values(['Loc','Date']).reset_index(drop=True)
# for y in yv:
#     ey = 'extra_'+y
#     d[ey] = d[ey].fillna(0.)
#     d[ey] = d.groupby('Loc')[ey].cummax()


# In[ ]:


# d[['y0','y1','extra_y0','extra_y1']].describe()


# In[ ]:


# # impute us state data prior to march 10
# for i in range(ny):
#     ei = 'extra_'+yv[i]
#     qm = (d.Country_Region == 'US') & (d.Date < '2020-03-10') & (d[ei].notnull())
#     print(i,sum(qm))
#     d.loc[qm,yv[i]] = d.loc[qm,ei]


# In[ ]:


# d[['y0','y1']].describe()


# In[ ]:


# plt.plot(d.loc[d.Province_State=='New York','y0'])
dq = (d.Country_Region=='US') & (d.Province_State=='') & (d.County=='')
plt.plot(d.loc[dq,['y0','Date']].set_index('Date'))
# plt.plot(d.loc[dq,['extra_y0','Date']].set_index('Date'))


# In[ ]:


d.columns


# In[ ]:


# log rates
d['rate0'] = np.log1p(d.y0) - np.log(d['Population'])
d['rate1'] = np.log1p(d.y1) - np.log(d['Population'])
d[['rate0','rate1']].describe()


# In[ ]:


# # recovered data from hopkins, https://github.com/CSSEGISandData/COVID-19
# recovered = pd.read_csv(path+'COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
# recovered = recovered.rename(mapper={'Country/Region':'Country_Region','Province/State':'Province_State'}, axis=1)
# recovered[cp] = recovered[cp].fillna('')
# recovered = recovered.drop(['Lat','Long'], axis=1)
# recovered


# In[ ]:


# # replace US row with identical rows for every US state
# usp = d.loc[d.Country_Region=='US','Province_State'].unique()
# print(usp, len(usp))
# rus = recovered[recovered.Country_Region=='US']
# rus


# In[ ]:


# rus = rus.reindex(np.repeat(rus.index.values,len(usp)))
# rus.loc[:,'Province_State'] = usp
# rus


# In[ ]:


# recovered =  recovered[recovered.Country_Region!='US']
# recovered = pd.concat([recovered,rus]).reset_index(drop=True)
# recovered


# In[ ]:


# # melt and merge
# rm = pd.melt(recovered, id_vars=cp, var_name='d', value_name='recov')
# rm


# In[ ]:


# rm['Date'] = pd.to_datetime(rm.d)
# rm.drop('d',axis=1,inplace=True)
# rm['Date'] = rm['Date'].dt.strftime('%Y-%m-%d')
# rm


# In[ ]:


# d = d.merge(rm, how='left', on=['Country_Region','Province_State','Date'])
# d


# In[ ]:


# d['recov'].describe()


# In[ ]:


# # approximate US state recovery via proportion of confirmed cases
# d['ccsum'] = d.groupby(['Country_Region','Date'])['ConfirmedCases'].transform(lambda x: x.sum())
# d.loc[d.Country_Region=='US','recov'] = d.loc[d.Country_Region=='US','recov'] * \
#                                         d.loc[d.Country_Region=='US','ConfirmedCases'] / \
#                                         (d.loc[d.Country_Region=='US','ccsum'] + 1)


# In[ ]:


# d.loc[:,'recov'] = np.log1p(d.recov)
# # d.loc[:,'recov'] = d['recov'].fillna(0)


# In[ ]:


# # enforce monotonicity
# d = d.sort_values(['Loc','Date']).reset_index(drop=True)
# d['recov'] = d['recov'].fillna(0.)
# d['recov'] = d.groupby('Loc')['recov'].cummax()


# In[ ]:


# d.loc[d.Province_State=='North Carolina','recov'][45:55]


# In[ ]:


d = d.sort_values(['Loc','Date']).reset_index(drop=True)
d.shape


# In[ ]:


d['Date'] = pd.to_datetime(d['Date'])
d['Date'].describe()


# In[ ]:


# days since beginning
# basedate = train['Date'].min()
# train['dint'] = train.apply(lambda x: (x.name.to_datetime() - basedate).days, axis=1)
d['dint'] = (d['Date'] - d['Date'].min()).dt.days
d['dint'].describe()


# In[ ]:


d.shape


# In[ ]:


# reference days since exp(j)th occurrence
for i in range(ny):
    
    for j in range(3):

        ij = str(i)+'_'+str(j)
        
        cut = 2**j if i==0 else j
        
        qd1 = (d[yvc[i]] > cut) & (d[yvc[i]].notnull())
        d1 = d.loc[qd1,['Loc','dint']]
        # d1.shape
        # d1.head()

        # get min for each location
        d1['dmin'] = d1.groupby('Loc')['dint'].transform(lambda x: x.min())
        # dintmax = d1['dint'].max()
        # print(i,j,'dintmax',dintmax)
        # d1.head()

        d1.drop('dint',axis=1,inplace=True)
        d1 = d1.drop_duplicates()
        d = d.merge(d1,how='left',on=['Loc'])
 
        # if dmin is missing then the series had no occurrences in the training set
        # go ahead and assume there will be one at the beginning of the test period
        # the average time between first occurrence and first death is 14 days
        # if j==0: d[dmi] = d[dmi].fillna(dintmax + 1 + i*14)

        # ref day is days since dmin, must clip at zero to avoid leakage
        d['ref_day'+ij] = np.clip(d.dint - d.dmin, 0, None)
        d['ref_day'+ij] = d['ref_day'+ij].fillna(0)
        d.drop('dmin',axis=1,inplace=True)

        # asymptotic curve may bin differently
        d['recip_day'+ij] = 1 / (1 + (1 + d['ref_day'+ij])**(-1.0))
    

gc.collect()


# In[ ]:


d['dint'].value_counts().std()


# In[ ]:


d[[f for f in d.columns if f.startswith('ref_day')]].describe()


# In[ ]:


d[['scale_factor0','y0_w7_max']].describe()


# In[ ]:


# diffs and rolling means
# note lags are taken dynamically at run time
e = 1
# r = 5
r = 7
wr = [2]
# wr = [2,5]
gb = d.groupby('Loc')

for i in range(ny):
    yi = 'y'+str(i)
    yis = yi+'_scaled'
    dd = '_d'+str(e)
    rr = '_r'+str(r)
    
    for w in wr:
        ww = '_w'+str(w)
        d[yi+ww] = gb[yi].transform(lambda x: ewma(x,w))
        d['rate'+str(i)+ww] = gb['rate'+str(i)].transform(lambda x: ewma(x,w))
        
    # scale target by smooth max + scale factor penalty for recent max
    d[yis] = d['scale_factor'+str(i)] * d[yi+ww] / (1e-4 + d[yi+'_w7_max'])
    print(d[yis].describe())
    
    for j in range(7):
        d[yi+'_d'+str(1+j)] = gb[yi].transform(lambda x: x.diff(1+j))
        d[yi+'_l'+str(1+j)] = gb[yi].transform(lambda x: x.shift(1+j))
        d[yis+'_d'+str(1+j)] = gb[yis].transform(lambda x: x.diff(1+j))
        d[yis+'_l'+str(1+j)] = gb[yis].transform(lambda x: x.shift(1+j))
    
    d[yi+rr] = gb[yi].transform(lambda x: x.rolling(r).mean())
    d[yis+rr] = gb[yis].transform(lambda x: x.rolling(r).mean())
        
    d['rate'+str(i)+dd] = gb['rate'+str(i)].transform(lambda x: x.diff(e))
    d['rate'+str(i)+rr] = gb['rate'+str(i)].transform(lambda x: x.rolling(r).mean())
    
#     d['extra_y'+str(i)+dd] = gb['extra_y'+str(i)].transform(lambda x: x.diff(e))
#     d['extra_y'+str(i)+rr] = gb['extra_y'+str(i)].transform(lambda x: x.rolling(r).mean())
#     d['extra_y'+str(i)+ww] = gb['extra_y'+str(i)].transform(lambda x: ewma(x,w))
        
# vlist = ['recov'] + google + wf

# for v in vlist:
#     d[v+dd] = gb[v].transform(lambda x: x.diff(e))
#     d[v+rr] = gb[v].transform(lambda x: x.rolling(r).mean())
#     d[v+ww] = gb[v].transform(lambda x: ewma(x,w))


# In[ ]:


d['y0'+ww].describe()


# In[ ]:


# compute nearest neighbors
regions = d[['Loc','lat','lon']].drop_duplicates('Loc').reset_index(drop=True)
regions


# In[ ]:


# regions.to_csv('regions.csv', index=False)


# In[ ]:


# knn max features
k = kv[0]
nn = NearestNeighbors(k)
nn.fit(regions[['lat','lon']])


# In[ ]:


# first matrix is distances, second indices to nearest neighbors including self
# note two cruise ships have identical lat, lon values
knn = nn.kneighbors(regions[['lat','lon']])
knn


# In[ ]:


ns = d['Loc'].nunique()


# In[ ]:


# time series matrix
ky = d['y0'].values.reshape(ns,-1)
print(ky.shape)

print(ky[0])

# use knn indices to create neighbors
knny = ky[knn[1]]
print(knny.shape)

knny = knny.transpose((0,2,1)).reshape(-1,k)
print(knny.shape)


# In[ ]:


# knn max features
nk = len(kv)
kp = []
kd = []
ns = regions.shape[0]
for k in kv:
    nn = NearestNeighbors(k)
    nn.fit(regions[['lat','lon']])
    knn = nn.kneighbors(regions[['lat','lon']])
    kp.append('knn'+str(k)+'_')
    kd.append('kd'+str(k)+'_')
    for i in range(ny):
        yis = 'y'+str(i)+'_scaled'
        kc = kp[-1]+yis
        # time series matrix
        ky = d[yis].values.reshape(ns,-1)
        # use knn indices to create neighbor matrix
        km = ky[knn[1]].transpose((0,2,1)).reshape(-1,k)
        
        # take maximum value over all neighbors to approximate spreading
        d[kc] = np.amax(km, axis=1)
        print(d[kc].describe())
        print()
        
        # distance to max
        kc = kd[-1]+yis
        ki = np.argmax(km, axis=1).reshape(ns,-1)
        kw = np.zeros_like(ki).astype(float)
        # inefficient indexing, surely some way to do it faster
        for j in range(ns): 
            kw[j] = knn[0][j,ki[j]]
        d[kc] = kw.flatten()
        print(d[kc].describe())
        print()


# In[ ]:


# correlations for knn features
cols = []
for i in range(ny):
    yi = yv[i]
    yis = yi+'_scaled'
    cols.append(yi)
    cols.append(yis)
    for k in kp:
        cols.append(k+yis)
d.loc[:,cols].corr()


# In[ ]:


# smooth knn features
for i in range(ny):
    yi = 'y'+str(i)
    yis = yi+'_scaled'
    dd = '_d'+str(e)
    rr = '_r'+str(r)
    
    for k in kp:
        d[k+yis+dd] = gb[k+yis].transform(lambda x: x.diff(e))
        d[k+yis+rr] = gb[k+yis].transform(lambda x: x.rolling(r).mean())
        d[k+yis+ww] = gb[k+yis].transform(lambda x: ewma(x,w))

    for k in kd:
        d[k+yis+dd] = gb[k+yis].transform(lambda x: x.diff(e))
        d[k+yis+rr] = gb[k+yis].transform(lambda x: x.rolling(r).mean())
        d[k+yis+ww] = gb[k+yis].transform(lambda x: ewma(x,w))


# In[ ]:


# remove US data before 2020-03-10, instead might should impute from ny times
# must do this after knn because it assumes balanced data
# note also 4/5 NY boroughs are all 0s:  Bronx, Queens, Kings, Richmond
d = d[(d.Country_Region!='US') | (d.Date >= '2020-03-10')]
# d = d[d.Date >= '2020-03-10']
print(d.shape)


# In[ ]:


# drop leading zeros, must do after knn
qr = (d.ref_day0_0 > 0) | (d.Date >= '2020-04-06')
d = d[qr]
print(d.shape)


# In[ ]:


# final sort before training
d = d.sort_values(['Loc','dint']).reset_index(drop=True)
d.shape


# In[ ]:


# # save data'
# if save_data:
#     fname = mname + '_data.csv'
#     d.to_csv(fname, index=False)
#     print(fname, d.shape)


# In[ ]:


# # range of dates for training
# # dates = d[~d.y0.isnull()]['Date'].drop_duplicates()
# dates = d[d.y0.notnull()]['Date'].drop_duplicates()
# dates


# In[ ]:


# initial continuous and categorical features
# dogs = tfeats
# ref_day0_0 is no longer leaky since every location has at least one confirmed case
# dogs = ['ref_day0_0']
dogs = ['Population','Weight','lat','lon']
# dogs = ['Population','Weight','dow']
# cats = ['Loc','dow','continent']
# cats = ['dow','continent']
cats = ['Country_Region','Province_State','dow','continent']
# cats = []
print(dogs, len(dogs))
print(cats, len(cats))


# In[ ]:


# one-hot encode categorical features
ohef = []
for i,c in enumerate(cats):
    print(c, d[c].nunique())
    ohe = pd.get_dummies(d[c], prefix=c)
    ohec = [f.translate({ord(c): "_" for c in " !@#$%^&*()[]{};:,./<>?\|`~-=_+"}) for f in list(ohe.columns)]
    ohe.columns = ohec
    d = pd.concat([d,ohe],axis=1)
    ohef = ohef + ohec


# In[ ]:


# d['Loc_US_North_Carolina'].describe()


# In[ ]:


# d['Loc_US_Colorado'].describe()


# In[ ]:


# must start cas server from gevmlax02 before running this cell
# ssh rdcgrd001 /opt/vb025/laxnd/TKGrid/bin/caslaunch stat -mode mpp -cfg /u/sasrdw/config.lua
if 'cas' in booster:
    from swat import *
    s = CAS('rdcgrd001.unx.sas.com', 16695)


# In[ ]:


# boosting hyperparameters
params = {}

# # from vopani
# SEED = 345
# LGB_PARAMS = {"objective": "regression",
#               "num_leaves": 5,
#               "learning_rate": 0.013,
#               "bagging_fraction": 0.91,
#               "feature_fraction": 0.81,
#               "reg_alpha": 0.13,
#               "reg_lambda": 0.13,
#               "metric": "rmse",
#               "seed": SEED
#              }

# from oscii
SEED = 42
LGB_PARAMS = {
    'objective': 'quantile',
    # 'metric': 'rmse',
    'metric': 'quantile',
    'alpha': 0.5,
    'num_leaves': 8,
    'min_data_in_leaf': 5,  # 42,
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
    'min_gain_to_split': 0.02,  # 0.01077313523861969,
    'min_child_weight': 5,  # 19.428902804238373,
    # 'num_threads': 6,
    # 'device_type': 'gpu'
}

params[('lgb','y0')] = LGB_PARAMS
params[('lgb','y1')] = LGB_PARAMS
# params[('lgb','y0')] = {'lambda_l2': 1.9079933811271934, 'max_depth': 5}
# params[('lgb','y1')] = {'lambda_l2': 1.690407455211948, 'max_depth': 3}
params[('xgb','y0')] = {'lambda_l2': 1.9079933811271934, 'max_depth': 5}
params[('xgb','y1')] = {'lambda_l2': 1.690407455211948, 'max_depth': 3}
params[('ctb','y0')] = {'l2_leaf_reg': 1.9079933811271934, 'max_depth': 5}
params[('ctb','y1')] = {'l2_leaf_reg': 1.690407455211948, 'max_depth': 3}

# fix number of estimators to avoid overfitting with early stopping
# index by target and quantile, based on previous early stopping results in iallv
lgb_nest = np.zeros((2,3)).astype(int)
lgb_nest[0,0] = 1000
lgb_nest[1,0] = 1000
lgb_nest[0,1] = 400
lgb_nest[1,1] = 475
lgb_nest[0,2] = 315
lgb_nest[1,2] = 540


# In[ ]:


# booster = ['rdg','lgb','xgb','ctb']
# booster = ['lgb','xgb']


# In[ ]:


# weighted quantile loss, used to compute total pinball loss
def wqloss(y,p,q,w):
    e = y-p
    a,s = np.average(np.maximum(q*e, (q-1)*e), weights=w, returned=True)
    # divide by length instead of sum of weights
    loss = a*s/len(y)
    return loss
    
# def quantile_loss(q, y_p, y):
#     e = y_p-y
#     return tf.keras.backend.mean(tf.keras.backend.maximum(q*e, (q-1)*e))


# In[ ]:


# main training and validation loop
nb = len(booster)
nls = np.zeros((nhorizon//win,ny,nq,nb))
rallv = np.zeros((nhorizon//win,ny,nq,nb))
iallv = np.zeros((nhorizon//win,ny,nq,nb)).astype(int)
yallv = []
pallv = []
imps = []
MAD_FACTOR = 0.5
 
# loop over horizons
for horizon in range(win,nhorizon+1,win):
    
    print()
    gc.collect()
    
    hs = str(horizon)
    if horizon < 10: hs = '0' + hs
    z = horizon // win - 1
    
    # build lists of features
    lags = []
    # must lag reference days to avoid validation leakage
    for i in range(ny):
        for j in range(3):
            # omit ref_day0_0 since it is no longer leaky
            # if (i > 0) | (j > 0): lags.append('ref_day'+str(i)+'_'+str(j))
            lags.append('ref_day'+str(i)+'_'+str(j))
            
    # lag all time-varying features
    for i in range(ny):
        yi = 'y'+str(i)
        yis = yi + '_scaled'
        lags.append(yi)
        lags.append(yis)
#         lags.append('extra_'+yi)
        lags.append('rate'+str(i))
        for j in range(5):
            lags.append(yi+'_d'+str(1+j))
            lags.append(yi+'_l'+str(1+j))
            lags.append(yis+'_d'+str(1+j))
            lags.append(yis+'_l'+str(1+j))
#         lags.append('extra_'+yi+dd)
        lags.append('rate'+str(i)+dd)
        lags.append(yi+rr)
        lags.append(yis+rr)
#         lags.append('extra_'+yi+rr)
        lags.append('rate'+str(i)+rr)
    
        for w in wr:
            lags.append(yi+ww)
            lags.append('rate'+str(i)+ww)
        
#         lags.append('extra_'+yi+ww)
    
        for k in kp:
            lags.append(k+yis)
            lags.append(k+yis+dd)
            lags.append(k+yis+rr)
            lags.append(k+yis+ww)
        for k in kd:
            lags.append(k+yis)
            lags.append(k+yis+dd)
            lags.append(k+yis+rr)
            lags.append(k+yis+ww)
       
#     lags.append('recov')
    
#     lags = lags + wmf + google + wf + ckeep

    lags = lags + fbf + ckeep
    
#     cinfo = ['pop', 'tests', 'testpop', 'density', 'medianage',
#        'urbanpop', 'hospibed', 'smokers']


    cinfo0 = []
    cinfo1 = []
    
#     cinfo0 = ['testpop']
#     cinfo1 = ['testpop','medianage']
    
    f0 = dogs + lags + cinfo0 + ohef
    f1 = dogs + lags + cinfo1 + ohef
    
    # remove some features based on validation experiments
#     f0 = [f for f in f0 if not f.startswith('knn11') and not f.startswith('kd') \
#          and not f.startswith('rate') and not f.endswith(dd) and not f.endswith(rr)]

#     f0 = [f for f in f0 if not f.startswith('knn11') and not f.startswith('kd11')]
#     f1 = [f for f in f1 if not f.startswith('knn6') and not f.startswith('kd6')]
    
    # remove any duplicates
    # f0 = list(set(f0))
    # f1 = list(set(f1))
    
    features = []
    features.append(f0)
    features.append(f1)
    
    nf = []
    for i in range(ny):
        nf.append(len(features[i]))
        # print(nf[i], features[i][:10])
     
    if val_scheme == 'forward':
        # ddate is the last day of training for validation
        # training data stays constant
        ddate = dmax - timedelta(days=nhorizon)
        qtrain = d['Date'] <= ddate.isoformat()
        # validation day moves forward
        vdate0 = ddate + timedelta(days=horizon-win)
        vdate = ddate + timedelta(days=horizon)
        qval = (vdate0.isoformat() < d['Date']) & (d['Date'] <= vdate.isoformat())
        # lag day is last day of training
        ddate0 = ddate - timedelta(days=win)
        qvallag = (ddate0.isoformat() < d['Date']) & (d['Date'] <= ddate.isoformat())
        # for saving predictions into main table
        qsave = qval
    else: 
        # ddate is the last day of training for validation
        # training data moves backwards
        ddate = dmax - timedelta(days=horizon)
        qtrain = d['Date'] <= ddate.isoformat()
        # validate using the last day with data
        # validation day stays constant
        vdate = dmax
        qval = d['Date'] == vdate.isoformat()
        # lag day is last day of training
        qvallag = d['Date'] == ddate.isoformat()
        # for saving predictions into table, expected rise going backwards
        sdate = dmax - timedelta(days=horizon-1)
        qsave = d['Date'] == sdate.isoformat()

    x_train = d[qtrain].copy()
    x_val = d[qval].copy()
    
    # y training data
    y_train = []
    yd_train = [] 
    w_train = []
    y_val = []
    y_val2 = []
    yd_val = []
    w_val = []
    for i in range(ny):
        y_train.append(d.loc[qtrain,yv[i]+'_scaled'].values)
        y_val2.append(d.loc[qval,yv[i]+'_scaled'].values)            
        y_val.append(d.loc[qval,yv[i]+'_w2'].values)
        yd_train.append(d.loc[qtrain,yvc[i]].values)
        yd_val.append(d.loc[qval,yvc[i]].values)
        
        # fatality weight is 10x larger
        wm = 9*i + 1.0
        w_train.append(wm * d.loc[qtrain,'Weight'].values)
        w_val.append(wm * d.loc[qval,'Weight'].values)
        

    yallv.append(y_val)
    
    # lag time-varying features
    x_train.loc[:,lags] = x_train.groupby('Loc')[lags].transform(lambda x: x.shift(horizon))
    x_val.loc[:,lags] = d.loc[qvallag,lags].values
    
    # paulo mad features
    for i in range(ny):
        for x in [x_train, x_val]:
            x['avg_diff_'+yvs[i]] = (x[yvs[i]] - x[yvs[i]+'_l3']) / 3
            x['mad_'+yvs[i]] = x[yvs[i]] + horizon * x['avg_diff_'+yvs[i]] -                 (1 - MAD_FACTOR) * x['avg_diff_'+yvs[i]] *                 np.sum([j for j in range(horizon)]) / nhorizon
            
        features[i] = features[i] + ['avg_diff_'+yvs[0], 'mad_'+yvs[0]] +                                     ['avg_diff_'+yvs[1], 'mad_'+yvs[1]]
            
    print()
    print(horizon, 'x_train', x_train.shape)
    print(horizon, 'x_val', x_val.shape)
    
    if train_full:
        
        qfull = (d['Date'] <= tmax)
        
        tdate0 = dmax + timedelta(days=horizon-win)
        tdate = dmax + timedelta(days=horizon)
        qtest = (tdate0.isoformat() < d['Date']) & (d['Date'] <= tdate.isoformat())
                
        dmax0 = dmax - timedelta(days=win)
        qtestlag = (dmax0.isoformat() < d['Date']) & (d['Date'] <= dmax.isoformat())
    
        x_full = d[qfull].copy()

        y_full = []
        yd_full = []
        w_full = []
        for i in range(ny):
            y_full.append(d.loc[qfull,yv[i]+'_scaled'].values)
            yd_full.append(d.loc[qfull,yvc[i]].values)
            wm = 9*i + 1.0
            w_full.append(wm * d.loc[qfull,'Weight'].values)
                    
        x_test = d[qtest].copy()
        # y_fulllag = [d.loc[qtestlag,'y0'].values, d.loc[qtestlag,'y1'].values]
        
        # lag features
        x_full.loc[:,lags] = x_full.groupby('Loc')[lags].transform(lambda x: x.shift(horizon))
        x_test.loc[:,lags] = d.loc[qtestlag,lags].values

        # paulo mad features
        for i in range(ny):
            for x in [x_full, x_test]:
                x['avg_diff_'+yvs[i]] = (x[yvs[i]] - x[yvs[i]+'_l3']) / 3
                x['mad_'+yvs[i]] = x[yvs[i]] + horizon * x['avg_diff_'+yvs[i]] -                     (1 - MAD_FACTOR) * x['avg_diff_'+yvs[i]] *                     np.sum([j for j in range(horizon)]) / nhorizon
        
        print(horizon, 'x_full', x_full.shape)
        print(horizon, 'x_test', x_test.shape)

    train_set = []
    val_set = []
    ny = len(y_train)

#     for i in range(ny):
#         train_set.append(xgb.DMatrix(x_train[features[i]], y_train[i]))
#         val_set.append(xgb.DMatrix(x_val[features[i]], y_val[i]))

    gc.collect()

    # loop over multiple targets
    mod = []
    pred = []
    rez = []
    iters = []
    
    for i in range(ny):
#     for i in range(1):
        print()
        print('*'*40)
        print(f'horizon {horizon} {yv[i]} {ynames[i]} {vdate}')
        print('*'*40)
        
        # x_train[features[i]] = x_train[features[i]].fillna(0)
        # x_val[features[i]] = x_val[features[i]].fillna(0)
        
        # use catboost only for y1
        # nb = 2 if i==0 else 3
       
        # matrices to store predictions
        vpm = np.zeros((x_val.shape[0],nq,nb))
        if train_full:
            tpm = np.zeros((x_test.shape[0],nq,nb))
         
        # loop over quantiles
        for q in range(nq):    
        
            print()
            print('quantile',quant[q])
            ql = qlab[q]
        
            # loop over boosters
            for b in range(nb):
                
                params[(booster[b],yv[i])]['alpha'] = quant[q]
                restore_features = False

                # loop over validation or full training
                for tset in ['val','full']:

                    if (tset=='full') & (not train_full): continue

                    if tset=='full':
                        print()
                        print(f'{booster[b]} training with full data and predicting', tdate.isoformat())
                        # params[(booster[b],yv[i])]['n_estimators'] = iallv[horizon//win-1,i,q,b]

#                     else:
#                         params[(booster[b],yv[i])]['n_estimators'] = 10000
                        
                    # scikit interface automatically uses best model for predictions
                    # params[(booster[b],yv[i])]['n_estimators'] = 5000

                    kwargs = {'verbose':1000}
                    if booster[b]=='lgb':
                        params[(booster[b],yv[i])]['n_estimators'] = lgb_nest[i,q]
                        model = lgb.LGBMRegressor(**params[(booster[b],yv[i])]) 
                    elif booster[b]=='xgb':
                         #params[(booster[b],yv[i])]['n_estimators'] = 75 if i==0 else 50
                        params[(booster[b],yv[i])]['base_score'] = np.mean(y_train[i])
                        model = xgb.XGBRegressor(**params[(booster[b],yv[i])])
                    elif booster[b]=='ctb':
                         #params[(booster[b],yv[i])]['n_estimators'] = 400 if i==0 else 350
                        # change feature list for categorical features
                        features_save = features[i].copy()
                        features[i] = [f for f in features[i] if not f.startswith('Loc_')] + ['Loc']
                        params[(booster[b],yv[i])]['cat_features'] = ['Loc']
                        restore_features = True
                        model = ctb.CatBoostRegressor(**params[(booster[b],yv[i])])
                    elif booster[b]=='rdg':
                        # alpha from cpmp
                        model = Ridge(alpha=3, fit_intercept=True)
                        kwargs = {}
                    else:
                        raise ValueError(f'Unrecognized booster {booster[b]}')

                    if tset=='val':
                        xtrn = x_train[features[i]].copy()
                        xval = x_val[features[i]].copy()
                        ytrn = y_train[i]
                        yval = y_val[i]
                        wtrn = w_train[i]
                        wval = w_val[i]
                        eset = [(xtrn, ytrn), (xval, y_val2[i])] 
                        esw = [wtrn,wval]      
                    else:
                        xtrn = x_full[features[i]].copy()
                        ytrn = y_full[i]
                        xtest = x_test[features[i]].copy()
                        wtrn = w_full[i]
                        eset = [(xtrn, ytrn)]
                        esw = [wtrn]


                    if booster[b]=='rdg':
                        s = StandardScaler()
                        xtrn = s.fit_transform(xtrn)
                        xtrn = np.nan_to_num(xtrn)
                        xtrn = pd.DataFrame(xtrn, columns=features[i])
                        if tset=='val':
                            xval = s.transform(xval)
                            xval = pd.DataFrame(xval, columns=features[i])
                            xval = np.nan_to_num(xval)

                    # fit y
                    model.fit(xtrn, ytrn, sample_weight=wtrn,
                          eval_set=eset, eval_sample_weight=esw,
                          # early_stopping_rounds=50,
                          **kwargs
                    )

                    if tset=='val':
                        vp = model.predict(xval)
                        vp = vp * x_val[yv[i]+'_w7_max'] / x_val['scale_factor'+str(i)]
                        vp = vp.clip(0,None)
                        val_score = wqloss(yval, vp, quant[q], wval)
                        print(f'wqloss {quant[q]} {val_score:.6f}')
                    else:
                        tp = model.predict(xtest)
                        tp = tp * x_test[yv[i]+'_w7_max'] / x_test['scale_factor'+str(i)]
                        tp = tp.clip(0,None)

                    mod.append(model)
                    
#                     if tset=='val':
#                         iallv[horizon//win-1,i,q,b] = model._best_iteration if booster[b]=='lgb' else \
#                                                       model.best_iteration if booster[b]=='xgb' else \
#                                                       model.best_iteration_

                    # fit cum y
                    if fit_cum:
                        # kwargs = {'verbose':100}
                        if booster[b]=='lgb':
                            # params[(booster[b],yv[i])]['n_estimators'] = 125 if i==0 else 75
                            model = lgb.LGBMRegressor(**params[(booster[b],yv[i])]) 
                        elif booster[b]=='xgb':
                            # params[(booster[b],yv[i])]['n_estimators'] = 75 if i==0 else 30
                            params[(booster[b],yv[i])]['base_score'] = np.mean(yd_train[i])
                            model = xgb.XGBRegressor(**params[(booster[b],yv[i])])
                        elif booster[b]=='ctb':
                            # params[(booster[b],yv[i])]['n_estimators'] = 400 if i==0 else 200
                            # hack for categorical features, ctb must be last in booster list
                            # features[i] = [f for f in features[i] if not f.startswith('Loc_')] + ['Loc']
                            # params[(booster[b],yv[i])]['cat_features'] = ['Loc']
                            model = ctb.CatBoostRegressor(**params[(booster[b],yv[i])])
                        elif booster[b]=='rdg':
                            # alpha from cpmp
                            model = Ridge(alpha=3, fit_intercept=True)
                            kwargs = {}
                        else:
                            raise ValueError(f'Unrecognized booster {booster[b]}')

                        model.fit(xtrn, yd_train[i], sample_weight=w_train[i],
                              eval_set=[(xtrn, yd_train[i]), (xval, yd_val[i])],
                              eval_sample_weight=[w_train[i],w_val[i]],
                              # early_stopping_rounds=50,
                              **kwargs
                        )

                        vpd = model.predict(xval)
                        vpd = np.clip(vpd,0,None)
                        # vpd = y_vallag[i] + vpd

                        # blend two predictions based on horizon
                        # alpha = 0.1 + 0.8*(horizon-1)/29
                        alpha = 1.0
                        vp = alpha*vp + (1-alpha)*vpd

                        mod.append(model)


                    gain = np.abs(model.coef_) if booster[b]=='rdg' else model.feature_importances_
            #         gain = model.get_score(importance_type='gain')
            #         split = model.get_score(importance_type='weight')   
                #     gain = model.feature_importance(importance_type='gain')
                #     split = model.feature_importance(importance_type='split').astype(float)  
                #     imp = pd.DataFrame({'feature':features,'gain':gain,'split':split})
                    imp = pd.DataFrame({'feature':features[i],'gain':gain})
            #         imp = pd.DataFrame({'feature':features[i]})
            #         imp['gain'] = imp['feature'].map(gain)
            #         imp['split'] = imp['feature'].map(split)

                    imp.set_index(['feature'],inplace=True)

                    imp.gain /= np.sum(imp.gain)
            #         imp.split /= np.sum(imp.split)

                    imp.sort_values(['gain'], ascending=False, inplace=True)

                    print()
                    print(imp.head(n=10))
                    # print(imp.shape)

                    imp.reset_index(inplace=True)
                    imp['horizon'] = horizon
                    imp['target'] = yv[i]
                    imp['set'] = 'valid'
                    imp['booster'] = booster[b]

                    imps.append(imp)

                    # make sure horizon 1 prediction is not smaller than first lag
                    # because we know series is monotonic
                    # if horizon==1+skip:
                    if False:
                        a = np.zeros((len(vp),2))
                        a[:,0] = vp
                        # note yv is lagged here
                        a[:,1] = x_val[yv[i]].values
                        vp = np.nanmax(a,axis=1)

                    if tset=='val':
                        val_score = wqloss(yval, vp, quant[q], wval)
                        vpm[:,q,b] = vp

                        print()
                        print(f'{booster[b]} validation wqloss {quant[q]} {val_score:.6f}')
                        rallv[horizon//win-1,i,q,b] = val_score

                    else:
                        tpm[:,q,b] = tp


                    gc.collect()
    
                
                # restore feature list
                if restore_features:
                    features[i] = features_save
                    restore_features = False
                
    #         # concat team predictions
    #         if len(tfeats[i]):
    #             vpm = np.concatenate([vpm,d.loc[qval,tfeats[i]].values], axis=1)
    #             if train_full:
    #                 tpm = np.concatenate([tpm,d.loc[qtest,tfeats[i]].values], axis=1)

            # nonnegative least squares to estimate ensemble weights
            # x, rnorm = nnls(vpm, y_val[i])

            # smooth weights by shrinking towards all equal
            # x = (x + np.ones(3)/3.)/2

            nm = vpm.shape[-1]
            x = np.ones(nm)/nm

            # simple averaging to avoid overfitting
            # drop ridge from y0
    #         if i==0:
    #             x = np.array([1., 1., 1., 0.])/3.
    #         else:
    #             nm = vpm.shape[1]
    #             x = np.ones(nm)/nm

    #         # drop catboost from y0
    #         if i == 0:  
    #             x = np.array([0.5, 0.5, 0.0])
    #         else: 
    #             nm = vpm.shape[1]
    #             x = np.ones(nm)/nm

            # smooth weights with rolling mean, ewma
            # alpha = 0.1
            # if horizon-skip > 1: x = alpha * x + (1 - alpha) * nls[horizon-skip-2,i]

            nls[horizon//win-1,i,q] = x

            val_pred = np.matmul(vpm[:,q], x)
            if train_full:
                test_pred = np.matmul(tpm[:,q], x)

            # save validation and test predictions back into main table
            d.loc[qsave,yv[i]+'_pred_'+ql] = val_pred
            if train_full:
                d.loc[qtest,yv[i]+'_pred_'+ql] = test_pred
                
#             d.loc[qsave,yv[i]+'_pred_'+ql] = val_pred + x_val[yv[i]+'_median'].values
#             if train_full:
#                 d.loc[qtest,yv[i]+'_pred_'+ql] = test_pred + x_test[yv[i]+'_median'].values

            # ensemble validation score
            # val_score = np.sqrt(rnorm/vpm.shape[0])
            val_score = wqloss(y_val[i], val_pred, q, w_val[i])

            rez.append(val_score)
            pred.append(val_pred)

    pallv.append(pred)
    
#     # construct strings of nnls weights for printing
#     w0 = ''
#     w1 = ''
#     for b in range(nb+tf2):
#         w0 = w0 + f' {nls[horizon-skip-1,0,b]:.2f}'
#         w1 = w1 + f' {nls[horizon-skip-1,1,b]:.2f}'
        
#     print()
#     print('         Validation RMSLE  ', ' '.join(booster), ' '.join(tfeats[0]))
#     print(f'{ynames[0]} \t {rez[0]:.6f}  ' + w0)
#     print(f'{ynames[1]} \t {rez[1]:.6f}  ' + w1)
#     print(f'Mean \t \t {np.mean(rez):.6f}')

#     # break down RMSLE by day
#     rp = np.zeros((2,7))
#     for i in range(ny):
#         for di in range(50,57):
#             j = di - 50
#             qf = x_val.dint == di
#             rp[i,j] = np.sqrt(mean_squared_error(pred[i][qf], y_val[i][qf]))
#             print(i,di,f'{rp[i,j]:.6f}')
#         print(i,f'{np.mean(rp[i,:]):.6f}')
#         plt.plot(rp[i])
#         plt.title(ynames[i] + ' RMSLE')
#         plt.show()
        
    # plot actual vs predicted
    plt.figure(figsize=(10, 15))
    for q in range(nq):
        for i in range(ny):
            idx = ny*q+i+1
            plt.subplot(nq,ny,idx)
            # plt.plot([0, 12], [0, 12], 'black')
            plt.plot(np.log1p(pred[i*nq+q]), np.log1p(y_val[i]), '.')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(ynames[i] + ' ' + qlab[q])
            plt.grid()
    plt.show()
    
    print()
    print(f'Weighted Pinball Loss for Validation Horizon {horizon}')
    print(f'    {np.mean(rallv[horizon//win-1,0].flatten()):.5f} {ynames[0]}')
    print(f'    {np.mean(rallv[horizon//win-1,1].flatten()):.5f} {ynames[1]}')
    print(f'    {np.mean(rallv[horizon//win-1].flatten()):.5f} Average')
    print()

# save one big table of importances
impall = pd.concat(imps)

# remove number suffixes from lag names to aid in analysis
# impall['feature1'] = impall['feature'].replace(to_replace='lag..', value='lag', regex=True)

os.makedirs('imp', exist_ok=True)
fname = 'imp/' + mname + '_imp.csv'
impall.to_csv(fname, index=False)
print()
print(fname, impall.shape)

# save scores and weights
os.makedirs('rez', exist_ok=True)
fname = 'rez/' + mname+'_rallv.npy'
np.save(fname, rallv)
print(fname, rallv.shape)

fname = 'rez/' + mname+'_nnls.npy'
np.save(fname, nls)
print(fname, nls.shape)


# In[ ]:


if 'cas' in booster: s.shutdown()


# In[ ]:


np.mean(iallv, axis=0)


# In[ ]:


# plot pinball loss, note this is on smoothed target so is better than original
plt.figure(figsize=(10, 4))
for i in range(ny):
    plt.subplot(1,2,1+i)
    plt.plot(np.mean(rallv[:,i],axis=1).flatten())
    plt.title(ynames[i] + ' Weighted Pinball Loss vs Horizon')
    plt.grid()
    plt.legend(booster)
    
#     plt.subplot(2,2,3+i)
#     plt.plot(nls[:,i])
#     plt.title(ynames[i] + ' Ensemble Weights')
#     plt.grid()
#     plt.legend(booster+tfeats[i])
plt.show()


# In[ ]:


# weighted pinball loss with smooth target, better than with original target
np.mean(rallv.flatten())


# In[ ]:


d['Weight'].describe()


# In[ ]:


# # take differences and compute weighted pinball loss
# # skip over first horizon so difference is not nan
# vdate0 = ddate + timedelta(2+skip)
# vdate1 = ddate + timedelta(nhorizon)
# qval = (vdate0.isoformat() <= d['Date']) & (d['Date'] <= vdate1.isoformat())
# for i in range(ny):
#     d[yv[i]+'_pdiff'] = d.groupby('Loc')[yv[i]+'_pred'].transform(lambda x: x.diff())
#     # exponentiate if original target is on log scale
#     # d[yv[i]+'_exp'] = d.groupby('Loc')[yv[i]+'_pred'].transform(lambda x: np.expm1(x))
#     # d[yv[i]+'_pdiff'] = d.groupby('Loc')[yv[i]+'_exp'].transform(lambda x: x.diff())
#     vp = d.loc[qval,yv[i]+'_pdiff']
#     vy = d.loc[qval,ynames[i]]
#     vw = d.loc[qval,'Weight']
#     a,s = np.average(np.abs(vp-vy), weights=vw, returned=True)
#     if i==1: a *= 10
#     print(a*s/len(vy))


# In[ ]:


# break down validation pinball loss for each location
locs = d.loc[:,['Loc','Weight']].drop_duplicates().reset_index(drop=True)
wqa = []
for i in range(ny):
    for q in range(nq):
        wq = yv[i]+'_wqloss_'+qlab[q]
        locs[wq] = np.nan
        wqa.append(wq)

# locs = x_val.copy().reset_index(drop=True)
# print(locs.shape)
# y_truea = []
# y_preda = []

vdate0 = ddate + timedelta(days=1+skip)
vdate1 = ddate + timedelta(days=nhorizon)
qv = (vdate0.isoformat() <= d['Date']) & (d['Date'] <= vdate1.isoformat())

# print(f'# {mname}')
for j,loc in enumerate(locs['Loc']):
    if j % 1000 == 0: print(j)
    qvl = qv & (d['Loc']==loc)
    w = d.loc[qvl,'Weight']
    for i in range(ny):
        wm = i*9 + 1
        for q in range(nq):
            # y = d.loc[qvl,yv[i]].values
            y = d.loc[qvl,ynames[i]].values
            p = d.loc[qvl,yv[i]+'_pred_'+qlab[q]]
            wq = yv[i]+'_wqloss_'+qlab[q]
            locs.loc[locs.Loc==loc,wq] = wqloss(y,p,quant[q],wm*w)
            
        # make each series monotonic increasing
        # for j in range(y_pred.shape[1]): 
        #     y_pred[:,j] = np.maximum.accumulate(y_pred[:,j])
        # copy updated predictions into main table
locs


# In[ ]:


# overall validation pinball loss
np.mean(locs[wqa].values)


# In[ ]:


np.mean(locs.loc[locs.Loc!='US',wqa].values)


# In[ ]:


np.mean(locs.loc[locs.Loc=='US',wqa].values)


# In[ ]:


os.makedirs('locs',exist_ok=True)
fname = 'locs/' + mname + '_locs.csv'
locs.to_csv(fname, index=False)
print(fname, locs.shape)


# In[ ]:


yv[0]


# In[ ]:


np.nansum(np.abs(d[[yv[0]]].values.flatten() - d.ConfirmedCases.values.flatten()))


# In[ ]:


# # enforce monotonicity of forecasts in test set after last date in training
# if train_full:
#     # loc = d['Loc'].unique()
#     locs1 = d['Loc'].drop_duplicates()
#     for loc in locs1:
#         # q = (d.Loc==loc) & (d.ForecastId > 0)
#         q = (d.Loc==loc) & (d.Date > tmax)
#         # if skip, fill in last observed value
#         if skip: qs0 = (d.Loc==loc) & (d.Date == dmax.isoformat())
#         for yi in yv:
#             yp = yi+'_pred'
#             d.loc[q,yp] = np.maximum.accumulate(d.loc[q,yp])
#             if skip:
#                 for j in range(skip):
#                     qs1 = (d.Loc==loc) & (d.Date == (dmax + timedelta(1+j)).isoformat())
#                     d.loc[qs1,yp] = d.loc[qs0,yi].values


# In[ ]:


# sort to find worst predictions of y0
locs = locs.sort_values('y0_wqloss_q50', ascending=False)
locs[:10]


# In[ ]:


# plot worst fits 
for j in range(10):
    plt.figure(figsize=(14,4))
    lj = locs.index[j]
    loc = locs.loc[lj,'Loc']
    qvl = (d['Loc']==loc) & (d['Date'] <= tmax)
    for i in range(ny):
        plt.subplot(1,2,i+1)
        plt.plot(d.loc[qvl,[ynames[i],'Date']].set_index('Date'))
        plt.plot(d.loc[qvl,[yv[i]+'_w2','Date']].set_index('Date'))
        plt.plot(d.loc[qvl,[yv[i]+'_w7','Date']].set_index('Date'))
        # plt.plot(d.loc[qvl,[yv[i]+'_w5','Date']].set_index('Date'))
        for q in range(nq):
            plt.plot(d.loc[qvl,[yv[i]+'_pred_'+qlab[q],'Date']].set_index('Date'))

    #     qvl = qv & (d['Loc']==loc)
    #     y = d.loc[qvl,'y0'].values
    #     plt.plot(y)
    #     for q in range(nq):
    #         p = d.loc[qvl,'y0_pred_'+qlab[q]].values
    #         if quant[q] == 0.5: plt.plot(p, c='r')
    #         else: plt.plot(p, '--', c='r')

        plt.xticks([])
        plt.title(loc + ' ' + ynames[i])
    plt.show()


# In[ ]:


'dow' in features[0]


# In[ ]:


# plot individual location
loc = 'US North Carolina'
# loc = 'US North Carolina Wake'
# loc = 'US Ohio'
# loc = 'US Ohio Morrow'
qvl = (d['Loc']==loc) & (d['Date'] <= tmax)
plt.figure(figsize=(14,6))
for i in range(ny):
    plt.subplot(1,2,i+1)
    plt.plot(d.loc[qvl,[ynames[i],'Date']].set_index('Date'))
    for q in range(nq):
        plt.plot(d.loc[qvl,[yv[i]+'_pred_'+qlab[q],'Date']].set_index('Date'))

    plt.xticks([])
    plt.title(loc + ' ' + ynames[i])
plt.show()


# In[ ]:


# d.loc[d.Loc=='US','Loc_US'].describe()


# In[ ]:


# sort to find worst predictions of y0
locs = locs.sort_values('y1_wqloss_q50', ascending=False)


# In[ ]:


# # plot worst fits of y1
# i = 1
# for j in range(5):
#     lj = locs.index[j]
#     loc = locs.loc[lj,'Loc']
#     qvl = (d['Loc']==loc) & (d['Date'] <= tmax)
#     plt.plot(d.loc[qvl,[ynames[i],'Date']].set_index('Date'))
#     for q in range(nq):
#         plt.plot(d.loc[qvl,[yv[i]+'_pred_'+qlab[q],'Date']].set_index('Date'))

# #     qvl = qv & (d['Loc']==loc)
# #     y = d.loc[qvl,'y0'].values
# #     plt.plot(y)
# #     for q in range(nq):
# #         p = d.loc[qvl,'y0_pred_'+qlab[q]].values
# #         if quant[q] == 0.5: plt.plot(p, c='r')
# #         else: plt.plot(p, '--', c='r')
     
#     plt.xticks([])
#     plt.title(loc + ' ' + ynames[i])
#     plt.show()


# In[ ]:


# compute public lb score
if not prev_test:
    ys = ['CC','FT']
    print(f'# {ddate.isoformat()} {fmin} {tmax} {mname}')
    qvl = (d.Date >= '2020-04-27') & (d.Date <= tmax)
    w = d.loc[qvl,'Weight']
    a = []
    for i in range(ny):
        wm = i*9 + 1
        for q in range(nq):
            # y = d.loc[qvl,yv[i]].values
            y = d.loc[qvl,ynames[i]].values
            p = d.loc[qvl,yv[i]+'_pred_'+qlab[q]]
            loss = wqloss(y,p,quant[q],wm*w)
            print(f'# {ys[i]} \t {quant[q]} \t {loss:.5f}')
            a.append(loss)
            
    print()
    print(f'# {ys[0]} \t {np.mean(a[:3]):.5f}')
    print(f'# {ys[1]} \t {np.mean(a[3:]):.5f}')
    print(f'# Avg \t {np.mean(a):.5f}')


# In[ ]:


# # compute public lb score
# if not prev_test:
#     # q = (d.Date >= fmin) & (d.Date > ddate.isoformat()) & (d.Date <= tmax)
#     q = (d.Date >= '2020-04-02') & (d.Date <= tmax)
#     # q = (d.Date >= tmax) & (d.Date <= tmax)
#     print(f'# {fmin} {ddate.isoformat()} {tmax} {sum(q)//ns} {mname}')
#     s0 = np.sqrt(mean_squared_error(d.loc[q,'y0r'],d.loc[q,'y0_pred']))
#     s1 = np.sqrt(mean_squared_error(d.loc[q,'y1r'],d.loc[q,'y1_pred']))
#     print(f'# CC \t {s0:.6f}')
#     print(f'# Fa \t {s1:.6f}')
#     print(f'# Mean \t {(s0+s1)/2:.6f}')
    
#     s0 = np.sqrt(mean_squared_error(d.loc[q,'y0r'],d.loc[q,'y0_preda']))
#     s1 = np.sqrt(mean_squared_error(d.loc[q,'y1r'],d.loc[q,'y1_preda']))
#     print()
#     print(f'# CC \t {s0:.6f}')
#     print(f'# Fa \t {s1:.6f}')
#     print(f'# Mean \t {(s0+s1)/2:.6f}')


# In[ ]:


# nnls to estimate blending weights
if blend:
    print('blending with',blender)
    sub = d.loc[d.ForecastId > 0, ['ForecastId','ConfirmedCases','Fatalities',
                                   'y0','y1','y0_preda','y1_preda','Date','dint']]
    sub['dint'] = sub['dint'] - sub['dint'].min()
    # original data, nonmonotonic in some places
    sub['y0r'] = np.log1p(sub.ConfirmedCases)
    sub['y1r'] = np.log1p(sub.Fatalities)
    sub['ConfirmedCases'] = sub.ConfirmedCases.astype(float)
    sub['Fatalities'] = sub.Fatalities.astype(float)

    print(sub.shape)
    print(sub['dint'].describe())
    hmax = np.max(sub.dint.values) + 1
    print(hmax)
    
    # add nq
    bs = pd.read_csv('sub/'+blender[0]+'.csv')
    print(bs.shape)
    bs['nq0'] = np.log1p(bs.ConfirmedCases)
    bs['nq1'] = np.log1p(bs.Fatalities)
    bs.drop(['ConfirmedCases','Fatalities'],axis=1,inplace=True)
    sub = sub.merge(bs, how='left', on='ForecastId')
    sub['nq0'] = sub['nq0'].fillna(sub['y0'])
    sub['nq1'] = sub['nq1'].fillna(sub['y1'])

    # add kaz
    bs = pd.read_csv('sub/'+blender[1]+'.csv')
    print(bs.shape)
    bs['kaz0'] = np.log1p(bs.ConfirmedCases)
    bs['kaz1'] = np.log1p(bs.Fatalities)
    bs.drop(['ConfirmedCases','Fatalities'],axis=1,inplace=True)
    sub = sub.merge(bs, how='left', on='ForecastId')
    
    for i in range(ny): sub[mname+str(i)] = sub[yv[i]+'_preda']
        
    # qv = (sub.Date >= '2020-04-09') & (sub.Date <= tmax)
    qv = (sub.Date > tmax)
    a = sub[qv].copy()

#     # intercept estimate is 0
#     # a['intercept0'] = 1.0
#     # a['intercept1'] = 1.0
#     # m = ['intercept','b0g','v0e','o0e',mname]
#     # m = ['kaz',mname]
#     m = ['nq','kaz',mname]
#     print(m)
#     n = a.shape[0]
#     wt= np.zeros((2,len(m)))
#     s = 0
#     for i in range(ny):
#         mi = [c+str(i) for c in m]
#         wt[i], rnorm = nnls(a[mi].values, a[yv[i]+'r'].values)
#         r = rnorm/np.sqrt(n)
#         print(i, wt[i], f'{sum(wt[i]):.6f}', f'{r:.6f}')
#         s += 0.5*r
#     print(f'{s:.6f}')
#     print()
    print(a[['nq0','kaz0',mname+'0','nq1','kaz1',mname+'1']].corr())


# In[ ]:


d['ForecastId'].describe()


# In[ ]:


# create submission
ss = []
qf = d.ForecastId > 0
for q in range(nq):
    for i in range(ny):
        yp = yv[i]+'_pred_'+qlab[q]
        s = d.loc[qf,['ForecastId',yp]]
        s = s.rename(mapper={yp:'TargetValue'}, axis=1)
        s['Target'] = ynames[i]
        s['Quantile'] = quant[q]
        if i==1: s['ForecastId'] += 1
        s['ForecastId_Quantile'] = s.ForecastId.astype(str) + '_' + s.Quantile.astype(str)
        s = s.reset_index(drop=True)
        ss.append(s)
s


# In[ ]:


sub = pd.concat(ss).sort_values(['ForecastId','Quantile'])
sub


# In[ ]:


os.makedirs('sub', exist_ok=True)
sub1 = sub[['ForecastId_Quantile','TargetValue']]
fname = 'sub/'+mname+'.csv'
sub1.to_csv(fname, index=False)
print(fname, sub1.shape)


# In[ ]:


# # create blended submission, set weights by hand after looking at validation nnls
# if blend:
#     # blend
#     sub['ConfirmedCases'] = np.expm1(0.997*(0.5 * sub['nq0'] + \
#                                             0.2 * sub['kaz0'] + \
#                                             0.3 * sub['y0_preda']))
#     sub['Fatalities'] = np.expm1(0.996*(0.666666 * sub['nq1'] + \
#                                         0.133333 * sub['kaz1'] + \
#                                         0.2      * sub['y1_preda']))
            
# else:
#     # create submission without any blending with others
#     sub = d.loc[d.ForecastId > 0, ['ForecastId','y0_pred','y1_pred']]
#     print(sub.shape)

#     sub['ConfirmedCases'] = np.expm1(sub['y0_preda'])
#     sub['Fatalities'] = np.expm1(sub['y1_preda'])    

# sub0 = sub.copy()
# print(sub0.shape)
# sub = sub[['ForecastId','ConfirmedCases','Fatalities']]

# os.makedirs('sub',exist_ok=True)
# fname = 'sub/' + mname + '.csv'
# sub.to_csv(fname, index=False)
# print(fname, sub.shape)


# In[ ]:


# sub.describe()


# In[ ]:


# # final day adjustment as per northquay
# pname = mname
# # pred = sub.copy()
# pred = pd.read_csv('sub/' + mname + '.csv')

# # pname = 'kaz0m'
# # pred = pd.read_csv('../week3/sub/'+pname+'.csv')

# pred_orig = pred.copy()

# if prev_test:
#     test = pd.read_csv('../'+pw+'/test.csv')
# else:
#     test = pd.read_csv('test.csv')

# test[cp] = test[cp].fillna('')

# # test.Date = pd.to_datetime(test.Date)
# # train.Date = pd.to_datetime(train.Date)

# # TODAY = datetime.datetime(  *datetime.datetime.today().timetuple()[:3] )
# # TODAY = date(2020, 4, 7)

# # shift day back one to match wm adjustment

# print(TODAY)

# final_day = wm[wm.Date == TODAY].copy()
# final_day['cases_final'] = np.expm1(final_day.TotalCases)
# final_day['cases_chg'] = np.expm1(final_day.NewCases)
# final_day['deaths_final'] = np.expm1(final_day.TotalDeaths)
# final_day['deaths_chg'] = np.expm1(final_day.NewDeaths)


# # test.rename(columns={'Country_Region': 'Country'}, inplace=True)
# # test['Place'] = test.Country +  test.Province_State.fillna("")

# # final_day = pd.read_excel(path + '../week3/nq/' + 'final_day.xlsx')
# # final_day = final_day.iloc[1:, :5]
# # final_day = final_day.fillna(0)
# # final_day.columns = ['Country', 'cases_final', 'cases_chg', 
# #                      'deaths_final', 'deaths_chg']

# final_day = final_day[['Country_Region','Province_State','cases_final','cases_chg',
#                       'deaths_final','deaths_chg']].fillna(0)
# # final_day = final_day.drop('Date', axis=1).reset_index(drop=True)
# final_day = final_day.sort_values('cases_final', ascending=False)

# print()
# print('final_day')
# print(final_day.head(n=10), final_day.shape)

# # final_day.Country.replace({'Taiwan': 'Taiwan*',
# #                            'S. Korea': 'Korea, South',
# #                            'Myanmar': 'Burma',
# #                            'Vatican City': 'Holy See',
# #                            'Ivory Coast':  "Cote d'Ivoire",
                        
# #                           },
# #                          inplace=True)


# pred = pd.merge(pred, test, how='left', on='ForecastId')
# print()
# print('pred')
# print(pred.head(n=10), pred.shape)

# # pred = pd.merge(pred, test[test.Province_State.isnull()], how='left', on='ForecastId')

# # compare = pd.merge(pred[pred.Date == TODAY], final_day, on= [ 'Country'],
# #                            validate='1:1')

# compare = pd.merge(pred[pred.Date == TODAY], final_day, on=cp, validate='1:1')

# compare['c_li'] = np.round(np.log(compare.cases_final + 1) - np.log(compare.ConfirmedCases + 1), 2)
# compare['f_li'] = np.round(np.log(compare.deaths_final + 1) - np.log(compare.Fatalities + 1), 2)

# print()
# print('compare')
# print(compare.head(n=10), compare.shape)
# print(compare.describe())

# # compare[compare.c_li > 0.3][['Country', 'ConfirmedCases', 'Fatalities',
# #                                         'cases_final', 'cases_chg',
# #                                     'deaths_final', 'deaths_chg',
# #                                             'c_li', 'f_li']]

# # compare[compare.c_li > 0.15][['Country', 'ConfirmedCases', 'Fatalities',
# #                                         'cases_final', 'cases_chg',
# #                                     'deaths_final', 'deaths_chg',
# #                                             'c_li', 'f_li']]

# # compare[compare.f_li > 0.3][['Country', 'ConfirmedCases', 'Fatalities',
# #                                         'cases_final', 'cases_chg',
# #                                     'deaths_final', 'deaths_chg',
# #                                             'c_li', 'f_li']]


# # compare[compare.f_li > 0.15][['Country', 'ConfirmedCases', 'Fatalities',
# #                                         'cases_final', 'cases_chg',
# #                                     'deaths_final', 'deaths_chg',
# #                                             'c_li', 'f_li']]

# # compare[compare.c_li < -0.15][['Country', 'ConfirmedCases', 'Fatalities',
# #                                         'cases_final', 'cases_chg',
# #                                     'deaths_final', 'deaths_chg',
# #                                             'c_li', 'f_li']]

# # compare[compare.f_li < -0.2][['Country', 'ConfirmedCases', 'Fatalities',
# #                                         'cases_final', 'cases_chg',
# #                                     'deaths_final', 'deaths_chg',
# #                                             'c_li', 'f_li']]

# fixes = pd.merge(pred[pred.Date >= TODAY], 
#                      compare[cp + ['c_li', 'f_li']], on=cp)


# fixes['c_li'] = np.where( fixes.c_li < 0,
#                              0,
#                                  fixes.c_li)
# fixes['f_li'] = np.where( fixes.f_li < 0,
#                              0,
#                                  fixes.f_li)

# fixes['total_fixes'] = fixes.c_li**2 + fixes.f_li**2

# print()
# print('most fixes')
# print(fixes.groupby(cp).last().sort_values(['total_fixes','Date'], ascending = False).head(n=10))

# # adjustment
# fixes['Fatalities'] = np.round(np.exp((np.log(fixes.Fatalities + 1) + fixes.f_li))-1, 3)
# fixes['ConfirmedCases'] = np.round(np.exp((np.log(fixes.ConfirmedCases + 1) + fixes.c_li))-1, 3)


# fix_ids = fixes.ForecastId.unique()
# len(fix_ids)

# cols = ['ForecastId', 'ConfirmedCases', 'Fatalities']


# fixed = pd.concat((pred.loc[~pred.ForecastId.isin(fix_ids),cols],
#     fixes[cols])).sort_values('ForecastId')


# # fixed.head()
# # fixed.tail()

# # len(pred_orig)
# # len(fixed)

# fname = 'sub/' + pname + '_updated.csv'
# fixed.to_csv(fname, index=False)
# print(fname, fixed.shape)
# fixed.describe()


# In[ ]:


# !wget https://web.archive.org/web/20200408225104/https://www.worldometers.info/coronavirus/country/us/ -O us0408.html

# us = pd.read_html('us0408.html')
# len(us)

# # us[0].sort_values('USAState')
# us[0].sort_values('TotalCases', ascending=False)[2:12]


# In[ ]:


# !wget https://web.archive.org/web/20200408234045/https://www.worldometers.info/coronavirus/country/us/ -O us0408.html

# us = pd.read_html('us0408.html')
# len(us)

# # us[0].sort_values('USAState')
# us[0].sort_values('TotalCases', ascending=False)[2:12]


# In[ ]:


# compare[compare.Country_Region=='US'].describe()


# In[ ]:


# sum(qv)


# In[ ]:


# # merge final predictions back into main table
# sub1 = fixed.copy()
# for i in range(ny): 
#     mi = mname + str(i)
#     if mi in d.columns: d.drop(mi, axis=1, inplace=True)
#     sub1[mi] = np.log1p(sub1[ynames[i]])
#     sub1.drop(ynames[i],axis=1,inplace=True)
# d = d.merge(sub1, how='left', on='ForecastId')


# In[ ]:


# fixed.describe()


# In[ ]:


# # compute public lb score after averaging with others
# if not prev_test:
#     # q = (d.Date >= fmin) & (d.Date > ddate.isoformat()) & (d.Date <= tmax)
#     q = (d.Date >= '2020-04-02') & (d.Date <= tmax)
#     # q = (d.Date >= tmax) & (d.Date <= tmax)
#     print(f'# {fmin} {ddate.isoformat()} {tmax} {sum(q)/ns} {mname}')
#     s0 = np.sqrt(mean_squared_error(d.loc[q,'y0r'],d.loc[q,mname+'0']))
#     s1 = np.sqrt(mean_squared_error(d.loc[q,'y1r'],d.loc[q,mname+'1']))
#     print(f'# CC \t {s0:.6f}')
#     print(f'# Fa \t {s1:.6f}')
#     print(f'# Mean \t {(s0+s1)/2:.6f}')


# In[ ]:


# 2020-04-02 2020-03-15 2020-04-14 13.0 gbt5c
# CC 	 0.536442
# Fa 	 0.402339
# Mean 	 0.469391


# In[ ]:


# 2020-04-02 2020-03-15 2020-04-14 13.0 gbt5b
# CC 	 0.536442
# Fa 	 0.402339
# Mean 	 0.469391


# In[ ]:


# # check submissions
# if prev_test:
#     snames = nqs + ['nq0a', 'kaz','kaz0f','kaz0h','kaz0i','kaz0j', 'kaz0k', 
#                     'kaz0m','kaz0m_updated',
#                     'gbt3l','gbt3n',mname,mname+'_updated']
# #     snames = ['nq','kaz','kaz0f','kaz0h','kaz0i','kaz0j', 'kaz0k', 'kaz0m',
# #               'beluga0g',
# #               'vopani','jeremiah0a','jeremiah0b',
# #               'kgmon','isaac0a','oscii','oscii0g','pdd',
# #               'gbt3l','gbt3n',mname]
# #     snames = ['kaz','kaz0f','kaz0g']

#     for j,s in enumerate(snames):
#         if s != mname:
#             fname = '../'+pw+'/sub/'+s+'.csv'
#             if not os.path.exists(fname): fname = 'sub/'+s+'.csv'
#             sub = pd.read_csv(fname)
#             for i in range(ny): sub[s+str(i)] = np.log1p(sub[ynames[i]])
#             sub = sub.drop(['ConfirmedCases','Fatalities'], axis=1)
#             # print(sub.head(),sub.shape)

#             v = d.merge(sub, how='left', on='ForecastId')
#         else:
#             v = d.copy()

#         qv = (v.Date >= '2020-04-09') & (v.Date <= tmax)
#         v = v[qv]
#         v['y0r'] = np.log1p(v.ConfirmedCases)
#         v['y1r'] = np.log1p(v.Fatalities)
#         # print(v.shape)
#         cc = np.sqrt(mean_squared_error(v.y0r,v[s+'0']))
#         fa = np.sqrt(mean_squared_error(v.y1r,v[s+'1']))
#         print()
#         print(f'{s} CC   {cc:.6f}')
#         print(f'{s} Fa   {fa:.6f}')
#         print(f'{s} Mean {(cc+fa)/2:.6f}')
        
# else:
#     snames = ['nq0b', 'nq0g','nq0g_updated', 'nq0i','nq0i_updated',
#               'kaz0n', 'kaz0o', 'kaz0q', 'kaz0r', 'kaz0s', 'kaz0t',
#                mname, mname+'_updated']

#     for j,s in enumerate(snames):
#         if s != mname:
#             fname = 'sub/'+s+'.csv'
#             if not os.path.exists(fname): fname = '../'+pw+'/sub/'+s+'.csv'
#             sub = pd.read_csv(fname)
#             for i in range(ny): sub[s+str(i)] = np.log1p(sub[ynames[i]])
#             sub = sub.drop(['ConfirmedCases','Fatalities'], axis=1)
#             # print(sub.head(),sub.shape)

#             v = d.merge(sub, how='left', on='ForecastId')
#         else:
#             v = d.copy()

#         qv = (v.Date >= '2020-04-09') & (v.Date <= tmax)
#         v = v[qv]
#         v['y0r'] = np.log1p(v.ConfirmedCases)
#         v['y1r'] = np.log1p(v.Fatalities)
#         # print(v.shape)
#         cc = np.sqrt(mean_squared_error(v.y0r,v[s+'0']))
#         fa = np.sqrt(mean_squared_error(v.y1r,v[s+'1']))
#         print()
#         print(f'{s} CC   {cc:.6f}')
#         print(f'{s} Fa   {fa:.6f}')
#         print(f'{s} Mean {(cc+fa)/2:.6f}')


# In[ ]:


# gbt5b CC   0.582307
# gbt5b Fa   0.445966
# gbt5b Mean 0.514136

# gbt5b_updated CC   0.582307
# gbt5b_updated Fa   0.445966
# gbt5b_updated Mean 0.514136


# In[ ]:


# gbt5b CC   0.175608
# gbt5b Fa   0.123974
# gbt5b Mean 0.149791

# gbt5b_updated CC   0.175608
# gbt5b_updated Fa   0.123974
# gbt5b_updated Mean 0.149791


# In[ ]:


# gbt5a CC   0.152043
# gbt5a Fa   0.157290
# gbt5a Mean 0.154666

# gbt5a_updated CC   0.145772
# gbt5a_updated Fa   0.154880
# gbt5a_updated Mean 0.150326


# In[ ]:


wqa


# In[ ]:


# save oof predictions
ovars = ['Id','ForecastId','Country_Region','Province_State','County','Loc','Date',
         'y0_pred_q05', 'y0_pred_q50', 'y0_pred_q95',
         'y0_pred_q05', 'y0_pred_q50', 'y0_pred_q95']
qd = (d.Date >= '2020-04-27') & (d.Date <= '2020-05-22')
oof = d.loc[qd,ovars]
# oof = oof.rename(mapper={'y0_pred':mname+'0','y1_pred':mname+'1'}, axis=1)
os.makedirs('oof',exist_ok=True)
fname = 'oof/' + mname + '_oof.csv'
oof.to_csv(fname, index=False)
print(fname, oof.shape)


# In[ ]:


if save_data:
    os.makedirs('data',exist_ok=True)
    fname = 'data/' + mname + '_d.csv'
    d.to_csv(fname, index=False)
    print(fname, d.shape)
    
    fname = 'data/' + mname + '_x_train.csv'
    x_train.to_csv(fname, index=False)
    print(fname, d.shape)
    
    fname = 'data/' + mname + '_x_val.csv'
    x_val.to_csv(fname, index=False)
    print(fname, d.shape)
    
    fname = 'data/' + mname + '_x_full.csv'
    x_full.to_csv(fname, index=False)
    print(fname, d.shape)
    
    fname = 'data/' + mname + '_x_test.csv'
    x_test.to_csv(fname, index=False)
    print(fname, d.shape)
    
#     fname = 'data/' + mname + '_y_train.csv'
#     y_train[0].to_csv(fname, index=False)
#     print(fname, d.shape)
    
#     fname = 'data/' + mname + '_y_val.csv'
#     y_val[0].to_csv(fname, index=False)
#     print(fname, d.shape)
    
#     fname = 'data/' + mname + '_y_full.csv'
#     y_full[0].to_csv(fname, index=False)
#     print(fname, d.shape)


# In[ ]:


# set(features[i]) - set(lags)


# In[ ]:


# set(lags) - set(features[i])


# In[ ]:


len(features[0])


# In[ ]:


# pd.set_option('display.max_rows', 150)


# In[ ]:


# # q = (d.Date >= '2020-04-02') & (d.Loc=='Cabo Verde')
# # q = (d.Date >= '2020-04-02') & (d.Loc=='Congo (Brazzaville)')
# q = (d.Date >= '2020-04-02') & (d.Loc=='Somalia')
# d.loc[q,['Date','ForecastId','y0','y1','y0r','y1r',
#                         mname + str(0),mname+str(1)]]


# In[ ]:


# most fixes
#                                     ForecastId  ConfirmedCases   Fatalities  \
# Country_Region      Province_State                                            
# Cabo Verde                                1591       83.069233     2.661215   
# Congo (Brazzaville)                       3827      188.486683    10.366824   
# Jamaica                                   6364      310.090653     9.060071   
# Slovakia                                  9374     2407.449830    17.087912   
# Netherlands         Aruba                 7869      207.263786     1.729323   
# Timor-Leste                              10019       15.586367     0.892582   
# Tanzania                                  9933      250.855338    11.648181   
# Somalia                                   9460      213.936255    15.164921   
# Gabon                                     5375      169.672938     2.876582   
# US                  Maryland             11137    24302.549228  1205.949083   

#                                           Date  c_li  f_li  total_fixes  
# Country_Region      Province_State                                       
# Cabo Verde                          2020-05-14  0.49  0.00       0.2401  
# Congo (Brazzaville)                 2020-05-14  0.28  0.00       0.0784  
# Jamaica                             2020-05-14  0.18  0.03       0.0333  
# Slovakia                            2020-05-14 -0.00  0.17       0.0289  
# Netherlands         Aruba           2020-05-14  0.00  0.14       0.0196  
# Timor-Leste                         2020-05-14  0.13  0.00       0.0169  
# Tanzania                            2020-05-14  0.12  0.02       0.0148  
# Somalia                             2020-05-14  0.02  0.10       0.0104  
# Gabon                               2020-05-14  0.09  0.00       0.0081  
# US                  Maryland        2020-05-14  0.04  0.08       0.0080  


# In[ ]:


# # plot actual and predicted curves over time for specific locations
# # locs = ['China Tibet','China Xinjiang','China Hong Kong', 'China Macau',
# #         'Spain','Italy','India',
# #         'US Washington','US New York','US California',
# #         'US North Carolina','US Ohio']
# # xlab = ['03-12','03-18','03-25','04-01','04-08','04-15','04-22']
# # plot all locations
# locs = d['Loc'].drop_duplicates()
# for loc in locs:
#     plt.figure(figsize=(14,2))
    
#     # fig, ax = plt.subplots()
#     # fig.autofmt_xdate()
    
#     for i in range(ny):
    
#         plt.subplot(1,2,i+1)
#         plt.plot(d.loc[d.Loc==loc,[yv[i],'Date']].set_index('Date'))
#         plt.plot(d.loc[d.Loc==loc,[mname + str(i),'Date']].set_index('Date'))
#         # plt.plot(d.loc[d.Loc==loc,[yv[i]+'_pred','Date']].set_index('Date'))
#         # plt.plot(d.loc[d.Loc==loc,[yv[i]]])
#         # plt.plot(d.loc[d.Loc==loc,[yv[i]+'_pred']])
#         # plt.xticks(np.arange(len(xlab)), xlab, rotation=-45)
#         # plt.xticks(np.arange(12), calendar.month_name[3:5], rotation=20)
#         # plt.xticks(rotation=-45)
#         plt.xticks([])
#         plt.title(loc + ' ' + ynames[i])
       
#     plt.show()


# In[ ]:


# fixed.describe()


# In[ ]:




