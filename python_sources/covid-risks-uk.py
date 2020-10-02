#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests
from scipy import stats
from functools import partial
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, LeaveOneGroupOut, LeavePGroupsOut
import lightgbm as lgb
import os
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# Any results you write to the current directory are saved as output.


# In[ ]:


# def is_interactive():
#    return 'runtime' in get_ipython().config.IPKernelApp.connection_file

# if is_interactive(): 
#     plt.style.use('dark_background')

#     COLOR = 'white' # used 'white' when editing in interactive mode with dark theme ON
#     import matplotlib
#     matplotlib.rcParams['text.color'] = COLOR
#     matplotlib.rcParams['axes.labelcolor'] = COLOR
#     matplotlib.rcParams['xtick.color'] = COLOR
#     matplotlib.rcParams['ytick.color'] = COLOR
#     else: plt.style.use('ggplot')
        
plt.style.use('ggplot')


# In[ ]:


def create_axes_grid(numplots_y, numplots_x, plotsize_x=6, plotsize_y=3):
    fig, axes = plt.subplots(numplots_y, numplots_x)
    fig.set_size_inches(plotsize_x * numplots_x, plotsize_y * numplots_y)
    fig.subplots_adjust(wspace=0.1, hspace=0.25)
    return fig, axes

def set_axes(axes, use_grid=True):
    axes.grid(use_grid)
    axes.tick_params(which='both', direction='inout', top=True, right=True, labelbottom=True, labelleft=True)


# In[ ]:


import warnings

class generalCallback():
    def __init__(self, model_trainer, *args, **kwargs):
        self.model_trainer = model_trainer
        
    def onTrainStart(self, *args, **kwargs):
        pass
    def onFoldStart(self, *args, **kwargs):
        pass
    def onFoldEnd(self, *args, **kwargs):
        pass
    def onTrainEnd(self, *args, **kwargs):
        pass
        
        
class Base_Model(object):
    """
    Parent model class, contains functions universal for all models used. Initial implementation credits to Bruno Aquino.
    """
    def __init__(self, train_df, test_df, features, categoricals=[], n_splits=5, verbose=True, target='accuracy_group', val_metric=metrics.r2_score, params=None, cb=None):
        if not verbose: warnings.filterwarnings("ignore")
        self.val_metric = val_metric
        if cb is None: 
            self.cb = generalCallback(self)
        else:
            self.cb = cb(self)
        self.train_df = train_df
        self.test_df = test_df
        self.params = params
        self.features = features
        if verbose:print(f'Using {len(features)} features')
        self.n_splits = n_splits
        self.categoricals = categoricals
        self.target = target
        self.verbose = verbose
        self.all_data = []
        
    def __call__(self):
        self.params = self.get_params()
        self.cv = self.get_cv()
        self.fit()
        
    def train_model(self, train_set, val_set):
        raise NotImplementedError
        
    def get_cv(self):
        cv = GroupKFold(5)
        return cv.split(self.train_df, self.train_df[self.target], groups=self.train_df.area_code)
    
    def get_params(self):
        raise NotImplementedError
        
    def convert_dataset(self, x_train, y_train, x_val, y_val):
        raise NotImplementedError
        
    def convert_x(self, x):
        return x
        
    def fit(self):
        self.val_preds = []
        self.val_ys = []
        self.model = []
#         self.area_codes = []
        self.cb.onTrainStart()
        
        for fold, (train_idx, val_idx) in enumerate(self.cv):
#             self.area_codes.append(self.train_df['area_code'].iloc[val_idx])
            self.cb.onFoldStart(val_idx=val_idx, train_idx=train_idx)
            
#             if sum(self.train_df.iloc[train_idx]['area_code'].isin(self.train_df.iloc[val_idx]['area_code'])) > 0: raise Exception('Same area codes in trn and val sets, may be overfitting')
            
            
            x_train, x_val = self.train_df[self.features].iloc[train_idx], self.train_df[self.features].iloc[val_idx]
            y_train, y_val = self.train_df[self.target][train_idx], self.train_df[self.target][val_idx]
            
            all_data_fold = {
                'x_train': x_train,
                'x_val': x_val,
                'y_train': y_train,
                'y_val': y_val
            }
            self.all_data.append(all_data_fold)
            
            train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)
            
            model = self.train_model(train_set, val_set)
            self.model.append(model)
            conv_x_val = self.convert_x(x_val.reset_index(drop=True))
            
            preds_all = model.predict(conv_x_val)
            self.val_preds.append(preds_all)
            
            oof_score = self.val_metric(y_val, np.array(preds_all)) if type(self.val_metric)!=list else [i(y_val, np.array(preds_all)) for i in self.val_metric]
            if self.verbose: print(f'Partial score (all) of fold {fold} is: {oof_score}')

            self.val_ys.append(y_val.reset_index(drop=True).values)
            
            self.cb.onFoldEnd()

        self.val_ys = np.concatenate(self.val_ys)
        self.val_preds = np.concatenate(self.val_preds)
#         self.area_codes = np.concatenate(self.area_codes)
        self.cb.onTrainEnd()
        
        self.score = self.val_metric(self.val_ys, self.val_preds) if type(self.val_metric)!=list else [i(self.val_ys, self.val_preds) for i in self.val_metric]

        if self.verbose: print(f'Our oof rmse score (all) is: {self.score}')

class uk_hp_model_cb(generalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def onTrainStart(self, *args, **kwargs):
        self.model_trainer.area_codes = []
    
    def onFoldStart(self, *args, **kwargs):
        val_idx = kwargs['val_idx']
        train_idx = kwargs['train_idx']
        self.model_trainer.area_codes.append(self.model_trainer.train_df['area_code'].iloc[val_idx])
        if sum(self.model_trainer.train_df.iloc[train_idx]['area_code'].isin(self.model_trainer.train_df.iloc[val_idx]['area_code'])) > 0: 
            raise Exception('Same area codes in trn and val sets, may be overfitting')
        
    def onFoldEnd(self, *args, **kwargs):
        pass
    
    def onTrainEnd(self, *args, **kwargs):
        self.model_trainer.area_codes = np.concatenate(self.model_trainer.area_codes)   
    
    
class Lgb_Model(Base_Model):
    def train_model(self, train_set, val_set):
        verbosity = 100 if self.verbose else 0
        return lgb.train(self.params, train_set, valid_sets=[train_set, val_set], verbose_eval=verbosity)
        
    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = lgb.Dataset(x_train, y_train, categorical_feature=self.categoricals)
        val_set = lgb.Dataset(x_val, y_val, categorical_feature=self.categoricals)
        return train_set, val_set
        
    def get_params(self):
        params = {'n_estimators':6000,
                    'boosting_type': 'gbdt',
                    'objective': 'regression',
                    'metric': 'rmse',
                    'subsample': 0.75,
                    'subsample_freq': 1,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.8,
                    'max_depth': 150, # was 15
                  'num_leaves': 50,
                    'lambda_l1': 0.1,  
                    'lambda_l2': 0.1,
                    'early_stopping_rounds': 300,
                  'min_data_in_leaf': 1,
                          'min_gain_to_split': 0.01,
                          'max_bin': 400
                    } if not self.params else self.params
        return params


# In[ ]:


class check():
    def __init__(self, model):
        self.m = model
    def test(self):
        print(self.m.b)

class asd():
    def __init__(self):
        self.a = 123
        self.cb = check(self)
    
    def addNewVar(self):
        self.b = '1111111'
        
    def test(self):
        self.cb.test()

tobj = asd()
tobj.addNewVar()
tobj.test()


# In[ ]:


def prep_features_list(train_data, exc_col, cor_trh):
    train_features = [i for i in train_data.columns if i not in exc_col]
    # train_features = important_features
    counter = 0
    to_remove = []
    for feat_a in train_features:
        for feat_b in train_features:
            if (feat_a != feat_b) & (feat_a not in to_remove) & (feat_b not in to_remove):
                c = np.corrcoef(train_data[feat_a], train_data[feat_b])[0][1]
                if c > cor_trh:
                    counter += 1
                    to_remove.append(feat_a)
                    print(f'{counter}: FEAT_A: {feat_a} ||| FEAT_B: {feat_b} ||| Correlation: {np.round(c,3)}')


    train_features = [i for i in train_data.columns if i not in exc_col + to_remove]

    for i in to_remove:print(i)
    
    return train_features

def adj_r2_score(truth, preds, n_predictors):
    n = truth.shape[0]
    r2 = metrics.r2_score(truth, preds)
    res = 1-(1-r2)*(n-1)/(n-n_predictors-1)
    return res


# # Get health data from PHE

# I have downloaded UK data on various health profiles across UK regions
# 
# Link - https://fingertips.phe.org.uk/profile/health-profiles/data#page/6/gid/8000073/pat/6/par/E12000004/ati/201/are/E07000032/iid/20201/age/1/sex/2/cid/4/page-options/map-ao-4_cin-ci-4_ovw-tdo-0

# The dataset contains a lot of different indicators. Some of them contain values for different age groups, different sexes and for different time periods. The values of indicators themselves also contain confidence intervals and comparisons to other UK statistics.
# 
# Notes
# * Despite all that information, at the moment I only use value by indicator by group, everything else being summed up. 
# * Summing indicator values up is appropriate for indicators measured in absolute numbers, but is not very appropriate for percentages.

# In[ ]:


hp_original = pd.read_csv('../input/uk-covid-19-related-data/indicators-DistrictUA.data.csv')
hp = hp_original.copy()


# In[ ]:


def prep_health_profiles(hp):
    hp.rename({i:i.replace(' ','_').lower() for i in hp.columns}, axis=1,inplace=True)
    
    hp_cols_to_use = ['indicator_name','area_code','area_name','sex','age','category_type','category','time_period','value']#,'count','denominator']
    hp = hp.loc[~hp.value.isna(),hp_cols_to_use]
    
    last_periods = hp.groupby('indicator_name', as_index=False).last()[['indicator_name','time_period']]
    hp = last_periods.merge(hp, on=['indicator_name','time_period'], validate='one_to_many')
    
    hp = hp.groupby(['indicator_name', 'area_code'], as_index=False).mean()[['indicator_name', 'area_code','value']]#,'count','denominator']]
    
    hp = hp.pivot(columns='indicator_name',index='area_code',values='value')
    
    return hp


# In[ ]:


health_profiles = prep_health_profiles(hp)


# # Get covid mortality data

# The data comes from UK's Office for National Statistics
# 
# Link: https://www.ons.gov.uk/peoplepopulationandcommunity/healthandsocialcare/causesofdeath/datasets/deathregistrationsandoccurrencesbylocalauthorityandhealthboard

# The dataset contains registered deaths by UK regions, weeks, causes (covid vs all), place
# 
# Again, at the moment only used part of this data. Here using regions and weeks.

# In[ ]:


def prep_mortality_data(d):
    cols = d.loc[2,:]
    d = d.loc[3:,:]
    d.columns = cols
    
    d.reset_index(drop=True,inplace=True)
    d.index.name=None
    d.columns.name=None
    
    d.rename({i:i.replace(' ','_').lower() for i in d.columns}, axis=1,inplace=True)
    d.rename({'area_name_':'area_name'}, axis=1,inplace=True)
    
    d_weekly = d.groupby(['area_code', 'cause_of_death', 'week_number'], as_index=False).sum().drop(['place_of_death'], axis=1)
    d_all = d.groupby(['area_code', 'cause_of_death'], as_index=False).sum().drop(['week_number', 'place_of_death'], axis=1)
    
    return d_weekly, d_all
    
def get_mortality_ratios(d, USE_WEEK):
    d.drop(['geography_type', 'area_name'], axis=1, inplace=True)
    
    d_all = d.loc[d.cause_of_death=='All causes',:].drop('cause_of_death', axis=1).rename({'number_of_deaths':'dnum_all'}, axis=1)
    d_cv = d.loc[d.cause_of_death=='COVID 19',:].drop('cause_of_death', axis=1).rename({'number_of_deaths':'dnum_cv'}, axis=1)

    d_prop = d_cv.merge(d_all, how='left', on=['area_code', 'week_number'], validate='one_to_one') if USE_WEEK else d_cv.merge(d_all, how='left', on=['area_code'], validate='one_to_one')
    d_prop.loc[:,'ratio'] = d_prop.dnum_cv/d_prop.dnum_all

    d_prop.drop(['dnum_cv','dnum_all'], axis=1, inplace=True)
    
    d_prop = d_prop.loc[d_prop.ratio>0,:].reset_index(drop=True)
    
    return d_prop


# In[ ]:


d_original = pd.read_excel('../input/uk-covid-19-related-data/lahbtablesweek22.xlsx', sheet_name='Registrations - All data')
d = d_original.copy()


# In[ ]:


mortality_data_weekly, mortality_data_weekly_sum = prep_mortality_data(d)


# In[ ]:


weekly_ratios = get_mortality_ratios(mortality_data_weekly, USE_WEEK=True)
sum_ratios = get_mortality_ratios(mortality_data_weekly_sum, USE_WEEK=False)


# # Population data

# From https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/populationestimatesanalysistool

# In[ ]:


def get_population_data(cd):
    cd.rename({i:i.replace(' ','_').lower() for i in cd.columns}, axis=1,inplace=True)
    cd.loc[:,'internal_migration_ratio'] = cd.loc[:,'internal_migration_net']/cd.loc[:,'estimated_population_2016.']
    cd.loc[:,'international_migration_ratio'] = cd.loc[:,'international_migration_net']/cd.loc[:,'estimated_population_2016.']
    
    cd.rename({'la_code':'area_code'}, axis=1, inplace=True)
    cols_to_use = [i for i in cd.columns if i not in ['la_name','country_code','country_name','region_code','region_name','county_code','county_name','estimated_population_2015']]
    
    cd = cd[cols_to_use]
    
    return cd


# In[ ]:


# cd_original = pd.read_excel('../input/uk-covid-19-related-data/Analysis Tool mid-2016 UK.xlsx', sheet_name='Components of Change')
cd_original = pd.read_csv('../input/uk-covid-19-related-data/Analysis Tool mid-2016 UK.csv')
cd = cd_original.copy()


# In[ ]:


population_data = get_population_data(cd)


# ### Density

# From https://www.nomisweb.co.uk/census/2011/wd102ew

# In[ ]:


density = pd.read_csv('../input/uk-covid-19-related-data/density.csv')


# In[ ]:


density = density[['geography code', 'Area/Population Density: Density (number of persons per hectare); measures: Value']]
density = density.rename({
    'Area/Population Density: Density (number of persons per hectare); measures: Value':'density',
    'geography code':'area_code'
}, axis=1)


# In[ ]:


density.head()


# # Merging

# In[ ]:


def merge_all(mortality_ratios, health_profiles, population_data):
    hp_and_d = mortality_ratios.merge(health_profiles, left_on='area_code', right_index=True, how='left', validate='many_to_one')

    hp_and_d.rename({i:i.replace(' ','_').lower() for i in hp_and_d.columns}, axis=1,inplace=True)
    hp_and_d.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in hp_and_d.columns]
    
    hp_and_d.replace([np.inf, -np.inf], np.nan, inplace=True)

    hp_and_d.reset_index(drop=True,inplace=True)
    
    hp_d_cd = hp_and_d.merge(population_data, on='area_code', how='left')
    
    hp_d_cd = hp_d_cd.merge(density, on='area_code', how='left')
    
    return hp_d_cd


# In[ ]:


train_data_weekly = merge_all(weekly_ratios, health_profiles, population_data)
train_data_sum = merge_all(sum_ratios, health_profiles, population_data)


# # Map

# In[ ]:


import geopandas as gpd

ukmap = gpd.read_file('../input/uk-covid-19-related-data/Local_Authority_Districts__April_2019__Boundaries_UK_BFE-shp/Local_Authority_Districts__April_2019__Boundaries_UK_BFE.shp')


# In[ ]:


for_map = ukmap.merge(mortality_data_weekly_sum, left_on='LAD19CD', right_on='area_code', how='right')

fig, ax = plt.subplots(1, figsize=(20, 12))

sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
sm._A = []
cbar = fig.colorbar(sm)
ax.set_title('Ratio of covid-19 deaths to all deaths per UK local authrotity. Data as of w/e 12-Jun.')
for_map.plot(column='number_of_deaths', cmap='plasma', linewidth=0.8, ax=ax, edgecolor='0.8')


# In[ ]:


for_map = ukmap.merge(population_data, left_on='LAD19CD', right_on='area_code', how='right')

fig, ax = plt.subplots(1, figsize=(20, 12))

sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
sm._A = []
cbar = fig.colorbar(sm)
ax.set_title('Ratio of covid-19 deaths to all deaths per UK local authrotity. Data as of w/e 12-Jun.')
for_map.plot(column='births', cmap='plasma', linewidth=0.8, ax=ax, edgecolor='0.8')


# In[ ]:


for_map = ukmap.merge(health_profiles, left_on='LAD19CD', right_index=True, how='right')

fig, ax = plt.subplots(1, figsize=(20, 12))

sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
sm._A = []
cbar = fig.colorbar(sm)
ax.set_title('Ratio of covid-19 deaths to all deaths per UK local authrotity. Data as of w/e 12-Jun.')
for_map.plot(column='Suicide rate', cmap='plasma', linewidth=0.8, ax=ax, edgecolor='0.8')


# In[ ]:


for_map = ukmap.merge(train_data_sum, left_on='LAD19CD', right_on='area_code', how='right')

fig, ax = plt.subplots(1, figsize=(20, 12))

sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
sm._A = []
cbar = fig.colorbar(sm)
ax.set_title('Ratio of covid-19 deaths to all deaths per UK local authrotity')
for_map.plot(column='ratio', cmap='plasma', linewidth=0.8, ax=ax, edgecolor='0.8')


# In[ ]:


plt.hist(np.log(train_data_sum.density))


# In[ ]:


from textwrap import wrap


# In[ ]:


for_map = ukmap.merge(train_data_sum, left_on='LAD19CD', right_on='area_code', how='right')
from mpl_toolkits.axes_grid1 import make_axes_locatable

# for_map.loc[:,'supporting_information_____population_from_ethnic_minorities'] -= for_map.loc[:,'supporting_information_____population_from_ethnic_minorities']

fig, ax = plt.subplots(1,2, figsize=(20, 12))

var = 'density'
ax[0].set_title('Population density (persons per hectare) (log scale)', pad=20)
for_map.loc[:, var] = np.log(for_map.loc[:, var])
# for_map.loc[:,var] /= for_map['estimated_population_2016.']
for_map.plot(column=var, cmap='plasma', linewidth=0.8, ax=ax[0], edgecolor='0.8')

sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=for_map[var].min(), vmax=for_map[var].max()))
sm._A = []
cbar = fig.colorbar(sm, ax=ax[0])

# divider = make_axes_locatable(ax[0])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(im1, cax=cax, orientation='vertical')

var = 'supporting_information_____population_from_ethnic_minorities'
ax[1].set_title('Ethnic minorities (%)', pad=20)
for_map.plot(column=var, cmap='plasma', linewidth=0.8, ax=ax[1], edgecolor='0.8')

sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=for_map[var].min(), vmax=for_map[var].max()))
sm._A = []
cbar = fig.colorbar(sm, ax=ax[1])


# In[ ]:


for_map = ukmap.merge(train_data_sum, left_on='LAD19CD', right_on='area_code', how='right')
from mpl_toolkits.axes_grid1 import make_axes_locatable

# for_map.loc[:,'supporting_information_____population_from_ethnic_minorities'] -= for_map.loc[:,'supporting_information_____population_from_ethnic_minorities']

fig, ax = plt.subplots(1,2, figsize=(20, 12))

var = 'tb_incidence__three_year_average_'
ax[0].set_title('Tuberculosis incidence per 100,000 (3-year average)', pad=20)
# for_map.loc[:,var] /= 100
for_map.plot(column=var, cmap='plasma', linewidth=0.8, ax=ax[0], edgecolor='0.8')

sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=for_map[var].min(), vmax=for_map[var].max()))
sm._A = []
cbar = fig.colorbar(sm, ax=ax[0])

# divider = make_axes_locatable(ax[0])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(im1, cax=cax, orientation='vertical')

var = 'smoking_status_at_time_of_delivery'
ax[1].set_title('Smoking status at time of delivery (%)', pad=20)
# for_map.loc[:,var] /= 100
for_map.plot(column=var, cmap='plasma', linewidth=0.8, ax=ax[1], edgecolor='0.8')

sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=for_map[var].min(), vmax=for_map[var].max()))
sm._A = []
cbar = fig.colorbar(sm, ax=ax[1])


# In[ ]:


train_data_sum.sort_values('births', ascending=False)[['area_code','births']]


# In[ ]:


for_map = ukmap.merge(train_data_sum, left_on='LAD19CD', right_on='area_code', how='right')
from mpl_toolkits.axes_grid1 import make_axes_locatable

# for_map.loc[:,'supporting_information_____population_from_ethnic_minorities'] -= for_map.loc[:,'supporting_information_____population_from_ethnic_minorities']

fig, ax = plt.subplots(1,2, figsize=(20, 12))

var = 'supporting_information_____population_aged_under_18'
ax[0].set_title('Population aged under 18 (%)', pad=20)
# for_map.loc[:,var] /= 100
for_map.plot(column=var, cmap='plasma', linewidth=0.8, ax=ax[0], edgecolor='0.8')

sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=for_map[var].min(), vmax=for_map[var].max()))
sm._A = []
cbar = fig.colorbar(sm, ax=ax[0])

# divider = make_axes_locatable(ax[0])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(im1, cax=cax, orientation='vertical')

var = 'births'
ax[1].set_title('Births (log scale)', pad=20)
for_map.loc[:,var] = np.log(for_map.loc[:,var])
for_map.plot(column=var, cmap='plasma', linewidth=0.8, ax=ax[1], edgecolor='0.8')

sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=for_map[var].min(), vmax=for_map[var].max()))
sm._A = []
cbar = fig.colorbar(sm, ax=ax[1])


# In[ ]:


train_data_sum.head()


# # World data

# In[ ]:


os.listdir("../input/uk-covid-19-related-data")


# In[ ]:


wdata = pd.read_csv('../input/uk-covid-19-related-data/owid-covid-data.csv', parse_dates=['date'])


# In[ ]:


wdata.location.unique()


# In[ ]:


wdata.head()


# In[ ]:


w = wdata.loc[(wdata.total_cases_per_million!=0) & (~wdata.total_cases_per_million.isna())]
count_by_location = w.groupby('location').count().sort_values('date')
mode = count_by_location.date.mode().iloc[0]
print(mode)
countries_include = count_by_location.loc[count_by_location.date>=mode].index.values


# In[ ]:





# In[ ]:


w = w.loc[w.location.isin(countries_include)]

all_ts = np.zeros((countries_include.shape[0], mode, 1))
all_ts_scaled = np.zeros((countries_include.shape[0], mode, 1))
all_labels = np.zeros(countries_include.shape[0], dtype=object)

for idx, c in enumerate(countries_include):
    curr_country = w.loc[w.location==c]
    values = curr_country.total_cases_per_million.iloc[0:mode].values
    all_ts[idx, :, 0] = np.gradient(values)
    
    values -= np.min(values)
    values /= np.max(values)
    all_ts_scaled[idx, :, 0] = values

    all_labels[idx] = c


# In[ ]:


get_ipython().system('pip install tslearn')


# In[ ]:


from tslearn.clustering import TimeSeriesKMeans


# In[ ]:





# In[ ]:


n_clusters = 6
km = TimeSeriesKMeans(n_clusters, n_jobs=-1).fit_predict(all_ts)

fig, axes = create_axes_grid(n_clusters//2,2,10,5)
col = 0

for row, cluster_id in zip(np.repeat(np.arange(n_clusters/2, dtype='int'), 2), range(n_clusters)):
    names = all_labels[km==cluster_id]
    name = ', '.join([i for i in names[0:5]])
    
    cases = all_ts[km==cluster_id, :, 0]
    
    axes[row, col].set_title(f'C {cluster_id} | {len(names)} countries | {name}')
    axes[row, col].set_ylabel('Total cases per million')
    axes[row, col].set_xlabel('Days since case 1')
    
    days = np.arange(0, mode)
    
    for i in range(sum(km==cluster_id)):
        axes[row, col].plot(days, cases[i,:], color='blue')
        
    col = 1 if col == 0 else 0


# In[ ]:


n_clusters = 6
km = TimeSeriesKMeans(n_clusters, n_jobs=-1).fit_predict(all_ts_scaled)

fig, axes = create_axes_grid(n_clusters//2,2,10,5)
col = 0

for row, cluster_id in zip(np.repeat(np.arange(n_clusters/2, dtype='int'), 2), range(n_clusters)):
    names = all_labels[km==cluster_id]
    name = ', '.join([i for i in names[0:5]])
    
    cases = all_ts_scaled[km==cluster_id, :, 0]
    
    axes[row, col].set_title(f'C {cluster_id} | {len(names)} countries | {name}')
    axes[row, col].set_ylabel('Total cases per million')
    axes[row, col].set_xlabel('Days since case 1')
    
    days = np.arange(0, mode)
    
    for i in range(sum(km==cluster_id)):
        axes[row, col].plot(days, cases[i,:], color='blue')
        
    col = 1 if col == 0 else 0


# In[ ]:


for row, cluster_id in zip(np.repeat(np.arange(n_clusters/2, dtype='int'), 2), range(n_clusters)):
    cluster_countries = all_labels[km==cluster_id]
    
    wdata.loc[wdata.location.isin(cluster_countries), 'cluster_id'] = cluster_id
    
    names = sorted(cluster_countries)
    name = [', '.join(names[i:i+10]) for i in range(0, len(names), 10)]

    name = '\n '.join([i for i in name])
    
    print(f'C {cluster_id} | {len(names)} countries | {name}  \n')


# In[ ]:


cols_include = ['cluster_id','total_cases','total_cases_per_million','total_tests','total_tests_per_thousand','tests_units','population_density','median_age','aged_65_older','aged_70_older', 'gdp_per_capita', 'cvd_death_rate', 'diabetes_prevalence']
cluster_details = wdata[cols_include+['location']].groupby(['cluster_id','location'], as_index=False).last()
cluster_details


# In[ ]:


cluster_details_mean = cluster_details[cols_include].groupby('cluster_id').median()
cluster_details_mean


# In[ ]:


train_data = cluster_details

train_features = [i for i in cluster_details if i not in ['cluster_id','location','tests_units',]]#'total_cases', 'total_tests']]

def wrapper(y, yhat, func):
    y_m = np.zeros((yhat.shape[0],yhat.shape[1]))
    if type(y) == pd.Series:
        y_m[np.arange(0,y_m.shape[0],dtype=int), y.values.astype('int')] = 1
    else:
        y_m[np.arange(0,y_m.shape[0],dtype=int), y.astype('int')] = 1
    r = func(y_m, yhat)
    return r

# validation_metrics = [metrics.r2_score, partial(adj_r2_score, n_predictors=len(train_features)), lambda x,y: stats.pearsonr(x,y)[0]]
validation_metrics = [partial(wrapper, func=metrics.roc_auc_score)]

params = {'n_estimators':6000,
                    'boosting_type': 'gbdt',
                    'objective': 'multiclass',
                      'num_classes': n_clusters,
                    'metric': 'multiclass',
                    'subsample': 0.7,
                    'subsample_freq': 1,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.7,
                    'max_depth': 10, # was 15
                  'num_leaves': 5,
                    'lambda_l1': 0.1,  
                    'lambda_l2': 0.1,
                    'early_stopping_rounds': 100,
                  'min_data_in_leaf': 1,
                          'min_gain_to_split': 0.01,
                          'max_bin': 100
                    }

import types

lgb_model = Lgb_Model(train_df=train_data, test_df=None, features=train_features, categoricals=[], n_splits=5, verbose=True, target='cluster_id', val_metric=validation_metrics, params=params, cb=None)
# lgb_model.get_cv = types.MethodType( lambda x: KFold(2, shuffle=True, random_state=1997).split(x.train_df, x.train_df[x.target]), lgb_model )
lgb_model.get_cv = types.MethodType( lambda x: StratifiedKFold(2).split(x.train_df, x.train_df[x.target]), lgb_model )
lgb_model()


# In[ ]:


model = lgb_model.model[0]
fig, ax = plt.subplots(1,1,figsize=(15, 10))
lgb.plot_importance(model, max_num_features = 20, ax=ax)


# In[ ]:


fig, ax = plt.subplots(2,6,figsize=(33, 10))

var = 'total_cases'
ax[0, 0].set_title(var)
bplot0 = ax[0, 0].boxplot([train_data.loc[(train_data.cluster_id==i) & (train_data[var]<1e6), var] for i in range(6)], labels=range(6), patch_artist=True)

var = 'total_tests'
ax[0, 1].set_title(var)
bplot1 = ax[0, 1].boxplot([train_data.loc[(train_data.cluster_id==i) & (train_data[var]<0.6*1e7), var] for i in range(6)], labels=range(6), patch_artist=True)

var = 'cvd_death_rate'
ax[0, 2].set_title(var)
bplot2 = ax[0, 2].boxplot([train_data.loc[(train_data.cluster_id==i) & (train_data[var]<np.inf), var] for i in range(6)], labels=range(6), patch_artist=True)

var = 'total_cases_per_million'
ax[0, 3].set_title(var)
bplot3 = ax[0, 3].boxplot([train_data.loc[(train_data.cluster_id==i) & (train_data[var]<np.inf), var] for i in range(6)], labels=range(6), patch_artist=True)

var = 'median_age'
ax[0, 4].set_title(var)
bplot4 = ax[0, 4].boxplot([train_data.loc[(train_data.cluster_id==i) & (train_data[var]<np.inf), var] for i in range(6)], labels=range(6), patch_artist=True)

var = 'aged_65_older'
ax[0, 5].set_title(var)
bplot5 = ax[0, 5].boxplot([train_data.loc[(train_data.cluster_id==i) & (train_data[var]<np.inf), var] for i in range(6)], labels=range(6), patch_artist=True)

var = 'population_density'
ax[1, 0].set_title(var)
bplot6 = ax[1, 0].boxplot([train_data.loc[(train_data.cluster_id==i) & (train_data[var]<750), var] for i in range(6)], labels=range(6), patch_artist=True)

var = 'diabetes_prevalence'
ax[1, 1].set_title(var)
bplot7 = ax[1, 1].boxplot([train_data.loc[(train_data.cluster_id==i) & (train_data[var]<0.6*1e7), var] for i in range(6)], labels=range(6), patch_artist=True)

var = 'total_tests_per_thousand'
ax[1, 2].set_title(var)
bplot8 = ax[1, 2].boxplot([train_data.loc[(train_data.cluster_id==i) & (train_data[var]<np.inf), var] for i in range(6)], labels=range(6), patch_artist=True)

var = 'gdp_per_capita'
ax[1, 3].set_title(var)
bplot9 = ax[1, 3].boxplot([train_data.loc[(train_data.cluster_id==i) & (train_data[var]<np.inf), var] for i in range(6)], labels=range(6), patch_artist=True)

var = 'aged_70_older'
ax[1, 4].set_title(var)
bplot10 = ax[1, 4].boxplot([train_data.loc[(train_data.cluster_id==i) & (train_data[var]<np.inf), var] for i in range(6)], labels=range(6), patch_artist=True)

# var = 'aged_65_older'
# ax[1, 5].set_title(var)
# ax[1, 5].boxplot([train_data.loc[(train_data.cluster_id==i) & (train_data[var]<np.inf), var] for i in range(6)], labels=range(6))

for bp in [bplot0, bplot1, bplot2,bplot3,bplot4,bplot5,bplot6,bplot7,bplot8,bplot9,bplot10]:
    for b, c in zip(bp['boxes'], ['yellow','green','red','green','red','red']):
        b.set_facecolor(c)

plt.show()


# In[ ]:





# In[ ]:


model = lgb_model.model[0]

fi = [(i,f) for i, f in zip(model.feature_name(), model.feature_importance())]
fi = sorted(fi, key = lambda x: x[1], reverse=True)
cutoff_trh = np.percentile([i[1] for i in fi], 10)
print(cutoff_trh)
important_features = [i[0] for i in fi if i[1] > cutoff_trh]
for i,z in fi: print((i,z))


# In[ ]:


get_ipython().run_line_magic('time', '')

import shap
x_val = lgb_model.all_data[0]['x_val']

shap_values = shap.TreeExplainer(lgb_model.model[0]).shap_values(x_val)

shap.summary_plot(shap_values, x_val)


# In[ ]:


# for f in important_features[0:8]:
#     shap.dependence_plot(f, shap_values[-1], x_val)


# # Sweden

# In[ ]:


fig, axes = create_axes_grid(1,2,10,5)

y = 'total_cases_per_million'
x = 'United Kingdom'
x2 = 'Sweden'
axes[0].set_title(f'{y} in {x} &  {x2}')
axes[0].plot(wdata.loc[wdata.location==x, 'date'], wdata.loc[wdata.location==x, y], label=x);
axes[0].plot(wdata.loc[wdata.location==x2, 'date'], wdata.loc[wdata.location==x2, y], label=x2);
axes[0].legend()

y = 'total_deaths_per_million'
x = 'United Kingdom'
x2 = 'Sweden'
axes[1].set_title(f'{y} in {x} &  {x2}')
axes[1].plot(wdata.loc[wdata.location==x, 'date'], wdata.loc[wdata.location==x, y], label=x);
axes[1].plot(wdata.loc[wdata.location==x2, 'date'], wdata.loc[wdata.location==x2, y], label=x2);
axes[1].legend()


# In[ ]:


fig, axes = create_axes_grid(1,2,10,5)

y = 'new_cases_per_million'
x = 'United Kingdom'
x2 = 'Sweden'
axes[0].set_title(f'{y} in {x} &  {x2}')
axes[0].plot(wdata.loc[wdata.location==x, 'date'], wdata.loc[wdata.location==x, y].rolling(7).mean(), label=x);
axes[0].plot(wdata.loc[wdata.location==x2, 'date'], wdata.loc[wdata.location==x2, y].rolling(7).mean(), label=x2);
axes[0].legend()

y = 'new_deaths_per_million'
x = 'United Kingdom'
x2 = 'Sweden'
axes[1].set_title(f'{y} in {x} &  {x2}')
axes[1].plot(wdata.loc[wdata.location==x, 'date'], wdata.loc[wdata.location==x, y].rolling(7).mean(), label=x);
axes[1].plot(wdata.loc[wdata.location==x2, 'date'], wdata.loc[wdata.location==x2, y].rolling(7).mean(), label=x2);
axes[1].legend()


# In[ ]:





# In[ ]:


def norm(arr):
    return (arr - np.mean(arr))/ np.std(arr)

fig, axes = create_axes_grid(1,2,10,5)

y = 'total_cases_per_million'
x = 'United Kingdom'
x2 = 'Sweden'
axes[0].set_title(f'{y} in {x} &  {x2}')
axes[0].hist(norm(wdata.loc[(wdata.location==x) & (wdata[y]>0), y].values.reshape(-1,1)), label=x);
axes[0].hist(norm(wdata.loc[(wdata.location==x2) & (wdata[y]>0), y].values.reshape(-1,1)), label=x2);
axes[0].legend()

y = 'total_deaths_per_million'
x = 'United Kingdom'
x2 = 'Sweden'
axes[1].set_title(f'{y} in {x} &  {x2}')
axes[1].hist(norm(wdata.loc[(wdata.location==x) & (wdata[y]>0), y].values.reshape(-1,1)), label=x);
axes[1].hist(norm(wdata.loc[(wdata.location==x2) & (wdata[y]>0), y].values.reshape(-1,1)), label=x2);
axes[1].legend()


# In[ ]:





# In[ ]:


from scipy import stats

y = 'total_cases_per_million'
a = wdata.loc[(wdata.location==x) & (wdata[y]>0), y].values
b = wdata.loc[(wdata.location==x2) & (wdata[y]>0), y].values
# a = wdata.loc[(wdata.location==x), y].values
# b = wdata.loc[(wdata.location==x2), y].values
print(np.std(a), np.std(b), a.shape, b.shape)
stats.ttest_ind(a, b, equal_var=False), stats.mannwhitneyu(a, b)


# In[ ]:


y = 'total_deaths_per_million'
a = wdata.loc[(wdata.location==x) & (wdata[y]>0), y].values
b = wdata.loc[(wdata.location==x2) & (wdata[y]>0), y].values
# a = wdata.loc[(wdata.location==x), y].values
# b = wdata.loc[(wdata.location==x2), y].values
print(np.std(a), np.std(b), a.shape, b.shape)
stats.ttest_ind(a, b, equal_var=False), stats.mannwhitneyu(a, b)


# In[ ]:


y = 'new_cases_per_million'
a = wdata.loc[(wdata.location==x) & (wdata[y]>0), y].values
b = wdata.loc[(wdata.location==x2) & (wdata[y]>0), y].values
# a = wdata.loc[(wdata.location==x), y].values
# b = wdata.loc[(wdata.location==x2), y].values
print(np.std(a), np.std(b), a.shape, b.shape)
stats.ttest_ind(a, b, equal_var=False), stats.mannwhitneyu(a, b)


# In[ ]:


y = 'new_deaths_per_million'
a = wdata.loc[(wdata.location==x) & (wdata[y]>0), y].values
b = wdata.loc[(wdata.location==x2) & (wdata[y]>0), y].values
# a = wdata.loc[(wdata.location==x), y].values
# b = wdata.loc[(wdata.location==x2), y].values
print(np.std(a), np.std(b), a.shape, b.shape)
stats.ttest_ind(a, b, equal_var=False), stats.mannwhitneyu(a, b)


# In[ ]:





# In[ ]:


train_data = wdata.loc[wdata.location.isin([x, x2]), ['location', 'date', 'total_cases_per_million', 'total_deaths_per_million']].reset_index(drop=True)
train_data.loc[train_data.location==x, 'date'] = np.arange(0, train_data.loc[train_data.location==x, 'date'].shape[0])
train_data.loc[train_data.location==x2, 'date'] = np.arange(0, train_data.loc[train_data.location==x2, 'date'].shape[0])

np.random.seed(123321)
ridx = np.random.permutation(train_data.shape[0])
train_data = train_data.loc[ridx,:].reset_index(drop=True)
val_cutoff = round(0.8*train_data.shape[0])

cat_dict = {
    'United Kingdom': 0,
    'Sweden': 1
}
train_data.loc[:, 'location'] = train_data.loc[:, 'location'].map(cat_dict)

train_features = ['date', 'total_cases_per_million', 'total_deaths_per_million']
validation_metrics = [metrics.r2_score, partial(adj_r2_score, n_predictors=len(train_features)), lambda x,y: stats.pearsonr(x,y)[0]]

params = {'n_estimators':6000,
                    'boosting_type': 'gbdt',
                    'objective': 'regression',
                    'metric': 'rmse',
                    'subsample': 0.7,
                    'subsample_freq': 1,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.7,
                    'max_depth': 10, # was 15
                  'num_leaves': 5,
                    'lambda_l1': 0.1,  
                    'lambda_l2': 0.1,
                    'early_stopping_rounds': 100,
                  'min_data_in_leaf': 1,
                          'min_gain_to_split': 0.01,
                          'max_bin': 100
                    }


lgb_model = Lgb_Model(train_df=train_data, test_df=None, features=train_features, categoricals=[], n_splits=5, verbose=True, target='location', val_metric=validation_metrics, params=params, cb=None)
lgb_model.get_cv = lambda: [(np.arange(0, val_cutoff, dtype=int), np.arange(val_cutoff, train_data.shape[0], dtype=int))]
lgb_model()


# In[ ]:





# In[ ]:





# # Descriptive plots

# In[ ]:


area_code_to_name = cd_original.groupby(['LA Code','LA Name'], as_index=False).first()[['LA Code','LA Name']]
area_code_to_name.rename({'LA Code':'area_code','LA Name':'la_name'},  axis=1, inplace=True)

num_areas = 20
hp_d_cd = train_data_weekly.sort_values('ratio')
areas = hp_d_cd.area_code.unique()[0:num_areas]
hp_d_cd = hp_d_cd.sort_values(['area_code','week_number'])

fig, axes = create_axes_grid(num_areas//2,2,10,5)
col = 0

for row, area in zip(np.repeat(np.arange(num_areas/2, dtype='int'), 2), areas):
    name = area_code_to_name.loc[area_code_to_name.area_code==area,'la_name'].values[0]
    population = hp_d_cd.loc[hp_d_cd.area_code==area, 'estimated_population_2016.'].values[0]
    axes[row, col].set_title(f'{name} - {area} | Population: {population}')
    axes[row, col].set_ylabel('Cov deaths / all deaths ratio')
    
    weeks = hp_d_cd.loc[hp_d_cd.area_code==area, 'week_number']
    ratios = hp_d_cd.loc[hp_d_cd.area_code==area, 'ratio']
    
    axes[row, col].plot(weeks, ratios, color='blue')
    col = 1 if col == 0 else 0


# In[ ]:





# In[ ]:





# # Modelling

# Using light gbm, one of the most mainstream and most powerful and easy-to-use models
# 
# I did not adjust hyperparameters specifically, just used what I used in one of previous competitions I took part in.

# In[ ]:





# In[ ]:





# In[ ]:





# ## Weekly

# In[ ]:


original_train_data_weekly = train_data_weekly.copy()


# In[ ]:


train_data_weekly =original_train_data_weekly

smoking_related = [i for i in train_data_weekly.columns if 'smoking' in i]
# train_data_weekly[smoking_related] -= train_data_weekly[smoking_related].min()
# train_data_weekly[smoking_related] /= train_data_weekly[smoking_related].max()
# train_data_weekly.loc[:, 'smoking_related'] = train_data_weekly[smoking_related].mean(axis=1)
# train_data_weekly.drop(smoking_related, axis=1, inplace=True)


# In[ ]:


[i for i in train_data_weekly.columns if 'smoking' in i]


# In[ ]:


train_features = prep_features_list(train_data_weekly, exc_col = ['area_code', 'geography_type', 'area_name', 'ratio'], cor_trh=0.95)
validation_metrics = [metrics.r2_score, partial(adj_r2_score, n_predictors=len(train_features)), lambda x,y: stats.pearsonr(x,y)[0]]

params = {'n_estimators':6000,
                    'boosting_type': 'gbdt',
                    'objective': 'regression',
                    'metric': 'rmse',
                    'subsample': 0.7,
                    'subsample_freq': 1,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.7,
                    'max_depth': 10, # was 15
                  'num_leaves': 5,
                    'lambda_l1': 0.1,  
                    'lambda_l2': 0.1,
                    'early_stopping_rounds': 100,
                  'min_data_in_leaf': 1,
                          'min_gain_to_split': 0.01,
                          'max_bin': 100
                    }


lgb_model = Lgb_Model(train_df=train_data_weekly, test_df=None, features=train_features, categoricals=[], n_splits=5, verbose=True, target='ratio', val_metric=validation_metrics, params=params, cb=uk_hp_model_cb)


# In[ ]:


lgb_model()


# In[ ]:


'FINAL SCORE:', lgb_model.score


# In[ ]:


fig, axes = create_axes_grid(1,1,10,5)
axes.set_title('Truth vs preds')
axes.set_ylabel('True ratio')
axes.set_xlabel('Predicted ratio')
axes.scatter(lgb_model.val_preds, lgb_model.val_ys, color='blue')


# In[ ]:


model = lgb_model.model[0]

fi = [(i,f) for i, f in zip(model.feature_name(), model.feature_importance())]
fi = sorted(fi, key = lambda x: x[1], reverse=True)
cutoff_trh = np.percentile([i[1] for i in fi], 10)
print(cutoff_trh)
important_features = [i[0] for i in fi if i[1] > cutoff_trh]
for i,z in fi: print((i,z))


# In[ ]:


individual_weeks_fi = fi.copy()


# In[ ]:


fig, ax = plt.subplots(1,1,figsize=(15, 10))
lgb.plot_importance(model, max_num_features = 20, ax=ax)


# ## Summed weeks

# In[ ]:


original_train_data_sum = train_data_sum.copy()


# In[ ]:


train_data_sum.area_code.nunique()


# In[ ]:


train_data_sum = original_train_data_sum

smoking_related = [i for i in train_data_sum.columns if 'smoking' in i]
# train_data_sum[smoking_related] -= train_data_sum[smoking_related].min()
# train_data_sum[smoking_related] /= train_data_sum[smoking_related].max()
# train_data_sum.loc[:, 'smoking_related'] = train_data_sum[smoking_related].sum(axis=1)
# train_data_sum.drop(smoking_related, axis=1, inplace=True)


# In[ ]:


# train_data_sum.drop('smoking_status_at_time_of_delivery',axis=1,inplace=True)
[i for i in train_data_sum.columns if 'smoking' in i]


# In[ ]:


# from itertools import combinations

# for f1,f2 in combinations(smoking_related, 2):
#     print(f1,f2,stats.pearsonr(train_data_sum[f1],train_data_sum[f2]))
#     plt.scatter(train_data_sum[f1],train_data_sum[f2])
#     plt.show()


# In[ ]:


[i for i in train_data_sum.columns if 'smoking' in i]


# In[ ]:


train_features = prep_features_list(train_data_sum, exc_col = ['area_code', 'geography_type', 'area_name', 'ratio'], cor_trh=0.95)
validation_metrics = [metrics.r2_score, partial(adj_r2_score, n_predictors=len(train_features)), lambda x,y: stats.pearsonr(x,y)[0]]

params = {'n_estimators':6000,
                    'boosting_type': 'gbdt',
                    'objective': 'regression',
                    'metric': 'rmse',
                    'subsample': 0.7,
                    'subsample_freq': 1,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.7,
                    'max_depth': 10, # was 15
                  'num_leaves': 5,
                    'lambda_l1': 0.1,  
                    'lambda_l2': 0.1,
                    'early_stopping_rounds': 100,
                  'min_data_in_leaf': 1,
                          'min_gain_to_split': 0.01,
                          'max_bin': 100
                    }


lgb_model = Lgb_Model(train_df=train_data_sum, test_df=None, features=train_features, categoricals=[], n_splits=5, verbose=True, target='ratio', val_metric=validation_metrics, params=params, cb=uk_hp_model_cb)


# In[ ]:


lgb_model()


# In[ ]:


'FINAL SCORE:', lgb_model.score


# In[ ]:


fig, axes = create_axes_grid(1,1,10,5)
axes.set_title('Truth vs preds')
axes.set_ylabel('True ratio')
axes.set_xlabel('Predicted ratio')
axes.scatter(lgb_model.val_preds, lgb_model.val_ys, color='blue')


# In[ ]:


model = lgb_model.model[0]

fi = [(i,f) for i, f in zip(model.feature_name(), model.feature_importance())]
fi = sorted(fi, key = lambda x: x[1], reverse=True)
cutoff_trh = np.percentile([i[1] for i in fi], 10)
print(cutoff_trh)
important_features = [i[0] for i in fi if i[1] > cutoff_trh]
for i,z in fi: print((i,z))


# In[ ]:


summed_weeks_fi = fi.copy()


# In[ ]:


fig, ax = plt.subplots(1,1,figsize=(15, 10))
lgb.plot_importance(model, max_num_features = 20, ax=ax)


# ## Feature importance across 2 models

# In[ ]:





# In[ ]:


fi_both = pd.DataFrame({'Feature': [i for i,z in summed_weeks_fi]})
fi_both.loc[:,'Rank_summed'] = [i for i in range(1, fi_both.shape[0]+1)]
fi_both.loc[:,'Rank_weekly'] = fi_both.loc[:,'Feature'].map({i:r for r, (i,z) in enumerate(individual_weeks_fi, 1)})

fi_both.loc[:, 'Lowest_rank'] = fi_both.apply(lambda x: max([x['Rank_weekly'], x['Rank_summed']]), axis=1)
fi_both.loc[:, 'Diff_in_rank'] = np.abs(fi_both['Rank_weekly'] - fi_both['Rank_summed'])
fi_both = fi_both.sort_values('Lowest_rank')


# In[ ]:


fi_both


# In[ ]:


import re
def title_prettify(x):
    x = re.sub('_+', ' ', x)
    x = x[0].upper() + x[1:]
    return  x

title_prettify('under_18s_conception_rate___1_000')


# 

# In[ ]:


specific_week = None
train_data = train_data_sum

num_features = 6
fig, axes = create_axes_grid(num_features//2,2,10,5)
col = 0

do_log = ['Density','Births']

for row, f_idx in zip(np.repeat(np.arange(num_features/2, dtype='int'), 2), range(num_features)):
#     f = fi_both.iloc[f_idx]['Feature']
    f = summed_weeks_fi[f_idx][0]
#     axes[row, col].set_title(f'{f} (rank: {f_idx+1})')
    title = title_prettify(f)
    axes[row, col].set_title(title)
    axes[row, col].set_xlabel(title)
    axes[row, col].set_ylabel('Covid / all deaths ratio')
    
    if title not in do_log:
        if specific_week:
            axes[row, col].scatter(train_data.loc[train_data.week_number==specific_week, f], train_data.loc[train_data.week_number==specific_week, 'ratio'], color='blue')
        else:
            axes[row, col].scatter(train_data[f], train_data.ratio, color='blue')
    else:
        title = f"{title} (log scale)"
        axes[row, col].set_title(title)
        axes[row, col].set_xlabel(title)
        if specific_week:
            axes[row, col].scatter(np.log(train_data.loc[train_data.week_number==specific_week, f]), train_data.loc[train_data.week_number==specific_week, 'ratio'], color='blue')
        else:
            axes[row, col].scatter(np.log(train_data[f]), train_data.ratio, color='blue')
    col = 1 if col == 0 else 0


# In[ ]:





# In[ ]:





# In[ ]:


import shap


# In[ ]:


get_ipython().run_line_magic('time', '')
x_val = lgb_model.all_data[0]['x_val']

shap_values = shap.TreeExplainer(lgb_model.model[0]).shap_values(x_val)


# In[ ]:


shap.summary_plot(shap_values, x_val)


# In[ ]:


for f in fi_both.iloc[0:6]['Feature']:
    shap.dependence_plot(f, shap_values, x_val)


# In[ ]:





# In[ ]:


# top10_f = hp_and_d.sort_values('tb_incidence__three_year_average_',ascending=False)[['area_code','ratio','tb_incidence__three_year_average_']].iloc[0:10]
# top10_f = top10_f.merge(hp_original[['Area Code', 'Area Name', 'Area Type']].drop_duplicates(),  left_on='area_code', right_on='Area Code', how='left')
# top10_f


# In[ ]:





# # Prediction errors per region

# In[ ]:


errors = pd.DataFrame({'area_code':lgb_model.area_codes,
                      'prediction':lgb_model.val_preds,
                      'truth':lgb_model.val_ys})

errors.loc[:,'error'] = abs(errors.prediction - errors.truth)

errors = errors.merge(hp_original[['Area Code', 'Area Name', 'Area Type']].drop_duplicates(),  left_on='area_code', right_on='Area Code', how='left')

errors = errors.groupby('Area Name').mean()

errors = errors.sort_values('error',ascending=False)


# In[ ]:


errors.head(10)


# In[ ]:


errors.tail(10)


# In[ ]:





# In[ ]:





# In[ ]:




