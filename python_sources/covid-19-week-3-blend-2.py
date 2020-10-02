#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, gc, pickle, copy, datetime, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn import metrics
pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')


# In[ ]:


df_train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")
print(df_train.shape)
df_train.head()


# In[ ]:


df_test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")
print(df_test.shape)
df_test.head()


# In[ ]:


# concat train and test
df_traintest = pd.concat([df_train, df_test])
print(df_train.shape, df_test.shape, df_traintest.shape)


# In[ ]:


# process date
df_traintest['Date'] = pd.to_datetime(df_traintest['Date'])
df_traintest['day'] = df_traintest['Date'].apply(lambda x: x.dayofyear).astype(np.int16)
df_traintest.head()


# In[ ]:


day_before_valid = 71+7 # 3-18 day  before of validation
day_before_public = 78+7 # 3-25 last day of train
day_before_private = df_traintest['day'][pd.isna(df_traintest['ForecastId'])].max() # last day of train
print(df_traintest['Date'][df_traintest['day']==day_before_valid].values[0])
print(df_traintest['Date'][df_traintest['day']==day_before_public].values[0])
print(df_traintest['Date'][df_traintest['day']==day_before_private].values[0])


# In[ ]:


# concat Country/Region and Province/State
def func(x):
    try:
        x_new = x['Country_Region'] + "/" + x['Province_State']
    except:
        x_new = x['Country_Region']
    return x_new
        
df_traintest['place_id'] = df_traintest.apply(lambda x: func(x), axis=1)
df_traintest.head()


# In[ ]:


df_traintest[(df_traintest['day']>=day_before_public-3) & (df_traintest['place_id']=='China/Hubei')].head()


# In[ ]:


# concat lat and long
df_latlong = pd.read_csv("../input/smokingstats/df_Latlong.csv")
df_latlong.head()


# In[ ]:


# concat Country/Region and Province/State
def func(x):
    try:
        x_new = x['Country/Region'] + "/" + x['Province/State']
    except:
        x_new = x['Country/Region']
    return x_new
        
df_latlong['place_id'] = df_latlong.apply(lambda x: func(x), axis=1)
df_latlong = df_latlong[df_latlong['place_id'].duplicated()==False]
df_latlong.head()


# In[ ]:


df_traintest = pd.merge(df_traintest, df_latlong[['place_id', 'Lat', 'Long']], on='place_id', how='left')
df_traintest.head()


# In[ ]:


print(pd.isna(df_traintest['Lat']).sum()) # count Nan
df_traintest[pd.isna(df_traintest['Lat'])].head()


# In[ ]:


# get place list
places = np.sort(df_traintest['place_id'].unique())
print(len(places))


# In[ ]:


# calc cases, fatalities per day
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
print(df_traintest2.shape)
df_traintest2[df_traintest2['place_id']=='China/Hubei'].head()


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


df_traintest3 = []
for place in places[:]:
    df_tmp = df_traintest2[df_traintest2['place_id']==place].reset_index(drop=True)
    df_tmp = do_aggregations(df_tmp)
    df_traintest3.append(df_tmp)
df_traintest3 = pd.concat(df_traintest3).reset_index(drop=True)
df_traintest3[df_traintest3['place_id']=='China/Hubei'].head()


# In[ ]:


# add Smoking rate per country
# data of smoking rate is obtained from https://ourworldindata.org/smoking
df_smoking = pd.read_csv("../input/smokingstats/share-of-adults-who-smoke.csv")
print(np.sort(df_smoking['Entity'].unique())[:10])
df_smoking.head()


# In[ ]:


# extract newest data
df_smoking_recent = df_smoking.sort_values('Year', ascending=False).reset_index(drop=True)
df_smoking_recent = df_smoking_recent[df_smoking_recent['Entity'].duplicated()==False]
df_smoking_recent['Country_Region'] = df_smoking_recent['Entity']
df_smoking_recent['SmokingRate'] = df_smoking_recent['Smoking prevalence, total (ages 15+) (% of adults)']
df_smoking_recent.head()


# In[ ]:


# merge
df_traintest4 = pd.merge(df_traintest3, df_smoking_recent[['Country_Region', 'SmokingRate']], on='Country_Region', how='left')
print(df_traintest4.shape)
df_traintest4.head()


# In[ ]:


# fill na with world smoking rate
SmokingRate = df_smoking_recent['SmokingRate'][df_smoking_recent['Entity']=='World'].values[0]
print("Smoking rate of the world: {:.6f}".format(SmokingRate))
df_traintest4['SmokingRate'][pd.isna(df_traintest4['SmokingRate'])] = SmokingRate
df_traintest4.head()


# In[ ]:


# add data from World Economic Outlook Database
# https://www.imf.org/external/pubs/ft/weo/2017/01/weodata/index.aspx
df_weo = pd.read_csv("../input/smokingstats/WEO.csv")
df_weo.head()


# In[ ]:


print(df_weo['Subject Descriptor'].unique())


# In[ ]:


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
df_weo_agg.head()


# In[ ]:


# merge
df_traintest5 = pd.merge(df_traintest4, df_weo_agg, on='Country_Region', how='left')
print(df_traintest5.shape)
df_traintest5.head()


# In[ ]:


# add Life expectancy
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
df_life.head()


# In[ ]:


df_life = df_life[['Country', '2018']]
df_life.columns = ['Country_Region', 'LifeExpectancy']


# In[ ]:


# merge
df_traintest6 = pd.merge(df_traintest5, df_life, on='Country_Region', how='left')
print(len(df_traintest6))
df_traintest6.head()


# In[ ]:


# add additional info from countryinfo dataset
df_country = pd.read_csv("../input/countryinfo/covid19countryinfo.csv")
df_country.head()


# In[ ]:


df_country['Country_Region'] = df_country['country']
df_country = df_country[df_country['country'].duplicated()==False]


# In[ ]:


df_country[df_country['country'].duplicated()]


# In[ ]:


df_traintest7 = pd.merge(df_traintest6, 
                         df_country.drop(['tests', 'testpop', 'country'], axis=1), 
                         on=['Country_Region',], how='left')
print(df_traintest7.shape)
df_traintest7.head()


# In[ ]:


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

df_traintest7['id'] = np.arange(len(df_traintest7))
df_le = get_df_le(df_traintest7, 'id', ['Country_Region', 'Province_State'])
df_traintest8 = pd.merge(df_traintest7, df_le, on='id', how='left')


# In[ ]:


df_traintest8['cases/day'] = df_traintest8['cases/day'].astype(np.float)
df_traintest8['fatal/day'] = df_traintest8['fatal/day'].astype(np.float)


# In[ ]:


# covert object type to float
def func(x):
    x_new = 0
    try:
        x_new = float(x.replace(",", ""))
    except:
#         print(x)
        x_new = np.nan
    return x_new
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
for col in cols:
    df_traintest8[col] = df_traintest8[col].apply(lambda x: func(x))  
print(df_traintest8['pop'].dtype)


# In[ ]:


df_traintest8[df_traintest8['place_id']=='China/Hubei'].head()


# In[ ]:


day_before_valid = 71+7 # 3-18 day  before of validation
day_before_public = 78+7 # 3-25 last day of train
day_before_launch = 85+7 # 4-8 last day before launch


# In[ ]:


df_traintest8['cases/day_(1-1)']/df_traintest8['cases/day_(1-7)']


# In[ ]:


df_traintest8['cases/day_1/7_ratio'] = df_traintest8['cases/day_(1-1)']/df_traintest8['cases/day_(1-7)']


# In[ ]:


df_traintest8['fatal/day_1/7_ratio'] = df_traintest8['fatal/day_(1-1)']/df_traintest8['fatal/day_(1-7)']


# In[ ]:


df_traintest8['fatal/cases_1/1_ratio'] = df_traintest8['fatal/day_(1-1)']/df_traintest8['cases/day_(1-1)']


# In[ ]:


df_traintest8['fatal/cases_7/7_ratio'] = df_traintest8['fatal/day_(1-7)']/df_traintest8['cases/day_(1-7)']


# In[ ]:


df_traintest8.head()


# In[ ]:





# In[ ]:


def calc_score(y_true, y_pred):
    y_true[y_true<0] = 0
    score = metrics.mean_squared_error(np.log(y_true.clip(0, 1e10)+1), np.log(y_pred[:]+1))**0.5
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
    'cases/day_1/7_ratio',
    'fatal/day_1/7_ratio',
    'fatal/cases_1/1_ratio',
    'fatal/cases_7/7_ratio',
    
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
df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<=day_before_valid)]
df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (day_before_valid<df_traintest8['day']) & (df_traintest8['day']<=day_before_public)]
df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]
X_train = df_train[col_var]
X_valid = df_valid[col_var]
y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)
y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)
valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)
num_round = 15000
model = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],
                  verbose_eval=100,
                  early_stopping_rounds=150,)

best_itr = model.best_iteration


# In[ ]:


y_true = df_valid['fatal/day'].values
y_pred = np.exp(model.predict(X_valid))-1
score = calc_score(y_true, y_pred)
print("{:.6f}".format(score))


# In[ ]:


# display feature importance
tmp = pd.DataFrame()
tmp["feature"] = col_var
tmp["importance"] = model.feature_importance()
tmp = tmp.sort_values('importance', ascending=False)
tmp


# In[ ]:


# train with all data before public
df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<=day_before_public)]
df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<=day_before_public)]
df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]
X_train = df_train[col_var]
X_valid = df_valid[col_var]
y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)
y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)
valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)
model = lgb.train(params, train_data, best_itr, valid_sets=[train_data, valid_data],
                  verbose_eval=100,
                  early_stopping_rounds=150,)


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
    'cases/day_1/7_ratio',
    'fatal/day_1/7_ratio',
    'fatal/cases_1/1_ratio',
    'fatal/cases_7/7_ratio',
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
col_cat = []
df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<=day_before_valid)]
df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (day_before_valid<df_traintest8['day']) & (df_traintest8['day']<=day_before_public)]
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


y_true = df_valid['cases/day'].values
y_pred = np.exp(model2.predict(X_valid))-1
score = calc_score(y_true, y_pred)
print("{:.6f}".format(score))


# In[ ]:


# display feature importance
tmp = pd.DataFrame()
tmp["feature"] = col_var2
tmp["importance"] = model2.feature_importance()
tmp = tmp.sort_values('importance', ascending=False)
tmp


# In[ ]:


df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<=day_before_public)]
df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<=day_before_public)]
X_train = df_train[col_var2]
X_valid = df_valid[col_var2]
y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)
y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)
valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)
model2 = lgb.train(params, train_data, best_itr2, valid_sets=[train_data, valid_data],
                  verbose_eval=100,
                  early_stopping_rounds=150,)


# In[ ]:


# train model to predict fatalities/day
df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<=day_before_public)]
df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (day_before_public<df_traintest8['day'])]
df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]
X_train = df_train[col_var]
X_valid = df_valid[col_var]
y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)
y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)
valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)
num_round = 15000
model = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],
                  verbose_eval=100,
                  early_stopping_rounds=150,)

best_itr = model.best_iteration


# In[ ]:


# train with all data
df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId']))]
df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId']))]
X_train = df_train[col_var]
X_valid = df_valid[col_var]
y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)
y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)
valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)
model_pri = lgb.train(params, train_data, best_itr, valid_sets=[train_data, valid_data],
                  verbose_eval=100,
                  early_stopping_rounds=150,)


# In[ ]:


# train model to predict cases/day
df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<=day_before_public)]
df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (day_before_public<df_traintest8['day'])]
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


# train with all data
df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId']))]
df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId']))]
X_train = df_train[col_var2]
X_valid = df_valid[col_var2]
y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)
y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)
valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)
model2_pri = lgb.train(params, train_data, best_itr2, valid_sets=[train_data, valid_data],
                  verbose_eval=100,
                  early_stopping_rounds=150,)


# In[ ]:


# remove overlap for public LB prediction
df_tmp = df_traintest8[
    ((df_traintest8['day']<=day_before_public)  & (pd.isna(df_traintest8['ForecastId'])))
    | ((day_before_public<df_traintest8['day']) & (pd.isna(df_traintest8['ForecastId'])==False))].reset_index(drop=True)
df_tmp = df_tmp.drop([
    'cases/day_(1-1)', 'cases/day_(1-7)', 'cases/day_(8-14)', 'cases/day_(15-21)', 
    'fatal/day_(1-1)', 'fatal/day_(1-7)', 'fatal/day_(8-14)', 'fatal/day_(15-21)', 'cases/day_1/7_ratio', 'fatal/cases_1/1_ratio', 'fatal/cases_7/7_ratio', 'fatal/day_1/7_ratio',
    'days_since_1cases', 'days_since_10cases', 'days_since_100cases',
    'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',
                               ],  axis=1)
df_traintest9 = []
for i, place in enumerate(places[:]):
    df_tmp2 = df_tmp[df_tmp['place_id']==place].reset_index(drop=True)
    df_tmp2 = do_aggregations(df_tmp2)
    df_tmp2['cases/day_1/7_ratio'] = df_tmp2['cases/day_(1-1)']/df_tmp2['cases/day_(1-7)']
    df_tmp2['fatal/day_1/7_ratio'] = df_tmp2['fatal/day_(1-1)']/df_tmp2['fatal/day_(1-7)']
    df_tmp2['fatal/cases_1/1_ratio'] = df_tmp2['fatal/day_(1-1)']/df_tmp2['cases/day_(1-1)']
    df_tmp2['fatal/cases_7/7_ratio'] = df_tmp2['fatal/day_(1-7)']/df_tmp2['cases/day_(1-7)']
    df_traintest9.append(df_tmp2)
df_traintest9 = pd.concat(df_traintest9).reset_index(drop=True)
df_traintest9[df_traintest9['day']>day_before_public-2].head()


# In[ ]:


# remove overlap for private LB prediction
df_tmp = df_traintest8[
    ((df_traintest8['day']<=day_before_private)  & (pd.isna(df_traintest8['ForecastId'])))
    | ((day_before_private<df_traintest8['day']) & (pd.isna(df_traintest8['ForecastId'])==False))].reset_index(drop=True)
df_tmp = df_tmp.drop([
    'cases/day_(1-1)', 'cases/day_(1-7)', 'cases/day_(8-14)', 'cases/day_(15-21)', 
    'fatal/day_(1-1)', 'fatal/day_(1-7)', 'fatal/day_(8-14)', 'fatal/day_(15-21)', 'cases/day_1/7_ratio', 'fatal/cases_1/1_ratio', 'fatal/cases_7/7_ratio', 'fatal/day_1/7_ratio',
    'days_since_1cases', 'days_since_10cases', 'days_since_100cases',
    'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',
                               ],  axis=1)
df_traintest10 = []
for i, place in enumerate(places[:]):
    df_tmp2 = df_tmp[df_tmp['place_id']==place].reset_index(drop=True)
    df_tmp2 = do_aggregations(df_tmp2)
    df_tmp2['cases/day_1/7_ratio'] = df_tmp2['cases/day_(1-1)']/df_tmp2['cases/day_(1-7)']
    df_tmp2['fatal/day_1/7_ratio'] = df_tmp2['fatal/day_(1-1)']/df_tmp2['fatal/day_(1-7)']
    df_tmp2['fatal/cases_1/1_ratio'] = df_tmp2['fatal/day_(1-1)']/df_tmp2['cases/day_(1-1)']
    df_tmp2['fatal/cases_7/7_ratio'] = df_tmp2['fatal/day_(1-7)']/df_tmp2['cases/day_(1-7)']
    df_traintest10.append(df_tmp2)
df_traintest10 = pd.concat(df_traintest10).reset_index(drop=True)
df_traintest10[df_traintest10['day']>day_before_private-2].head()


# In[ ]:


1


# In[ ]:


# predict test data in public
# predict the cases and fatatilites one day at a time and use the predicts as next day's feature recursively.
df_preds = []
for i, place in enumerate(places[:]):
    df_interest = copy.deepcopy(df_traintest9[df_traintest9['place_id']==place].reset_index(drop=True))
    df_interest['cases/day'][(pd.isna(df_interest['ForecastId']))==False] = -1
    df_interest['fatal/day'][(pd.isna(df_interest['ForecastId']))==False] = -1
    len_known = (df_interest['day']<=day_before_public).sum()
    len_unknown = (day_before_public<df_interest['day']).sum()
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
            'fatal/day_(1-1)', 'fatal/day_(1-7)', 'fatal/day_(8-14)', 'fatal/day_(15-21)', 'cases/day_1/7_ratio', 'fatal/cases_1/1_ratio', 'fatal/cases_7/7_ratio', 'fatal/day_1/7_ratio',
            'days_since_1cases', 'days_since_10cases', 'days_since_100cases',
            'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',

                                       ],  axis=1)
        df_interest = do_aggregations(df_interest)
        df_interest['cases/day_1/7_ratio'] = df_interest['cases/day_(1-1)']/df_interest['cases/day_(1-7)']
        df_interest['fatal/day_1/7_ratio'] = df_interest['fatal/day_(1-1)']/df_interest['fatal/day_(1-7)']
        df_interest['fatal/cases_1/1_ratio'] = df_interest['fatal/day_(1-1)']/df_interest['cases/day_(1-1)']
        df_interest['fatal/cases_7/7_ratio'] = df_interest['fatal/day_(1-7)']/df_interest['cases/day_(1-7)']
    if (i+1)%10==0:
        print("{:3d}/{}  {}, len known: {}, len unknown: {}".format(i+1, len(places), place, len_known, len_unknown), df_interest.shape)
    df_interest['fatal_pred'] = np.cumsum(df_interest['fatal/day'].values)
    df_interest['cases_pred'] = np.cumsum(df_interest['cases/day'].values)
    df_preds.append(df_interest)
df_preds = pd.concat(df_preds)


# In[ ]:


# predict test data in public
df_preds_pri = []
for i, place in enumerate(places[:]):
    df_interest = copy.deepcopy(df_traintest10[df_traintest10['place_id']==place].reset_index(drop=True))
    df_interest['cases/day'][(pd.isna(df_interest['ForecastId']))==False] = -1
    df_interest['fatal/day'][(pd.isna(df_interest['ForecastId']))==False] = -1
    len_known = (df_interest['day']<=day_before_private).sum()
    len_unknown = (day_before_private<df_interest['day']).sum()
    for j in range(len_unknown): # use predicted cases and fatal for next days' prediction
        X_valid = df_interest[col_var].iloc[j+len_known]
        X_valid2 = df_interest[col_var2].iloc[j+len_known]
        pred_f = model_pri.predict(X_valid)
        pred_c = model2_pri.predict(X_valid2)
        pred_c = (np.exp(pred_c)-1).clip(0, 1e10)
        pred_f = (np.exp(pred_f)-1).clip(0, 1e10)
        df_interest['fatal/day'][j+len_known] = pred_f
        df_interest['cases/day'][j+len_known] = pred_c
        df_interest['Fatalities'][j+len_known] = df_interest['Fatalities'][j+len_known-1] + pred_f
        df_interest['ConfirmedCases'][j+len_known] = df_interest['ConfirmedCases'][j+len_known-1] + pred_c
#         print(df_interest['ConfirmedCases'][j+len_known-1], df_interest['ConfirmedCases'][j+len_known], pred_c)
        df_interest = df_interest.drop([
            'cases/day_(1-1)', 'cases/day_(1-7)', 'cases/day_(8-14)', 'cases/day_(15-21)', 
            'fatal/day_(1-1)', 'fatal/day_(1-7)', 'fatal/day_(8-14)', 'fatal/day_(15-21)', 'cases/day_1/7_ratio', 'fatal/cases_1/1_ratio', 'fatal/cases_7/7_ratio', 'fatal/day_1/7_ratio',
            'days_since_1cases', 'days_since_10cases', 'days_since_100cases',
            'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',

                                       ],  axis=1)
        df_interest = do_aggregations(df_interest)
        df_interest['cases/day_1/7_ratio'] = df_interest['cases/day_(1-1)']/df_interest['cases/day_(1-7)']
        df_interest['fatal/day_1/7_ratio'] = df_interest['fatal/day_(1-1)']/df_interest['fatal/day_(1-7)']
        df_interest['fatal/cases_1/1_ratio'] = df_interest['fatal/day_(1-1)']/df_interest['cases/day_(1-1)']
        df_interest['fatal/cases_7/7_ratio'] = df_interest['fatal/day_(1-7)']/df_interest['cases/day_(1-7)']
    if (i+1)%10==0:
        print("{:3d}/{}  {}, len known: {}, len unknown: {}".format(i+1, len(places), place, len_known, len_unknown), df_interest.shape)
    df_interest['fatal_pred'] = np.cumsum(df_interest['fatal/day'].values)
    df_interest['cases_pred'] = np.cumsum(df_interest['cases/day'].values)
    df_preds_pri.append(df_interest)
df_preds_pri = pd.concat(df_preds_pri)


# In[ ]:


places_sort = df_traintest10[['place_id', 'ConfirmedCases']][df_traintest10['day']==day_before_private]
places_sort = places_sort.sort_values('ConfirmedCases', ascending=False).reset_index(drop=True)['place_id'].values
print(len(places_sort))
places_sort[:5]


# In[ ]:


print("Fatalities / Public")
plt.figure(figsize=(30,30))
for i in range(30):
    plt.subplot(5,6,i+1)
    idx = i * 10
    df_interest = df_preds[df_preds['place_id']==places_sort[idx]].reset_index(drop=True)
    tmp = df_interest['fatal/day'].values
    tmp = np.cumsum(tmp)
    sns.lineplot(x=df_interest['day'], y=tmp, label='pred')
    df_interest2 = df_traintest10[(df_traintest10['place_id']==places_sort[idx]) & (df_traintest10['day']<=day_before_private)].reset_index(drop=True)
    sns.lineplot(x=df_interest2['day'].values, y=df_interest2['Fatalities'].values, label='true')
    plt.title(places_sort[idx])
plt.show()


# In[ ]:


print("Confirmed Cases / Public")
plt.figure(figsize=(30,30))
for i in range(30):
    plt.subplot(5,6,i+1)
    idx = i * 10
    df_interest = df_preds[df_preds['place_id']==places_sort[idx]].reset_index(drop=True)
    tmp = df_interest['cases/day'].values
    tmp = np.cumsum(tmp)
    sns.lineplot(x=df_interest['day'], y=tmp, label='pred')
    df_interest2 = df_traintest10[(df_traintest10['place_id']==places_sort[idx]) & (df_traintest10['day']<=day_before_private)].reset_index(drop=True)
    sns.lineplot(x=df_interest2['day'].values, y=df_interest2['ConfirmedCases'].values, label='true')
    plt.title(places_sort[idx])
plt.show()


# In[ ]:


print("Fatalities / Private")
plt.figure(figsize=(30,30))
for i in range(30):
    plt.subplot(5,6,i+1)
    idx = i * 10
    df_interest = df_preds_pri[df_preds_pri['place_id']==places_sort[idx]].reset_index(drop=True)
    tmp = df_interest['fatal/day'].values
    tmp = np.cumsum(tmp)
    sns.lineplot(x=df_interest['day'], y=tmp, label='pred')
    df_interest2 = df_traintest10[(df_traintest10['place_id']==places_sort[idx]) & (df_traintest10['day']<=day_before_private)].reset_index(drop=True)
    sns.lineplot(x=df_interest2['day'].values, y=df_interest2['Fatalities'].values, label='true')
    plt.title(places_sort[idx])
plt.show()


# In[ ]:


print("ConfirmedCases / Private")
plt.figure(figsize=(30,30))
for i in range(30):
    plt.subplot(5,6,i+1)
    idx = i * 10
    df_interest = df_preds_pri[df_preds_pri['place_id']==places_sort[idx]].reset_index(drop=True)
    tmp = df_interest['cases/day'].values
    tmp = np.cumsum(tmp)
    sns.lineplot(x=df_interest['day'], y=tmp, label='pred')
    df_interest2 = df_traintest10[(df_traintest10['place_id']==places_sort[idx]) & (df_traintest10['day']<=day_before_private)].reset_index(drop=True)
    sns.lineplot(x=df_interest2['day'].values, y=df_interest2['ConfirmedCases'].values, label='true')
    plt.title(places_sort[idx])
plt.show()


# In[ ]:


# merge 2 preds
df_preds[df_preds['day']>day_before_private] = df_preds_pri[df_preds['day']>day_before_private]


# In[ ]:


df_preds.to_csv("df_preds.csv", index=None)


# In[ ]:


# load sample submission
df_sub = pd.read_csv("../input/covid19-global-forecasting-week-3/submission.csv")
print(len(df_sub))
df_sub.head()


# In[ ]:


# merge prediction with sub
df_sub = pd.merge(df_sub, df_traintest3[['ForecastId', 'place_id', 'day']])
df_sub = pd.merge(df_sub, df_preds[['place_id', 'day', 'cases_pred', 'fatal_pred']], on=['place_id', 'day',], how='left')
df_sub.head(10)


# In[ ]:


# save
df_sub['ConfirmedCases'] = df_sub['cases_pred']
df_sub['Fatalities'] = df_sub['fatal_pred']
df_sub = df_sub[['ForecastId', 'ConfirmedCases', 'Fatalities']]
df_sub.to_csv("submission_0.csv", index=None)
df_sub.head(10)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.set_option('display.max_columns', 99)
pd.set_option('display.max_rows', 99)
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import datetime as dt


# In[ ]:


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
import seaborn as sns
sns.set_palette(sns.color_palette('tab20', 20))

import plotly.express as px
import plotly.graph_objects as go


# In[ ]:


COMP = '../input/covid19-global-forecasting-week-3'
DATEFORMAT = '%Y-%m-%d'


def get_comp_data(COMP):
    train = pd.read_csv(f'{COMP}/train.csv')
    test = pd.read_csv(f'{COMP}/test.csv')
    submission = pd.read_csv(f'{COMP}/submission.csv')
    print(train.shape, test.shape, submission.shape)
    train['Country_Region'] = train['Country_Region'].str.replace(',', '')
    test['Country_Region'] = test['Country_Region'].str.replace(',', '')

    train['Location'] = train['Country_Region'] + '-' + train['Province_State'].fillna('')

    test['Location'] = test['Country_Region'] + '-' + test['Province_State'].fillna('')

    train['LogConfirmed'] = to_log(train.ConfirmedCases)
    train['LogFatalities'] = to_log(train.Fatalities)
    train = train.drop(columns=['Province_State'])
    test = test.drop(columns=['Province_State'])

    country_codes = pd.read_csv('../input/covid19-metadata/country_codes.csv', keep_default_na=False)
    train = train.merge(country_codes, on='Country_Region', how='left')
    test = test.merge(country_codes, on='Country_Region', how='left')

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


# In[ ]:


start = dt.datetime.now()
train, test, submission = get_comp_data(COMP)
train.shape, test.shape, submission.shape
train.head(2)
test.head(2)


# In[ ]:


train[train.geo_region.isna()].Country_Region.unique()
train = train.fillna('#N/A')
test = test.fillna('#N/A')

train[train.duplicated(['Date', 'Location'])]
train.count()


# In[ ]:


train.describe()
train.nunique()
train.dtypes
train.count()

TRAIN_START = train.Date.min()
TEST_START = test.Date.min()
TRAIN_END = train.Date.max()
TEST_END = test.Date.max()
TRAIN_START, TRAIN_END, TEST_START, TEST_END


# In[ ]:


train = train.sort_values(by='Date')
countries_latest_state = train[train['Date'] == TRAIN_END].groupby([
    'Country_Region', 'continent', 'geo_region', 'country_iso_code_3']).sum()[[
    'ConfirmedCases', 'Fatalities']].reset_index()
countries_latest_state['Log10Confirmed'] = np.log10(countries_latest_state.ConfirmedCases + 1)
countries_latest_state['Log10Fatalities'] = np.log10(countries_latest_state.Fatalities + 1)
countries_latest_state = countries_latest_state.sort_values(by='Fatalities', ascending=False)
countries_latest_state.to_csv('countries_latest_state.csv', index=False)

countries_latest_state.shape
countries_latest_state.head()


# In[ ]:


# The source dataset is not necessary cumulative we will force it
latest_loc = train[train['Date'] == TRAIN_END][['Location', 'ConfirmedCases', 'Fatalities']]
max_loc = train.groupby(['Location'])[['ConfirmedCases', 'Fatalities']].max().reset_index()
check = pd.merge(latest_loc, max_loc, on='Location')
np.mean(check.ConfirmedCases_x == check.ConfirmedCases_y)
np.mean(check.Fatalities_x == check.Fatalities_y)
check[check.Fatalities_x != check.Fatalities_y]
check[check.ConfirmedCases_x != check.ConfirmedCases_y]


# In[ ]:


train_clean = process_each_location(train)

train_clean.shape
train_clean.tail()


# In[ ]:


regional_progress = train_clean.groupby(['DateTime', 'continent']).sum()[['ConfirmedCases', 'Fatalities']].reset_index()
regional_progress['Log10Confirmed'] = np.log10(regional_progress.ConfirmedCases + 1)
regional_progress['Log10Fatalities'] = np.log10(regional_progress.Fatalities + 1)
regional_progress = regional_progress[regional_progress.continent != '#N/A']


# In[ ]:


china = train_clean[train_clean.Location.str.startswith('China')]
top10_locations = china.groupby('Location')[['ConfirmedCases']].max().sort_values(
    by='ConfirmedCases', ascending=False).reset_index().Location.values[:10]


# In[ ]:


country_progress = train_clean.groupby(['Date', 'DateTime', 'Country_Region']).sum()[[
    'ConfirmedCases', 'Fatalities', 'ConfirmedDelta', 'FatalitiesDelta']].reset_index()
top10_countries = country_progress.groupby('Country_Region')[['Fatalities']].max().sort_values(
    by='Fatalities', ascending=False).reset_index().Country_Region.values[:10]


# In[ ]:


countries_0301 = country_progress[country_progress.Date == '2020-03-01'][[
    'Country_Region', 'ConfirmedCases', 'Fatalities']]
countries_0331 = country_progress[country_progress.Date == '2020-03-31'][[
    'Country_Region', 'ConfirmedCases', 'Fatalities']]
countries_in_march = pd.merge(countries_0301, countries_0331, on='Country_Region', suffixes=['_0301', '_0331'])
countries_in_march['IncreaseInMarch'] = countries_in_march.ConfirmedCases_0331 / (countries_in_march.ConfirmedCases_0301 + 1)
countries_in_march = countries_in_march[countries_in_march.ConfirmedCases_0331 > 200].sort_values(
    by='IncreaseInMarch', ascending=False)
countries_in_march.tail(15)


# In[ ]:


selected_countries = [
    'Italy', 'Vietnam', 'Bahrain', 'Singapore', 'Taiwan*', 'Japan', 'Kuwait', 'Korea, South', 'China']


# In[ ]:


deltas = train_clean[np.logical_and(
        train_clean.LogConfirmed > 0,
        ~train_clean.Location.str.startswith('China')
)].dropna().sort_values(by='LogConfirmedDelta', ascending=False)
deltas = deltas[deltas['Date'] >= '2020-03-12']

confirmed_deltas = pd.concat([
    deltas.groupby('Location')[['LogConfirmedDelta']].mean(),
    deltas.groupby('Location')[['LogConfirmedDelta']].std(),
    deltas.groupby('Location')[['LogConfirmedDelta']].count(),
    deltas.groupby('Location')[['LogConfirmed']].max()
], axis=1)
confirmed_deltas.columns = ['avg', 'std', 'cnt', 'max']

confirmed_deltas.sort_values(by='avg').head(10)
confirmed_deltas.sort_values(by='avg').tail(10)
confirmed_deltas.to_csv('confirmed_deltas.csv')


# In[ ]:


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
confirmed_deltas.loc[confirmed_deltas.Location=='France-', 'DELTA'] = 0.1
confirmed_deltas.loc[confirmed_deltas.Location=='United Kingdom-', 'DELTA'] = 0.12
confirmed_deltas.loc[confirmed_deltas.Location=='Diamond Princess-', 'DELTA'] = 0.00
confirmed_deltas.loc[confirmed_deltas.Location=='China-Hong Kong', 'DELTA'] = 0.08
confirmed_deltas.loc[confirmed_deltas.Location=='San Marino-', 'DELTA'] = 0.03


confirmed_deltas.shape, confirmed_deltas.DELTA.mean()

confirmed_deltas[confirmed_deltas.DELTA != GLOBAL_DELTA].shape, confirmed_deltas[confirmed_deltas.DELTA != GLOBAL_DELTA].DELTA.mean()
confirmed_deltas[confirmed_deltas.DELTA != GLOBAL_DELTA]
confirmed_deltas.describe()


# In[ ]:


daily_log_confirmed = train_clean.pivot('Location', 'Date', 'LogConfirmed').reset_index()
daily_log_confirmed = daily_log_confirmed.sort_values(TRAIN_END, ascending=False)
daily_log_confirmed.to_csv('daily_log_confirmed.csv', index=False)

for i, d in tqdm(enumerate(pd.date_range(add_days(TRAIN_END, 1), add_days(TEST_END, 1)))):
    new_day = str(d).split(' ')[0]
    last_day = dt.datetime.strptime(new_day, DATEFORMAT) - dt.timedelta(days=1)
    last_day = last_day.strftime(DATEFORMAT)
    for loc in confirmed_deltas.Location.values:
        confirmed_delta = confirmed_deltas.loc[confirmed_deltas.Location == loc, 'DELTA'].values[0]
        daily_log_confirmed.loc[daily_log_confirmed.Location == loc, new_day] = daily_log_confirmed.loc[daily_log_confirmed.Location == loc, last_day] +             confirmed_delta * DECAY ** i


# In[ ]:


daily_log_confirmed.head()


# In[ ]:


confirmed_prediciton = pd.melt(daily_log_confirmed[:25], id_vars='Location')
confirmed_prediciton['ConfirmedCases'] = to_exp(confirmed_prediciton['value'])


# In[ ]:


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
death_deltas.loc[death_deltas.Location=='France-', 'DELTA'] = 0.14
death_deltas.loc[death_deltas.Location=='United Kingdom-', 'DELTA'] = 0.14
death_deltas.loc[death_deltas.Location=='Diamond Princess-', 'DELTA'] = 0.00

death_deltas.loc[death_deltas.Location=='China-Hong Kong', 'DELTA'] = 0.01
death_deltas.loc[death_deltas.Location=='San Marino-', 'DELTA'] = 0.05


death_deltas.shape
death_deltas.DELTA.mean()

death_deltas[death_deltas.DELTA != GLOBAL_DELTA].shape
death_deltas[death_deltas.DELTA != GLOBAL_DELTA].DELTA.mean()
death_deltas[death_deltas.DELTA != GLOBAL_DELTA]
death_deltas.describe()


# In[ ]:


daily_log_deaths = train_clean.pivot('Location', 'Date', 'LogFatalities').reset_index()
daily_log_deaths = daily_log_deaths.sort_values(TRAIN_END, ascending=False)
daily_log_deaths.to_csv('daily_log_deaths.csv', index=False)

for i, d in tqdm(enumerate(pd.date_range(add_days(TRAIN_END, 1), add_days(TEST_END, 1)))):
    new_day = str(d).split(' ')[0]
    last_day = dt.datetime.strptime(new_day, DATEFORMAT) - dt.timedelta(days=1)
    last_day = last_day.strftime(DATEFORMAT)
    for loc in death_deltas.Location:
        death_delta = death_deltas.loc[death_deltas.Location == loc, 'DELTA'].values[0]
        daily_log_deaths.loc[daily_log_deaths.Location == loc, new_day] = daily_log_deaths.loc[daily_log_deaths.Location == loc, last_day] +             death_delta * DECAY ** i


# In[ ]:


confirmed_prediciton = pd.melt(daily_log_deaths[:25], id_vars='Location')
confirmed_prediciton['Fatalities'] = to_exp(confirmed_prediciton['value'])


# In[ ]:


confirmed = []
fatalities = []
for id, d, loc in tqdm(test[['ForecastId', 'Date', 'Location']].values):
    c = to_exp(daily_log_confirmed.loc[daily_log_confirmed.Location == loc, d].values[0])
    f = to_exp(daily_log_deaths.loc[daily_log_deaths.Location == loc, d].values[0])
    confirmed.append(c)
    fatalities.append(f)


# In[ ]:


my_submission = test.copy()
my_submission['ConfirmedCases'] = confirmed
my_submission['Fatalities'] = fatalities
my_submission.shape
my_submission.head()


# In[ ]:


my_submission.groupby('Date').sum().tail()


# In[ ]:


total = my_submission.groupby('Date')[['ConfirmedCases', 'Fatalities']].sum().reset_index()

fig2 = px.line(pd.melt(total, id_vars=['Date']), x='Date', y='value', color='variable')
_ = fig2.update_layout(
    yaxis_type="log",
    title_text=f'COVID-19 Cumulative Prediction Total [Updated: {TRAIN_END}]'
)
fig2.show()


# In[ ]:


my_submission[[
    'ForecastId', 'ConfirmedCases', 'Fatalities'
]].to_csv('submission_1.csv', index=False)
print(DECAY)
my_submission.head()
my_submission.tail()
my_submission.shape


# In[ ]:


end = dt.datetime.now()
print('Finished', end, (end - start).seconds, 's')


# In[ ]:


## importing packages
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


## defining constants
PATH_TRAIN = "/kaggle/input/covid19-global-forecasting-week-3/train.csv"
PATH_TEST = "/kaggle/input/covid19-global-forecasting-week-3/test.csv"

PATH_SUBMISSION = "submission.csv"
PATH_OUTPUT = "output.csv"

PATH_REGION_METADATA = "/kaggle/input/covid19-forecasting-metadata/region_metadata.csv"
PATH_REGION_DATE_METADATA = "/kaggle/input/covid19-forecasting-metadata/region_date_metadata.csv"

VAL_DAYS = 7
MAD_FACTOR = 0.5
DAYS_SINCE_CASES = [1, 10, 50, 100, 500, 1000, 5000, 10000]

SEED = 137

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


# In[ ]:


## reading data
train = pd.read_csv(PATH_TRAIN)
test = pd.read_csv(PATH_TEST)

region_metadata = pd.read_csv(PATH_REGION_METADATA)
region_date_metadata = pd.read_csv(PATH_REGION_DATE_METADATA)


# In[ ]:


## preparing data
train = train.merge(test[["ForecastId", "Province_State", "Country_Region", "Date"]], on = ["Province_State", "Country_Region", "Date"], how = "left")
test = test[~test.Date.isin(train.Date.unique())]

df_panel = pd.concat([train, test], sort = False)

# combining state and country into 'geography'
df_panel["geography"] = df_panel.Country_Region.astype(str) + ": " + df_panel.Province_State.astype(str)
df_panel.loc[df_panel.Province_State.isna(), "geography"] = df_panel[df_panel.Province_State.isna()].Country_Region

# fixing data issues with cummax
df_panel.ConfirmedCases = df_panel.groupby("geography")["ConfirmedCases"].cummax()
df_panel.Fatalities = df_panel.groupby("geography")["Fatalities"].cummax()

# merging external metadata
df_panel = df_panel.merge(region_metadata, on = ["Country_Region", "Province_State"])
df_panel = df_panel.merge(region_date_metadata, on = ["Country_Region", "Province_State", "Date"], how = "left")

# label encoding continent
df_panel.continent = LabelEncoder().fit_transform(df_panel.continent)
df_panel.Date = pd.to_datetime(df_panel.Date, format = "%Y-%m-%d")

df_panel.sort_values(["geography", "Date"], inplace = True)


# In[ ]:


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


## function for preparing features
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


# In[ ]:


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


# In[ ]:


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
df_panel = df_panel.merge(pd.concat(df_preds_val, sort = False), on = "Id", how = "left")
df_panel = df_panel.merge(pd.concat(df_preds_test, sort = False), on = "ForecastId", how = "left")

rmsle_cc_lgb = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel.ConfirmedCases_val_lgb.isna()].ConfirmedCases), np.log1p(df_panel[~df_panel.ConfirmedCases_val_lgb.isna()].ConfirmedCases_val_lgb)))
rmsle_ft_lgb = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel.Fatalities_val_lgb.isna()].Fatalities), np.log1p(df_panel[~df_panel.Fatalities_val_lgb.isna()].Fatalities_val_lgb)))

rmsle_cc_mad = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel.ConfirmedCases_val_mad.isna()].ConfirmedCases), np.log1p(df_panel[~df_panel.ConfirmedCases_val_mad.isna()].ConfirmedCases_val_mad)))
rmsle_ft_mad = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel.Fatalities_val_mad.isna()].Fatalities), np.log1p(df_panel[~df_panel.Fatalities_val_mad.isna()].Fatalities_val_mad)))

print("LGB CC RMSLE Val of", VAL_DAYS, "days for CC:", round(rmsle_cc_lgb, 2))
print("LGB FT RMSLE Val of", VAL_DAYS, "days for FT:", round(rmsle_ft_lgb, 2))
print("LGB Overall RMSLE Val of", VAL_DAYS, "days:", round((rmsle_cc_lgb + rmsle_ft_lgb) / 2, 2))
print("\n")
print("MAD CC RMSLE Val of", VAL_DAYS, "days for CC:", round(rmsle_cc_mad, 2))
print("MAD FT RMSLE Val of", VAL_DAYS, "days for FT:", round(rmsle_ft_mad, 2))
print("MAD Overall RMSLE Val of", VAL_DAYS, "days:", round((rmsle_cc_mad + rmsle_ft_mad) / 2, 2))


# In[ ]:


## preparing submission file
df_test = df_panel.loc[~df_panel.ForecastId.isna(), ["ForecastId", "Country_Region", "Province_State", "Date",
                                                     "ConfirmedCases_test", "ConfirmedCases_test_lgb", "ConfirmedCases_test_mad",
                                                     "Fatalities_test", "Fatalities_test_lgb", "Fatalities_test_mad"]].reset_index()

df_test["ConfirmedCases"] = 0.13 * df_test.ConfirmedCases_test_lgb + 0.87 * df_test.ConfirmedCases_test_mad
df_test["Fatalities"] = 0.13 * df_test.Fatalities_test_lgb + 0.87 * df_test.Fatalities_test_mad

# Since LGB models don't predict these geographies well
df_test.loc[df_test.Country_Region.isin(["Diamond Princess", "MS Zaandam"]), "ConfirmedCases"] = df_test[df_test.Country_Region.isin(["Diamond Princess", "MS Zaandam"])].ConfirmedCases_test_mad.values
df_test.loc[df_test.Country_Region.isin(["Diamond Princess", "MS Zaandam"]), "Fatalities"] = df_test[df_test.Country_Region.isin(["Diamond Princess", "MS Zaandam"])].Fatalities_test_mad.values

df_test.loc[df_test.Date.isin(df_train.Date.values), "ConfirmedCases"] = df_test[df_test.Date.isin(df_train.Date.values)].ConfirmedCases_test.values
df_test.loc[df_test.Date.isin(df_train.Date.values), "Fatalities"] = df_test[df_test.Date.isin(df_train.Date.values)].Fatalities_test.values

df_submission = df_test[["ForecastId", "ConfirmedCases", "Fatalities"]]
df_submission.ForecastId = df_submission.ForecastId.astype(int)

df_submission


# In[ ]:


## writing final submission and complete output
df_submission.to_csv('submission_2.csv', index = False)
df_test.to_csv(PATH_OUTPUT, index = False)


# In[ ]:


submission_0 = pd.read_csv('submission_0.csv')
submission_1 = pd.read_csv('submission_1.csv')
submission_2 = pd.read_csv('submission_2.csv')


# In[ ]:


submission_1.tail()


# In[ ]:


submission_2.tail()


# In[ ]:


submission = submission_1.copy()


# In[ ]:


submission['ConfirmedCases'] = 0.8*(0.6*submission_1['ConfirmedCases'].values+0.4*submission_2['ConfirmedCases'].values)+0.2*submission_0['ConfirmedCases'].values
submission['Fatalities'] = 0.8*(0.6*submission_1['Fatalities'].values+0.4*submission_2['Fatalities'].values)+0.2*submission_0['Fatalities'].values
submission.tail()


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)

