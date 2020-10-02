#!/usr/bin/env python
# coding: utf-8

# # Summary
# 
# I will start with linear regression as base model. 
# 
# + I add quantitative features incrementally to see their impact on prediction perf
# + I will also figure out what is the best perf of a linear model
# 
# Then we will switch to ensemble models:
# + random forest
# + tune hyperparams of RF to achieve a better model. 

# We will pick features based on analysis from my [EDA notebook](https://www.kaggle.com/victor191/eda-and-smart-feature-engineer).

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV


# In[ ]:


folder = '/kaggle/input/house-prices-advanced-regression-techniques/'
out_dir = '/kaggle/working'

# local run
# folder = 'data'
# out_dir = 'output/'

train = pd.read_csv(os.path.join(folder, 'train.csv'))
test = pd.read_csv(os.path.join(folder, 'test.csv'))

print(train.shape)
print(test.shape)


# In[ ]:


# concat train and test sets st we always perform transformation on both sets
test['SalePrice'] = 0
data = pd.concat([train, test])

print(data.shape)

# lowercase all column names for convenience
data.columns = [str.lower(cc) for cc in data.columns]

# sale price per square feet is also interested
# data['sale_price_per_sf'] = data['saleprice'] / data['grlivarea']


# There are two approaches:
# + predict directly sale price
# + predict price per SF, then multiply with living area to estimate sale price
# 
# I plan to try both, but first we need some helpers.

# ## Helper methods

# In[ ]:


def cal_age_from_built(row):
    return row['yrsold'] - row['yearbuilt']

def cal_age_from_remodel(row):
    return row['yrsold'] - row['yearremodadd']


def fold_zone_type(ms_zone):
    if ms_zone in ['FV', 'RH', 'C (all)']:
        return 'Other'
    else:
        return ms_zone
#         return {'RL': 'Residential Low Density'.lower() , 
#                 'RM': 'Residential Medium Density'.lower(),
#                 None: 'NA'
#                }[ms_zone]    

def to_adjacency(cond):
    if 'RR' in cond:
        return 'Railroad'
    if 'Pos' in cond:
        return 'Positive feature'
    return {
        'Artery': 'Arterial street',
        'Feedr': 'Feeder street',
        'Norm': 'Normal'    
        }[cond]


# In[ ]:


def onehot_encode(cat_feat, data, dummy_na=False):
    # given a categorical column,
    # perform onehot encode and return encoded DF together with names of new binary columns
    categories = data[cat_feat].unique()
    print('there are', len(categories), 'categories as follows:')
    print(categories)
    
    encoded = pd.get_dummies(data[cat_feat], prefix=cat_feat, dummy_na=dummy_na)
    res = pd.concat([data.drop(columns=[cat_feat]), encoded], axis='columns')
    new_feat_names = ['_'.join([cat_feat, cc]) for cc in categories]
    return res, new_feat_names

def encode_cat_feats(data, cat_feats, dummy_na=False):
    print('Onehot encode categorical features: ', cat_feats)

    encoded_df = data.copy()
    # encode 1 cat feature at a time
    for cf in cat_feats:
        encoded_df, _ = onehot_encode(cf, encoded_df, dummy_na=dummy_na)

    return encoded_df


# In[ ]:


def list_numeric_columns(data):
    return list(data.columns[np.where(data.dtypes != 'object')])

def list_string_columns(data):
    return list(data.columns[np.where(data.dtypes == 'object')])

def split_train_valid(data, target):
    y = data.pop(target)
    X = data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.1, 
                                                          random_state=1
                                                         )
    return X_train, X_valid, y_train, y_valid

def check_na(data):
    # check if any NA left
    na_count = [sum(data[ff].isnull()) for ff in data.columns]
    return pd.DataFrame({'column': data.columns, 'na_count': na_count}).              query('na_count > 0')   


# In[ ]:


def to_quantitative(text_feat, df, scoring):
    '''
    Given a feature stored in data as text but actually a quantitative feat, convert it to numerical values
    via given encoding
    :param scoring:
    :param text_feat:
    :return:
    '''
    n_na = sum(df[text_feat].isnull())
    print('\t Feature {0} has {1} NAs, they will be filled by 0'.format(text_feat, n_na))

    res = df.copy()
    res[text_feat].fillna("NA", inplace=True)
    res[text_feat] = res[text_feat].apply(lambda form: scoring[form])
    return res

def quant_to_scores(quant_feats, data, scorings):
    print('\n Converting quantitative text features to scores...')
    score_dict = dict(zip(quant_feats, scorings))
    
    for tf in quant_feats:  
        data = to_quantitative(text_feat=tf, df=data, scoring=score_dict[tf])

    return data


# In[ ]:


def make_output(y_pred):
    test_index = range(len(train) + 1, len(data) + 1)
    return pd.DataFrame({'Id': test_index, 'SalePrice': y_pred})


# In[ ]:


def get_train_tests(data, target):
    train_part = data.loc[data[target] > 0]
    test_part = data.loc[data[target] == 0]
    return train_part, test_part


# ## Define target

# In[ ]:


target = 'saleprice'
y_train = data.loc[data[target] > 0][target]


# ## Preprocessing

# ### Handle missing values

# In[ ]:


num_vars = list_numeric_columns(data)
na_checker = check_na(data[num_vars] ).sort_values('na_count', ascending=False).    reset_index(drop=True)
print(na_checker)


# Fill NAs in certain numeric vars by their means.

# In[ ]:


to_fill = na_checker['column'].values[3:]
to_fill


# In[ ]:


data[to_fill] = data[to_fill].fillna(data[to_fill].mean())


# In[ ]:


data['masvnrarea'].fillna(data['masvnrarea'].mean(), inplace=True)


# In[ ]:


train_part, test_part = get_train_tests(data, target)


# TODO: Drop columns with lots of NAs.

# In[ ]:


# columns_with_lots_of_na = na_checker.head(11)['column']
# print(columns_with_lots_of_na)
# data = data.drop(columns=columns_with_lots_of_na)


# # Simple model
# LInear regressors with no derived features.

# In[ ]:


feats0 = ['overallqual',
          'yearbuilt', 'mosold', 'yrsold', 'grlivarea', 'lotarea'
         ]
# overallcond


# In[ ]:


y_train = train_part[target]


# In[ ]:


X_train = train_part[feats0]
X_test = test_part[feats0]


# In[ ]:


lr = LinearRegression()
lr.fit(X_train, y_train)
print('score of linear regressor', lr.score(X_train, y_train))

alphas = np.linspace(-3, 0, 4)
ridge = RidgeCV(alphas=alphas, cv=5)
ridge.fit(X_train, y_train)
print('score of ridge regressor', ridge.score(X_train, y_train))


# Ridge and base linear regressors have same score. So we only need linear regressor.

# ## Numeric features

# In[ ]:


num_cols = list_numeric_columns(data)
print(num_cols)


# In[ ]:


check_na(data[num_cols]).sort_values('na_count', ascending=False)


# In[ ]:


feats0 = ['overallqual',
          'yearbuilt', 'mosold', 'yrsold', 'grlivarea', 'lotarea'
         ]
# overallcond


# __note__: adding var "overallcond" pull down prediction perf a lot, so this var has problem.

# In[ ]:


lr = LinearRegression()


# In[ ]:


# features for bathrooms, bedrooms
room_feats = ['bedroomabvgr', 'fullbath', 'halfbath',
              'kitchenabvgr', 'totrmsabvgrd'
             ] # total_bath


# In[ ]:


X_train = train_part[feats0 + room_feats]
X_test = test_part[feats0 + room_feats]

lr.fit(X_train, y_train)
print('score of linear regressor', lr.score(X_train, y_train))


# In[ ]:


# area-related features
area_feats = ['1stflrsf', '2ndflrsf', 'lowqualfinsf', 'masvnrarea']


# + adding "masvnrarea" boost prediction a bit more

# In[ ]:


X_train = train_part[feats0 + room_feats + area_feats]
X_test = test_part[feats0 + room_feats + area_feats]

lr.fit(X_train, y_train)
print('score of linear regressor', lr.score(X_train, y_train))


# In[ ]:


check_na(data[['bsmtfinsf1', 'bsmtunfsf',]])


# In[ ]:


# basement features
# a potential feature is ratio between unfinished basement area and total area

# data['bsmt_unfinished_ratio'] = data['bsmtunfsf'] / data['totalbsmtsf']
bsmt_feats = [ 'totalbsmtsf', 
             ] 
# 'bsmtfullbath', 'bsmthalfbath',  
# 'bsmtunfsf', 'bsmtfinsf2', 'bsmtfinsf1'


# In[ ]:


X_train = train_part[feats0 + room_feats  + bsmt_feats + area_feats]
X_test = test_part[feats0 + room_feats  + bsmt_feats + area_feats]

lr.fit(X_train, y_train)
round(lr.score(X_train, y_train), 4)


# In[ ]:


make_output(lr.predict(X_test)).to_csv(os.path.join(out_dir, 'lin_res.csv') , index=False)


# + Adding "bsmtunfsf" and "bsmtfinsf1" helps reach better score on train, but pull down prediction perf on test set. This maybe overfitting.
# 
# + bsmt_unfinished_ratio has NA

# In[ ]:


# garage feats
gar_feats = [ 'garagecars', 'garagearea'] # garageyrblt: many NA


# In[ ]:


X_train = train_part[feats0 + room_feats + area_feats + bsmt_feats + gar_feats]
X_test = test_part[feats0 + room_feats + area_feats + bsmt_feats + gar_feats]


# In[ ]:


check_na(X_train)


# In[ ]:


lr.fit(X_train, y_train)
round(lr.score(X_train, y_train), 4)


# In[ ]:


make_output(lr.predict(X_test)).to_csv(os.path.join(out_dir, 'lin_res_2.csv') , index=False)


# + adding garage features not help, so drop them

# In[ ]:


feats = feats0 + room_feats  + bsmt_feats + area_feats


# # Categorical features
# 
# To pick good features among categorical vars, we need a way to measure the correlation between a categorical var and our
# continuous target. Here comes eta correlation.

# ## Helpers for categorical correlaton

# In[ ]:


get_ipython().system('pip install dython')


# In[ ]:


from dython.nominal import correlation_ratio

from dython.nominal import associations


# + for correlation between 2 categorical vars, check method theils_u in dython

# In[ ]:


cat_feats = list_string_columns(data)
print('# cat feats: ', len(cat_feats))
print(cat_feats)


# In[ ]:


eta_corrs = [correlation_ratio(train_part[cf], train_part[target]) for cf in cat_feats]
corr_target_cat_feats = pd.DataFrame({'cat_feat': cat_feats, 'corr_with_target': eta_corrs
                                     }).sort_values('corr_with_target', ascending=False)
corr_target_cat_feats


# As expected: 
# + "neighborhood" has very high correlation with target, as it is purely categorical feats, I will deal with it later
# + "exterqual" and "kitchenqual" also has high correlation with target, so I will encode them by proper scales.

# In[ ]:


qual_feats = ['exterqual', 'kitchenqual', 'bsmtqual']
data[qual_feats].describe()


# In[ ]:


data['exterqual'].unique()
data['kitchenqual'].unique()
data['bsmtqual'].unique()


# In[ ]:


six_scale = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}
scorings = [six_scale]*len(qual_feats)

data = quant_to_scores(qual_feats, data, scorings)
train_part, test_part = get_train_tests(data, target)


# In[ ]:


feats += qual_feats
X_train = train_part[feats]
X_test = test_part[feats]


# In[ ]:


lr.fit(X_train, y_train)
lr_score = round(lr.score(X_train, y_train), 4)


# In[ ]:


make_output(lr.predict(X_test)).to_csv(os.path.join(out_dir, 'lin_res_2.csv') , index=False)


# In[ ]:


# plot corr between 10 cat features with highest corr to target
top10_feats = corr_target_cat_feats.head(10)['cat_feat']
associations(data[top10_feats], theil_u=True, figsize=(10, 10))


# In[ ]:


# to reduce minority 
data['zone_type'] = data['mszoning'].apply(fold_zone_type)
data['adjacency'] = data['condition1'].apply(to_adjacency)
train_part, test_part = get_train_tests(data, target)


# In[ ]:


[correlation_ratio(train_part[cf], train_part[target]) 
 for cf in ['zone_type', 'adjacency']
]


# ## Encode categorical features
# 
# I will perform onehot encoding incrementally, starting with features having highest correlations with target.

# In[ ]:


check_na(data[['neighborhood']])


# In[ ]:


# nbhood
encoded_data, nbh_feats = onehot_encode('neighborhood', data)
print(encoded_data.shape)


# In[ ]:


encoded_data[nbh_feats] .head()


# In[ ]:


train_part, test_part = get_train_tests(encoded_data, target)


# In[ ]:


feats += nbh_feats
X_train = train_part[feats]
X_test = test_part[feats]


# ## TODO: drop non-important feats

# In[ ]:


to_drop = ['saletype', 'salecondition']


# # Random forest
# 
# 

# ## Base RF
# 
# No tuning yet.

# In[ ]:


base_rf = RandomForestRegressor(n_estimators=100, max_features=1.0, n_jobs=-1,
                               random_state=1,
                               )


# In[ ]:


base_rf.fit(X_train, y_train)
rf_score = round(base_rf.score(X_train, y_train), 4)
rf_score


# In[ ]:


y_pred = base_rf.predict(X_test)
make_output(y_pred).to_csv(os.path.join(out_dir, 'rf_res.csv') , index=False)


# In[ ]:


100 * (rf_score - lr_score)/lr_score


# + adding neighborhood has a whooping effect. It boosts score by >20% and advanced me >450 places on leaderboard.
# + now we add the next relevant categorical var, "foundation" (this is because kitchen and exterior qualities are already added). But should expect that the impact may be marginal, as the correlation is not that high.

# In[ ]:


# adding foundation
data.foundation.value_counts()


# In[ ]:


check_na(data[['foundation']])


# In[ ]:


encoded_data, fdn_feats = onehot_encode('foundation', encoded_data)


# In[ ]:


train_part, test_part = get_train_tests(encoded_data, target)


# In[ ]:


feats += fdn_feats
X_train = train_part[feats]
X_test = test_part[feats]


# In[ ]:


base_rf.fit(X_train, y_train)
rf_score = round(base_rf.score(X_train, y_train), 4)
rf_score


# + as expected "foundation" not help at all, it even pulls down perf.

# ## Tuning RF 
# 
# I will tune RF via randomized then grid searches. 
# + A randomized search allows a quick search over hyperparameter space, 
# + which then suggests a clearer direction for an exhaustive grid search.

# ### Random grid search

# In[ ]:


from pprint import pprint

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = [1., 'auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(2, 20, num = 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)


# In[ ]:


# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
random_rf = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                               n_iter = 100, cv = 3, verbose=2, random_state=42, 
                               n_jobs = -1)


random_rf.fit(X_train, y_train)

pprint(random_rf.best_params_)


# In[ ]:


random_search_score = random_rf.best_estimator_.score(X_train, y_train)
print(round(random_search_score, 4))


# # Others

# In[ ]:


get_ipython().system('pip install xgboost')


# In[ ]:


get_ipython().system('pip install lightgbm')


# ## Try other encoding schemes for nominal vars

# In[ ]:


get_ipython().system('pip install category_encoders')


# In[ ]:


import category_encoders as ce


# In[ ]:


# concat train and test sets st we always perform transformation on both sets
test['SalePrice'] = 0
data = pd.concat([train, test])

print(data.shape)

# lowercase all column names for convenience
data.columns = [str.lower(cc) for cc in data.columns]

# sale price per square feet is also interested
# data['sale_price_per_sf'] = data['saleprice'] / data['grlivarea']


# In[ ]:


cat_feats = list_string_columns(data)
print('# categ feats:', len(cat_feats))


# ### Target encoding

# In[ ]:


# try with nominal feats with high correlation with target first
# eta_corrs = [correlation_ratio(train_part[cf], train_part[target]) for cf in cat_feats]
# corr_target_cat_feats = pd.DataFrame({'cat_feat': cat_feats, 'corr_with_target': eta_corrs
#                                      }).sort_values('corr_with_target', ascending=False)
# corr_target_cat_feats


# In[ ]:


get_ipython().run_line_magic('pinfo', 'pd.DataFrame.join')


# In[ ]:


train.columns = [str.lower(cc) for cc in train.columns]
test.columns = [str.lower(cc) for cc in test.columns]


# In[ ]:


target_enc = ce.TargetEncoder(cols=cat_feats)
target_enc.fit(train[cat_feats] , train[target])
print(target_enc.transform(train[cat_feats]).head())


# In[ ]:


# Transform the features, 
# rename the columns with _target suffix, and join to dataframe
# also remove old categ vars
train_TE = train.join(target_enc.transform(train[cat_feats]).add_suffix('_target')).drop(columns=cat_feats)
test_TE = test.join(target_enc.transform(test[cat_feats]).add_suffix('_target')).drop(columns=cat_feats)


# In[ ]:


features = train_TE.columns.drop([target, 'id'])
print(features)


# In[ ]:


base_rf = RandomForestRegressor()


# In[ ]:


print(check_na(train_TE))
print(check_na(test_TE))


# In[ ]:


num_feats = list_numeric_columns(train_TE)
train_TE[num_feats] = train_TE[num_feats].fillna(train_TE[num_feats].mean())
test_TE[num_feats] = test_TE[num_feats].fillna(test_TE[num_feats].mean())


# In[ ]:


print(check_na(train_TE))
print(check_na(test_TE))


# In[ ]:


base_rf.fit(train_TE[features], train_TE[target])
base_rf.score(train_TE[features], train_TE[target])


# In[ ]:


y_pred = base_rf.predict(test_TE[features])
make_output(y_pred).to_csv(os.path.join(out_dir, 'rf_res_TE.csv') , index=False)


# ### CatBoost encoding

# In[ ]:




