#!/usr/bin/env python
# coding: utf-8

# # Ensemble of Voting Classifiers
# 
# **Edits by Eric Antoine Scuccimarra** - This is a fork of  https://www.kaggle.com/mlisovyi/feature-engineering-lighgbm-with-f1-macro, by Misha Losvyo, with a few changes:
#  - The LightGBM models have been replaced with XGBoost and the code has been updated accordingly.
#  - Some additional features have been added.
#  - Some features which were previously dropped have been retained.
#  - Some of the code has been reorganized.
#  - Rather than splitting the data once and using the validation data for the LGBM early stopping, I split the data during the training so the entire training set can be trained on. I found that this works better than a k-fold split in this case.
# 
# Some features were taken from the following kernels: 
#  - https://www.kaggle.com/kuriyaman1002/reduce-features-140-84-keeping-f1-score, by Kuriyaman.
#  - https://www.kaggle.com/gaxxxx/exploratory-data-analysis-lightgbm
#  
# **Notes from Original Kernel (edited by EAS):**
# 
# This kernel closely follows https://www.kaggle.com/mlisovyi/lighgbm-hyperoptimisation-with-f1-macro, but instead of running hyperparameter optimisation it uses optimal values from that kernel and thus runs faster. 
# 
# Several key points:
# - **This kernel runs training on the heads of housholds only** (after extracting aggregates over households). This follows the announced scoring startegy: *Note that ONLY the heads of household are used in scoring. All household members are included in test + the sample submission, but only heads of households are scored.* (from the data description). However, at the moment it seems that evaluation depends also on non-head household members, see https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403#360115. In practise, ful prediction gives ~0.4 PLB score, while replacing all non-head entries with class 1 leads to a drop down to ~0.2 PLB score
# - **It seems to be very important to balance class frequencies.** Without balancing a trained model gives ~0.39 PLB / ~0.43 local test, while adding balancing leads to ~0.42 PLB / 0.47 local test. One can do it by hand, one can achieve it by undersampling. But the simplest (and more powerful compared to undersampling) is to set `class_weight='balanced'` in the LightGBM model constructor in sklearn API.
# - **This kernel uses macro F1 score to early stopping in training**. This is done to align with the scoring strategy.
# - Categoricals are turned into numbers with proper mapping instead of blind label encoding. 
# - **OHE if reversed into label encoding, as it is easier to digest for a tree model.** This trick would be harmful for non-tree models, so be careful.
# - **idhogar is NOT used in training**. The only way it could have any info would be if there is a data leak. We are fighting with poverty here- exploiting leaks will not reduce poverty in any way :)
# - **There are aggregations done within households and new features are hand-crafted**. Note, that there are not so many features that can be aggregated, as most are already quoted on household level.
# - **A voting classifier is used to average over several LightGBM models**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd 

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import lightgbm as lgb
from scipy import stats
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.utils import class_weight

import warnings
warnings.filterwarnings("ignore")


# The following categorical mapping originates from [this kernel](https://www.kaggle.com/mlisovyi/categorical-variables-encoding-function).

# In[ ]:


from sklearn.preprocessing import LabelEncoder

# this only transforms the idhogar field, the other things this function used to do are done elsewhere
def encode_data(df):
    df['idhogar'] = LabelEncoder().fit_transform(df['idhogar'])

# plot feature importance for sklearn decision trees    
def feature_importance(forest, X_train, display_results=True):
    ranked_list = []
    zero_features = []
    
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]
    
    if display_results:
        # Print the feature ranking
        print("Feature ranking:")

    for f in range(X_train.shape[1]):
        if display_results:
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]) + " - " + X_train.columns[indices[f]])
        
        ranked_list.append(X_train.columns[indices[f]])
        
        if importances[indices[f]] == 0.0:
            zero_features.append(X_train.columns[indices[f]])
            
    return ranked_list, zero_features


# **There is also feature engineering magic happening here:**

# In[ ]:


def do_features(df):
    feats_div = [('children_fraction', 'r4t1', 'r4t3'), 
                 ('working_man_fraction', 'r4h2', 'r4t3'),
                 ('all_man_fraction', 'r4h3', 'r4t3'),
                 ('human_density', 'tamviv', 'rooms'),
                 ('human_bed_density', 'tamviv', 'bedrooms'),
                 ('rent_per_person', 'v2a1', 'r4t3'),
                 ('rent_per_room', 'v2a1', 'rooms'),
                 ('mobile_density', 'qmobilephone', 'r4t3'),
                 ('tablet_density', 'v18q1', 'r4t3'),
                 ('mobile_adult_density', 'qmobilephone', 'r4t2'),
                 ('tablet_adult_density', 'v18q1', 'r4t2'),
                ]
    
    feats_sub = [('people_not_living', 'tamhog', 'tamviv'),
                 ('people_weird_stat', 'tamhog', 'r4t3')]

    for f_new, f1, f2 in feats_div:
        df['fe_' + f_new] = (df[f1] / df[f2]).astype(np.float32)       
    for f_new, f1, f2 in feats_sub:
        df['fe_' + f_new] = (df[f1] - df[f2]).astype(np.float32)
    
    # aggregation rules over household
    aggs_num = {'age': ['min', 'max', 'mean'],
                'escolari': ['min', 'max', 'mean']
               }
    
    aggs_cat = {'dis': ['mean']}
    for s_ in ['estadocivil', 'parentesco', 'instlevel']:
        for f_ in [f_ for f_ in df.columns if f_.startswith(s_)]:
            aggs_cat[f_] = ['mean', 'count']

    # aggregation over household
    for name_, df_ in [('18', df.query('age >= 18'))]:
        df_agg = df_.groupby('idhogar').agg({**aggs_num, **aggs_cat}).astype(np.float32)
        df_agg.columns = pd.Index(['agg' + name_ + '_' + e[0] + "_" + e[1].upper() for e in df_agg.columns.tolist()])
        df = df.join(df_agg, how='left', on='idhogar')
        del df_agg

    # Drop id's
    df.drop(['Id'], axis=1, inplace=True)
    
    return df


# In[ ]:


# convert one hot encoded fields to label encoding
def convert_OHE2LE(df):
    tmp_df = df.copy(deep=True)
    for s_ in ['pared', 'piso', 'techo', 'abastagua', 'sanitario', 'energcocinar', 'elimbasu', 
               'epared', 'etecho', 'eviv', 'estadocivil', 'parentesco', 
               'instlevel', 'lugar', 'tipovivi',
               'manual_elec']:
        if 'manual_' not in s_:
            cols_s_ = [f_ for f_ in df.columns if f_.startswith(s_)]
        elif 'elec' in s_:
            cols_s_ = ['public', 'planpri', 'noelec', 'coopele']
        sum_ohe = tmp_df[cols_s_].sum(axis=1).unique()
        #deal with those OHE, where there is a sum over columns == 0
        if 0 in sum_ohe:
            print('The OHE in {} is incomplete. A new column will be added before label encoding'
                  .format(s_))
            # dummy colmn name to be added
            col_dummy = s_+'_dummy'
            # add the column to the dataframe
            tmp_df[col_dummy] = (tmp_df[cols_s_].sum(axis=1) == 0).astype(np.int8)
            # add the name to the list of columns to be label-encoded
            cols_s_.append(col_dummy)
            # proof-check, that now the category is complete
            sum_ohe = tmp_df[cols_s_].sum(axis=1).unique()
            if 0 in sum_ohe:
                 print("The category completion did not work")
        tmp_cat = tmp_df[cols_s_].idxmax(axis=1)
        tmp_df[s_ + '_LE'] = LabelEncoder().fit_transform(tmp_cat).astype(np.int16)
        if 'parentesco1' in cols_s_:
            cols_s_.remove('parentesco1')
        tmp_df.drop(cols_s_, axis=1, inplace=True)
    return tmp_df


# # Read in the data and clean it up

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

test_ids = test.Id


# In[ ]:


def process_df(df_):
    # encode the idhogar
    encode_data(df_)
    
    # create aggregate features
    return do_features(df_)

train = process_df(train)
test = process_df(test)


# Clean up some missing data and convert objects to numeric.

# In[ ]:


# some dependencies are Na, fill those with the square root of the square
train['dependency'] = np.sqrt(train['SQBdependency'])
test['dependency'] = np.sqrt(test['SQBdependency'])

# fill "no"s for education with 0s
train.loc[train['edjefa'] == "no", "edjefa"] = 0
train.loc[train['edjefe'] == "no", "edjefe"] = 0
test.loc[test['edjefa'] == "no", "edjefa"] = 0
test.loc[test['edjefe'] == "no", "edjefe"] = 0

# if education is "yes" and person is head of household, fill with escolari
train.loc[(train['edjefa'] == "yes") & (train['parentesco1'] == 1), "edjefa"] = train.loc[(train['edjefa'] == "yes") & (train['parentesco1'] == 1), "escolari"]
train.loc[(train['edjefe'] == "yes") & (train['parentesco1'] == 1), "edjefe"] = train.loc[(train['edjefe'] == "yes") & (train['parentesco1'] == 1), "escolari"]

test.loc[(test['edjefa'] == "yes") & (test['parentesco1'] == 1), "edjefa"] = test.loc[(test['edjefa'] == "yes") & (test['parentesco1'] == 1), "escolari"]
test.loc[(test['edjefe'] == "yes") & (test['parentesco1'] == 1), "edjefe"] = test.loc[(test['edjefe'] == "yes") & (test['parentesco1'] == 1), "escolari"]

# this field is supposed to be interaction between gender and escolari, but it isn't clear what "yes" means, let's fill it with 4
train.loc[train['edjefa'] == "yes", "edjefa"] = 4
train.loc[train['edjefe'] == "yes", "edjefe"] = 4

test.loc[test['edjefa'] == "yes", "edjefa"] = 4
test.loc[test['edjefe'] == "yes", "edjefe"] = 4

# convert to int for our models
train['edjefe'] = train['edjefe'].astype("int")
train['edjefa'] = train['edjefa'].astype("int")
test['edjefe'] = test['edjefe'].astype("int")
test['edjefa'] = test['edjefa'].astype("int")

# create feature with max education of either head of household
train['edjef'] = np.max(train[['edjefa','edjefe']], axis=1)
test['edjef'] = np.max(test[['edjefa','edjefe']], axis=1)

# fill some nas
train['v2a1']=train['v2a1'].fillna(0)
test['v2a1']=test['v2a1'].fillna(0)

test['v18q1']=test['v18q1'].fillna(0)
train['v18q1']=train['v18q1'].fillna(0)

train['rez_esc']=train['rez_esc'].fillna(0)
test['rez_esc']=test['rez_esc'].fillna(0)

train.loc[train.meaneduc.isnull(), "meaneduc"] = 0
train.loc[train.SQBmeaned.isnull(), "SQBmeaned"] = 0

test.loc[test.meaneduc.isnull(), "meaneduc"] = 0
test.loc[test.SQBmeaned.isnull(), "SQBmeaned"] = 0

# fix some inconsistencies in the data - some rows indicate both that the household does and does not have a toilet, 
# if there is no water we'll assume they do not
train.loc[(train.v14a ==  1) & (train.sanitario1 ==  1) & (train.abastaguano == 0), "v14a"] = 0
train.loc[(train.v14a ==  1) & (train.sanitario1 ==  1) & (train.abastaguano == 0), "sanitario1"] = 0

test.loc[(test.v14a ==  1) & (test.sanitario1 ==  1) & (test.abastaguano == 0), "v14a"] = 0
test.loc[(test.v14a ==  1) & (test.sanitario1 ==  1) & (test.abastaguano == 0), "sanitario1"] = 0


# In[ ]:


train['roof_waste_material'] = np.nan
test['roof_waste_material'] = np.nan
train['electricity_other'] = np.nan
test['electricity_other'] = np.nan

def fill_roof_exception(x):
    if (x['techozinc'] == 0) and (x['techoentrepiso'] == 0) and (x['techocane'] == 0) and (x['techootro'] == 0):
        return 1
    else:
        return 0
    
def fill_no_electricity(x):
    if (x['public'] == 0) and (x['planpri'] == 0) and (x['noelec'] == 0) and (x['coopele'] == 0):
        return 1
    else:
        return 0

train['roof_waste_material'] = train.apply(lambda x : fill_roof_exception(x),axis=1)
test['roof_waste_material'] = test.apply(lambda x : fill_roof_exception(x),axis=1)
train['electricity_other'] = train.apply(lambda x : fill_no_electricity(x),axis=1)
test['electricity_other'] = test.apply(lambda x : fill_no_electricity(x),axis=1)


# In[ ]:


def train_test_apply_func(train_, test_, func_):
    test_['Target'] = 0
    xx = pd.concat([train_, test_])

    xx_func = func_(xx)
    train_ = xx_func.iloc[:train_.shape[0], :]
    test_  = xx_func.iloc[train_.shape[0]:, :].drop('Target', axis=1)

    del xx, xx_func
    return train_, test_


# In[ ]:


# convert the one hot fields into label encoded
train, test = train_test_apply_func(train, test, convert_OHE2LE)


# # Geo aggregates

# In[ ]:


cols_2_ohe = ['eviv_LE', 'etecho_LE', 'epared_LE', 'elimbasu_LE', 
              'energcocinar_LE', 'sanitario_LE', 'manual_elec_LE',
              'pared_LE']
cols_nums = ['age', 'meaneduc', 'dependency', 
             'hogar_nin', 'hogar_adul', 'hogar_mayor', 'hogar_total',
             'bedrooms', 'overcrowding']

def convert_geo2aggs(df_):
    tmp_df = pd.concat([df_[(['lugar_LE', 'idhogar']+cols_nums)],
                        pd.get_dummies(df_[cols_2_ohe], 
                                       columns=cols_2_ohe)],axis=1)

    geo_agg = tmp_df.groupby(['lugar_LE','idhogar']).mean().groupby('lugar_LE').mean().astype(np.float32)
    geo_agg.columns = pd.Index(['geo_' + e for e in geo_agg.columns.tolist()])
    
    del tmp_df
    return df_.join(geo_agg, how='left', on='lugar_LE')

# add some aggregates by geography
train, test = train_test_apply_func(train, test, convert_geo2aggs)


# In[ ]:


# add the number of people over 18 in each household
train['num_over_18'] = 0
train['num_over_18'] = train[train.age >= 18].groupby('idhogar').transform("count")
train['num_over_18'] = train.groupby("idhogar")["num_over_18"].transform("max")
train['num_over_18'] = train['num_over_18'].fillna(0)

test['num_over_18'] = 0
test['num_over_18'] = test[test.age >= 18].groupby('idhogar').transform("count")
test['num_over_18'] = test.groupby("idhogar")["num_over_18"].transform("max")
test['num_over_18'] = test['num_over_18'].fillna(0)

# add some extra features, these were taken from another kernel
def extract_features(df):
    df['adult'] = df['hogar_adul'] - df['hogar_mayor']
    df['dependency_count'] = df['hogar_nin'] + df['hogar_mayor']
    df['dependency'] = df['dependency_count'] / df['adult']
    df['dependency'] = df['dependency'].replace({np.inf: 0})
    df['child_percent'] = df['hogar_nin']/df['hogar_total']
    df['elder_percent'] = df['hogar_mayor']/df['hogar_total']
    df['adult_percent'] = df['hogar_adul']/df['hogar_total']
    df['bedrooms_to_rooms'] = df['bedrooms']/df['rooms']
    df['rent_to_rooms'] = df['v2a1']/df['rooms']
    df['r4h1_percent_in_male'] = df['r4h1'] / df['r4h3']
    df['r4m1_percent_in_female'] = df['r4m1'] / df['r4m3']
    df['r4h1_percent_in_total'] = df['r4h1'] / df['hhsize']
    df['r4m1_percent_in_total'] = df['r4m1'] / df['hhsize']
    df['r4t1_percent_in_total'] = df['r4t1'] / df['hhsize']
    df['tamhog_to_rooms'] = df['tamhog']/df['rooms'] # tamhog - size of the household
    df['r4t3_to_tamhog'] = df['r4t3']/df['tamhog'] # r4t3 - Total persons in the household
    df['r4t3_to_rooms'] = df['r4t3']/df['rooms'] # r4t3 - Total persons in the household
    df['rent_per_adult'] = df['v2a1']/df['hogar_adul']
    df['v2a1_to_r4t3'] = df['v2a1']/df['r4t3'] # rent to people in household
    df['v2a1_to_r4t3'] = df['v2a1']/(df['r4t3'] - df['r4t1']) # rent to people under age 12
    df['hhsize_to_rooms'] = df['hhsize']/df['rooms'] # rooms per person
    df['rent_to_hhsize'] = df['v2a1']/df['hhsize'] # rent to household size
    df['rent_to_over_18'] = df['v2a1']/df['num_over_18']
    df['tablet_per_person_household'] = df['v18q1']/df['hhsize']
    df['phone_per_person_household'] = df['qmobilephone']/df['hhsize']
    df['rez_esc_escolari'] = df['rez_esc']/df['escolari']
    df['rez_esc_r4t1'] = df['rez_esc']/df['r4t1']
    df['rez_esc_r4t2'] = df['rez_esc']/df['r4t2']
    df['rez_esc_r4t3'] = df['rez_esc']/df['r4t3']
    df['rez_esc_age'] = df['rez_esc']/df['age']
    
    df['escolari_age'] = df['escolari']/df['age']
    df['escolari_age_mean'] = df.groupby("idhogar")["escolari_age"].transform("mean")
    df['age_sum'] = df.groupby("idhogar")["age"].transform("sum")
    df['age_std'] = df.groupby("idhogar")["age"].transform("std")
    df['escolari_age_std'] = df.groupby("idhogar")["escolari_age"].transform("std")
    df['escolari_age_max'] = df.groupby("idhogar")["escolari_age"].transform("max")
    df['escolari_age_min'] = df.groupby("idhogar")["escolari_age"].transform("min")
    df['escolari_std'] = df.groupby("idhogar")["escolari"].transform("min")
    # fill nas
    df['escolari_age'] = df['escolari_age'].fillna(0)
    df['escolari_age_std'] = df['escolari_age_std'].fillna(-1)
    df['age_std'] = df['age_std'].fillna(-1)
    # some households have no one over 18, use the total rent for those
    df.loc[df.num_over_18 == 0, "rent_to_over_18"] = df[df.num_over_18 == 0].v2a1
    
extract_features(train)    
extract_features(test)   


# In[ ]:


train.columns[train.isnull().any()]


# In[ ]:


# drop duplicated columns
needless_cols = ['r4t3', 'tamhog', 'tamviv', 'hhsize', 'v18q', 'v14a', 'agesq',
                 'mobilephone', 'female', ]

instlevel_cols = [s for s in train.columns.tolist() if 'instlevel' in s]

needless_cols.extend(instlevel_cols)

train = train.drop(needless_cols, axis=1)
test = test.drop(needless_cols, axis=1)


# ## Split the data
# 
# We split the data by household to avoid leakage, since rows belonging to the same household usually have the same target. Since we filter the data to only include heads of household this isn't technically necessary, but it provides an easy way to use the entire training data set if we want to do that.
# 
# Note that after splitting the data we overwrite the train data with the entire data set so we can train on all of the data. The split_data function does the same thing without overwriting the data, and is used within the training loop to (hopefully) approximate a K-Fold split. 

# In[ ]:


def split_data(train, y, sample_weight=None, households=None, test_percentage=0.20, seed=None):
    # uncomment for extra randomness
#     np.random.seed(seed=seed)
    
    train2 = train.copy()
    
    # pick some random households to use for the test data
    cv_hhs = np.random.choice(households, size=int(len(households) * test_percentage), replace=False)
    
    # select households which are in the random selection
    cv_idx = np.isin(households, cv_hhs)
    X_test = train2[cv_idx]
    y_test = y[cv_idx]

    X_train = train2[~cv_idx]
    y_train = y[~cv_idx]
    
    if sample_weight is not None:
        y_train_weights = sample_weight[~cv_idx]
        return X_train, y_train, X_test, y_test, y_train_weights
    
    return X_train, y_train, X_test, y_test


# In[ ]:


X = train.query('parentesco1==1')
# X = train.copy()

# pull out and drop the target variable
y = X['Target'] - 1
X = X.drop(['Target'], axis=1)

np.random.seed(seed=None)

train2 = X.copy()

train_hhs = train2.idhogar

households = train2.idhogar.unique()
cv_hhs = np.random.choice(households, size=int(len(households) * 0.15), replace=False)

cv_idx = np.isin(train2.idhogar, cv_hhs)

X_test = train2[cv_idx]
y_test = y[cv_idx]

X_train = train2[~cv_idx]
y_train = y[~cv_idx]

# train on entire dataset
X_train = train2
y_train = y

train_households = X_train.idhogar


# In[ ]:


# figure out the class weights for training with unbalanced classes
y_train_weights = class_weight.compute_sample_weight('balanced', y_train, indices=None)


# # Fit a voting classifier
# Define a derived VotingClassifier class to be able to pass `fit_params` for early stopping. Vote based on LGBM models with early stopping based on macro F1 and decaying learning rate.
# 
# The parameters are optimised with a random search in this kernel: https://www.kaggle.com/mlisovyi/lighgbm-hyperoptimisation-with-f1-macro

# In[ ]:


# drop some features which aren't used by the LGBM or have very low importance
extra_drop_features = ['agg18_estadocivil1_MEAN',
 'agg18_estadocivil4_COUNT',
 'agg18_estadocivil5_COUNT',
 'agg18_estadocivil6_COUNT',
 'agg18_estadocivil7_COUNT',
 'agg18_parentesco10_COUNT',
 'agg18_parentesco11_COUNT',
 'agg18_parentesco12_COUNT',
 'agg18_parentesco1_COUNT',
 'agg18_parentesco2_COUNT',
 'agg18_parentesco3_COUNT',
 'agg18_parentesco4_COUNT',
 'agg18_parentesco5_COUNT',
 'agg18_parentesco6_COUNT',
 'agg18_parentesco7_COUNT',
 'agg18_parentesco8_COUNT',
 'agg18_parentesco9_COUNT',
 'electricity_other',
 'geo_elimbasu_LE_4',
 'geo_energcocinar_LE_0',
 'geo_energcocinar_LE_1',
 'geo_energcocinar_LE_2',
 'geo_epared_LE_0',
 'geo_epared_LE_2',
 'geo_hogar_mayor',
 'geo_manual_elec_LE_2',
 'geo_pared_LE_0',
 'geo_pared_LE_3',
 'geo_pared_LE_4',
 'geo_pared_LE_5',
 'geo_pared_LE_6',
 'num_over_18',
 'parentesco_LE',
#  'rez_esc',
#  'rez_esc_age',
 'rez_esc_r4t2',
 'rez_esc_r4t3']


# In[ ]:


xgb_drop_cols = extra_drop_features + ["idhogar",  'parentesco1']


# In[ ]:


opt_parameters_1 = {'max_depth':35, 'learning_rate':0.125, 'silent':1, "n_estimators": 325, 'objective':'multi:softmax', 'min_child_weight': 2, 'num_class': 4, 'gamma': 2.0, 'colsample_bylevel': 0.9, 'subsample': 0.84, 'colsample_bytree': 0.88, 'reg_lambda': 0.4 }
opt_parameters_2 = {'max_depth':35, 'learning_rate':0.14, 'silent':1, "n_estimators": 275, 'objective':'multi:softmax', 'min_child_weight': 2, 'num_class': 4, 'gamma': 2.25, 'colsample_bylevel': 1, 'subsample': 0.95, 'colsample_bytree': 0.85, 'reg_lambda': 0.35 }
opt_parameters_3 = {'max_depth':35, 'learning_rate':0.15, 'silent':1, "n_estimators": 300, 'objective':'multi:softmax', 'min_child_weight': 2, 'num_class': 4, 'gamma': 2.75, 'colsample_bylevel': 0.95, 'subsample': 0.94, 'colsample_bytree': 0.9, 'reg_lambda': 0.375 }
opt_parameters_4 = {'max_depth':35, 'learning_rate':0.13, 'silent':1, "n_estimators": 350, 'objective':'multi:softmax', 'min_child_weight': 2, 'num_class': 4, 'gamma': 2.5, 'colsample_bylevel': 1, 'subsample': 0.95, 'colsample_bytree': 0.88, 'reg_lambda': 0.325 }

xgb_param_list = [opt_parameters_1,  opt_parameters_2,  opt_parameters_3, opt_parameters_4]


# In[ ]:


# since XGB minimizes the metric we need to subtract the F1 score from 1
def evaluate_macroF1_xgb(predictions, truth):  
    # this follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483
    pred_labels = predictions.argmax(axis=1)
    truth = truth.get_label()
    f1 = f1_score(truth, pred_labels, average='macro')
    return ('macroF1', 1-f1) 

fit_params={"early_stopping_rounds":150,
            "eval_metric" : evaluate_macroF1_xgb, 
            "eval_set" : [(X_test,y_test)],
            'verbose': False,
           }

fit_params['verbose'] = 50


# In[ ]:


np.random.seed(100)

def _parallel_fit_estimator(estimator1, X, y, drop_cols=[], use_cv=True, sample_weight=None, threshold=True, **fit_params):
    estimator = clone(estimator1)
    
    # randomly split the data so we have a test set for early stopping
    if use_cv:
        if sample_weight is not None:
            X_train, y_train, X_cv, y_cv, y_train_weight = split_data(X, y, sample_weight, households=train_households)
        else:
            X_train, y_train, X_cv, y_cv = split_data(X, y, None, households=train_households)
    else:
        X_train = X
        y_train = y
        X_cv = X
        y_cv = y
        
    # update the fit params with our new split
    if isinstance(estimator1, xgb.XGBClassifier):
        fit_params["eval_set"] = [(X_cv,y_cv)]
    else:
        fit_params["eval_set"] = [(X_train, y_train), (X_cv,y_cv)]
        
    # fit the estimator
    if sample_weight is not None:
        if isinstance(estimator1, ExtraTreesClassifier) or isinstance(estimator1, RandomForestClassifier):
            estimator.fit(X_train, y_train)
        else:
            _ = estimator.fit(X_train, y_train, sample_weight=y_train_weight, **fit_params)
    else:
        if isinstance(estimator1, ExtraTreesClassifier) or isinstance(estimator1, RandomForestClassifier):
            estimator.fit(X_train, y_train)
        else:
            _ = estimator.fit(X_train, y_train, **fit_params)
    
    if not isinstance(estimator1, ExtraTreesClassifier) and not isinstance(estimator1, RandomForestClassifier) and not isinstance(estimator1, xgb.XGBClassifier):
        best_cv_round = np.argmax(estimator.evals_result_['valid']['macroF1'])
        best_cv = np.max(estimator.evals_result_['valid']['macroF1'])
        best_train = estimator.evals_result_['train']['macroF1'][best_cv_round]
    else:
        best_train = f1_score(y_train, estimator.predict(X_train), average="macro")
        best_cv = f1_score(y_cv, estimator.predict(X_cv), average="macro")
        print("Train F1:", best_train)
        print("Test F1:", best_cv)
        
    # reject some estimators based on their performance on train and test sets
    if threshold:
        # if the valid score is very high we'll allow a little more leeway with the train scores
        if ((best_cv > 0.37) and (best_train > 0.75)) or ((best_cv > 0.44) and (best_train > 0.65)):
            return estimator

        # else recurse until we get a better one
        else:
            print("Unacceptable!!! Trying again...")
            return _parallel_fit_estimator(estimator1, X, y, sample_weight=sample_weight, **fit_params)
    
    else:
        return estimator
    
class VotingClassifierLGBM(VotingClassifier):
    '''
    This implements the fit method of the VotingClassifier propagating fit_params
    '''
    def fit(self, X, y, drop_cols=[], split_data=True, sample_weight=None, threshold=True, **fit_params):
        
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')

        if self.voting not in ('soft', 'hard'):
            raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
                             % self.voting)

        if self.estimators is None or len(self.estimators) == 0:
            raise AttributeError('Invalid `estimators` attribute, `estimators`'
                                 ' should be a list of (string, estimator)'
                                 ' tuples')

        if (self.weights is not None and
                len(self.weights) != len(self.estimators)):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d estimators'
                             % (len(self.weights), len(self.estimators)))

        names, clfs = zip(*self.estimators)
        self._validate_names(names)

        n_isnone = np.sum([clf is None for _, clf in self.estimators])
        if n_isnone == len(self.estimators):
            raise ValueError('All estimators are None. At least one is '
                             'required to be a classifier!')

        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        self.estimators_ = []

        transformed_y = self.le_.transform(y)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_parallel_fit_estimator)(clone(clf), X, transformed_y, drop_cols=drop_cols, use_cv=split_data,
                                                 sample_weight=sample_weight, threshold=threshold, **fit_params)
                for clf in clfs if clf is not None)

        return self


# In[ ]:


clfs = []
for i in range(5):
    clf = xgb.XGBClassifier(random_state=217+i, n_jobs=4, **xgb_param_list[i % len(xgb_param_list)])
    
    clfs.append(('xgb{}'.format(i), clf))
    
vc = VotingClassifierLGBM(clfs, voting='soft')
del(clfs)

#Train the final model with learning rate decay
_ = vc.fit(X_train.drop(xgb_drop_cols, axis=1), y_train, sample_weight=y_train_weights, drop_cols=[], threshold=False, **fit_params)

clf_final = vc.estimators_[0]


# In[ ]:


# l2 used features
global_score = f1_score(y_test, clf_final.predict(X_test.drop(xgb_drop_cols, axis=1)), average='macro')
vc.voting = 'soft'
global_score_soft = f1_score(y_test, vc.predict(X_test.drop(xgb_drop_cols, axis=1)), average='macro')
vc.voting = 'hard'
global_score_hard = f1_score(y_test, vc.predict(X_test.drop(xgb_drop_cols, axis=1)), average='macro')

print('Validation score of a single XGBM Classifier: {:.4f}'.format(global_score))
print('Validation score of a VotingClassifier on XGBMs with soft voting: {:.4f}'.format(global_score_soft))
print('Validation score of a VotingClassifier on XGBMs with hard voting: {:.4f}'.format(global_score_hard))


# In[ ]:


_ = feature_importance(vc.estimators_[0], X_train.drop(xgb_drop_cols, axis=1), display_results=True)


# In[ ]:


# see which features are not used by ANY models
useless_features = []
drop_features = set()
counter = 0
for est in vc.estimators_:
    ranked_features, unused_features = feature_importance(est, X_train.drop(xgb_drop_cols, axis=1), display_results=False)
    useless_features.append(unused_features)
    if counter == 0:
        drop_features = set(unused_features)
    else:
        drop_features = drop_features.intersection(set(unused_features))
    counter += 1
    
drop_features


# ### LGB

# In[ ]:


extra_drop_features = ['agg18_estadocivil1_MEAN',
 'agg18_estadocivil4_COUNT',
 'agg18_estadocivil5_COUNT',
 'agg18_estadocivil6_COUNT',
 'agg18_estadocivil7_COUNT',
 'agg18_parentesco10_COUNT',
 'agg18_parentesco10_MEAN',
 'agg18_parentesco11_COUNT',
 'agg18_parentesco11_MEAN',
 'agg18_parentesco12_COUNT',
 'agg18_parentesco12_MEAN',
 'agg18_parentesco1_COUNT',
 'agg18_parentesco2_COUNT',
 'agg18_parentesco3_COUNT',
 'agg18_parentesco4_COUNT',
 'agg18_parentesco4_MEAN',
 'agg18_parentesco5_COUNT',
 'agg18_parentesco6_COUNT',
 'agg18_parentesco6_MEAN',
 'agg18_parentesco7_COUNT',
 'agg18_parentesco8_COUNT',
 'agg18_parentesco8_MEAN',
 'agg18_parentesco9_COUNT',]

lgb_drop_cols = extra_drop_features + ["idhogar",  'parentesco1']


# In[ ]:


lgb_parameters_1 = {'max_depth': -1, 'learning_rate': 0.125, 'colsample_bytree': 0.88, 'min_child_samples': 90, 'num_leaves': 16, 'subsample': 0.94, 'reg_lambda': 0.425, "n_estimators": 5000}
lgb_parameters_2 = {'max_depth': 50, 'learning_rate': 0.11, 'colsample_bytree': 0.95, 'min_child_samples': 90, 'num_leaves': 25, 'subsample': 0.94, 'reg_lambda': 0.34, "n_estimators": 3000}
lgb_parameters_3 = {'max_depth': 35, 'learning_rate': 0.12, 'colsample_bytree': 0.88, 'min_child_samples': 56, 'num_leaves': 20, 'subsample': 0.95, 'reg_lambda': 0.375, "n_estimators": 4000}
lgb_parameters_4 = {'max_depth': 20, 'learning_rate': 0.13, 'colsample_bytree': 0.89, 'min_child_samples': 90, 'num_leaves': 14, 'subsample': 0.96, 'reg_lambda': 0.39, "n_estimators": 2500}
lgb_parameters_5 = {'max_depth': 25, 'learning_rate': 0.14, 'colsample_bytree': 0.93, 'min_child_samples': 70, 'num_leaves': 17, 'subsample': 0.95, 'reg_lambda': 0.35, "n_estimators": 4500 }

lgb_param_list = [lgb_parameters_1, lgb_parameters_2, lgb_parameters_3, lgb_parameters_4, lgb_parameters_5]


# In[ ]:


opt_parameters = {'colsample_bytree': 0.88, 'min_child_samples': 90, 'num_leaves': 16, 'subsample': 0.94, 'reg_lambda': 0.5, }
opt_parameters = {'colsample_bytree': 0.88, 'min_child_samples': 90, 'num_leaves': 25, 'subsample': 0.94, 'reg_lambda': 0.5, }

def evaluate_macroF1_lgb(truth, predictions):  
    # this follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483
    pred_labels = predictions.reshape(len(np.unique(truth)),-1).argmax(axis=0)
    f1 = f1_score(truth, pred_labels, average='macro')
    return ('macroF1', f1, True) 

fit_params={"early_stopping_rounds":150,
            "eval_metric" : evaluate_macroF1_lgb, 
            "eval_set" : [(X_train,y_train), (X_test,y_test)],
            'eval_names': ['train', 'valid'],
            'verbose': False,
            'categorical_feature': 'auto'}

def learning_rate_power_0997(current_iter):
    base_learning_rate = 0.1
    min_learning_rate = 0.02
    lr = base_learning_rate  * np.power(.995, current_iter)
    return max(lr, min_learning_rate)

fit_params['verbose'] = 200
fit_params['callbacks'] = [lgb.reset_parameter(learning_rate=learning_rate_power_0997)]


# In[ ]:


clfs = []
for i in range(15):
    clf = lgb.LGBMClassifier(objective='multiclass',
                             random_state=314+i, silent=True, metric='None', 
                             n_jobs=4, class_weight='balanced', **lgb_param_list[i % len(lgb_param_list)])
    
    param_set = i % len(lgb_param_list)
    clf.set_params(**lgb_param_list[param_set])
    clfs.append(('lgbm{}'.format(i), clf))
    
vc3 = VotingClassifierLGBM(clfs, voting='soft')
del clfs

#Train the final model with learning rate decay
_ = vc3.fit(X_train.drop(lgb_drop_cols, axis=1), y_train, threshold=True, **fit_params)

clf_final = vc3.estimators_[0]


# In[ ]:


# l1 used features
global_lgb_score = f1_score(y_test, clf_final.predict(X_test.drop(lgb_drop_cols, axis=1)), average='macro')
vc3.voting = 'soft'
global_lgb_score_soft = f1_score(y_test, vc3.predict(X_test.drop(lgb_drop_cols, axis=1)), average='macro')
vc3.voting = 'hard'
global_lgb_score_hard = f1_score(y_test, vc3.predict(X_test.drop(lgb_drop_cols, axis=1)), average='macro')

print('Validation score of a single LGBM Classifier: {:.4f}'.format(global_lgb_score))
print('Validation score of a VotingClassifier on 3 LGBMs with soft voting strategy: {:.4f}'.format(global_lgb_score_soft))
print('Validation score of a VotingClassifier on 3 LGBMs with hard voting strategy: {:.4f}'.format(global_lgb_score_hard))


# In[ ]:


# see which features are not used by ANY models
useless_features = []
drop_features = set()
counter = 0
for est in vc3.estimators_:
    ranked_features, unused_features = feature_importance(est, X_train.drop(lgb_drop_cols, axis=1), display_results=False)
    useless_features.append(unused_features)
    if counter == 0:
        drop_features = set(unused_features)
    else:
        drop_features = drop_features.intersection(set(unused_features))
    counter += 1
    
drop_features


# In[ ]:


_ = feature_importance(vc3.estimators_[0], X_train.drop(lgb_drop_cols, axis=1), display_results=True)


# ### Random Forest

# In[ ]:


et_drop_cols = ['agg18_age_MAX', 'agg18_age_MEAN', 'agg18_age_MIN', 'agg18_dis_MEAN',
       'agg18_escolari_MAX', 'agg18_escolari_MEAN', 'agg18_escolari_MIN',
       'agg18_estadocivil1_COUNT', 'agg18_estadocivil1_MEAN',
       'agg18_estadocivil2_COUNT', 'agg18_estadocivil2_MEAN',
       'agg18_estadocivil3_COUNT', 'agg18_estadocivil3_MEAN',
       'agg18_estadocivil4_COUNT', 'agg18_estadocivil4_MEAN',
       'agg18_estadocivil5_COUNT', 'agg18_estadocivil5_MEAN',
       'agg18_estadocivil6_COUNT', 'agg18_estadocivil6_MEAN',
       'agg18_estadocivil7_COUNT', 'agg18_estadocivil7_MEAN',
       'agg18_parentesco10_COUNT', 'agg18_parentesco10_MEAN',
       'agg18_parentesco11_COUNT', 'agg18_parentesco11_MEAN',
       'agg18_parentesco12_COUNT', 'agg18_parentesco12_MEAN',
       'agg18_parentesco1_COUNT', 'agg18_parentesco1_MEAN',
       'agg18_parentesco2_COUNT', 'agg18_parentesco2_MEAN',
       'agg18_parentesco3_COUNT', 'agg18_parentesco3_MEAN',
       'agg18_parentesco4_COUNT', 'agg18_parentesco4_MEAN',
       'agg18_parentesco5_COUNT', 'agg18_parentesco5_MEAN',
       'agg18_parentesco6_COUNT', 'agg18_parentesco6_MEAN',
       'agg18_parentesco7_COUNT', 'agg18_parentesco7_MEAN',
       'agg18_parentesco8_COUNT', 'agg18_parentesco8_MEAN',
       'agg18_parentesco9_COUNT', 'agg18_parentesco9_MEAN',
       'fe_rent_per_person', 'fe_rent_per_room', 'fe_tablet_adult_density',
       'fe_tablet_density', 'r4h1_percent_in_male', 'r4m1_percent_in_female',
       'rent_per_adult', 'rez_esc_escolari', 'rez_esc_r4t1', 'rez_esc_age']# + ['parentesco_LE', 'rez_esc', 'rez_esc_r4t2', 'rez_esc_r4t3']

et_drop_cols.extend(["idhogar", "parentesco1"])


# In[ ]:


rf_params_1 = {"max_depth": None, "min_impurity_decrease": 1e-3, "min_samples_leaf": 2, "min_samples_split": 3, "n_estimators":750 }
rf_params_2 = {"max_depth": 25, "min_impurity_decrease": 1e-3, "min_samples_leaf": 2, "min_samples_split": 3, "n_estimators":700 }
rf_params_3 = {"max_depth": 31, "min_impurity_decrease": 1e-4, "min_samples_leaf": 3, "min_samples_split": 2, "n_estimators":850 }
rf_params_4 = {"max_depth": 32, "min_impurity_decrease": 1e-3, "min_samples_leaf": 2, "min_samples_split": 2, "n_estimators":950 }
rf_params_5 = {"max_depth": 29, "min_impurity_decrease": 1e-3, "min_samples_leaf": 2, "min_samples_split": 2, "n_estimators":800 }

rf_param_list = [rf_params_1, rf_params_2, rf_params_3, rf_params_4, rf_params_5]


# In[ ]:


# do the same thing for some extra trees classifiers
ets = []    
for i in range(10):
    rf_params = rf_param_list[i % len(rf_param_list)]
    rf = RandomForestClassifier(random_state=217+i, n_jobs=4, verbose=0, class_weight="balanced", **rf_params)
    ets.append(('rf{}'.format(i), rf))   

vc2 = VotingClassifier(ets, voting='soft')    
_ = vc2.fit(X_train.drop(et_drop_cols, axis=1), y_train)    


# In[ ]:


# all used non-null features + greater depth
vc2.voting = 'soft'
global_rf_score_soft = f1_score(y_test, vc2.predict(X_test.drop(et_drop_cols, axis=1)), average='macro')
vc2.voting = 'hard'
global_rf_score_hard = f1_score(y_test, vc2.predict(X_test.drop(et_drop_cols, axis=1)), average='macro')

print('Validation score of a VotingClassifier on RF with soft voting strategy: {:.4f}'.format(global_rf_score_soft))
print('Validation score of a VotingClassifier on RF with hard voting strategy: {:.4f}'.format(global_rf_score_hard))


# In[ ]:


# all used non-null features
vc2.voting = 'soft'
global_rf_score_soft = f1_score(y_test, vc2.predict(X_test.drop(et_drop_cols, axis=1)), average='macro')
vc2.voting = 'hard'
global_rf_score_hard = f1_score(y_test, vc2.predict(X_test.drop(et_drop_cols, axis=1)), average='macro')

print('Validation score of a VotingClassifier on RF with soft voting strategy: {:.4f}'.format(global_rf_score_soft))
print('Validation score of a VotingClassifier on RF with hard voting strategy: {:.4f}'.format(global_rf_score_hard))


# In[ ]:


# all non-null test features
vc2.voting = 'soft'
global_rf_score_soft = f1_score(y_test, vc2.predict(X_test.drop(et_drop_cols, axis=1)), average='macro')
vc2.voting = 'hard'
global_rf_score_hard = f1_score(y_test, vc2.predict(X_test.drop(et_drop_cols, axis=1)), average='macro')

print('Validation score of a VotingClassifier on RF with soft voting strategy: {:.4f}'.format(global_rf_score_soft))
print('Validation score of a VotingClassifier on RF with hard voting strategy: {:.4f}'.format(global_rf_score_hard))


# In[ ]:


# see which features are not used by ANY models
useless_features = []
drop_features = set()
counter = 0
for est in vc2.estimators_:
    ranked_features, unused_features = feature_importance(est, X_train.drop(et_drop_cols, axis=1), display_results=False)
    useless_features.append(unused_features)
    if counter == 0:
        drop_features = set(unused_features)
    else:
        drop_features = drop_features.intersection(set(unused_features))
    counter += 1
    
drop_features


# ### Combine The Voters

# In[ ]:


def combine_voters(data, weights=[0.3, 0.3, 0.4], voting="soft"):
    # do soft voting with both classifiers

    vc.voting="soft"
    vc1_probs = vc.predict_proba(data.drop(xgb_drop_cols, axis=1))
    vc2.voting="soft"
    vc2_probs = vc2.predict_proba(data.drop(et_drop_cols, axis=1))
    vc3.voting="soft"
    vc3_probs = vc3.predict_proba(data.drop(lgb_drop_cols, axis=1))

    final_vote = (vc1_probs * weights[0]) + (vc2_probs * weights[1]) + (vc3_probs * weights[2])
    predictions = np.argmax(final_vote, axis=1)
    
    if voting=="hard":
        vc.voting="hard"
        vc1_preds = vc.predict(data.drop(xgb_drop_cols, axis=1))
        vc2.voting="hard"
        vc2_preds = vc2.predict(data.drop(et_drop_cols, axis=1))
        vc3.voting="hard"
        vc3_preds = vc3.predict(data.drop(lgb_drop_cols, axis=1))
        
        # cstack the predictions, add the soft votes in, and get the mode
        predictions = np.array(stats.mode(np.column_stack([vc1_preds, vc2_preds, vc3_preds, predictions]), axis=1)[0])[:,0]
        
    return predictions


# In[ ]:


# equal weight
combo_preds = combine_voters(X_test, weights=[0.33, 0.34, 0.33])
global_combo_score_soft = f1_score(y_test, combo_preds, average='macro')
global_combo_score_soft


# In[ ]:


# hard voting
combo_preds = combine_voters(X_test, weights=[0.4, 0.2, 0.4], voting="hard")
global_combo_score_soft = f1_score(y_test, combo_preds, average='macro')
global_combo_score_soft


# In[ ]:


# xgb weighted higher
combo_preds = combine_voters(X_test, weights=[0.4, 0.3, 0.3])
global_combo_score_soft= f1_score(y_test, combo_preds, average='macro')
global_combo_score_soft


# In[ ]:


# rf weighted higher
combo_preds = combine_voters(X_test, weights=[0.3, 0.4, 0.3])
global_combo_score_soft = f1_score(y_test, combo_preds, average='macro')
global_combo_score_soft


# In[ ]:


# rf weighted higher
combo_preds = combine_voters(X_test, weights=[0.4, 0.5, 0.2])
global_combo_score_soft = f1_score(y_test, combo_preds, average='macro')
global_combo_score_soft


# In[ ]:


# lgb weighted higher
combo_preds = combine_voters(X_test, weights=[0.3, 0.3, 0.4])
global_combo_score_soft = f1_score(y_test, combo_preds, average='macro')
global_combo_score_soft


# # Prepare submission

# In[ ]:


y_subm = pd.DataFrame()
y_subm['Id'] = test_ids


# In[ ]:


vc.voting = 'soft'
y_subm_xgb = y_subm.copy(deep=True)
y_subm_xgb['Target'] = vc.predict(test.drop(xgb_drop_cols, axis=1)) + 1

vc2.voting = 'soft'
y_subm_rf = y_subm.copy(deep=True)
y_subm_rf['Target'] = vc2.predict(test.drop(et_drop_cols, axis=1)) + 1

vc3.voting = 'soft'
y_subm_lgb = y_subm.copy(deep=True)
y_subm_lgb['Target'] = vc3.predict(test.drop(lgb_drop_cols, axis=1)) + 1

y_subm_ens = y_subm.copy(deep=True)
y_subm_ens['Target'] = combine_voters(test) + 1


# In[ ]:


from datetime import datetime
now = datetime.now()

sub_file_xgb = 'submission_soft_XGB_{:.4f}_{}.csv'.format(global_score_soft, str(now.strftime('%Y-%m-%d-%H-%M')))
sub_file_lgb = 'submission_soft_LGB_{:.4f}_{}.csv'.format(global_lgb_score_soft, str(now.strftime('%Y-%m-%d-%H-%M')))
sub_file_rf = 'submission_soft_RF_{:.4f}_{}.csv'.format(global_rf_score_soft, str(now.strftime('%Y-%m-%d-%H-%M')))
sub_file_ens = 'submission_ens_{:.4f}_{}.csv'.format(global_combo_score_soft, str(now.strftime('%Y-%m-%d-%H-%M')))

y_subm_xgb.to_csv(sub_file_xgb, index=False)
y_subm_lgb.to_csv(sub_file_lgb, index=False)
y_subm_rf.to_csv(sub_file_rf, index=False)
y_subm_ens.to_csv(sub_file_ens, index=False)


# In[ ]:




