#!/usr/bin/env python
# coding: utf-8

# Some features taken from: https://www.kaggle.com/mlisovyi/feature-engineering-lighgbm-with-f1-macro by Misha Lisovyi
# 
# Second LGB model taken from: https://www.kaggle.com/opanichev/lgb-as-always by Oleg Panichev

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import datetime

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, RepeatedKFold, GroupKFold

import lightgbm as lgb
import gc

from sklearn.utils.class_weight import compute_sample_weight
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score


# In[ ]:


def feature_importance(forest, X_train):
    ranked_list = []
    
    importances = forest.feature_importances_

    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]) + " - " + X_train.columns[indices[f]])
        ranked_list.append(X_train.columns[indices[f]])
    
    return ranked_list

def evaluate_macroF1_lgb(truth, predictions):  
    # this follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483
    pred_labels = predictions.reshape(len(np.unique(truth)),-1).argmax(axis=0)
    f1 = f1_score(truth, pred_labels, average='macro')
    return ('macroF1', f1, True) 


# In[ ]:


data_path = "../input"
train = pd.read_csv(os.path.join(data_path, "train.csv"))
test = pd.read_csv(os.path.join(data_path, "test.csv"))


# ## Clean Up Data

# In[ ]:


# some dependencies are Na, fill those with the square root of the square
train['dependency'] = np.sqrt(train['SQBdependency'])
test['dependency'] = np.sqrt(test['SQBdependency'])

# change education to a number instead of a string, combine the two fields and drop the originals
train.loc[train['edjefa'] == "no", "edjefa"] = 0
train.loc[train['edjefa'] == "yes", "edjefa"] = 1
train['edjefa'] = train['edjefa'].astype("int")

train.loc[train['edjefe'] == "no", "edjefe"] = 0
train.loc[train['edjefe'] == "yes", "edjefe"] = 1
train['edjefe'] = train['edjefe'].astype("int")

test.loc[test['edjefa'] == "no", "edjefa"] = 0
test.loc[test['edjefa'] == "yes", "edjefa"] = 4
test['edjefa'] = test['edjefa'].astype("int")

test.loc[test['edjefe'] == "no", "edjefe"] = 0
test.loc[test['edjefe'] == "yes", "edjefe"] = 4
test['edjefe'] = test['edjefe'].astype("int")

train['edjef'] = np.max(train[['edjefa','edjefe']], axis=1)
test['edjef'] = np.max(test[['edjefa','edjefe']], axis=1)

# let's keep these features for now
# train.drop(["edjefe", "edjefa"], axis=1, inplace=True)
# test.drop(["edjefe", "edjefa"], axis=1, inplace=True)

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

# some rows indicate both that the household does and does not have a toilet, if there is no water we'll assume they do not
train.loc[(train.v14a ==  1) & (train.sanitario1 ==  1) & (train.abastaguano == 0), "v14a"] = 0
train.loc[(train.v14a ==  1) & (train.sanitario1 ==  1) & (train.abastaguano == 0), "sanitario1"] = 0

test.loc[(test.v14a ==  1) & (test.sanitario1 ==  1) & (test.abastaguano == 0), "v14a"] = 0
test.loc[(test.v14a ==  1) & (test.sanitario1 ==  1) & (test.abastaguano == 0), "sanitario1"] = 0


# In[ ]:


# this came from another kernel, some households have different targets for the same household,
# we should make the target for each household be the target for the head of that household
d={}
weird=[]
for row in train.iterrows():
    idhogar=row[1]['idhogar']
    target=row[1]['Target']
    if idhogar in d:
        if d[idhogar]!=target:
            weird.append(idhogar)
    else:
        d[idhogar]=target
        
for i in set(weird):
    hhold=train[train['idhogar']==i][['idhogar', 'parentesco1', 'Target']]
    target=hhold[hhold['parentesco1']==1]['Target'].tolist()[0]
    for row in hhold.iterrows():
        idx=row[0]
        if row[1]['parentesco1']!=1:
            train.at[idx, 'Target']=target        


# ## Some EDA

# In[ ]:


print("Unique Households:", train.idhogar.nunique())
print("Total Rows:", len(train))


# In[ ]:


train[['Id','idhogar', 'v2a1', 'hacdor', 'rooms', 'hacapo', 'v14a', 'refrig',
       'v18q', 'v18q1', 'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3',
       'r4t1', 'r4t2', 'r4t3', 'tamhog', 'tamviv', 'escolari', 'rez_esc',
       'hhsize', 'paredblolad', 'paredzocalo', 'paredpreb', 'pareddes',
       'paredmad', 'paredzinc', 'paredfibras', 'paredother', 'pisomoscer',
       'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene',
       'pisomadera', 'techozinc', 'techoentrepiso', 'techocane',
       'techootro', 'cielorazo', 'abastaguadentro', 'abastaguafuera',
       'abastaguano', 'public', 'planpri', 'noelec', 'coopele',
       'sanitario1', 'sanitario2', 'sanitario3', 'sanitario5',
       'sanitario6', 'energcocinar1', 'energcocinar2', 'energcocinar3',
       'energcocinar4', 'elimbasu1', 'elimbasu2', 'elimbasu3',
       'elimbasu4', 'elimbasu5', 'elimbasu6', 'epared1', 'epared2',
       'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2',
       'eviv3', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2',
       'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6',
       'estadocivil7', 'parentesco1', 'parentesco2', 'parentesco3',
       'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7',
       'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11',
       'parentesco12', 'hogar_nin', 'hogar_adul',
       'hogar_mayor', 'hogar_total', 'dependency', 
       'meaneduc', 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4',
       'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8',
       'instlevel9', 'bedrooms', 'overcrowding', 'tipovivi1', 'tipovivi2',
       'tipovivi3', 'tipovivi4', 'tipovivi5', 'computer', 'television',
       'mobilephone', 'qmobilephone', 'lugar1', 'lugar2', 'lugar3',
       'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'age',
       'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe',
       'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned',
       'agesq', 'Target']].sort_values("idhogar").head(20)


# The only features which vary within a household are related to the individuals, we will consolidate these and add some household level features.

# ### Add Some Features

# In[ ]:


# min/max education by household
train['max_esc'] = train.groupby('idhogar')['escolari'].transform('max')
train['min_esc'] = train.groupby('idhogar')['escolari'].transform('min')

test['max_esc'] = test.groupby('idhogar')['escolari'].transform('max')
test['min_esc'] = test.groupby('idhogar')['escolari'].transform('min')


# In[ ]:


# min/max/mean behind in education by household
train['max_rez'] = train.groupby('idhogar')['rez_esc'].transform('max')
train['min_rez'] = train.groupby('idhogar')['rez_esc'].transform('min')
train['mean_rez'] = train.groupby('idhogar')['rez_esc'].transform('mean')

test['max_rez'] = test.groupby('idhogar')['rez_esc'].transform('max')
test['min_rez'] = test.groupby('idhogar')['rez_esc'].transform('min')
test['mean_rez'] = test.groupby('idhogar')['rez_esc'].transform('mean')

# these features are added by a function, so there is no need to add them twice
# train['max_age'] = train.groupby('idhogar')['age'].transform('max')
# train['min_age'] = train.groupby('idhogar')['age'].transform('min')

# test['max_age'] = test.groupby('idhogar')['age'].transform('max')
# test['min_age'] = test.groupby('idhogar')['age'].transform('min')

# we'll init the feature to 0, then count the people over 18 in the household and use the max of that
train['num_over_18'] = 0
train['num_over_18'] = train[train.age >= 18].groupby('idhogar').transform("count")
train['num_over_18'] = train.groupby("idhogar")["num_over_18"].transform("max")
train['num_over_18'] = train['num_over_18'].fillna(0)

test['num_over_18'] = 0
test['num_over_18'] = test[test.age >= 18].groupby('idhogar').transform("count")
test['num_over_18'] = test.groupby("idhogar")["num_over_18"].transform("max")
test['num_over_18'] = test['num_over_18'].fillna(0)


# In[ ]:


def extract_features(df):
    df['bedrooms_to_rooms'] = df['bedrooms']/df['rooms']
    df['rent_to_rooms'] = df['v2a1']/df['rooms']
    df['tamhog_to_rooms'] = df['tamhog']/df['rooms'] # tamhog - size of the household
    df['r4t3_to_tamhog'] = df['r4t3']/df['tamhog'] # r4t3 - Total persons in the household
    df['r4t3_to_rooms'] = df['r4t3']/df['rooms'] # r4t3 - Total persons in the household
    df['v2a1_to_r4t3'] = df['v2a1']/df['r4t3'] # rent to people in household
    df['v2a1_to_r4t3'] = df['v2a1']/(df['r4t3'] - df['r4t1']) # rent to people under age 12
    df['hhsize_to_rooms'] = df['hhsize']/df['rooms'] # rooms per person
    df['rent_to_hhsize'] = df['v2a1']/df['hhsize'] # rent to household size
    df['rent_to_over_18'] = df['v2a1']/df['num_over_18']
    # some households have no one over 18, use the total rent for those
    df.loc[df.num_over_18 == 0, "rent_to_over_18"] = df[df.num_over_18 == 0].v2a1
    df['dependency_yes'] = df['dependency'].apply(lambda x: 1 if x == 'yes' else 0)
    df['dependency_no'] = df['dependency'].apply(lambda x: 1 if x == 'no' else 0)
    
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
                 #('', '', ''),
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

    return df    

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

def process_df(df_):
    # fix categorical features
#     encode_data(df_)
    #fill in missing values based on https://www.kaggle.com/mlisovyi/missing-values-in-the-data
    for f_ in ['v2a1', 'v18q1', 'meaneduc', 'SQBmeaned']:
        df_[f_] = df_[f_].fillna(0)
    df_['rez_esc'] = df_['rez_esc'].fillna(-1)
    # do feature engineering and drop useless columns
    return do_features(df_)


# In[ ]:


extract_features(train)
extract_features(test)

train = process_df(train)
test = process_df(test)


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
    #print(pd.get_dummies(train[cols_2_ohe], 
    #                                   columns=cols_2_ohe).head())
    #print(tmp_df.head())
    #print(tmp_df.groupby(['lugar_LE','idhogar']).mean().head())
    geo_agg = tmp_df.groupby(['lugar_LE','idhogar']).mean().groupby('lugar_LE').mean().astype(np.float32)
    geo_agg.columns = pd.Index(['geo_' + e for e in geo_agg.columns.tolist()])
    
    #print(gb.T)
    del tmp_df
    return df_.join(geo_agg, how='left', on='lugar_LE')

#train, test = train_test_apply_func(train, test, convert_geo2aggs)


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


train, test = train_test_apply_func(train, test, convert_OHE2LE)


# In[ ]:


print("train:", train.shape)
print("test:", test.shape)


# ### Split Data
# 
# We'll split the data by household to avoid leakage.

# In[ ]:


# only train on head of household records since apparently only they are scored?
# train2 = train.query('parentesco1==1')

train2 = train.copy()
target = train2.Target
train2 = train2.drop("Target", axis=1)

train_hhs = train2.idhogar

households = train2.idhogar.unique()
cv_hhs = np.random.choice(households, size=int(len(households) * 0.25), replace=False)

cv_idx = np.isin(train2.idhogar, cv_hhs)

# train2 = train2.dropna(axis=1, how='any')
# test = test.dropna(axis=1, how='any')

X_cv = train2[cv_idx]
y_cv = target[cv_idx]

X_tr = train2[~cv_idx]
y_tr = target[~cv_idx]

# train on entire dataset
X_tr = train2
y_tr = target


# In[ ]:


drop_cols = train.columns[train.isnull().any()]

X_tr.drop(drop_cols, axis=1, inplace=True)
X_cv.drop(drop_cols, axis=1, inplace=True)
test.drop(drop_cols, axis=1, inplace=True)


# In[ ]:


print("X_tr:", X_tr.shape)
print("X_cv:", X_cv.shape)
print("test:", test.shape)


# ### First Model with ExtraTrees

# In[ ]:


extra_drop_cols = []


# In[ ]:


# we got this list of columns by training on all columns and looking at the importances
extra_drop_cols = ['fe_people_weird_stat',
 'min_rez',
 'dependency_no',
 'rez_esc',
 'parentesco1',
 'parentesco_LE',
 'dependency_yes']


# In[ ]:


et_drop_cols = ['Id', 'idhogar'] + extra_drop_cols

et = ExtraTreesClassifier(n_estimators=250, max_depth=24, min_impurity_decrease=1e-10, min_samples_leaf=1, min_samples_split=2, 
            min_weight_fraction_leaf=0.0, n_jobs=-1, random_state=1, verbose=1)
et.fit(X_tr.drop(et_drop_cols, axis=1), y_tr)


# In[ ]:


et_cv_preds = et.predict(X_cv.drop(et_drop_cols, axis=1))
et_cv_probs = et.predict_proba(X_cv.drop(et_drop_cols, axis=1)) * 2

print("Accuracy:", et.score(X_cv.drop(et_drop_cols, axis=1), y_cv))
print("F1:", f1_score(y_cv, et_cv_preds, average="micro"))


# In[ ]:


# depth 24
print("Accuracy:", et.score(X_cv.drop(et_drop_cols, axis=1), y_cv))
print("F1:", f1_score(y_cv, et_cv_preds, average="micro"))


# In[ ]:


et_preds = et.predict(test.drop(et_drop_cols, axis=1))
et_probs = et.predict_proba(test.drop(et_drop_cols, axis=1)) * 2
model_count = 2


# In[ ]:


cv_submission = pd.DataFrame()
cv_submission['Id'] = X_cv.Id
cv_submission['Target_et'] = et_cv_preds


# In[ ]:


submission = pd.DataFrame()
submission['Id'] = test.Id
submission['Target_et'] = et_preds


# In[ ]:


submission[["Id", "Target_et"]].to_csv("20180725_etc_1.csv", header=["Id", "Target"], index=False)


# In[ ]:


ranked_features = feature_importance(et, X_tr.drop(et_drop_cols, axis=1))


# In[ ]:


extra_drop_cols = ranked_features[95:]
extra_drop_cols


# ### LGB

# In[ ]:


extra_lgb_drop_cols = ['techo_LE',
 'abastagua_LE',
 'hogar_total',
 'v18q',
 'mobilephone',
 'hacdor',
 'tamhog',
 'hacapo',
 'parentesco1'
]


# In[ ]:


lgb_drop_cols = ['Id', 'idhogar'] + extra_lgb_drop_cols
# use the full training set for our cv, and then train only on training set so we can validate
# train_all = lgb.Dataset(train[feature_names],signal)
train_data = lgb.Dataset(X_tr.drop(lgb_drop_cols, axis=1), y_tr-1)

max_depth = 15

params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'max_depth': max_depth,
    'num_leaves': 2**max_depth-1,
    'learning_rate': 0.05,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'verbose': -1,
    'num_threads': 4,
    'lambda_l2': 0.85,
    'min_gain_to_split': 0,
    'num_class': len(np.unique(y_tr)),
}
model = lgb.train(params, train_data, num_boost_round=500)


# In[ ]:


lgb_cv_probs = model.predict(X_cv.drop(lgb_drop_cols, axis=1))
lgb_cv_preds = np.argmax(lgb_cv_probs, axis=1) + 1

print("Accuracy:", accuracy_score(y_cv, lgb_cv_preds))
print("F1:", f1_score(y_cv, lgb_cv_preds, average="micro"))


# In[ ]:


# lambda 0.85
# lr 0.05
# ff 0.75
# bf 0.85
# depth 15
print("Accuracy:", accuracy_score(y_cv, lgb_cv_preds))
print("F1:", f1_score(y_cv, lgb_cv_preds, average="micro"))


# In[ ]:


lgb_probs = model.predict(test.drop(lgb_drop_cols, axis=1))
lgb_preds = np.argmax(lgb_probs, axis=1) + 1


# In[ ]:


# give this model a lot of weight
cv_probs = et_cv_probs + lgb_cv_probs + lgb_cv_probs + lgb_cv_probs
test_probs = et_probs + lgb_probs + lgb_probs + lgb_probs
model_count += 3


# In[ ]:


# since this is the best model we'll give it more weight in the vote by adding the predictions twice
submission['Target_lgb'] = lgb_preds
submission['Target_lgb4'] = lgb_preds
submission[["Id", "Target_lgb"]].to_csv("20180725_lgb_1.csv", header=["Id", "Target"], index=False)


# In[ ]:


# since this is the best model we'll give it more weight in the vote by adding the predictions twice
cv_submission['Target_lgb'] = lgb_cv_preds
cv_submission['Target_lgb4'] = lgb_cv_preds


# In[ ]:


importance = model.feature_importance()
model_fnames = model.feature_name()
tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
tuples = [x for x in tuples if x[1] > 0]
ranked_features = []

print('Important features:')
for i in range(len(model_fnames)):
    if i < len(tuples):
        print(i, tuples[i])
        ranked_features.append(tuples[i][0])
    else:
        break

del importance, model_fnames, tuples


# In[ ]:


extra_lgb_drop_cols = ranked_features[85:]
extra_lgb_drop_cols


# ### Random Forest

# In[ ]:


extra_rf_drop_cols = []


# In[ ]:


extra_rf_drop_cols = ['r4t3_to_tamhog', 'min_rez', 'dependency_no', 'dependency_yes']


# In[ ]:


rf_drop_cols = ['Id', 'idhogar', 'parentesco1'] + extra_rf_drop_cols

rf = RandomForestClassifier(n_estimators=350, max_depth=31, verbose=1, min_impurity_decrease=1e-6, n_jobs=-1)
rf.fit(X_tr.drop(rf_drop_cols, axis=1), y_tr)


# In[ ]:


rf_cv_preds = rf.predict(X_cv.drop(rf_drop_cols, axis=1))
rf_cv_probs = rf.predict_proba(X_cv.drop(rf_drop_cols, axis=1))

print("Accuracy:", accuracy_score(y_cv, rf_cv_preds))
print("F1:", f1_score(y_cv, rf_cv_preds, average="micro"))


# In[ ]:


# 95 cols
print("Accuracy:", accuracy_score(y_cv, rf_cv_preds))
print("F1:", f1_score(y_cv, rf_cv_preds, average="micro"))


# In[ ]:


rf_preds = rf.predict(test.drop(rf_drop_cols, axis=1))
rf_probs = rf.predict_proba(test.drop(rf_drop_cols, axis=1))


# In[ ]:


# weight this model more
cv_probs = cv_probs + rf_cv_probs
test_probs = test_probs + rf_probs
model_count += 1


# In[ ]:


submission['Target_rf'] = rf_preds
submission[["Id", "Target_rf"]].to_csv("20180725_rf_1.csv", header=["Id", "Target"], index=False)


# In[ ]:


cv_submission['Target_rf'] = rf_cv_preds


# In[ ]:


rf_features = feature_importance(rf, X_tr.drop(rf_drop_cols, axis=1))


# In[ ]:


extra_rf_drop_cols = rf_features[96:]
extra_rf_drop_cols


# ### LGB 2
# 
# Originally from https://www.kaggle.com/opanichev/lgb-as-always by Oleg Panichev

# In[ ]:


def dprint(*args, **kwargs):
    print("[{}] ".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) +         " ".join(map(str,args)), **kwargs)

id_name = 'Id'
target_name = 'Target'

df_all = pd.concat([train, test], axis=0)
cols = [f_ for f_ in df_all.columns if df_all[f_].dtype == 'object' and f_ != id_name]

for c in cols:
    le = LabelEncoder()
    le.fit(df_all[c].astype(str))
    train[c] = le.transform(train[c].astype(str))
    test[c] = le.transform(test[c].astype(str))

gc.collect()
print("Done.")

# Build the model
cnt = 0
p_buf = []
n_splits = 5
n_repeats = 1
kf = RepeatedKFold(
    n_splits=n_splits, 
    n_repeats=n_repeats, 
    random_state=0)
err_buf = []   

cols_to_drop = [
    id_name, 
    target_name,
    'idhogar',
]

X = train2.drop(cols_to_drop, axis=1, errors='ignore')
feature_names = list(X.columns)

X = X.fillna(0)
X = X.values
y = target

classes = np.unique(y)
dprint('Number of classes: {}'.format(len(classes)))
c2i = {}
i2c = {}
for i, c in enumerate(classes):
    c2i[c] = i
    i2c[i] = c

y_le = np.array([c2i[c] for c in y])

X_test = test.drop(cols_to_drop, axis=1, errors='ignore')
X_test = X_test.fillna(0)
X_test = X_test.values
id_test = test[id_name].values

dprint(X.shape, y.shape)
dprint(X_test.shape)

n_features = X.shape[1]

max_depth = 12

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'max_depth': max_depth,
    'num_leaves': (2**max_depth)-1,
    'learning_rate': 0.05,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'bagging_freq': 5,
    'verbose': -1,
    'num_threads': 4,
    'lambda_l2': 0.85,
    'min_gain_to_split': 0,
    'num_class': len(np.unique(y)),
}

for i in range(5):
    cv_hhs = np.random.choice(households, size=int(len(households) * 0.25), replace=False)

    valid_index = np.isin(train_hhs, cv_hhs)
    train_index = ~np.isin(train_hhs, cv_hhs)
    
    print('Fold {}/{}*{}'.format(cnt + 1, n_splits, n_repeats))
    params = lgb_params.copy() 

    sampler = ADASYN(random_state=0)
    X_train, y_train = sampler.fit_sample(X[train_index], y_le[train_index])

    lgb_train = lgb.Dataset(
        X_train, 
        y_train, 
        feature_name=feature_names,
        )
    lgb_train.raw_data = None

    lgb_valid = lgb.Dataset(
        X[valid_index], 
        y_le[valid_index],
        feature_name=feature_names,
        )
    lgb_valid.raw_data = None

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=99999,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=200, 
        verbose_eval=100, 
    )

    if cnt == 0:
        importance = model.feature_importance()
        model_fnames = model.feature_name()
        tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
        tuples = [x for x in tuples if x[1] > 0]
        print('Important features:')
        for i in range(10):
            if i < len(tuples):
                print(i, tuples[i])
            else:
                break

        del importance, model_fnames, tuples

    p = model.predict(X[valid_index], num_iteration=model.best_iteration)
    
    err = f1_score(y_le[valid_index], np.argmax(p, axis=1), average='micro')

    dprint('{} F1: {}'.format(cnt + 1, err))

    p = model.predict(X_test, num_iteration=model.best_iteration)
    
    if len(p_buf) == 0:
        p_buf = np.array(p, dtype=np.float16)
    else:
        p_buf += np.array(p, dtype=np.float16)
    err_buf.append(err)

    cnt += 1

    del model, lgb_train, lgb_valid, p
    gc.collect()


err_mean = np.mean(err_buf)
err_std = np.std(err_buf)
print('F1 = {:.6f} +/- {:.6f}'.format(err_mean, err_std))

preds_lgb2 = p_buf/cnt

# Prepare probas
subm = pd.DataFrame()
subm[id_name] = id_test
for i in range(preds_lgb2.shape[1]):
    subm[i2c[i]] = preds_lgb2[:, i]

# Prepare submission
submission["Target_lgb2"] = [i2c[np.argmax(p)] for p in preds_lgb2]
submission[["Id", "Target_lgb2"]].to_csv("20180725_lgb2_preds.csv", header=["Id", "Target"], index=False)


# In[ ]:


test_probs = test_probs + preds_lgb2
model_count += 1


# ### One More LGB
# 
# Taken from https://www.kaggle.com/mlisovyi/feature-engineering-lighgbm-with-f1-macro by Misha Lisovyi, with some slight modifications.

# In[ ]:


lgb_drop_cols = ['Id', 'idhogar']
opt_parameters = {'colsample_bytree': 0.93, 'min_child_samples': 56, 'num_leaves': 19, 'subsample': 0.84}


# In[ ]:


def evaluate_macroF1_lgb(truth, predictions):  
    # this follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483
    pred_labels = predictions.reshape(len(np.unique(truth)),-1).argmax(axis=0)
    f1 = f1_score(truth, pred_labels, average='macro')
    return ('macroF1', f1, True) 

fit_params={"early_stopping_rounds":300, 
            "eval_metric" : evaluate_macroF1_lgb, 
            "eval_set" : [(X_tr.drop(lgb_drop_cols, axis=1),y_tr-1), (X_cv.drop(lgb_drop_cols, axis=1),y_cv-1)],
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


from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier

def _parallel_fit_estimator(estimator, X, y, sample_weight=None, **fit_params):
    """Private function used to fit an estimator within a job."""
    if sample_weight is not None:
        estimator.fit(X, y, sample_weight=sample_weight, **fit_params)
    else:
        estimator.fit(X, y, **fit_params)
    return estimator

class VotingClassifierLGBM(VotingClassifier):
    '''
    This implements the fit method of the VotingClassifier propagating fit_params
    '''
    def fit(self, X, y, sample_weight=None, **fit_params):
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

        if sample_weight is not None:
            for name, step in self.estimators:
                if not has_fit_parameter(step, 'sample_weight'):
                    raise ValueError('Underlying estimator \'%s\' does not'
                                     ' support sample weights.' % name)
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
                delayed(_parallel_fit_estimator)(clone(clf), X, transformed_y,
                                                 sample_weight=sample_weight, **fit_params)
                for clf in clfs if clf is not None)

        return self


# In[ ]:


clfs = []
for i in range(10):
    clf = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                             random_state=314+i, silent=True, metric='None', 
                             n_jobs=4, n_estimators=5000, class_weight='balanced')
    clf.set_params(**opt_parameters)
    clfs.append(('lgbm{}'.format(i), clf))
    
vc = VotingClassifierLGBM(clfs, voting='soft')
del clfs
#Train the final model with learning rate decay
_ = vc.fit(X_tr.drop(lgb_drop_cols, axis=1), y_tr-1, **fit_params)

clf_final = vc.estimators_[0]


# In[ ]:


global_score = f1_score(y_cv-1, clf_final.predict(X_cv.drop(lgb_drop_cols, axis=1)), average='macro')
vc.voting = 'soft'
global_score_soft = f1_score(y_cv-1, vc.predict(X_cv.drop(lgb_drop_cols, axis=1)), average='macro')

print('Validation score of a single LGBM Classifier: {:.4f}'.format(global_score))
print('Validation score of a VotingClassifier on 3 LGBMs with soft voting strategy: {:.4f}'.format(global_score_soft))


# In[ ]:


lgb_cv_probs2 = clf_final.predict_proba(X_cv.drop(lgb_drop_cols, axis=1))

lgb_probs2 = clf_final.predict_proba(test.drop(lgb_drop_cols, axis=1))
lgb_preds2 = np.argmax(lgb_probs2, axis=1) + 1
submission['Target_lgb3'] = lgb_preds2


# In[ ]:


# weight this model more
cv_probs = cv_probs + lgb_cv_probs2 + lgb_cv_probs2
test_probs = test_probs + lgb_probs2 + lgb_probs2
model_count += 3


# In[ ]:


submission[["Id", "Target_lgb3"]].to_csv("20180725_lgb3_preds.csv", header=["Id", "Target"], index=False)


# ### Average Probabilities and Make Predictions

# In[ ]:


cv_probs = cv_probs / model_count
test_probs = test_probs / model_count


# In[ ]:


cv_predictions = np.argmax(cv_probs, axis=1) + 1
test_predictions = np.argmax(test_probs, axis=1) + 1


# In[ ]:


print("Accuracy:", accuracy_score(y_cv, cv_predictions))
print("F1:", f1_score(y_cv, cv_predictions, average="micro"))


# In[ ]:


submission['Avg_probs'] = test_predictions
submission[["Id", "Avg_probs"]].to_csv("20180725_avg_preds.csv", header=["Id", "Target"], index=False)


# In[ ]:




