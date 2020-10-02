#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.


# ## Initialize

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
import gc


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'png')
pd.options.display.float_format = '{:.2f}'.format
rc={'savefig.dpi': 75, 'figure.autolayout': False, 'figure.figsize': [12, 8], 'axes.labelsize': 18,   'axes.titlesize': 18, 'font.size': 18, 'lines.linewidth': 2.0, 'lines.markersize': 8, 'legend.fontsize': 16,   'xtick.labelsize': 16, 'ytick.labelsize': 16}

sns.set(style='dark',rc=rc)


# In[ ]:


default_color = '#56B4E9'
colormap = plt.cm.cool


# In[ ]:


# Setting working directory

path = '../input/'
path_result = '../output/'


# ## Loading files

# In[ ]:


train = pd.read_csv(path + 'train_data.csv')
test = pd.read_csv(path + 'teste_data.csv')
train = train.rename(columns={"default": "target", "ids":"id"})
test = test.rename(columns={"ids":"id"})


# In[ ]:


train.shape


# In[ ]:


test.head()


# In[ ]:


test.shape


# In[ ]:


test_id = test.id


# In[ ]:


train.head()


# In[ ]:


train.describe()


# train.dtypes

# ## Missing Values

# In[ ]:


def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns


# In[ ]:


missing_values_table(train)


# Ops! Target has missing values!
# 
# **For next versions**: iterate predictions: predict target missing values, model again...

# Missing correlation heat map before droping missing values in target.

# In[ ]:


missingValueColumns = train.columns[train.isnull().any()].tolist()
df_null = train[missingValueColumns]


# In[ ]:


msno.bar(df_null,figsize=(20,8),color=default_color,fontsize=18,labels=True)


# In[ ]:


msno.heatmap(df_null,figsize=(20,8),cmap=colormap)


# In[ ]:


msno.dendrogram(df_null,figsize=(20,8))


# In[ ]:


sorted_data = msno.nullity_sort(df_null, sort='descending') # or sort='ascending'
msno.matrix(sorted_data,figsize=(20,8),fontsize=14)


# In[ ]:


train = train.dropna(subset=['target'])


# In[ ]:


missingValueColumns = train.columns[train.isnull().any()].tolist()
df_null = train[missingValueColumns]


# In[ ]:


msno.bar(df_null,figsize=(20,8),color=default_color,fontsize=18,labels=True)


# In[ ]:


msno.heatmap(df_null,figsize=(20,8),cmap=colormap)


# In[ ]:


msno.dendrogram(df_null,figsize=(20,8))


# In[ ]:


sorted_data = msno.nullity_sort(df_null, sort='descending') # or sort='ascending'
msno.matrix(sorted_data,figsize=(20,8),fontsize=14)


# 

# In[ ]:





# ## Target Analysis

# In[ ]:


plt.figure(figsize=(15,5))

ax = sns.countplot('target',data=train,color=default_color)
for p in ax.patches:
    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(train['target'])), (p.get_x()+ 0.3, p.get_height()+0.2))


# ## Data Analysis

# In[ ]:


def get_meta(train):
    data = []
    for col in train.columns:
        # Defining the role
        if col == 'target':
            role = 'target'
        elif col == 'id':
            role = 'id'
        else:
            role = 'input'

        # Defining the level
        if col == 'target' or 'facebook' in col or col == 'gender' or 'bin_' in col:
            level = 'binary'
        elif train[col].dtype == np.object or col == 'id':
            level = 'nominal'
        elif train[col].dtype == np.float64:
            level = 'interval'
        elif train[col].dtype == np.int64:
            level = 'ordinal'

        # Initialize keep to True for all variables except for id
        keep = True
        if col == 'id':
            keep = False

        # Defining the data type 
        dtype = train[col].dtype

        # Creating a Dict that contains all the metadata for the variable
        col_dict = {
            'varname': col,
            'role'   : role,
            'level'  : level,
            'keep'   : keep,
            'dtype'  : dtype
        }
        data.append(col_dict)
    meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
    meta.set_index('varname', inplace=True)
    return meta


# In[ ]:


meta_data = get_meta(train)
meta_data


# In[ ]:


meta_counts = meta_data.groupby(['role', 'level']).agg({'dtype': lambda x: x.count()}).reset_index()
meta_counts


# In[ ]:


fig,ax = plt.subplots()
fig.set_size_inches(20,5)
sns.barplot(data=meta_counts[(meta_counts.role != 'target') & (meta_counts.role != 'id') ],x="level",y="dtype",ax=ax,color=default_color)
ax.set(xlabel='Variable Type', ylabel='Count',title="Variables Count Across Datatype")


# In[ ]:


col_ordinal   = meta_data[(meta_data.level == 'ordinal') & (meta_data.keep)].index
col_nominal   = meta_data[(meta_data.level == 'nominal') & (meta_data.keep)& (meta_data.role != 'target')& (meta_data.role != 'id')].index
col_interval = meta_data[(meta_data.level == 'interval') & (meta_data.keep)].index
col_binary    = meta_data[(meta_data.level == 'binary') & (meta_data.keep) & (meta_data.role != 'target')].index


# # Input variables analysis

# ## Categorical features analysis

# In[ ]:


list(col_nominal)


# %matplotlib inline
# import seaborn as sns
# import matplotlib.pyplot as plt
# 
# for i in list(col_nominal):
# 
#     carrier_count = train[i].value_counts()
#     sns.set(style="darkgrid")
#     sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
#     plt.title('Frequency Distribution of '+i)
#     plt.ylabel('Number of Occurrences', fontsize=12)
#     plt.xlabel(i, fontsize=12)
#     plt.show()

# ## Count label encoding

# In[ ]:


def new_missing_columns(df):
    for i in missingValueColumns:
        if 'target' not in i:
            new_col = 'bin_missing_'+ i
            df[new_col] = np.where(df[i].isnull(), True, False)
    return df


# In[ ]:


train = new_missing_columns(train)
test = new_missing_columns(test)


# In[ ]:


def count_label_encoding(train, test,col):
    for i in col:
        df1 = train[i].value_counts().reset_index(name='freq_'+ i).rename(columns={'index': 'lc_'+ i})
        train = pd.merge(train,df1,left_on=i, right_on='lc_'+ i, how='left')
        test = pd.merge(test,df1,left_on=i, right_on='lc_'+ i, how='left')
        
    for i in list(train):
        if 'lc_' in i:
            train = train.drop(i, axis = 1)
            test = test.drop(i, axis = 1)
    return train, test


# In[ ]:


train, test = count_label_encoding(train, test,col_nominal)
train, test = count_label_encoding(train, test,col_binary)


# meta_data = get_meta(train)

# In[ ]:


meta_data = get_meta(train)
col_ordinal   = meta_data[(meta_data.level == 'ordinal') & (meta_data.keep)& (meta_data.role != 'target')].index
col_nominal   = meta_data[(meta_data.level == 'nominal') & (meta_data.keep)& (meta_data.role != 'target')].index
col_interval = meta_data[(meta_data.level == 'interval') & (meta_data.keep)& (meta_data.role != 'target')].index
col_binary    = meta_data[(meta_data.level == 'binary') & (meta_data.keep) & (meta_data.role != 'target')].index


# In[ ]:


meta_counts = meta_data.groupby(['role', 'level']).agg({'dtype': lambda x: x.count()}).reset_index()
meta_counts


# In[ ]:


fig,ax = plt.subplots()
fig.set_size_inches(20,5)
sns.barplot(data=meta_counts[(meta_counts.role != 'target') & (meta_counts.role != 'id') ],x="level",y="dtype",ax=ax,color=default_color)
ax.set(xlabel='Variable Type', ylabel='Count',title="Variables Count Across Datatype")


# ## Continuous features analysis

# In[ ]:


plt.figure(figsize=(18,16))
plt.title('Pearson correlation of continuous features', y=1.05, size=15)
sns.heatmap(train[col_interval].corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True, fmt = '.2f')


# train.fillna(-1, inplace=True) 
# test.fillna(-1, inplace=True)
# #I should improve this...

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train.head()


# In[ ]:


X = pd.concat([train[col_interval],train[col_ordinal],pd.get_dummies(train[col_binary])], axis=1)
y = pd.DataFrame(train.target)
X.fillna(-1, inplace=True) 
y.fillna(-1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


y.shape


# In[ ]:


X.shape


# In[ ]:


X.head()


# In[ ]:


plt.figure(figsize=(18,16))
plt.title('Pearson correlation of continuous features', y=1.05, size=15)
sns.heatmap(X.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=False, fmt = '.1f')


# # Testing feature importance with random forests

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=30, max_features=0.2, n_jobs=-1, random_state=0)
rf.fit(X_train, y_train['target'])
features = X_train.columns.values
print("----- Training Done -----")


# In[ ]:


from sklearn.metrics import accuracy_score, roc_auc_score


# In[ ]:


acc = accuracy_score(y_train, rf.predict(X_train))
auc = roc_auc_score(y_train, rf.predict(X_train))
print("Accuracy: %.4f" % acc)
print("AUC: %.4f" % auc)


# In[ ]:


acc = accuracy_score(y_test, rf.predict(X_test))
auc = roc_auc_score(y_test, rf.predict(X_test))
print("Accuracy: %.4f" % acc)
print("AUC: %.4f" % auc)


# In[ ]:


def get_feature_importance_df(feature_importances, 
                              column_names, 
                              top_n=25):
    """Get feature importance data frame.
 
    Parameters
    ----------
    feature_importances : numpy ndarray
        Feature importances computed by an ensemble 
            model like random forest or boosting
    column_names : array-like
        Names of the columns in the same order as feature 
            importances
    top_n : integer
        Number of top features
 
    Returns
    -------
    df : a Pandas data frame
 
    """
     
    imp_dict = dict(zip(column_names, 
                        feature_importances))
    top_features = sorted(imp_dict, 
                          key=imp_dict.get, 
                          reverse=True)[0:top_n]
    top_importances = [imp_dict[feature] for feature 
                          in top_features]
    df = pd.DataFrame(data={'feature': top_features, 
                            'importance': top_importances})
    return df


# In[ ]:


feature_importance = get_feature_importance_df(rf.feature_importances_, features)
feature_importance


# In[ ]:


fig,ax = plt.subplots()
fig.set_size_inches(20,10)
sns.barplot(data=feature_importance[:10],x="feature",y="importance",ax=ax,color=default_color,)
ax.set(xlabel='Variable name', ylabel='Importance',title="Variable importances")


# # Baseline Models

# In[ ]:





# In[ ]:


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


def cross_val_model(X,y, model, n_splits=3):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_score
    X = np.array(X)
    y = np.array(y)
    

    folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2017).split(X, y))

    for j, (train_idx, test_idx) in enumerate(folds):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_holdout = X[test_idx]
        y_holdout = y[test_idx]

        print ("Fit %s fold %d" % (str(model).split('(')[0], j+1))
        m = model.fit(X_train, y_train)
        cross_score = cross_val_score(model, X_holdout, y_holdout, cv=3, scoring='roc_auc')
        print("    cross_score: %.5f" % cross_score.mean())
    return m


# ## Random Forest Model

# In[ ]:


#RandomForest params
rf_params = {}
rf_params['n_estimators'] = 200
rf_params['max_depth'] = 6
rf_params['min_samples_split'] = 70
rf_params['min_samples_leaf'] = 30


# In[ ]:


rf_model = RandomForestClassifier(**rf_params, random_state=29,n_jobs = -1)


# In[ ]:


cross_val_model(X_train, y_train['target'], rf_model)


# ## XGBoost Model

# In[ ]:


# XGBoost params
xgb_params = {}
xgb_params['learning_rate'] = 0.02
xgb_params['n_estimators'] = 1000
xgb_params['max_depth'] = 4
xgb_params['subsample'] = 0.9
xgb_params['colsample_bytree'] = 0.9


# In[ ]:


XGB_model = XGBClassifier(**rf_params, random_state=29,n_jobs=-1)


# In[ ]:


cross_val_model(X_train, y_train['target'], XGB_model)


# ## Extra features effect

# In[ ]:


train_ext = train.copy()
test_ext = test.copy()
#train_ext.fillna(-1,inplace = True)
missing_values_table(train_ext)


# In[ ]:


train.head()


# In[ ]:


def create_extra_features(train_ext):
    train_ext['null_sum'] = train_ext[train_ext==-1].count(axis=1)
    #train_ext['bin_sum']  = train_ext[col_binary].sum(axis=1)
    train_ext['ord_sum']  = train_ext[col_ordinal].sum(axis=1)
    train_ext['interval_median']  = train_ext[col_interval].sum(axis=1)
    train_ext['new_amount_borrowed_by_income']  = train_ext['amount_borrowed']/train_ext['income']
    train_ext['new_amount_borrowed_by_months']  = train_ext['amount_borrowed']/train_ext['borrowed_in_months']
    return train_ext


# get_meta(train_ext)

# In[ ]:


train_ext = create_extra_features(train_ext)
test_ext = create_extra_features(test_ext)

train_backup = train_ext.copy()
test_backup = test_ext.copy()


# In[ ]:


meta_data = get_meta(train_ext)
col_ordinal   = meta_data[(meta_data.level == 'ordinal') & (meta_data.keep)& (meta_data.role != 'target')].index
col_nominal   = meta_data[(meta_data.level == 'nominal') & (meta_data.keep)& (meta_data.role != 'target')].index
col_interval = meta_data[(meta_data.level == 'interval') & (meta_data.keep)& (meta_data.role != 'target')].index
col_binary    = meta_data[(meta_data.level == 'binary') & (meta_data.keep) & (meta_data.role != 'target')].index
#meta_data


# In[ ]:


meta_counts = meta_data.groupby(['role', 'level']).agg({'dtype': lambda x: x.count()}).reset_index()
meta_counts


# In[ ]:


ids_targets = meta_data[meta_data['role'] != 'input'].index


# In[ ]:


train_ext.head()


# In[ ]:


test_ext.head()


# In[ ]:


train_ext.fillna(-1, inplace = True)

X_ext = pd.concat([train_ext[col_interval],train_ext[col_ordinal], pd.get_dummies(train_ext[col_binary])], axis=1)
X_ext.head()


# In[ ]:


X_ext = X_ext.drop(columns = ['facebook_profile_False','gender_f'], axis=1)


# In[ ]:


test_ext = pd.concat([test_ext[col_interval],test_ext[col_ordinal], pd.get_dummies(test_ext[col_binary])], axis=1)
test_ext.fillna(-1, inplace = True)
#X_ext = X_ext.drop(columns = ids_targets, axis =1)
y_ext = pd.DataFrame(train_ext.target) #train_lc.target_default.ravel(order='K') #pd.DataFrame(train_ext.target)
y_ext=y_ext.astype('bool')
y_ext = y_ext.values
y_ext = y_ext.reshape(-1)


# In[ ]:


test_ext['gender_-1'] = 0
test_ext['facebook_profile_-1'] = 0
test_ext=test_ext.drop(columns = ['facebook_profile_False', 'gender_f'], axis = 1)
test_ext.head()


# In[ ]:


cols = list(X_ext)
test_ext = test_ext[cols]


# In[ ]:


X_ext.head()


# In[ ]:


test_ext.head()


# In[ ]:


from sklearn.utils.multiclass import type_of_target
type_of_target(y_ext)


# from sklearn import preprocessing

# lb = preprocessing.LabelBinarizer()
# y_ext = lb.fit_transform(y_ext)

# In[ ]:


X_ext.shape


# In[ ]:


test_ext.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_ext, y_ext, test_size=0.2, random_state=42)


# In[ ]:


X_ext.head()


# In[ ]:


test_ext.head()


# Question: if test_ext and X_ext have different columns, why didn't the code break? 

# In[ ]:


cross_val_model(X_ext, y_ext, rf_model)


# In[ ]:


cross_val_model(X_ext, y_ext, XGB_model)


# In[ ]:


gc.collect


# ## Tuning Parameters

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


tuned_parameters = [{'max_depth': [4,5,6,7,8,9,10],
                     'max_features': [4,5,6,7,8,9,10],
                    'n_estimators':[10,25,50,75]}]

clf = GridSearchCV(RandomForestClassifier(random_state=29), tuned_parameters, cv=3, scoring='roc_auc')
clf.fit(X_train, y_train)


# In[ ]:


print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
        % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
from sklearn.metrics import accuracy_score, roc_auc_score
acc = accuracy_score(y_test, clf.predict(X_test))
auc = roc_auc_score(y_test, clf.predict(X_test))
print("Accuracy: %.4f" % acc)
print()
print("AUC: %.4f" % auc)
print()


# ## Parameter optimization

# In[ ]:





# In[ ]:


from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


# In[ ]:


import random
import itertools
N_HYPEROPT_PROBES = 10
EARLY_STOPPING = 80
HOLDOUT_SEED = 123456
HOLDOUT_SIZE = 0.10
HYPEROPT_ALGO = tpe.suggest  #  tpe.suggest OR hyperopt.rand.suggest
DATASET = 'clean' # 'raw' | 'clean' | 'extended'
SEED0 = random.randint(1,1000000000)
NB_CV_FOLDS = 5


# In[ ]:


obj_call_count = 0
cur_best_score = 0


# ### Random Forest

# In[ ]:


space_RF ={
    'n_estimators'           : hp.choice('n_estimators',         np.arange(10, 200,  dtype=int)),       
    'max_depth'              : hp.choice("max_depth",            np.arange(3, 15,    dtype=int)),
    'min_samples_split'      : hp.choice("min_samples_split",    np.arange(20, 100,    dtype=int)),
    'min_samples_leaf'       : hp.choice("min_samples_leaf",    np.arange(10, 100,    dtype=int)),
    'criterion'              : hp.choice('criterion', ["gini", "entropy"]),
    'class_weight'           : hp.choice('class_weight', ['balanced_subsample', None]),
    'n_jobs'                 : -1,
    'oob_score'              : True,
    'random_state'           :  hp.randint('random_state',2000000)
   }
#{'class_weight': 1, 'criterion': 1, 'max_depth': 9, 'min_samples_leaf': 74, 'min_samples_split': 12, 'n_estimators': 134, 'random_state': 1433254}
#Params: class_weight=balanced_subsample criterion=entropy max_depth=11 min_samples_leaf=2 min_samples_split=29 n_estimators=89 n_jobs=-1 oob_score=True
#Params: class_weight=balanced_subsample criterion=entropy max_depth=10 min_samples_leaf=2 min_samples_split=17 n_estimators=38 n_jobs=-1 oob_score=True


# In[ ]:


def objective_RF(space):
    import uuid
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    
    global obj_call_count, cur_best_score, X_train, y_train, test, X_test, y_test, test_id

    
    obj_call_count += 1
    print('\nLightGBM objective call #{} cur_best_score={:7.5f}'.format(obj_call_count,cur_best_score) )

    sorted_params = sorted(space.items(), key=lambda z: z[0])
    print('Params:', str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params if not k.startswith('column:')]))


    params = sample(space)
        
    mdl = RandomForestClassifier(**params)
    
    cv_score = cross_val_score(mdl, X_train, y_train).mean()

    print( 'CV finished ; \n cv_score={:7.5f}'.format(cv_score ) )
    
    _model = mdl.fit(X_train, y_train)
    
    predictions = _model.predict_proba(X_test)[:,1]#(X_test)
    
    score = roc_auc_score(y_test, predictions)
    print('valid score={}'.format(score))
    
    
    do_submit = score > 0.64

    if score > cur_best_score:
        cur_best_score = score
        print('NEW BEST SCORE={}'.format(cur_best_score))
        do_submit = True

    if do_submit:
        submit_guid = uuid.uuid4()

        print('Compute submissions guid={}'.format(submit_guid))

        y_submission = _model.predict_proba(test_ext)[:,1] #, num_iteration=n_rounds)
        submission_filename = 'rf_score_{:13.11f}_submission_guid_{}.csv'.format(score,submit_guid)
        pd.DataFrame(
        {'ids':test_id, 'prob':y_submission}
        ).to_csv(submission_filename, index=False)
       
    loss = 1 - score
    return {'loss': loss, 'status': STATUS_OK}


# In[ ]:


trials = Trials()
best = fmin(fn=objective_RF,
                     space=space_RF,
                     algo=HYPEROPT_ALGO,
                     max_evals=N_HYPEROPT_PROBES,
                     trials=trials,
                     verbose=1)

print('-'*50)
print('The best params for RF:')
print( best )
print('\n\n')


# ## XGBoost

# In[ ]:


space_XGB ={
    'max_depth'        : hp.choice("max_depth", np.arange(5, 15,dtype=int)), 
    'learning_rate'    : hp.loguniform('learning_rate', -4.9, -3.0),
    'n_estimators'     : hp.choice('n_estimators', np.arange(10, 100,dtype=int)),
    'objective'        : 'binary:logistic',
    'booster'          : 'gbtree',       
    'reg_alpha'        :  hp.uniform('reg_alpha', 1e-5, 1e-1),
    'reg_lambda'       :  hp.uniform('reg_lambda', 1e-5, 1e-1), 
    'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 0.8),
    'min_child_weight ': hp.uniform('min_child_weight', 0.5, 0.8),   
    'random_state'     :  hp.randint('random_state',2000000)
   }


# In[ ]:


def objective_XGB(space):
    import uuid
    from sklearn.model_selection import cross_val_score
    from xgboost import XGBClassifier
   
    global obj_call_count, cur_best_score, X_train, y_train, test, X_test, y_test, test_id

    
    obj_call_count += 1
    print('\nLightGBM objective call #{} cur_best_score={:7.5f}'.format(obj_call_count,cur_best_score) )

    sorted_params = sorted(space.items(), key=lambda z: z[0])
    print('Params:', str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params if not k.startswith('column:')]))


    params = sample(space)
        
    mdl = XGBClassifier(**params)
    
    cv_score = cross_val_score(mdl, X_train, y_train).mean()

    
    print( 'CV finished ; \n cv_score={:7.5f}'.format(cv_score ) )
    
    _model = mdl.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='auc',verbose=True,early_stopping_rounds =30)
    
    params.update({'n_estimators': _model.best_iteration})
    
    predictions = _model.predict_proba(X_test)[:,1]#(X_test)
    
    score = roc_auc_score(y_test, predictions)
    print('valid score={}'.format(score))
    
    
    do_submit = score > 0.64

    if score > cur_best_score:
        cur_best_score = score
        print('NEW BEST SCORE={}'.format(cur_best_score))
        do_submit = True

    if do_submit:
        submit_guid = uuid.uuid4()

        print('Compute submissions guid={}'.format(submit_guid))

        y_submission = _model.predict_proba(test_ext)[:,1]#, num_iteration=n_rounds)
        submission_filename = 'xgb_score_{:13.11f}_submission_guid_{}.csv'.format(score,submit_guid)
        pd.DataFrame(
        {'ids':test_id, 'prob':y_submission}
        ).to_csv(submission_filename, index=False)
       
    loss = 1 - score
    return {'loss': loss, 'status': STATUS_OK}


# In[ ]:


trials = Trials()
best = fmin(fn=objective_XGB,
                     space=space_XGB,
                     algo=HYPEROPT_ALGO,
                     max_evals=N_HYPEROPT_PROBES,
                     trials=trials,
                     verbose=1)

print('-'*50)
print('The best params for XGB:')
print( best )
print('\n\n')


# # Stacking Models

# In[ ]:





# In[ ]:


train_stack = train_backup.copy()
test_stack  = test_backup.copy()


# In[ ]:


train_stack.shape


# In[ ]:


test_stack.shape


# list(train_ext)

# list(test_ext)

# In[ ]:


meta_data = get_meta(train_stack)
col_ordinal   = meta_data[(meta_data.level == 'ordinal') & (meta_data.keep)& (meta_data.role != 'target')].index
col_nominal   = meta_data[(meta_data.level == 'nominal') & (meta_data.keep)& (meta_data.role != 'target')].index
col_interval = meta_data[(meta_data.level == 'interval') & (meta_data.keep)& (meta_data.role != 'target')].index
col_binary    = meta_data[(meta_data.level == 'binary') & (meta_data.keep) & (meta_data.role != 'target')].index
#meta_data


# In[ ]:


one_hot = {c: list(train_stack[c].unique()) for c in col_nominal}


# In[ ]:


train_stack = train_stack.replace(-1, np.NaN)

d_median    = train_stack.median(axis=0)
d_mean      = train_stack.mean(axis=0)

train_stack = train_stack.fillna(-1)


# In[ ]:


from sklearn import preprocessing


# In[ ]:


def transform(df, ohe, d_median, d_mean):
    
   
    dcol = [c for c in d_median.index if c in d_mean.index and c !='target']
    
    #df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    #df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
    
    for c in dcol:
        if 'bin_' not in c:
            df[c+'_median_range'] = (df[c].values > d_median[c])#.astype(np.int)
            df[c+'_mean_range']   = (df[c].values > d_mean[c])#.astype(np.int)

    for c in one_hot:
        if len(one_hot[c])>2 and len(one_hot[c]) < 7:
            for val in one_hot[c]:
                df[c+'_oh_' + str(val)] = (df[c].values == val)#.astype(np.int)
    return df


# In[ ]:


train_stack = transform(train_stack, one_hot,d_median, d_mean)


# In[ ]:


test_stack = transform(test_stack, one_hot, d_median, d_mean)


# In[ ]:



train_stack = create_extra_features(train_stack)
test_stack = create_extra_features(test_stack)
train_stack['bin_sum']  = train_stack[col_binary].sum(axis=1)
test_stack['bin_sum']  = test_stack[col_binary].sum(axis=1)


# In[ ]:





# In[ ]:


col = [c for c in train_stack.columns if c not in ['id','target']]
col = [c for c in col if not c.startswith('ps_calc_')] ## Droping ps_cal_ vars


# In[ ]:


dups = train_stack[train_stack.duplicated(subset=col, keep=False)]


# In[ ]:


train_stack = train_stack[~(train_stack.index.isin(dups.index))]


# In[ ]:


target_stack = train_stack['target']


# In[ ]:


train_stack = train_stack[col]


# In[ ]:


target_stack.shape


# In[ ]:


test_stack = test_stack[col]


# In[ ]:


train_stack.shape


# In[ ]:


test_stack.shape


# ## Testing combined transformations without stacking

# In[ ]:


meta_data = get_meta(train_stack)
col_ordinal   = meta_data[(meta_data.level == 'ordinal') & (meta_data.keep)& (meta_data.role != 'target')].index
col_nominal   = meta_data[(meta_data.level == 'nominal') & (meta_data.keep)& (meta_data.role != 'target')].index
col_interval = meta_data[(meta_data.level == 'interval') & (meta_data.keep)& (meta_data.role != 'target')].index
col_binary    = meta_data[(meta_data.level == 'binary') & (meta_data.keep) & (meta_data.role != 'target')].index
#meta_data


# In[ ]:


X_stack = pd.concat([train_stack[col_interval],train_stack[col_ordinal], pd.get_dummies(train_stack[col_binary])], axis=1)
test_stack_val = pd.concat([test_stack[col_interval],test_stack[col_ordinal], pd.get_dummies(test_stack[col_binary])], axis=1)
y_stack = target_stack


# In[ ]:


X_stack.shape


# In[ ]:


test_stack_val.shape


# In[ ]:


X_stack =  X_stack.drop(columns=['gender_-1','facebook_profile_-1'], axis = 1)


# In[ ]:


cross_val_model(X_stack, y_stack, rf_model)


# In[ ]:


cross_val_model(X_stack, y_stack, XGB_model)


# # Ensemble CV

# In[ ]:


class Ensemble(object):

    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.model_selection import cross_val_score
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2017).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]

                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                clf.fit(X_train, y_train)
                y_pred = clf.predict_proba(X_holdout)[:,1]                

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
            S_test[:, i] = S_test_i.mean(axis=1)

        results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:,1]
        return res


# In[ ]:


#RandomForest params
rf_params = {}
rf_params['n_estimators'] = 80
rf_params['max_depth'] = 12
rf_params['min_samples_split'] = 50
rf_params['min_samples_leaf'] = 23
#rf_params['class_weight'] = "balanced_subsample"# "balanced" # "balanced_subsample"
#rf_params['criterion'] = 1
#{'class_weight': 1, 'criterion': 1, 'max_depth': 10, 'min_samples_leaf': 23, 'min_samples_split': 88, 'n_estimators': 66, 'random_state': 584867}
#{'class_weight': 0, 'criterion': 1, 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 15, 'n_estimators': 31}
#{'class_weight': 1, 'criterion': 1, 'max_depth': 7, 'min_samples_leaf': 1, 'min_samples_split': 25, 'n_estimators': 52}


# In[ ]:


# XGBoost params
xgb_params = {}
xgb_params['learning_rate'] =0.03660642032718193
xgb_params['n_estimators'] = 70
xgb_params['max_depth'] = 7
xgb_params['reg_alpha'] = 0.1
xgb_params['reg_lambda'] = 0.1
xgb_params['colsample_bytree'] = 0.6162725690461764 
xgb_params['min_child_weight'] =  0.751826989118936
#{'colsample_bytree': 0.6162725690461764, 'learning_rate': 0.07660642032718193, 'max_depth': 1, 'min_child_weight': 0.751826989118936, 'n_estimators': 51, 'random_state': 2943, 'reg_alpha': 8.447744027604217e-05, 'reg_lambda': 2.506380824011793e-05}
#{'colsample_bytree': 0.6669680642534331, 'learning_rate': 0.0027697150000431693, 'max_depth': 2, 'min_child_weight': 0.7842089630474731, 'n_estimators': 58, 'random_state': 194789, 'reg_alpha': 6.334122926125054e-05, 'reg_lambda': 7.725227814541321e-05}
#{'colsample_bytree': 0.7185209051997172, 'learning_rate': 0.09634564047154007, 'max_depth': 1, 'min_child_weight': 0.7765683660381831, 'n_estimators': 60, 'random_state': 1791482, 'reg_alpha': 1.5998181299665275e-05, 'reg_lambda': 9.446368653609355e-05}
#{'colsample_bytree': 0.785981949747911, 'learning_rate': 0.07697973917507268, 'max_depth': 0, 'min_child_weight': 0.7528834859046539, 'n_estimators': 48, 'random_state': 1038594, 'reg_alpha': 9.730513129698628e-05, 'reg_lambda': 9.804649087783435e-05}


# In[ ]:


rf_model = RandomForestClassifier(**rf_params, random_state=584867)


# In[ ]:


xgb_model = XGBClassifier(**xgb_params, random_state=2943)


# In[ ]:


log_model = LogisticRegression(random_state=29)


# In[ ]:


stack = Ensemble(n_splits=3,
        stacker = log_model,
        base_models = (rf_model, xgb_model))


# In[ ]:


X_stack.fillna(-1, inplace = True)
test_stack_val.fillna(-1,inplace=True)
y_pred = stack.fit_predict(X_stack, target_stack, test_stack_val)


# # Make submission

# In[ ]:


sub = pd.DataFrame()
sub['ids'] = test_id
sub['prob'] = y_pred
sub.to_csv('stacked_main.csv', index=False)


# In[ ]:




