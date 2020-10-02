#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
from collections import Counter
from sklearn.feature_selection import mutual_info_classif
from sklearn.externals import joblib
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


round(0)


# In[ ]:


test = pd.read_csv('../input/test.csv')
test.tail()


# In[ ]:


print(train.shape, test.shape)


# ## Construction of metadata

# In[ ]:


data = []
for col in train.columns:
    # Defining the role
    if col == 'target' or col == 'id':
        role = col
    else:
        role = 'input'
         
    # Defining the level
    if 'bin' in col or col == 'target':
        level = 'binary'
    elif 'cat' in col or col == 'id':
        level = 'nominal'
    elif train[col].dtype == float:
        level = 'interval'
    elif train[col].dtype == int:
        level = 'ordinal'
        
    # Initialize keep to True for all variables except for id
    keep = True
    if col == 'id':
        keep = False
    
    # Defining the data type 
    dtype = train[col].dtype
    
    # Creating a Dict that contains all the metadata for the variable
    col_dict = {'col_name': col, 'role': role, 'level': level, 'keep': keep, 'dtype': dtype}
    data.append(col_dict)
    
meta = pd.DataFrame(data, columns=['col_name', 'role', 'level', 'keep', 'dtype'])
meta.set_index('col_name', inplace=True)


# In[ ]:


bin_cols = meta[(meta.level=='binary') & meta.keep].index
train[bin_cols].describe()


# ## Balance training data distribution

# In[ ]:


# from sklearn.utils import shuffle
# desired_apriori=0.10

# # Get the indices per target value
# idx_0 = train[train.target == 0].index
# idx_1 = train[train.target == 1].index

# # Get original number of records per target value
# nb_0 = len(train.loc[idx_0])
# nb_1 = len(train.loc[idx_1])

# undersampling_rate = ((1 - desired_apriori) * nb_1) / (nb_0 * desired_apriori)
# undersampling_nb_0 = int(nb_0 * undersampling_rate)
# print('Number of training samples with target == 0 after undersampling', undersampling_nb_0)

# undersampled_idx = shuffle(idx_0, random_state=77, n_samples=undersampling_nb_0)
# idx_list = list(undersampled_idx) + list(idx_1)
# balanced_train = train.loc[idx_list].reset_index(drop=True)


# ## 1. Data quality checks

# ### Null or missing values check

# In[ ]:


train.isnull().any().any()


# In[ ]:


ms_cols = []
train_copy = train
train_copy = train_copy.replace(-1, np.NaN)
for col in train_copy.columns:
    ms_nb = train_copy[col].isnull().sum()
    if ms_nb > 0:
        ms_cols.append(col)
        print('Column {} has {} records ({:.2%}) with missing values'.format(col, ms_nb, ms_nb/train_copy.shape[0]))


# In[ ]:


# import missingno as msno
# msno.matrix(df=train_copy.iloc[:, 2:40], figsize=(20, 14),
#             color=(0.42, 0.1, 0.05))


# ### Dropout columns with too many missing values and imputing

# In[ ]:


from sklearn.preprocessing import Imputer

drop_cols = ['ps_car_03_cat', 'ps_car_05_cat']
real_train = train.drop(drop_cols, axis=1)
real_test = test.drop(drop_cols, axis=1)
meta.loc[drop_cols, 'keep'] = False

mean_imp = Imputer(missing_values=-1, strategy='mean', axis=0)
mode_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)

# Imputing training data
real_train['ps_reg_03'] = mean_imp.fit_transform(X=real_train[['ps_reg_03']]).ravel()
real_train['ps_car_11'] = mode_imp.fit_transform(X=real_train[['ps_car_11']]).ravel()
real_train['ps_car_12'] = mode_imp.fit_transform(X=real_train[['ps_car_12']]).ravel()
real_train['ps_car_14'] = mean_imp.fit_transform(X=real_train[['ps_car_14']]).ravel()

# Imputing test data
real_test['ps_reg_03'] = mean_imp.fit_transform(X=real_test[['ps_reg_03']]).ravel()
real_test['ps_car_11'] = mode_imp.fit_transform(X=real_test[['ps_car_11']]).ravel()
real_test['ps_car_12'] = mode_imp.fit_transform(X=real_test[['ps_car_12']]).ravel()
real_test['ps_car_14'] = mean_imp.fit_transform(X=real_test[['ps_car_14']]).ravel()


# We didn't impute the missing values in the categorical columns, instead, we kept it as a seperate category. As later on we can see that customers with a missing value in these variables appear to have a much higher possibility to file an insurance claim (a good takeaway for the future data preproccessing method)

# In[ ]:


# Replace missing values with 999
# real_train1 = real_train.replace(-1, 999)
# real_test1 = real_test.replace(-1, 999)


# ## Check the cardinality of categorical columns

# In[ ]:


cat_cols = meta[(meta.level=='nominal') & meta.keep].index

for col in cat_cols:
    distinct_values = real_train[col].value_counts().shape[0]
    print('Categorical column {} has {} distinct values'.format(col, distinct_values))


# ## Handling column "ps_car_11_cat" specifically as it has too many cardinality

# In[ ]:


# Script by https://www.kaggle.com/ogrellier
# Code: https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


# In[ ]:


train_encoded, test_encoded = target_encode(train["ps_car_11_cat"], 
                             test["ps_car_11_cat"], 
                             target=train.target, 
                             min_samples_leaf=100,
                             smoothing=10,
                             noise_level=0.01)
    
real_train['ps_car_11_cat_te'] = train_encoded
real_train.drop('ps_car_11_cat', axis=1, inplace=True)
meta.loc['ps_car_11_cat','keep'] = False  # Updating the meta
real_test['ps_car_11_cat_te'] = test_encoded
real_test.drop('ps_car_11_cat', axis=1, inplace=True)


# ## Dropout calculated columns 

# In[ ]:


# calc_cols = [col for col in train.columns if 'calc' in col]

# ###=========== A data exploration of calc features from armamut ==============###
# # Script: https://www.kaggle.com/armamut/ps-calc-15-bin-ps-calc-20-bin

# # Columns -> binary decoded.

# tmp  =real_train['ps_calc_15_bin'] * 32 + real_train['ps_calc_16_bin'] * 16 + real_train['ps_calc_17_bin'] * 8
# tmp += real_train['ps_calc_18_bin'] * 4 + real_train['ps_calc_19_bin'] * 2 + real_train['ps_calc_20_bin'] * 1

# tmp2 = [5, 22, 9, 32, 13, 38, 20, 47, 2, 19, 8, 30, 10, 35, 17, 45, 1,
#         15, 4, 24, 7, 29, 14, 40, 0, 12, 3, 21, 6, 26, 11, 36, 27, 52,
#         37, 57, 42, 60, 51, 63, 23, 49, 34, 56, 39, 59, 48, 62, 18, 46,
#         28, 53, 33, 55, 44, 61, 16, 43, 25, 50, 31, 54, 41, 58]
# tmp2 = pd.Series(tmp2)

# real_train['ps_calc_15_16_17_18_19_20'] = tmp.map(tmp2)
# real_test['ps_calc_15_16_17_18_19_20'] = tmp.map(tmp2)
# # You may now drop the others peacefully.
# # real_train_nocalc = real_train.drop(['ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin',
# #               'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin'], axis=1, inplace=False)
# ###============================================================================###

# real_train_nocalc = real_train.drop(calc_cols, axis=1)
# real_test_nocalc = real_test.drop(calc_cols, axis=1)
# real_train_nocalc.shape


# In[ ]:


calc_cols = [col for col in train.columns if 'calc' in col]

real_train_nocalc = real_train.drop(calc_cols, axis=1)
real_test_nocalc = real_test.drop(calc_cols, axis=1)
real_train_nocalc.shape


# ### Target variable inspection

# In[ ]:


def plotTargetDistribution(dataset):
    data = [go.Bar(
                x = dataset['target'].value_counts().index.values,
                y = dataset['target'].value_counts().values,
                text = 'Distribution of target variable')]
    layout = go.Layout(
                title = 'Distribution of target variable')
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='basic-bar')
plotTargetDistribution(real_train)


# ### Typedata check

# In[ ]:


# Counter(real_train.dtypes.values)
# train_float = real_train.select_dtypes(include=['float64'])
# train_int = real_train.select_dtypes(include=['int64'])


# ## 2. Correlation plots

# ### Correlation of float features

# In[ ]:


float_int_cols = meta[((meta.level == 'interval') | (meta.level == 'ordinal')) & meta.keep].index
float_int_train = real_train[float_int_cols]

colormap = plt.cm.cubehelix_r
plt.figure(figsize=(16,12))
plt.title('Pearson correlation of continuous features', y=1.05, size=15)
sns.heatmap(float_int_train.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# Column pairs with high correlation:
# 1. ps_reg_01 and ps_reg_02: 0.47
# 2. ps_reg_02 and ps_reg_03: 0.7
# 3. ps_car_12 and ps_car_13: 0.67
# 4. ps_car_12 and ps_car_14: 0.58
# 5. ps_car_13 and ps_car_14: 0.44
# 6. ps_car_13 and ps_car_15: 0.52
# 
# Maybe we need to do some domensionality reduction by using PCA...

# ### Dropout float features with high correlation to another column

# Within the two groups of highest correlation features, there is one column "ps_car_13" shared by both, so we will throw that column directly as to preserve as much information

# In[ ]:


# realTrain = realTrain.drop('ps_car_13', axis=1)


# ## 3. Binary features inspection

# In[ ]:


bin_cols = meta[(meta.level == 'binary') & (meta.role != 'target') & meta.keep].index
ones_list = []
zeros_list = []
for col in bin_cols:
    zeros_nb = (real_train[col] == 0).sum()
    ones_nb = real_train.shape[0] - zeros_nb
    ones_list.append(ones_nb)
    zeros_list.append(zeros_nb)
    print('Binary column {} has {} records ({:.2%}) with value zero'.format(col, zeros_nb, zeros_nb/real_train.shape[0]))


# In[ ]:


trace0 = go.Bar(x=bin_cols, y=zeros_list, name='Zeros count')
trace1 = go.Bar(x=bin_cols, y=ones_list, name='Ones count')

data = [trace0, trace1]
layout = go.Layout(barmode='stack', title='Count of zeros and ones')

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='stacked-bar')


# ### Dropout binary features dominated by zeros

# In[ ]:


imbalanced_cols = ['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_13_bin']
real_train = real_train.drop(imbalanced_cols, axis=1)
real_test = real_test.drop(imbalanced_cols, axis=1)

real_train_nocalc = real_train_nocalc.drop(imbalanced_cols, axis=1)
real_test_nocalc = real_test_nocalc.drop(imbalanced_cols, axis=1)


# In[ ]:


# plotTargetDistribution(train_data)
# plotTargetDistribution(val_data)


# ## Feature engineering

# ### Create dummy variables

# In[ ]:


cat_cols = meta[(meta.level=='nominal') & meta.keep].index
# cat_cols = cat_cols.drop('ps_car_11_cat')

print('Nb of columns in train data before dummification: {}'.format(real_train.shape[1]))
real_train = pd.get_dummies(data=real_train, columns=cat_cols, drop_first=True)
print('Nb of columns in train data before dummification: {}'.format(real_train.shape[1]))

print('Nb of columns in test data before dummification: {}'.format(real_test.shape[1]))
real_test = pd.get_dummies(data=real_test, columns=cat_cols, drop_first=True)
print('Nb of columns in test data after dummification: {}'.format(real_test.shape[1]))

real_train_nocalc = pd.get_dummies(data=real_train_nocalc, columns=cat_cols, drop_first=True)
real_test_nocalc = pd.get_dummies(data=real_test_nocalc, columns=cat_cols, drop_first=True)


# ### Create interaction variables

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
float_cols = meta[(meta.level=='interval') & meta.keep].index

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
interactions = pd.DataFrame(data=poly.fit_transform(real_train[float_cols]), columns=poly.get_feature_names(float_cols))
interactions.drop(float_cols, axis=1, inplace=True)
interacted_train = pd.concat(objs=[real_train, interactions], axis=1)
interacted_test = pd.concat(objs=[real_test, interactions], axis=1)


# ## 4. Learning models and predictions

# In[ ]:


from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(real_train, train_size=0.9, random_state=77)


# In[ ]:


# _, compressed_train = train_test_split(real_train_nocalc, train_size=0.95, random_state=777)
# train_data, val_data = train_test_split(compressed_train, train_size=0.8, random_state=777)


# ### Feature importance via random forest

# In[ ]:


# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=4,
#                             n_jobs=-1, random_state=77, max_features=1, class_weight={0:1, 1:700})
# rf.fit(X=train_data.drop(['id', 'target'], axis=1), y=train_data['target'])
# features = train_data.drop(['id', 'target'], axis=1).columns.values


# In[ ]:


# predVal = rf.predict(X=val_data.drop(['id', 'target'], axis=1))


# ### Visualisation of features importances

# In[ ]:


# x, y = (list(x) for x in zip(*sorted(zip(rf.feature_importances_, features), reverse=False)))
# trace = go.Bar(x=x, y=y, marker=dict(color=x, colorscale='Viridis'), 
#                name='Random Forest feature importance', orientation='h')
# layout = dict(title='Barplot of reature importance', width=900, height=2000, 
#              yaxis=dict(showgrid=False, showline=False, showticklabels=True))
# fig = go.Figure(data=[trace])
# fig['layout'].update(layout)
# py.iplot(figure_or_data=fig, filename='Barplots')


# ### Select features with a given threshold feature importance with SelectFromModel method

# In[ ]:


# from sklearn.feature_selection import SelectFromModel
# sfm = SelectFromModel(estimator=rf, threshold=0.001, prefit=True)
# sfm.transform(X=train_data.drop(['id', 'target'], axis=1))

# train_data = train_data.iloc[:, sfm.get_support(indices=True)]
# val_data = val_data.iloc[:, sfm.get_support(indices=True)]


# ### Training a neural network

# In[ ]:


# from sklearn.neural_network import MLPClassifier
# nnet = MLPClassifier(hidden_layer_sizes=(7, 7, 7), max_iter=250, batch_size=700, 
#                      random_state=777, verbose=True, tol=1e-7)
# nnet.fit(X=train_data.drop(['id', 'target'], axis=1), y=train_data[['target']])


# ### Introduce imblearn package to hopefully resolve skewed data problem

# In[ ]:


# from imblearn.ensemble import BalanceCascade
# bc = BalanceCascade(random_state=7)
# X_resampled, y_resampled = bc.fit_sample(realTrain.drop(['id', 'target'], axis=1), 
#                                          realTrain[['target']])


# ## Compute gini coefficient

# In[ ]:


# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897
def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def gini_xgb(preds, d_train):
    targets = d_train.get_label()
    gini_score = gini_normalized(targets, preds)
    return [('gini', gini_score)]
    


# 
# ## XGBoost

# In[ ]:


import xgboost as xgb
d_train = xgb.DMatrix(real_train.drop(['id', 'target'], axis=1), real_train[['target']])
# d_val = xgb.DMatrix(val_data.drop(['id', 'target'], axis=1), val_data[['target']])
d_test = xgb.DMatrix(real_test.drop(['id'], axis=1))

n_splits = 5
n_estimators = 7
folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=77) 

params = {
        'objective': 'binary:logistic', 
        'eta': 0.015,
        'eval_metric': 'auc', 
        'max_depth': 6, 
        'min_child_weight': 10,
        'gamma': 1, 
        'reg_lambda': 0.3, 
        'reg_alpha': 0.07, 
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'silent': False, 
        }

# watchlist = [(d_train, 'train'), (d_val, 'valid')]

xgbCV = xgboost.cv(params=params, dtrain=d_train, stratified=True, num_boost_round=10000, feval=gini_xgb,
                   early_stopping_rounds=100, maximize=True, verbose_eval=5, nfold=3)


# In[ ]:


from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
import gc
from numba import jit
from sklearn.preprocessing import LabelEncoder
import time 

@jit
def eval_gini(y_true, y_prob):
    """
    Original author CPMP : https://www.kaggle.com/cpmpml
    In kernel : https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = eval_gini(labels, preds)
    return [('gini', gini_score)]

trn_df = real_train
sub_df = real_test

target = trn_df.target
del trn_df["target"]

n_splits = 5
n_estimators = 7
folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=77) 
imp_df = np.zeros((len(trn_df.columns), n_splits))
xgb_evals = np.zeros((n_estimators, n_splits))
oof = np.empty(len(trn_df))
sub_preds = np.zeros(len(sub_df))
increase = True
np.random.seed(0)

print('Start fitting: ')
for fold_, (trn_idx, val_idx) in enumerate(folds.split(target, target)):
    trn_dat, trn_tgt = trn_df.iloc[trn_idx], target.iloc[trn_idx]
    val_dat, val_tgt = trn_df.iloc[val_idx], target.iloc[val_idx]

    clf = XGBClassifier(n_estimators=n_estimators,
                        max_depth=4,
                        objective="binary:logistic",
                        learning_rate=0.1, 
                        subsample=.8, 
                        colsample_bytree=.8,
                        gamma=1,
                        reg_alpha=0,
                        reg_lambda=1,
                      #  min_child_weight=10, 
                        nthread=2)
    # Upsample during cross validation to avoid having the same samples
    # in both train and validation sets
    # Validation set is not up-sampled to monitor overfitting
    if increase:
        # Get positive examples
        pos = pd.Series(trn_tgt == 1)
        # Add positive examples
        trn_dat = pd.concat([trn_dat, trn_dat.loc[pos]], axis=0)
        trn_tgt = pd.concat([trn_tgt, trn_tgt.loc[pos]], axis=0)
        # Shuffle data
        idx = np.arange(len(trn_dat))
        np.random.shuffle(idx)
        trn_dat = trn_dat.iloc[idx]
        trn_tgt = trn_tgt.iloc[idx]
    
    print('{}th fold'.format(fold_))
    clf.fit(trn_dat, trn_tgt, 
            eval_set=[(trn_dat, trn_tgt), (val_dat, val_tgt)],
            eval_metric=gini_xgb,
            early_stopping_rounds=None,
            verbose=False)
            
    imp_df[:, fold_] = clf.feature_importances_
    oof[val_idx] = clf.predict_proba(val_dat)[:, 1]
    
    # Find best round for validation set
    xgb_evals[:, fold_] = clf.evals_result_["validation_1"]["gini"]
    best_round = np.argsort(xgb_evals[:, fold_])[::-1][0]
    
    # Display results
    print("Fold %2d : %.6f @%4d / best score is %.6f @%4d" 
          % (fold_ + 1, 
             eval_gini(val_tgt, oof[val_idx]),
             n_estimators,
             xgb_evals[best_round, fold_],
             best_round))
             
    # Update submission
    sub_preds += clf.predict_proba(sub_df)[:, 1] / n_splits 
          
print("Full OOF score : %.6f" % eval_gini(target, oof))

# Compute mean score and std
mean_eval = np.mean(xgb_evals, axis=1)
std_eval = np.std(xgb_evals, axis=1)
best_round = np.argsort(mean_eval)[::-1][0]

print("Best mean score : %.6f + %.6f @%4d"
      % (mean_eval[best_round], std_eval[best_round], best_round))
    
importances = imp_df.mean(axis=1)
for i, imp in enumerate(importances):
    print("%-20s : %10.4f" % (trn_df.columns[i], imp))
    
sub_df["target"] = sub_preds

sub_df[["target"]].to_csv("submission_20fold.csv", index=True, float_format="%.9f")


# In[ ]:


# from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from xgboost import XGBClassifier

# X_train = real_train.drop(['id', 'target'], axis=1)
# y_train = real_train['target'].values
# # A parameter grid for XGBoost
# params = {
#         'objective': ['binary:logistic'], 
#         'min_child_weight': [10],
#         'gamma': [1], 
#         'reg_lambda': [0.3], 
#         'subsample': [0.8],
#         'colsample_bytree': [0.8],
#         'max_depth': [6], 
#         'learning_rate': [0.015],
#         'n_estimators': [700],
#         'silent': [True], 
#        # 'nthread': [1], 
#         }
# folds = 4

# xgbc = XGBClassifier()
# # xgbc.get_params().keys() # get the name of all parameters
# skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=77)
# grid_search = GridSearchCV(param_grid=params, estimator=xgbc, verbose=45, 
#                            cv=skf.split(X_train, y_train), n_jobs=4, scoring='roc_auc')


# In[ ]:


# grid_search.fit(X_train, y_train)
# print('Best estimator: ', grid_search.best_estimator_)
# print('Best score: ', grid_search.best_score_ * 2 - 1)
# print('Best parameters: ', grid_search.best_params_)


# In[ ]:


import xgboost as xgb
d_train = xgb.DMatrix(train_data.drop(['id', 'target'], axis=1), train_data[['target']])
d_val = xgb.DMatrix(val_data.drop(['id', 'target'], axis=1), val_data[['target']])
d_test = xgb.DMatrix(real_test.drop(['id'], axis=1))

# xgboost parameters
# params = {}
# params['objective'] = 'binary:logistic'
# params['eta'] = 0.04
# # params['max_delta_step'] = 8
# params['eval_metric'] ='auc'
# params['silent'] = True
# params['max_depth'] = 6
# params['subsample'] = 0.9
# params['colsample_bytree'] = 0.9

params = {
        'objective': 'binary:logistic', 
        'eta': 0.015,
        'eval_metric': 'auc', 
        'max_depth': 6, 
        'min_child_weight': 10,
        'gamma': 1, 
        'reg_lambda': 0.3, 
        'reg_alpha': 0.07, 
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'silent': True, 
        'n_estimators': 400,
        }

watchlist = [(d_train, 'train'), (d_val, 'valid')]

mdl_xgb = xgb.train(params, d_train, num_boost_round=10000, evals=watchlist, early_stopping_rounds=100, 
                    feval=gini_xgb, maximize=True, verbose_eval=10)

# Save model to disk as soon as learning process finishes in case the kernel dies
filename = 'xgb_model.joblib.pkl'
_ = joblib.dump(mdl_xgb, filename, compress=9)


# In[ ]:





# In[ ]:


# Prediction on test set
test_pred = mdl_xgb.predict(data=d_test)


# In[ ]:


# val_pred = mdl_xgb.predict(data=d_val)
# gini_xgb(d_train=d_val, preds=val_pred)


# In[ ]:


# Save prediction results to csv
submissions = pd.DataFrame()
submissions['id'] = real_test.iloc[:, 0]
submissions['target'] = test_pred
submissions.to_csv('xgb_model.csv', index=False)


# ### Evaluation the model on validation data

# In[ ]:


# from sklearn.metrics import confusion_matrix
# predVal = nnet.predict(X=train_data.drop(['id', 'target'], axis=1))
# conf_mat = confusion_matrix(y_pred=predVal, y_true=train_data.target).transpose()
# print(conf_mat)
# precision = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
# recall = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[1, 0])
# f1 = 2 * precision * recall / (precision + recall)
# f1


# ### Predictions on test data

# In[ ]:




