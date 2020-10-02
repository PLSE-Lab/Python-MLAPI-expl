#!/usr/bin/env python
# coding: utf-8

# ## This kernel basic on the kernels:
# 
# * [Logistic Regression](https://www.kaggle.com/martin1234567890/logistic-regression)
# 
# * [Feature importance - xgb, lgbm, logreg, linreg](https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg)
# 
# * [One-Hot, Stratified, Logistic Regression](https://www.kaggle.com/bustam/one-hot-stratified-logistic-regression)

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import lightgbm as lgbm
import xgboost as xgb
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


submission = pd.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv")
train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")
test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")

labels = train.pop('target')
labels = labels.values
train_id = train.pop("id")
test_id = test.pop("id")


# In[ ]:


train.head(5)


# In[ ]:


# Thanks to https://www.kaggle.com/bustam/one-hot-stratified-logistic-regression
def ord_to_num(df, col):
    keys=np.sort(df[col].unique())
    values=np.arange(len(keys))
    map = dict(zip(keys, values))
    df[col] = df[col].replace(map)
ord_to_num(train, 'ord_4')
ord_to_num(test,'ord_4')
ord_to_num(train, 'ord_5')
ord_to_num(test,'ord_5')

# ord_4
train['ord_4_band'] = pd.qcut(train['ord_4'], 6)
bands=train.ord_4_band.unique()
keys_bands=np.sort(bands)
values_bands=np.arange(len(keys_bands))
map_bands = dict(zip(keys_bands, values_bands))
train['ord_4_band'] = train['ord_4_band'].replace(map_bands)
test['ord_4_band']=pd.cut(test.ord_4,pd.IntervalIndex(keys_bands))
test['ord_4_band'] = test['ord_4_band'].replace(map_bands)

# ord_5
train['ord_5_band'] = pd.qcut(train['ord_5'], 6)
bands=train.ord_5_band.unique()
keys_bands=np.sort(bands)
values_bands=np.arange(len(keys_bands))
map_bands = dict(zip(keys_bands, values_bands))
train['ord_5_band'] = train['ord_5_band'].replace(map_bands)
test['ord_5_band']=pd.cut(test.ord_5,pd.IntervalIndex(keys_bands))
test['ord_5_band'] = test['ord_5_band'].replace(map_bands)

# "nom_7", "nom_8", "nom_9" - x for values is absent in both train and test
for col in ["nom_7", "nom_8", "nom_9"]:
    train_vals = set(train[col].unique())
    test_vals = set(test[col].unique())
   
    ex=train_vals ^ test_vals
    if ex:
        train.loc[train[col].isin(ex), col]="x"
        test.loc[test[col].isin(ex), col]="x"


# In[ ]:


# Preprocessing
acc = []
data = pd.concat([train, test])
# Determination categorical features
numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorical_columns = []
features = data.columns.values.tolist()
for col in features:
    if data[col].dtype in numerics: continue
    categorical_columns.append(col)
    
# Encoding categorical features
data2 = data
for col in categorical_columns:
    if col in data.columns:
        le = LabelEncoder()
        le.fit(list(data[col].astype(str).values))
        data2[col] = le.transform(list(data[col].astype(str).values))
train = data2.iloc[:train.shape[0], :]
test = data2.iloc[train.shape[0]:, :]


# ## EDA & FE by the XGB, LGBM, Logistic Regression and Linear Regression

# In[ ]:


# Building the feature importance diagrams and prediction by 4 models
# Thanks to https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
  
# LGBM
#%% split training set to validation set
Xtrain, Xval, Ztrain, Zval = train_test_split(train, labels, test_size=0.2, random_state=0)
train_set = lgbm.Dataset(Xtrain, Ztrain, silent=False)
valid_set = lgbm.Dataset(Xval, Zval, silent=False)
params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'max_depth': -1,
        'subsample': 0.8,
        'bagging_fraction' : 1,
        'max_bin' : 5000 ,
        'bagging_freq': 20,
        'colsample_bytree': 0.6,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1,
        'zero_as_missing': True,
        'seed':0,        
    }

modelL = lgbm.train(params, train_set = train_set, num_boost_round=1000,
                   early_stopping_rounds=50, verbose_eval=50, valid_sets=valid_set)
feature_score = pd.DataFrame(train.columns, columns = ['feature'])
pred_train = pd.DataFrame()
pred_test = pd.DataFrame()
feature_score['score_lgb'] = modelL.feature_importance()
pred_train['lgb_pred'] = modelL.predict(train)
pred_test['lgb_pred'] = modelL.predict(test)
acc.append(round(r2_score(labels, pred_train['lgb_pred']) * 100, 2))

#Plot FI
fig =  plt.figure(figsize = (15,15))
axes = fig.add_subplot(111)
lgbm.plot_importance(modelL,ax = axes,height = 0.5)
plt.show();plt.close()


# In[ ]:


#XGBM
#%% split training set to validation set 
data_tr  = xgb.DMatrix(Xtrain, label=Ztrain)
data_cv  = xgb.DMatrix(Xval   , label=Zval)
data_train = xgb.DMatrix(train)
data_test = xgb.DMatrix(test)
evallist = [(data_tr, 'train'), (data_cv, 'valid')]
parms = {'max_depth':8, #maximum depth of a tree
         'objective':'reg:logistic',
         'eta'      :0.3,
         'subsample':0.8,#SGD will use this percentage of data
         'lambda '  :4, #L2 regularization term,>1 more conservative 
         'colsample_bytree ':0.9,
         'colsample_bylevel':1,
         'min_child_weight': 10}
modelx = xgb.train(parms, data_tr, num_boost_round=200, evals = evallist,
                  early_stopping_rounds=30, maximize=False, 
                  verbose_eval=10)
feature_score['score_xgb'] = feature_score['feature'].map(modelx.get_score(importance_type='weight'))
pred_train['xgb_pred'] = modelx.predict(data_train)
pred_test['xgb_pred'] = modelx.predict(data_test)
acc.append(round(r2_score(labels, pred_train['xgb_pred']) * 100, 2))

#Plot FI
fig =  plt.figure(figsize = (15,15))
axes = fig.add_subplot(111)
xgb.plot_importance(modelx,ax = axes,height = 0.5)
plt.show();plt.close()


# In[ ]:


# Regression models
# Standardization for regression models
train2 = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(train),
    columns=train.columns, index=train.index)
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(train2, labels)
coeff_logreg = pd.DataFrame(train2.columns.delete(0))
coeff_logreg.columns = ['feature']
coeff_logreg["score_logreg"] = pd.Series(logreg.coef_[0])
pred_train['log_pred'] = logreg.predict(train)
pred_test['log_pred'] = logreg.predict(test)
acc.append(round(r2_score(labels, pred_train['log_pred']) * 100, 2))
coeff_logreg.sort_values(by='score_logreg', ascending=False)


# In[ ]:


coeff_logreg["score_logreg"] = coeff_logreg["score_logreg"].abs() # the level of importance of features is not associated with the sign
feature_score = pd.merge(feature_score, coeff_logreg, on='feature')


# In[ ]:


# Linear Regression
linreg = LinearRegression()
linreg.fit(train2, labels)
coeff_linreg = pd.DataFrame(train2.columns.delete(0))
coeff_linreg.columns = ['feature']
coeff_linreg["score_linreg"] = pd.Series(linreg.coef_)
pred_train['lin_pred'] = linreg.predict(train)
pred_test['lin_pred'] = linreg.predict(test)
acc.append(round(r2_score(labels, pred_train['lin_pred']) * 100, 2))
coeff_linreg.sort_values(by='score_linreg', ascending=False)


# In[ ]:


coeff_linreg["score_linreg"] = coeff_linreg["score_linreg"].abs() # the level of importance of features is not associated with the sign
feature_score = pd.merge(feature_score, coeff_linreg, on='feature')


# In[ ]:


# Comparison of the all feature importance diagrams
feature_score = feature_score.fillna(0)
feature_score = feature_score.set_index('feature')

# MinMax scale all importances
feature_score = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(feature_score),
    columns=feature_score.columns, index=feature_score.index)

# Create mean column
feature_score['mean'] = feature_score.mean(axis=1)
pred_train['mean'] = pred_train.mean(axis=1)
pred_test['mean'] = pred_test.mean(axis=1)
acc.append(round(r2_score(labels, pred_train['mean']) * 100, 2))

# Create total column with different weights
fL = 0.2
flog = 0.5
flin = 0.15
fx = 1-fL-flog-flin
feature_score['total'] = fL*feature_score['score_lgb'] + fx*feature_score['score_xgb']                        + flog*feature_score['score_logreg'] + flin*feature_score['score_linreg']
pred_train['total'] = fL*pred_train['lgb_pred'] + fx*pred_train['xgb_pred'] +                     + flog*pred_train['log_pred'] + flin*pred_train['lin_pred']
pred_test['total'] = fL*pred_test['lgb_pred'] + fx*pred_test['xgb_pred'] +                     + flog*pred_test['log_pred'] + flin*pred_test['lin_pred']
acc.append(round(r2_score(labels, pred_train['total']) * 100, 2))

# Plot the feature importances
feature_score.sort_values('total', ascending=False).plot(kind='bar', figsize=(20, 10))


# In[ ]:


feature_score.sort_values('score_logreg', ascending=False)


# In[ ]:


acc


# In[ ]:


# to_drop = feature_score[(feature_score['score_logreg'] < 0.0025)].index.tolist()
# to_drop


# In[ ]:


# data = data.drop(to_drop,axis=1)
data.info()


# ## Apply Logistic Regression

# In[ ]:


columns = [i for i in data.columns]

dummies = pd.get_dummies(data,
                         columns=columns,
                         drop_first=True,
                         sparse=True)

del data


# In[ ]:


train = dummies.iloc[:train.shape[0], :]
test = dummies.iloc[train.shape[0]:, :]

del dummies


# In[ ]:


train.head(5)


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


train = train.sparse.to_coo().tocsr()
test = test.sparse.to_coo().tocsr()

train = train.astype("float32")
test = test.astype("float32")


# In[ ]:


lr = LogisticRegression(C=0.12,
                        solver="lbfgs",
                        tol=0.002,
                        max_iter=10000)

lr.fit(train, labels)

lr_pred = lr.predict_proba(train)[:, 1]
score = roc_auc_score(labels, lr_pred)

print("score: ", score)


# In[ ]:


submission["id"] = test_id
submission["target"] = lr.predict_proba(test)[:, 1]


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv("submission.csv", index=False)

