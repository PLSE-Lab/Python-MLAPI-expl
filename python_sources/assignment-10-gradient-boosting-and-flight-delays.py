#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import warnings
# warnings.filterwarnings('ignore')
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier
# import xgboost as xgb
# from sklearn.metrics import roc_auc_score


# In[ ]:


# train = pd.read_csv('../input/flight_delays_train.csv')
# test = pd.read_csv('../input/flight_delays_test.csv')


# In[ ]:


# train.head()


# In[ ]:


# test.head()


# In[ ]:


# categorylist = ["DayOfWeek", 'Month', "DayofMonth", "UniqueCarrier", "dep_delayed_15min", "Origin", "Dest"]
# categorylist_test = ["DayOfWeek", 'Month', "DayofMonth", "UniqueCarrier", "Origin", "Dest"]


# In[ ]:


# unrelevant_cats = ["dep_delayed_15min", "DayOfWeek", "UniqueCarrier", 'Month', "DayofMonth"]
# unrelevant_cats_test = ["DayOfWeek", "UniqueCarrier", 'Month', "DayofMonth"]
# #
# # unrelevant_cats = ["DayOfWeek"]


# In[ ]:


# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()


# Search for important columns

# In[ ]:


# col_X_train, col_X_test, col_y_train, col_y_test = train_test_split(m_train.drop('dep_delayed_15min', axis=1), m_train['dep_delayed_15min'],
#                                                     test_size=0.3, stratify=m_train['dep_delayed_15min'], random_state=17)
# col_dtrain = xgb.DMatrix(col_X_train, col_y_train)
# col_dtest = xgb.DMatrix(col_X_test, col_y_test)


# In[ ]:


# col_params = {
#     'objective':'binary:logistic',
#     'max_depth':5,
#     'silent':1,
#     'eta':0.02
# }

# col_num_rounds = 200


# In[ ]:


# col_watchlist  = [(col_dtest,'test'), (col_dtrain,'train')] 
# col_xgb_model = xgb.train(col_params, col_dtrain, col_num_rounds, col_watchlist)


# In[ ]:


# print("Train dataset contains {0} rows and {1} columns".format(col_dtrain.num_row(), col_dtrain.num_col()))
# print("Test dataset contains {0} rows and {1} columns".format(col_dtest.num_row(), col_dtest.num_col()))


# In[ ]:


# print("Train mean target: ")
# print(np.mean(col_dtrain.get_label()))

# print("\nTest mean target: ")
# print(np.mean(col_dtest.get_label()))


# In[ ]:


# xgb.plot_importance(col_xgb_model);


# 1. 'DepTime' = 2033
# 2. 'Distance' = 1095
# 3. 'Flight' = 890
# 4. 'UniqueCarrier' = 757
# 5. 'Month' = 552
# 6. "DayofMonth" = 423 
# 7. "DayOfWeek" = 395

# CV

# In[ ]:





# Scoring

# In[ ]:


# params = {'max_depth': 11, 'silent':0,'n_estimators':200, 'seed':17, 'subsample':0.7, 'colsample_bytree': 0.8, 'eta': 0.04,  'gamma': 0.52,  'min_child_weight': 4.0}


# In[ ]:


# 'objective':'binary:logistic'


# In[ ]:


# logit = LogisticRegression()


# In[ ]:


# train["Flight"] = train["Origin"] + train["Dest"]
# train = train.drop(columns=["Origin", "Dest"], axis=1)
# train.head()


# In[ ]:


# test["Flight"] = test["Origin"] + test["Dest"]
# test = test.drop(columns=["Origin", "Dest"], axis=1)
# test.head()


# In[ ]:


# train = train.drop(['Month', "DayofMonth", "DayOfWeek"], axis=1)


# In[ ]:


# # test = test.drop(columns=unrelevant_cats, axis=1)
# # train = train.drop(columns=unrelevant_cats, axis=1)
# a=1


# In[ ]:


# for col in categorylist:
#     one_hot = pd.get_dummies(train[col], prefix=col)
#     train = train.drop(col,axis = 1)
#     train = train.join(one_hot)
# #     train[col] = le.fit_transform(train[col])


# In[ ]:


# # train['dep_delayed_15min'] = train['dep_delayed_15min_Y']
# # train = train.drop(['dep_delayed_15min_Y','dep_delayed_15min_N'], axis=1)
# # train.head()
# a=1


# In[ ]:


# for col in categorylist_test:
#     one_hot = pd.get_dummies(test[col], prefix=col)
#     test = test.drop(col,axis = 1)
#     test = test.join(one_hot)
# #     test[col] = le.fit_transform(test[col])


# In[ ]:


# train['dep_delayed_15min'] = train['dep_delayed_15min_Y']
# train = train.drop(['dep_delayed_15min_Y','dep_delayed_15min_N'], axis=1)
# # train.head()


# In[ ]:


# test.columns


# In[ ]:


# train.columns


# In[ ]:


# y_train = train['dep_delayed_15min'].values


# In[ ]:


# train = train.drop("dep_delayed_15min", axis=1)


# In[ ]:


# for col in test.columns.values.tolist():
#     if not col in train.columns.values.tolist():
#         train[col] = 0


# In[ ]:


# train.columns


# In[ ]:


# for col in train.columns.values.tolist():
#     if not col in test.columns.values.tolist():
#         test[col] = 0


# In[ ]:


# test.columns


# In[ ]:


# test.head()


# In[ ]:


# train.head()


# Tunung params

# In[ ]:


# X_train = train.values
# X_test = test.values

# X_train_part, X_valid, y_train_part, y_valid = \
#     train_test_split(X_train, 
#                      y_train, 
#                      test_size=0.3, 
#                      random_state=17)


# In[ ]:


# from hyperopt import hp
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


# In[ ]:


# dtrain = xgb.DMatrix(X_train_part, label=y_train_part)
# dvalid = xgb.DMatrix(X_valid, label=y_valid)


# In[ ]:





# In[ ]:





# In[ ]:


# def score(params):
#     from sklearn.metrics import log_loss
#     print("Training with params:")
#     print(params)
#     params['max_depth'] = int(params['max_depth'])
#     model = xgb.train(params, dtrain, params['num_round'])
#     predictions = model.predict(dvalid).reshape((X_valid.shape[0], 2))
#     score = log_loss(y_valid, predictions)
#     print("\tScore {0}\n\n".format(score))
#     return {'loss': score, 'status': STATUS_OK}


# In[ ]:




# def optimize(trials):
#     space = {
#              'num_round': 100,
#              'learning_rate': hp.quniform('eta', 0.005, 0.05, 0.005),
#              'max_depth': hp.quniform('max_depth', 3, 14, 1),
#              'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
#              'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
#              'gamma': hp.quniform('gamma', 0.5, 1, 0.01),
#              'colsample_bytree': hp.quniform('colsample_bytree', 0.4, 1, 0.05),
#              'num_class' : 2,
#              'eval_metric': 'merror',
#              'objective': 'multi:softprob',
#              'nthread' : 1,
#              'silent' : 1
#              }
    
#     best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=10)
#     return best


# In[ ]:


# trials = Trials()
# best_params = optimize(trials)
# best_params


# {'colsample_bytree': 0.75,
# 'eta': 0.03,
# 'gamma': 0.65,
# 'max_depth': 11.0,
# 'min_child_weight': 9.0,
# 'subsample': 0.9500000000000001}

# In[ ]:


# best_params['max_depth'] = int(best_params['max_depth'])
# best_params['num_class'] = 2
# # best_params['eval_metric'] = 'merror'
# # best_params['objective'] = 'multi:softprob'
# # best_params['nthread'] = 4
# # best_params['silent'] = 1


# In[ ]:


# dtrain = xgb.DMatrix(train, y_train)


# In[ ]:


# %%time
# xgbCvResult = xgb.cv(best_params, dtrain, 
#                       num_boost_round=500,  
#                       nfold=3, early_stopping_rounds=50)


# In[ ]:


# %matplotlib inline
# import matplotlib.pyplot as plt


# In[ ]:


# plt.plot(range(xgbCvResult.shape[0]), xgbCvResult['test-merror-mean'])
# plt.plot(range(xgbCvResult.shape[0]), xgbCvResult['train-merror-mean']);


# In[ ]:


# best_num_round = np.argmin(xgbCvResult['test-merror-mean'])
# best_num_round


# In[ ]:


# %reset


# In[ ]:


params = {
    'colsample_bytree': 0.75,
    'eta': 0.03,
    'gamma': 0.65,
    'max_depth': 11,
    'min_child_weight': 9, 
    'subsample': 0.95,
    'n_estimators': 348
}


# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


# In[ ]:


from sklearn.model_selection import cross_val_predict


# In[ ]:





# In[ ]:


train = pd.read_csv('../input/flight_delays_train.csv')
test = pd.read_csv('../input/flight_delays_test.csv')


# In[ ]:


categorylist = ["DayOfWeek", 'Month', "DayofMonth", "UniqueCarrier", "dep_delayed_15min", "Origin", "Dest"]
categorylist_test = ["DayOfWeek", 'Month', "DayofMonth", "UniqueCarrier", "Origin", "Dest"]


# In[ ]:


for col in categorylist:
    one_hot = pd.get_dummies(train[col], prefix=col)
    train = train.drop(col,axis = 1)
    train = train.join(one_hot)


# In[ ]:


for col in categorylist_test:
    one_hot = pd.get_dummies(test[col], prefix=col)
    test = test.drop(col,axis = 1)
    test = test.join(one_hot)


# In[ ]:


train['dep_delayed_15min'] = train['dep_delayed_15min_Y']
train = train.drop(['dep_delayed_15min_Y','dep_delayed_15min_N'], axis=1)


# In[ ]:


y_train = train['dep_delayed_15min'].values


# In[ ]:


train = train.drop("dep_delayed_15min", axis=1)


# In[ ]:


for col in test.columns.values.tolist():
    if not col in train.columns.values.tolist():
        train[col] = 0


# In[ ]:


for col in train.columns.values.tolist():
    if not col in test.columns.values.tolist():
        test[col] = 0


# In[ ]:


X_train = train.values
X_test = test.values

X_train_part, X_valid, y_train_part, y_valid =     train_test_split(X_train, 
                     y_train, 
                     test_size=0.3, 
                     random_state=17)


# In[ ]:


xgb_model = XGBClassifier(**params)


# In[ ]:


predicted = cross_val_predict(xgb_model, X_train_part, y_train_part, cv=5)


# In[ ]:


roc_auc_score(y_train_part, predicted)


# In[ ]:



pd.Series(predicted, 
          name='dep_delayed_15min').to_csv('xgbcv.csv', 
                                           index_label='id', header=True)


# In[ ]:


xgb_model.fit(X_train_part, y_train_part)

xgb_valid_pred = xgb_model.predict_proba(X_valid)[:, 1]

roc_auc_score(y_valid, xgb_valid_pred)


# In[ ]:


xgb_model.fit(X_train, y_train)
xgb_test_pred = xgb_model.predict_proba(X_test)[:, 1]

pd.Series(xgb_test_pred, 
          name='dep_delayed_15min').to_csv('xgb.csv', 
                                           index_label='id', header=True)

