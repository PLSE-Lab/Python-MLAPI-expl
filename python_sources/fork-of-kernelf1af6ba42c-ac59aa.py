#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Here we will use simple one-hot encoding with bayesian hyperparameter tuning for
#XGB regressor to predict fraud probability over attached dataset

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from fancyimpute import KNN
import gc
from bayes_opt import BayesianOptimization
import warnings

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
#id_test = pd.read_csv('../input/test_identity.csv')
s_s = pd.read_csv('../input/sample_submission.csv')
id_train = pd.read_csv('../input/train_identity.csv')
trans_train = pd.read_csv('../input/train_transaction.csv')
#trans_test = pd.read_csv('../input/test_transaction.csv')
# Any results you write to the current directory are saved as output.
trans_train.set_index('TransactionID').join(id_train.set_index('TransactionID'))
gc.collect()
train = trans_train
#trans_test.set_index('TransactionID').join(id_test.set_index('TransactionID'))
#del [id_test, trans_test]


# In[ ]:


#let's drop cols that are mostly nulls or 1 value
#There's over 50 of them!
one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]
many_null_cols = [col for col in train.columns if train[col].isnull().sum() / train.shape[0] > 0.9]
big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

cols_to_drop = list(set(many_null_cols + big_top_value_cols + one_value_cols))
cols_to_drop.remove('isFraud')
len(cols_to_drop)


# In[ ]:


train = train.drop(cols_to_drop, axis=1)
train = train.head(10000)
#train.to_csv('train_cols_trimmed.csv', index=False)


# In[ ]:



t_train = train.copy()
y = train.isFraud
X = t_train.drop(['isFraud'], axis = 1, inplace=True)
X_train_full, X_valid_full, y_train_full, y_valid_full = train_test_split(t_train, y, test_size = 0.2, random_state = 0)


# In[ ]:


#I've commited the sin of verbosity here, but it's just plain label encoding
#I've dropped categorical columns with too much categories
#Could have limit memory usage by changing types

from sklearn.preprocessing import LabelEncoder

categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and 
                    X_train_full[cname].dtype == "object"]
num_cols = [cname for cname in X_train_full.columns if
           X_train_full[cname].dtype in ['int64','float64']]
my_cols = categorical_cols + num_cols

X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
my_cols2 = [col for col in X_train.columns if col in X_valid]
for i in my_cols2:
    if i in X_train: X_train.drop(i, axis=1)
    if i in X_valid: X_valid.drop(i, axis=1)

numerical_transformer = SimpleImputer(strategy='mean')
cat_imp = SimpleImputer(strategy='most_frequent')

X_train_n = pd.DataFrame(numerical_transformer.fit_transform(X_train[num_cols]))
X_train_c = pd.DataFrame(cat_imp.fit_transform(X_train[categorical_cols]))
X_train_n.columns = num_cols
X_train_c.columns = categorical_cols
X_train2 = X_train_n.join(X_train_c,how = 'left')

X_vn = pd.DataFrame(numerical_transformer.fit_transform(X_valid[num_cols]))
X_vc = pd.DataFrame(cat_imp.fit_transform(X_valid[categorical_cols]))
X_vn.columns = num_cols
X_vc.columns = categorical_cols
X_valid2 = X_vn.join(X_vc,how = 'left')

for f in X_train2.columns:
    if  X_train2[f].dtype=='object': 
        lbl = LabelEncoder()
        lbl.fit(list(X_train2[f].values) + list(X_valid2[f].values))
        X_train2[f] = lbl.transform(list(X_train2[f].values))
        X_valid2[f] = lbl.transform(list(X_valid2[f].values))  
X_train2 = X_train2.reset_index()
X_valid2 = X_valid2.reset_index()


'''
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
'''


# In[ ]:


#Now I'm doing same thing for our test set

trans_test = pd.read_csv('../input/test_transaction.csv').head(10000)
id_test = pd.read_csv('../input/test_identity.csv').head(10000)
trans_test.set_index('TransactionID').join(id_test.set_index('TransactionID'))
test0 = trans_test
test0 = test0.drop(cols_to_drop, axis=1)
for i in my_cols2:
    if i in test0: test0.drop(i, axis=1)
del trans_test, id_test
gc.collect()
test0_n = pd.DataFrame(numerical_transformer.fit_transform(test0[num_cols]))
test0_c = pd.DataFrame(cat_imp.fit_transform(test0[categorical_cols]))
test0_n.columns = num_cols
test0_c.columns = categorical_cols
test00 = test0_n.join(test0_c,how = 'left')
del test0_c, test0_n, test0

for f in test00.columns:
    if  test00[f].dtype=='object': 
        lbl = LabelEncoder()
        lbl.fit(list(test00[f].values))
        test00[f] = lbl.transform(list(test00[f].values))
test00 = test00.reset_index()


# In[ ]:


#This is by far the coolest part, so let me explain what is going to happen.
#We will use XGB regressor as our model (tried also random forest, but it a bit
#worse), but we don't know what our hyperparameters should be. But we won't use
#grid, we will minimize the number of searches by using bayesian optimization!

import warnings
model = XGBRegressor()
#model = RandomForestRegressor(n_estimators=400, random_state=0)
X_train1 = X_train2
y_train1 = y_train_full
y_valid = y_valid_full.head(3000)
X_valid = X_valid2.head(3000)

def BB_fun(max_depth, n_estimators, learning_rate):# reg_alpha, reg_lambda):
    param = {
        'max_depth':int(max_depth),
        'n_estimators':int(n_estimators),
        'objective':'binary:logistic',
        'learning_rate':learning_rate,
        #'reg_alpha':reg_alpha,
        #'reg_lambda':reg_lambda,
        'random_state':0
    }
    model = XGBRegressor(**param)#max_depth=int(max_depth), n_estimators=int(n_estimators),objective='binary:logistic', learning_rate=learning_rate, reg_alpha=reg_alpha, reg_lambda=reg_lambda)
    clf = model.fit(X_train1,y_train1)
    YY = clf.predict(X_valid)
    score = roc_auc_score(y_valid,YY)
    return score


#clf = Pipeline(steps=[('preprocessor', preprocessor),
#                      ('model', model)
#                     ])

pbounds = {'max_depth':(2,5),
           'n_estimators':(50,400),
           'learning_rate':(0.04,0.2),
           #'reg_alpha':(0,0.5),
           #'reg_lambda':(0.5,1.5),
          }
optimizer = BayesianOptimization(
        f = BB_fun,
        pbounds = pbounds,
        verbose = 2,
        random_state=0
)
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    optimizer.maximize(init_points=10, n_iter=10, acq='ucb', xi=0.0, alpha=1e-6)
    


# In[ ]:


model = XGBRegressor(learning_rate=0.1278,max_depth=4,n_estimators=261)
model.fit(X_train2,y_train_full)
print("gotowe")
del X_valid_full, X_train_full, y_train_full, y_valid_full
gc.collect()
YY = model.predict(test00)


# In[ ]:


output = pd.DataFrame({'Id': s_s.index,
                       'isFraud': YY})
output.to_csv('submission.csv', index=False)

