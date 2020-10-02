#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling


# In[ ]:


train = pd.read_csv('../input/hackathon/train.csv')
test = pd.read_csv('../input/new-test/test.csv')


# In[ ]:


train.describe()


# In[ ]:


X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]


# In[ ]:


X_train = X_train.drop(columns = ['loan_id','financial_institution', 'loan_purpose'])


# In[ ]:


test = test.drop(columns = ['financial_institution', 'loan_purpose'])


# In[ ]:


X_train['source'].unique()


# In[ ]:


def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()


# In[ ]:


def plotCorrelationMatrix(df, graphWidth):
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for give dataframe', fontsize=15)
    plt.show()


# In[ ]:


def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()


# plotPerColumnDistribution(X_train, 25, 1)

# plotCorrelationMatrix(X_train[predictors], 8)

# plotCorrelationMatrix(X_train, 8)

# In[ ]:


X_train['first_payment_date'].unique()


# In[ ]:


test['first_payment_date'].unique()


# In[ ]:


test['first_payment_date'] = test['first_payment_date'].map({'Apr-12': '04/2012', 'Mar-12':'03/2012', 'May-12': '05/2012', 'Feb-12':'02/2012'})


# In[ ]:


X_train['origination_date'].unique()


# In[ ]:


test['origination_date'].unique()


# In[ ]:


test['origination_date'] = test['origination_date'].map({'01/02/12': '2012-02-01', '01/01/12': '2012-01-01', '01/03/12': '2012-03-01'})


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


le = LabelEncoder()
X_train['source'] = le.fit_transform(X_train['source'])
test['source'] = le.transform(test['source'])


# In[ ]:


le1 = LabelEncoder()
X_train['first_payment_date'] = le1.fit_transform(X_train['first_payment_date'])
test['first_payment_date'] = le1.transform(test['first_payment_date'])


# In[ ]:


le2 = LabelEncoder()
X_train['origination_date'] = le2.fit_transform(X_train['origination_date'])
test['origination_date'] = le2.transform(test['origination_date'])


# In[ ]:


X_train['to_pay'] = X_train['unpaid_principal_bal'] + (X_train['unpaid_principal_bal']*X_train['loan_term']*X_train['interest_rate'])/100
test['to_pay'] = test['unpaid_principal_bal'] + (test['unpaid_principal_bal']*test['loan_term']*X_train['interest_rate'])/100


# In[ ]:


X_train['loan%'] = X_train['loan_to_value']/(1/X_train['debt_to_income_ratio'])/X_train['insurance_percent']
test['loan%'] = test['loan_to_value']/(1/test['debt_to_income_ratio'])/test['insurance_percent']


# In[ ]:


X_train['interest'] = (X_train['unpaid_principal_bal']*X_train['loan_term']*X_train['interest_rate'])/36000
test['interest'] = (test['unpaid_principal_bal']*test['loan_term']*X_train['interest_rate'])/36000


# In[ ]:


X_train['comp_interest'] = X_train['unpaid_principal_bal']*((1 + (X_train['interest_rate']/12))**(X_train['loan_term']/360))
test['comp_interest'] = test['unpaid_principal_bal']*((1 + test['interest_rate']/12)**(test['loan_term']/360))


# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


remove = ['number_of_borrowers', 'm1', 'm2', 'co-borrower_credit_score',
          'source', 'insurance_type', 'first_payment_date',
          'origination_date', 'm13', 'loan%', 'loan_term',
          'insurance_percent', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8','comp_interest']


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


X_train['m13'] = y_train


# In[ ]:


X_train['m13'].value_counts()


# In[ ]:


target = 'm13'
IDcol = ['loan_id']
from sklearn import model_selection, metrics


# In[ ]:


clf = XGBClassifier()


# In[ ]:


predictors = [x for x in X_train.columns if x not in [target]+IDcol + remove]


# In[ ]:


X_train[predictors] = X_train[predictors].astype(np.float64)


# In[ ]:


predictors


# In[ ]:


clf.fit(X_train[predictors], X_train[target])


# In[ ]:


train_pred = clf.predict(X_train[predictors])


# In[ ]:


n = X_train.shape[0]
p = X_train.shape[1] - 1


# In[ ]:


from sklearn.metrics import r2_score
r2 = r2_score(X_train['m13'], train_pred)
adj_r2 = 1 - ((1-r2)*((n-1)/(n-p-1)))
print(adj_r2)


# In[ ]:


test[target] = clf.predict(test[predictors])


# In[ ]:


IDcol.append(target)


# In[ ]:


submission = pd.DataFrame({x: test[x] for x in IDcol})


# In[ ]:


submission['m13'].sum()


# In[ ]:


submission.to_csv('alg0.csv', index = False)


# In[ ]:


featimp = pd.Series(clf.feature_importances_,index= predictors).sort_values(ascending=False)
featimp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')

X_train.shape[0]
# #undersampling due to imbalanced dataset

# In[ ]:


Count_paid_del = len(X_train[X_train['m13'] == 0])
Count_unpaid_del = len(X_train[X_train['m13'] == 1])
percentage_of_paid_del = Count_paid_del/(Count_paid_del+Count_unpaid_del)
print('percentage of paid delequency is',percentage_of_paid_del*100)


# In[ ]:


def undersample(df, times):
    unpaid_indices = np.array(df[df.m13 == 1].index)
    paid_indices = np.array(df[df.m13 == 0].index) 
    paid_indices_undersample = np.array(np.random.choice(paid_indices, (times*Count_unpaid_del), replace = False))
    undersample_data = np.concatenate([unpaid_indices, paid_indices_undersample])
    undersample_data = df.iloc[undersample_data, :]
    return(undersample_data)


# In[ ]:





# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report


# for i in range(23,24):
#     print('the undersample data for {} proportion'.format(i))
#     print()
#     train_undersample = undersample(X_train, i)
#     test_undersample = undersample(test, i)
#     model = XGBClassifier()
#     sub = modelfit(model, train_undersample, test, predictors, target, IDcol)
#     print('m13 sum = ', sub[target].sum())
#     featimp = pd.Series(model.feature_importances_,index= predictors).sort_values(ascending=False)
#     print(max(featimp))

# featimp = pd.Series(model.feature_importances_,index= predictors).sort_values(ascending=False)
# print(featimp)

# In[ ]:


predictors = [x for x in X_train.columns if x not in [target]+IDcol + remove]


# i = 35
# train_undersample = undersample(X_train, i)
# test_undersample = undersample(test, i)
# model = XGBClassifier(n_estimators = 100, early_stopping_rounds = 20)
# model.fit(train_undersample[predictors], train_undersample[target])
# test[target] = model.predict(test[predictors])
# IDcol.append(target)
# subm = pd.DataFrame({x: test[x] for x in IDcol})
# subm.to_csv('XGB23.csv', index = False)

# In[ ]:


from sklearn.model_selection import cross_validate 
from sklearn.model_selection import learning_curve,GridSearchCV


# In[ ]:


import xgboost as xgb


# In[ ]:


def modelfit(alg, train, test, predictors, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train[predictors].values, label=train[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='error', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(train[predictors], train[target],eval_metric='error')
        
    #Predict training set:
    train_predictions = alg.predict(train[predictors])
    train_predprob = alg.predict_proba(train[predictors])[:,1]
        
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(train[target].values, train_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(train[target], train_predprob))
    featimp = pd.Series(alg.feature_importances_,index= predictors).sort_values(ascending=False)
    featimp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


# xgb1 = XGBClassifier(
#  learning_rate =0.1,
#  n_estimators=1000,
#  max_depth=5,
#  min_child_weight=1,
#  gamma=0,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  objective= 'binary:logistic',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27)
# modelfit(xgb1, X_train, test, predictors, target)

# param_test1 = {
#  'max_depth':range(3,10,2),
#  'min_child_weight':range(1,6,2)
# }
# gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
# min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
# objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
# param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch1.fit(X_train[predictors],X_train[target])
# 

# gsearch1.best_params_, gsearch1.best_score_

# param_test2 = {
#  'max_depth':[3,4],
#  'min_child_weight':[1,2,3]
# }
# gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=3,
#  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
#  param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch2.fit(X_train[predictors],X_train[target])

# gsearch2.best_params_, gsearch2.best_score_

# param_test3 = {
#  'gamma':[i/10.0 for i in range(0,5)]
# }
# gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=3,
#  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
#  param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch3.fit(X_train[predictors],X_train[target])
# gsearch3.best_params_, gsearch3.best_score_

# xgb1 = XGBClassifier(
#  learning_rate =0.1,
#  n_estimators=1000,
#  max_depth=3,
#  min_child_weight=1,
#  gamma=0,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  objective= 'binary:logistic',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27)
# modelfit(xgb1, X_train, test, predictors, target)

# param_test4 = {
#  'subsample':[i/10.0 for i in range(6,10)],
#  'colsample_bytree':[i/10.0 for i in range(6,10)]
# }
# gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=3,
#  min_child_weight=1, gamma=0, subsample=0.9, colsample_bytree=0.9,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
# param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch4.fit(X_train[predictors],X_train[target])
# gsearch4.best_params_, gsearch4.best_score_

# param_test5 = {
#  'subsample':[i/100.0 for i in range(80,100,5)],
#  'colsample_bytree':[i/100.0 for i in range(80,100,5)]
# }
# gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
#  min_child_weight=6, gamma=0, subsample=0.9, colsample_bytree=0.9,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
#  param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch5.fit(X_train[predictors],X_train[target])

# gsearch5.best_params_, gsearch5.best_score_

# i = 35
# train_undersample = undersample(X_train, i)
# test_undersample = undersample(test, i)
# model = XGBClassifier( learning_rate =0.1, n_estimators=100, max_depth=4,
#  min_child_weight=6, gamma=0, subsample=0.9, colsample_bytree=0.85,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27)
# model.fit(train_undersample[predictors], train_undersample[target])
# test[target] = model.predict(test[predictors])
# IDcol.append(target)
# subm = pd.DataFrame({x: test[x] for x in IDcol})
# print(subm['m13'].sum())
# subm.to_csv('XGB35.csv', index = False)

# param_test7 = {
#  'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
# }
# gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
#  min_child_weight=6, gamma=0, subsample=0.9, colsample_bytree=0.85, reg_alpha = 0.01,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
#  param_grid = param_test7, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch7.fit(X_train[predictors],X_train[target])
# gsearch7.best_params_, gsearch7.best_score_

# i = 30
# train_undersample = undersample(X_train, i)
# test_undersample = undersample(test, i)
# 

# In[ ]:


from imblearn.under_sampling import (RandomUnderSampler, 
                                     ClusterCentroids,
                                     TomekLinks,
                                     NeighbourhoodCleaningRule,
                                     NearMiss,
                                    CondensedNearestNeighbour)
from imblearn.combine import (SMOTETomek, SMOTEENN)


# In[ ]:


from collections import Counter


# In[ ]:


print('Original dataset shape %s' % Counter(X_train[target]))
rus = SMOTETomek(ratio = 'auto')
X_res, y_res = rus.fit_resample(X_train[predictors], X_train[target])
print('Resampled dataset shape %s' % Counter(y_res))


# In[ ]:


X_res = pd.DataFrame(X_res, columns = predictors)


# In[ ]:


X_res.head()


# In[ ]:


y_res = pd.DataFrame(y_res, columns = [target])


# In[ ]:


y_res.head()


# model = XGBClassifier( learning_rate =0.001, n_estimators=1000, max_depth=4,
#  min_child_weight=6, gamma=0, subsample=0.9, colsample_bytree=0.85, reg_alpha = 0.01,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27)
# model.fit(train_undersample[predictors], train_undersample[target])
# test[target] = model.predict(test[predictors])
# IDcol.append(target)
# subm = pd.DataFrame({x: test[x] for x in IDcol})
# print(subm['m13'].sum())
# subm.to_csv('XGB35.csv', index = False)

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# clf = KNeighborsClassifier(n_neighbors = 5)
# param_test1 = {
#     'weights': ['uniform', 'distance']
# }
# gsearch1 = GridSearchCV(estimator = KNeighborsClassifier(n_neighbors = 8, leaf_size = 25, algorithm = 'brute'), 
# param_grid = param_test1, scoring='roc_auc',n_jobs =-1,iid=False, cv=5)
# gsearch1.fit(X_train[predictors],X_train[target])

# gsearch1.best_params_, gsearch1.best_score_

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# param_test1 = {
#     'n_estimators': [300, 500, 1000]
# }
# gsearch1 = GridSearchCV(estimator = RandomForestClassifier(n_estimators = 1000, max_features = 'sqrt', max_depth = 90, min_samples_leaf = 5, min_samples_split = 9), 
# param_grid = param_test1, scoring='roc_auc',n_jobs =-1,iid=False, cv=5)
# gsearch1.fit(X_train[predictors],X_train[target])

# gsearch1.best_params_, gsearch1.best_score_

# In[ ]:


from sklearn.metrics import accuracy_score
from vecstack import stacking


# In[ ]:


predictors


# In[ ]:


col = X_train.columns
for col in predictors:
    print(max(X_train[col]))


# In[ ]:


sc = StandardScaler()
X_train[predictors] = sc.fit_transform(X_train[predictors])
test[predictors] = sc.transform(test[predictors])


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# In[ ]:


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# model1 = [
#     XGBClassifier( learning_rate =0.0001, n_estimators=1000, max_depth=4,
#  min_child_weight=6, gamma=0, subsample=0.9, colsample_bytree=0.85, reg_alpha = 0.01,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
#     RandomForestClassifier(n_estimators = 1000, max_features = 'sqrt', max_depth = 90, min_samples_leaf = 5, min_samples_split = 9),
#     XGBClassifier( learning_rate =0.001, n_estimators=1000, max_depth=4,
#  min_child_weight=6, gamma=0, subsample=0.9, colsample_bytree=0.85, reg_alpha = 0.01,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27)
# ]

# models = [classifier(batch_size = 1000, epochs = 100),
#     XGBClassifier( learning_rate =0.001, n_estimators=1000, max_depth=4,
#  min_child_weight=6, gamma=0, subsample=0.9, colsample_bytree=0.85, reg_alpha = 0.01,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
#     KNeighborsClassifier(n_neighbors = 8, leaf_size = 25, algorithm = 'brute', weights = 'distance'),
#     RandomForestClassifier(n_estimators = 1000, max_features = 'sqrt', max_depth = 90, min_samples_leaf = 5, min_samples_split = 9),
#     XGBClassifier( learning_rate =0.001, n_estimators=1000, max_depth=4,
#  min_child_weight=6, gamma=0, subsample=0.9, colsample_bytree=0.85, reg_alpha = 0.01,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27)
# ]

# In[ ]:


model2 = [
    RandomForestClassifier(n_estimators = 1000, max_features = 'sqrt', max_depth = 45, min_samples_leaf = 5, min_samples_split = 9),
    RandomForestClassifier(n_estimators = 1000, max_features = 'sqrt', max_depth = 90, min_samples_leaf = 5, min_samples_split = 9),
    XGBClassifier( learning_rate =0.01, n_estimators=1000, max_depth=2,
 min_child_weight=6, gamma=0, subsample=0.9, colsample_bytree=0.85, reg_alpha = 0.01,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
    XGBClassifier( learning_rate =0.01, n_estimators=1000, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.9, colsample_bytree=0.85, reg_alpha = 0.01,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
    XGBClassifier( learning_rate =0.01, n_estimators=1000, max_depth=8,
 min_child_weight=6, gamma=0, subsample=0.9, colsample_bytree=0.85, reg_alpha = 0.01,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27)
]


# In[ ]:


S_train1, S_test1 = stacking(model2, X_res[predictors], y_res[target], test[predictors],
                           regression = False,
                           mode = 'oof_pred_bag',
                           needs_proba = False,
                           save_dir = None,
                           metric = accuracy_score,
                           n_folds = 4,
                           stratified = True, 
                           shuffle = True,
                           random_state = 0,
                           verbose = 2)


# model2 = [
#     XGBClassifier( learning_rate =0.1, n_estimators=1000, max_depth=4,
#  min_child_weight=6, gamma=0, subsample=0.9, colsample_bytree=0.85, reg_alpha = 0.01,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
#     XGBClassifier( learning_rate =0.01, n_estimators=1000, max_depth=4,
#  min_child_weight=6, gamma=0, subsample=0.9, colsample_bytree=0.85, reg_alpha = 0.01,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
#     XGBClassifier( learning_rate =0.001, n_estimators=1000, max_depth=4,
#  min_child_weight=6, gamma=0, subsample=0.9, colsample_bytree=0.85, reg_alpha = 0.01,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
#     XGBClassifier( learning_rate =0.0001, n_estimators=1000, max_depth=4,
#  min_child_weight=6, gamma=0, subsample=0.9, colsample_bytree=0.85, reg_alpha = 0.01,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
#     XGBClassifier( learning_rate =0.00001, n_estimators=1000, max_depth=4,
#  min_child_weight=6, gamma=0, subsample=0.9, colsample_bytree=0.85, reg_alpha = 0.01,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27)
# ]

# S_train2, S_test2 = stacking(model2, S_train1, train_undersample[target], S_test1,
#                            regression = False,
#                            mode = 'oof_pred_bag',
#                            needs_proba = False,
#                            save_dir = None,
#                            metric = accuracy_score,
#                            n_folds = 4,
#                            stratified = True, 
#                            shuffle = True,
#                            random_state = 0,
#                            verbose = 2)

# In[ ]:


S_test1.shape


# In[ ]:


model = XGBClassifier( learning_rate =0.001, n_estimators=1000, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.9, colsample_bytree=0.85, reg_alpha = 0.01,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27)
model.fit(S_train1, y_res[target])
test[target] = model.predict(S_test1)
IDcol.append(target)
subm = pd.DataFrame({x: test[x] for x in IDcol})
print(subm['m13'].sum())
subm.to_csv('XGB35S.csv', index = False)


# In[ ]:





# In[ ]:




