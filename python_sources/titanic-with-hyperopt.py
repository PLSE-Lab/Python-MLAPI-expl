#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.linear_model import LogisticRegression
from scipy.stats import iqr
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import recall_score, precision_score, accuracy_score, precision_recall_curve, roc_auc_score, roc_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, ShuffleSplit

import gc


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")


def clean(data):
    data.columns = data.columns.str.strip().str.upper().str    .replace(' ', '_').str.replace('(', '').str.replace(')', '')
    return data

for item in (train,test):
    item = clean(item)
    numeric_cols = [cname for cname in item.columns if
                item[cname].dtype in ['int64', 'float64']]
    for columns in numeric_cols:
            item[columns].fillna(0, inplace=True)
train.head()


# In[ ]:


#need to credit original scripter
import seaborn as sns
def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(1,1,figsize=(30,30))

    for feature in features:
        i += 1
        plt.subplot(10,10,i)
        sns.distplot(df1[feature], hist=False,label=label1)
        sns.distplot(df2[feature], hist=False,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show();


numeric_cols = [cname for cname in train.columns if
                train[cname].dtype in ['int64', 'float64']]
X=train[numeric_cols].drop(['PASSENGERID'],axis=1)
t0 = X.loc[X['SURVIVED'] == 0]
t1 = X.loc[X['SURVIVED'] == 1]
features = X.columns.values[2:102]
plot_feature_distribution(t0, t1, '0', '1', features)


# In[ ]:


proposed_X = train.drop(['PASSENGERID','SURVIVED'],axis=1)

low_cardinality_cols = [cname for cname in proposed_X.columns if
                        proposed_X[cname].nunique() < 50 and
                        proposed_X[cname].dtype == "object"]

numeric_cols = [cname for cname in proposed_X.columns if
                proposed_X[cname].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numeric_cols

# One hot encoded
X=pd.get_dummies(proposed_X[my_cols])
print("# of columns: {0}"      .format(len(train.columns)))

FEATURES=list(X.columns)
X['NEW_FEAT']=X.SIBSP+X.PARCH


y=train['SURVIVED']

df_test = test.drop(['PASSENGERID'],axis=1)

df_test=pd.get_dummies(df_test
                       [my_cols])
df_test['NEW_FEAT']=df_test.SIBSP+df_test.PARCH
del proposed_X
gc.collect()
   #%%         
import pandas as pd
from sklearn.linear_model import LogisticRegression
from collections import Counter

xx=X[['AGE','SEX_male','SEX_female','PCLASS']]
propensity = LogisticRegression(solver='lbfgs',penalty='l2')
propensity = propensity.fit(xx,y)

pscore = (propensity.predict_proba(xx)[:,1])
test_xx=df_test[['AGE','SEX_male','SEX_female','PCLASS']]
test_pscore = (propensity.predict_proba(test_xx)[:,1])

#pscore=pscore.round()
#test_pscore = test_pscore.round()
print(pscore[:5])
print(test_pscore[:5])
X['Propensity'] = pscore
df_test['Propensity'] = test_pscore

(Counter(pscore))
Counter(test_pscore)

df_pdf=[]
for data in (X,df_test):
    for item in ('NEW_FEAT','Propensity','AGE'):
        n=round((max((data[item]))-min(data[item])) / (2 * iqr(data[item]) / len(data[item])**(1/3))).astype(int)
        data['new'+item] = (pd.cut(data[item], n, labels= list(range(1,n+1)))).astype(int)


# In[ ]:



import warnings
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,test_size=0.2,
                                                   random_state=4)

def objective(space):

    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    classifier = xgb.XGBClassifier(n_estimators = space['n_estimators'],
                            max_depth = (space['max_depth']),
                            learning_rate = space['learning_rate'],
#                            gamma = space['gamma'],
#                            min_child_weight = space['min_child_weight'],
#                            subsample = space['subsample'],
#                            colsample_bytree = space['colsample_bytree'],
#                            
#                            base_score=space['base_score'], 
# #                           max_delta_step=space['max_delta_step'], 
#                            scale_pos_weight=space['scale_pos_weight'],
                            reg_alpha=space['reg_alpha'], 
                            reg_lambda=space['reg_lambda'], 
#                            
#                            random_state=0,                             
#                            seed=None, 
#                            silent=True, 
##                            booster='gbtree'
##                            objective='binary:logistic',
#                            missing=None,
#                            nthread=-1
#                            #n_jobs=1,
                            )
    
    classifier.fit(X_train, y_train)

    # Applying Cross Validation
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier, 
                                 X = X_train, y = y_train, scoring='roc_auc', cv = 10)
    CrossValMean = accuracies.mean()

    print("CrossValMean:", CrossValMean)

    return{'loss':1-CrossValMean, 'status': STATUS_OK }

space = {
#    'base_score' : hp.quniform('base_score', .25, .75, .5),
##    'max_delta_step' : hp.choice('max_delta_step', range(5, 30, 1)),
#    'scale_pos_weight' :  hp.quniform('scale_pos_weight', .25, .75, .5),
    'reg_alpha' : hp.choice('reg_alpha', 2. **np.arange(-20, 20, 1)),
    'reg_lambda' : hp.choice('reg_lambda', range(1, 300, 1)),
    'max_depth' : hp.choice('max_depth', range(1, 300, 1)),
    'learning_rate' : hp.quniform('learning_rate', 0.01, 0.5, 0.01),
    'n_estimators' : hp.choice('n_estimators', range(1, 2000, 5)),
#    'gamma' : hp.quniform('gamma', 0, 0.95, 0.01),
#    'min_child_weight' : hp.quniform('min_child_weight', 1, 10, 1),
#    'subsample' : hp.quniform('subsample', 0.1, 1, 0.01),
#    'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1.0, 0.01)
    }

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=40,
            trials=trials)

print("Best: ", best)


# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier(n_estimators = best['n_estimators'],
                            max_depth = (best['max_depth']),
                            learning_rate = best['learning_rate'],
#                            gamma = best['gamma'],
#                            min_child_weight = best['min_child_weight'],
#                            subsample = best['subsample'],
#                            colsample_bytree = best['colsample_bytree'],
#                            
#                            base_score=best['base_score'], 
# #                           max_delta_step=best['max_delta_step'], 
#                            scale_pos_weight=best['scale_pos_weight'],
                            reg_alpha=best['reg_alpha'], 
                           reg_lambda=best['reg_lambda'], 
#                            
#                            random_state=0,                             
#                            seed=None, 
#                            silent=True, 
#                            booster='gbtree'
#                            objective='binary:logistic',
                            missing=None,
                            nthread=-1
                            #n_jobs=1,
                            )

classifier.fit(X_train, y_train)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
CrossValMean = 1-accuracies.mean()
print("Final CrossValMean: ", CrossValMean)

CrossValSTD = accuracies.std()
X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,test_size=0.2,
                                                   random_state=4)


# In[ ]:


print('Overall AUC:', roc_auc_score(y_train, classifier.predict_proba(X_train)[:,1]))
#print('Predict the probabilities on test set')
#pred = clf.predict_proba(X_train, ntree_limit=cvresult.shape[0])

xgb.plot_importance(classifier,max_num_features=10)


# In[ ]:


xgb.to_graphviz(classifier,size="10,10!")


# In[ ]:



y_pred_xgb = classifier.predict_proba(X_test)[:, 1]
fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_xgb)

plt.figure(2)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_grd, tpr_grd, label='GBT')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
plt.close()
average_precision = average_precision_score(y_test, y_pred_xgb)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))


# In[ ]:


precision,recall,_ =precision_recall_curve(y_test, y_pred_xgb)
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.show()
plt.close()


# In[ ]:


predictions = classifier.predict(df_test)
id_test = pd.read_csv("../input/test.csv")

submission=test[['PASSENGERID']]
submission['SURVIVED']=predictions
print(submission)

