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


dataset_train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
dataset_test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


# In[ ]:


dataset_train.head(10)


# In[ ]:


dataset_train.describe()


# In[ ]:


dataset_test.head(10)


# In[ ]:


dataset_test.describe()


# In[ ]:


dataset_train.info()


# In[ ]:


dataset_train_32 = dataset_train.drop(['ID_code','target'], axis=1).astype('float16')


# In[ ]:


dataset_train_32.info()


# In[ ]:


X_train = dataset_train_32.values
X_train


# In[ ]:


y_train = dataset_train.target.astype('uint8').values
y_train


# In[ ]:


X_test = dataset_test.iloc[:, 1:].astype('float16').values
X_test


# In[ ]:


X = X_train
y = y_train


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, stratify = y, random_state = 0)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[ ]:


x_test = dataset_test.iloc[:,2:]


# In[ ]:


x_test2 = dataset_test.iloc[:,1:].values


# In[ ]:


y_pred = logreg.predict(x_test2)
#print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:


id_n = dataset_test.ID_code.values


# In[ ]:





# In[ ]:


submission_logreg = pd.DataFrame({
    "ID_code": dataset_test["ID_code"],
    "target": y_pred
})
submission_logreg.to_csv('submission_logreg.csv', index=False)


# ***Naive Bayes***

# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


gnb = GaussianNB()


# In[ ]:


gnb.fit(X_train, y_train)


# In[ ]:


y_pred = gnb.predict(X_test)


# In[ ]:


from sklearn import metrics


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[ ]:


print("Accuracy:",metrics.classification_report(y_test, y_pred))


# In[ ]:


from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split


# In[ ]:


bnb = BernoulliNB(binarize=0.0)


# In[ ]:


bnb.fit(X_train, y_train)


# In[ ]:


bnb.score(X_test, y_test)


# In[ ]:


y_pred = bnb.predict(x_test2)


# In[ ]:


#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


#print("Accuracy:",metrics.classification_report(y_test, y_pred))


# In[ ]:


submission_naive_bayes = pd.DataFrame({
    "ID_code": dataset_test["ID_code"],
    "target": y_pred
})
submission_naive_bayes.to_csv('submission_naive_bayes.csv', index=False)


# ***XGBoots***

# In[ ]:


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[ ]:


import xgboost as xgb


# In[ ]:


xg_cl = xgb.XGBClassifier(objective = 'binary:logistic', n_estimators = 10, seed=123 )


# In[ ]:


xg_cl.fit(X_train, y_train)


# In[ ]:


y_pred = xg_cl.predict(x_test2)


# In[ ]:


import numpy as np
accuracy = float(np.sum(y_pred==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))


# In[ ]:


param_grid = {'max_depth': [6,7,8], 'gamma': [1, 2, 4], 'learning_rate': [1, 0.1, 0.01], 'objective':['binary:logistic'], 'eval_metric': ['auc'],'tree_method': ['gpu_hist'],'n_gpus': [1]}


# In[ ]:


dataset_dmatrix = xgb.DMatrix(data = X,label = y)
dataset_dmatrix


# In[ ]:


params = {"objective":"reg:logistic", "max_depth":3}
params


# In[ ]:


# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain = dataset_dmatrix, params = params, num_boost_round = 5, nfold = 3, metrics = "rmse", as_pandas = True, seed = 123)


# In[ ]:


print(cv_results)


# In[ ]:


print(1-cv_results["test-rmse-mean"].tail(1))


# In[ ]:


# Perform cross_validation: cv_results
cv_results = xgb.cv(dtrain = dataset_dmatrix, params = params, num_boost_round = 5, nfold = 3, metrics = "auc", as_pandas = True, seed = 123)# Perform cross_validation: cv_results
cv_results = xgb.cv(dtrain = dataset_dmatrix, params = params, num_boost_round = 5, nfold = 3, metrics = "auc", as_pandas = True, seed = 123)


# In[ ]:


print(cv_results)


# In[ ]:


print(cv_results["test-auc-mean"].tail(1))


# In[ ]:


submission_xgboots = pd.DataFrame({
    "ID_code": dataset_test["ID_code"],
    "target": y_pred
})
submission_xgboots.to_csv('submission_xgboots.csv', index=False)

