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
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/heart.csv")


# In[ ]:


df.head()


# In[ ]:


cp = df.iloc[:,2].values
cp


# In[ ]:


dummy_var = pd.get_dummies(cp, drop_first=True)

newdf = df.drop(["cp"], axis = 1)


# In[ ]:


mergeddf = pd.concat([newdf,dummy_var],axis=1)
target = mergeddf.pop("target").values


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(mergeddf)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(mergeddf, target, test_size=0.33, random_state=42)


# In[ ]:


## Applying Linear Regression ##
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)
linear_pred = reg.predict(X_test)


# In[ ]:



linear_pred = linear_pred>0.5


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, linear_pred)


# In[ ]:


## Let's check F1 Score 
from sklearn.metrics import f1_score
f1_score(y_test, linear_pred, average='macro')


# In[ ]:


# since this is a medical problem we should focus on Precision rather than Recall 
from sklearn.metrics import fbeta_score
fbeta_score(y_test, linear_pred, average='macro', beta=0.1)


# In[ ]:


## lets Apply SVM to it 
from sklearn import svm
clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train) 


# In[ ]:


svm_pred = clf.predict(X_test)


# In[ ]:


fbeta_score(y_test, svm_pred, average='macro', beta=0.1)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
clf.fit(X_train, y_train)
rf_pred = clf.predict(X_test)


# In[ ]:


fbeta_score(y_test, rf_pred, average='macro', beta=0.1)


# In[ ]:


from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression(random_state=0, solver='liblinear',multi_class='auto').fit(X_train, y_train)
l_regression = clf_lr.predict(X_test)


# In[ ]:


fbeta_score(y_test, l_regression, average='macro', beta=0.1)


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=4)
principalComponents  = pca.fit_transform(mergeddf) 
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4'])
principalDf


# In[ ]:


from sklearn.model_selection import train_test_split
X_trainpc, X_testpc, y_trainpc, y_testpc = train_test_split(principalDf, target, test_size=0.33, random_state=42)


# In[ ]:


from sklearn import svm
clf = svm.SVC(gamma='scale')
clf.fit(X_trainpc, y_trainpc) 
svm_pca = clf.predict(X_testpc)


# In[ ]:


fbeta_score(y_testpc, svm_pca, average='macro', beta=0.1)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
clf.fit(X_trainpc, y_trainpc)
rf_pred_pc = clf.predict(X_testpc)


# In[ ]:


fbeta_score(y_testpc, rf_pred_pc, average='macro', beta=0.1)


# In[ ]:


parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}
train_data = lightgbm.Dataset(scaled_df, label=target)
test_data = lightgbm.Dataset(X_test, label=y_test)

model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=5000,
                       early_stopping_rounds=100)


# In[ ]:


lgb = model.predict(X_test)


# In[ ]:


lgb = lgb>0.5


# In[ ]:


lgb


# In[ ]:


fbeta_score(y_test, lgb, average='macro', beta=0.1)


# In[ ]:




