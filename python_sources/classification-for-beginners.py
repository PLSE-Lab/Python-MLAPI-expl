#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:


input_df = pd.read_csv("../input/train.csv")


# In[ ]:


input_df.head()


# In[ ]:


input_df = input_df.drop("Name",axis=1)


# In[ ]:


input_df["Ticket"].describe()


# In[ ]:


input_df = input_df.drop("Ticket",axis=1)


# In[ ]:


input_df["Cabin"].describe()


# In[ ]:


input_df["Cabin"] = input_df["Cabin"].apply(lambda x : x[0] if x is not np.NaN else 0)


# In[ ]:


dct = {"A": 8, "B":7,"C":6,"D":5,"E":4,"F":3,"G":2,"T":1}


# In[ ]:


for i in range(len(input_df["Cabin"])):
    for k,v in dct.items():
        if input_df["Cabin"][i] == k:
            input_df["Cabin"][i] = int(v)
            break


# In[ ]:


input_df["Cabin"] = pd.to_numeric(input_df["Cabin"])


# In[ ]:


input_df = pd.get_dummies(input_df)


# In[ ]:


input_df.shape


# In[ ]:


input_df.head()


# In[ ]:


y = input_df["Survived"]


# In[ ]:


input_df = input_df.drop("Survived",axis=1)


# In[ ]:


input_df.info()


# In[ ]:


imputer = SimpleImputer()


# In[ ]:


input_df["Age"] = imputer.fit_transform(np.array(input_df["Age"]).reshape(-1,1))


# In[ ]:


test = pd.read_csv("../input/test.csv")


# In[ ]:


test.head()


# In[ ]:


test = test.drop("Name",axis=1)


# In[ ]:


test = test.drop(["Ticket", "Cabin"],axis = 1)


# In[ ]:


test.info()


# In[ ]:


test = pd.get_dummies(test)


# In[ ]:


test["Age"] = imputer.transform(np.array(test["Age"]).reshape(-1,1))


# In[ ]:


test["Fare"] = imputer.transform(np.array(test["Fare"]).reshape(-1,1))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(input_df.iloc[:,1:],y, test_size= 0.3, random_state=42)


# In[ ]:


X_train.info()


# In[ ]:


clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:


confusion_matrix(y_test, y_pred)


# In[ ]:


clf.score(X_test,y_test)


# In[ ]:


rf = RandomForestClassifier()


# In[ ]:


rf.fit(X_train, y_train)


# In[ ]:


y_pred = rf.predict(X_test)


# In[ ]:


confusion_matrix(y_test, y_pred)


# In[ ]:


rf.score(X_test,y_test)


# In[ ]:


xgb = XGBClassifier()


# In[ ]:


xgb.fit(X_train, y_train)


# In[ ]:


y_pred = xgb.predict(X_test)


# In[ ]:


confusion_matrix(y_test, y_pred)


# In[ ]:


xgb.score(X_test,y_test)


# In[ ]:


xgb_tuning = XGBClassifier(learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)


# In[ ]:


param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}


# In[ ]:


gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
                                                 min_child_weight=1, gamma=0, 
                                                  subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic',
                                                  nthread=4, scale_pos_weight=1, seed=27), 
                        param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(input_df.iloc[:,1:],y)


# In[ ]:


gsearch1.best_params_, gsearch1.best_score_


# In[ ]:


param_test2 = {
 'max_depth':[8,9,10],
 'min_child_weight':[4,5,6]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(input_df.iloc[:,1:],y)


# In[ ]:


gsearch2.best_params_, gsearch2.best_score_


# In[ ]:


param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=9,
 min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(input_df.iloc[:,1:],y)


# In[ ]:


gsearch3.best_params_, gsearch3.best_score_


# In[ ]:


param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=9,
 min_child_weight=5, gamma=0.0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(input_df.iloc[:,1:],y)


# In[ ]:


gsearch4.best_params_, gsearch4.best_score_


# In[ ]:


param_test5 = {
 'subsample':[i/100.0 for i in range(8,12)],
 'colsample_bytree':[i/100.0 for i in range(6,10)]
}
gsearch5 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=9,
 min_child_weight=5, gamma=0.0, subsample=0.9, colsample_bytree=0.9,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch5.fit(input_df.iloc[:,1:],y)


# In[ ]:


gsearch5.best_params_, gsearch5.best_score_


# In[ ]:





# In[ ]:





# In[ ]:


xgb_gamma_tuning = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=6,
 min_child_weight=3, gamma=0.4, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27)


# In[ ]:


xgb_gamma_tuning.fit(input_df.iloc[:,1:],y)


# In[ ]:


y_pred = pd.DataFrame({"Survived" : xgb_gamma_tuning.predict(test.iloc[:,1:])})


# In[ ]:


y_pred.shape


# In[ ]:


y_pred["PassengerId"] = test["PassengerId"]


# In[ ]:


y_pred.to_csv("submission with gamma tuning.csv", index = False)


# In[ ]:


y_pred.head()


# In[ ]:


input_df.head()


# In[ ]:


param_test3 = {
 'gamma':[0.4]
}
gscv = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=6,
 min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gscv.fit(input_df.iloc[:,1:],y)


# In[ ]:


input_wo_Fare = input_df.drop("Fare",axis=1)


# In[ ]:


gscv.best_params_, gscv.best_score_


# ### **Lets find out if having cabin in our input_df helps the accuracy**

# In[ ]:





# In[ ]:





# In[ ]:




