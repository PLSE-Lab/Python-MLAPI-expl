#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


import missingno


# In[ ]:


df = pd.read_csv("../input/train.csv")


# In[ ]:


df.head()


# In[ ]:


df["target"].value_counts()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


missingno.dendrogram(df)


# ### No missing  values in any of the columns. Lets go ahead and build some models. And get their accuracies.

# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb

classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(),
    XGBClassifier()]


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


ss = StandardScaler()


# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:


sm = SMOTE(random_state=42, k_neighbors=3, n_jobs = 5)


# In[ ]:


X,y = sm.fit_resample(df.iloc[:,2:],df["target"])


# In[ ]:


y.shape, X.shape


# In[ ]:


X = ss.fit_transform(X)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)


# In[ ]:


for clf in classifiers:
    name = clf.__class__.__name__
    clf.fit(X_train, y_train)
    acc = clf.score(X_test,y_test)
    print("{0}: {1}".format(name,acc))


# In[ ]:


test_df = pd.read_csv("../input/test.csv")


# In[ ]:


test_df.head()


# In[ ]:


missingno.dendrogram(test_df)


# ### Lets create a model on GaussianNB and use it on test data

# In[ ]:


gnb = GaussianNB()
gnb.fit(X,y)
test_df_pred = pd.DataFrame({"target": gnb.predict_proba(test_df.iloc[:,1:])})


# In[ ]:


pd.read_csv("../input/sample_submission.csv").head()


# In[ ]:


test_df_pred["id"] = test_df["id"]


# In[ ]:


test_df_pred.head()


# In[ ]:


test_df_pred.to_csv("Submission.csv", index=False)


# ### So we have 0.611 as our score. We can improve this with a bit of a feature engineering.  Lets apply decomposition on the dataset and see what happens.

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,250,1)


# In[ ]:


sklearn_pca = PCA(n_components=250)
print(sklearn_pca)


# In[ ]:


X_train_pca = sklearn_pca.fit_transform(X_train)
print(X_train_pca.shape)

X_test_pca = sklearn_pca.transform(X_test)
print(X_test_pca.shape)


# In[ ]:


classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(),
    XGBClassifier()]
for clf in classifiers:
    name = clf.__class__.__name__
    clf.fit(X_train_pca, y_train)
    acc = clf.score(X_test_pca,y_test)
    print("{0}: {1}".format(name,acc))


# In[ ]:


sklearn_pca = PCA(n_components=200)
print(sklearn_pca)


# In[ ]:


X_pca = sklearn_pca.fit_transform(X)
print(X_pca.shape)

test_pca = sklearn_pca.transform(test_df.iloc[:,1:])
print(test_pca.shape)


# In[ ]:


qda = QuadraticDiscriminantAnalysis()
qda.fit(X_pca,y)
test_df_pred = pd.DataFrame({"target": qda.predict_proba(test_pca)})
test_df_pred["id"] = test_df["id"]
test_df_pred.to_csv("Submission with pca and qda.csv", index=False)


# In[ ]:


qda = QuadraticDiscriminantAnalysis()
qda.fit(X,y)
test_df_pred = pd.DataFrame({"target": qda.predict_proba(test_df)})
test_df_pred["id"] = test_df["id"]
test_df_pred.to_csv("Submission with qda.csv", index=False)


# ## GridSearchCV on QDA

# In[ ]:


from sklearn.model_selection import GridSearchCV


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
 seed=27,
n_gpus=1)


# In[ ]:


xgb_tuning.fit(X_train,y_train)


# In[ ]:


xgb_tuning.score(X_test,y_test)


# In[ ]:


param_test1 = {
 'max_depth':[7,8,9,10],
 'min_child_weight':[0,1,2]
}


# In[ ]:


gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
                                                 min_child_weight=1, gamma=0, 
                                                  subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic',
                                                  nthread=32, scale_pos_weight=1, seed=27,n_gpus=1), 
                        param_grid = param_test1, scoring='roc_auc',n_jobs=32,iid=False, cv=5,verbose=4)
gsearch1.fit(X_train,y_train)
gsearch1.best_params_, gsearch1.best_score_


# In[ ]:


xgboost_params = { "n_estimators": 400, 'tree_method':'gpu_hist', 'predictor':'gpu_predictor' }


# In[ ]:


param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, max_depth=9,
 min_child_weight=0, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=32, scale_pos_weight=1,seed=27,n_gpus=1,**xgboost_params), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=32,iid=False,verbose=4 ,cv=5)
gsearch3.fit(X,y)
gsearch3.best_params_, gsearch3.best_score_


# In[ ]:


param_test4 = {
 'colsample_bytree':[i/10.0 for i in range(7,10)],
    'subsample': [i/10.0 for i in range(7,10)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, max_depth=9,
 min_child_weight=0, gamma=0.0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=32, scale_pos_weight=1,seed=27,n_gpus=1,**xgboost_params), 
 param_grid = param_test4, scoring='roc_auc',n_jobs=32,iid=False,verbose=5 ,cv=5)
gsearch4.fit(X,y)
gsearch4.best_params_, gsearch4.best_score_


# In[ ]:


param_test5 = {
    'subsample': [i/10.0 for i in range(5,8)]
}
gsearch5 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, max_depth=9,
 min_child_weight=0, gamma=0.0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=32, scale_pos_weight=1,seed=27,n_gpus=1,**xgboost_params), 
 param_grid = param_test5, scoring='roc_auc',n_jobs=32,iid=False,verbose=5 ,cv=5)
gsearch5.fit(X,y)
gsearch5.best_params_, gsearch5.best_score_


# In[ ]:


param_test7 = {
 'reg_alpha':[i/10.0 for i in range(0,3)],
    'reg_lambda':[i/10.0 for i in range(9,11)]
}
gsearch7 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.11, max_depth=9,
 min_child_weight=0, gamma=0.0, subsample=0.7, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=5, scale_pos_weight=1,seed=27,n_gpus=1,**xgboost_params), 
 param_grid = param_test7, scoring='roc_auc',n_jobs=32,iid=False,verbose=5 ,cv=5)
gsearch7.fit(X,y)
gsearch7.best_params_, gsearch7.best_score_


# In[ ]:


param_test8 = {
 'learning_rate' :[0.08,0.09,0.1,0.11,0.12]
}
gsearch8 = GridSearchCV(estimator = XGBClassifier(
                                learning_rate =0.1,max_depth=9,min_child_weight=0, gamma=0.0, 
                                subsample=0.7, colsample_bytree=0.8,objective= 'binary:logistic', 
                                nthread=5,reg_alpha= 0.1, reg_lambda= 0.9, 
                                scale_pos_weight=1,seed=27,n_gpus=1,**xgboost_params), 
                        param_grid = param_test8, scoring='roc_auc',
                        n_jobs=32,iid=False,verbose=5 ,cv=5
                       )
gsearch8.fit(X,y)
gsearch8.best_params_, gsearch8.best_score_


# In[ ]:


xgb_tuning = XGBClassifier(learning_rate =0.11,
 n_estimators=400,
 max_depth=9,
 min_child_weight=0,
 gamma=0.0,
 subsample=0.7,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27,
reg_alpha= 0.1, reg_lambda= 0.9,
n_gpus=1,
tree_method='gpu_hist',
predictor='gpu_predictor')


# In[ ]:


xgb_tuning.fit(X_train,y_train)


# In[ ]:


xgb_tuning.score(X_test,y_test)


# In[ ]:


X.shape


# In[ ]:


test_df.head()


# In[ ]:


xgb_tuning.fit(X,y)
test_df_pred = pd.DataFrame({"target": xgb_tuning.predict_proba(np.array(test_df.drop("id",axis=1)))})
test_df_pred["id"] = test_df["id"]
test_df_pred.to_csv("Submission xgb tuning.csv", index=False)


# ### XGBoostClassifier is not getting us anywhere. It needs more data to perform better. Lets try logistic regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lrcv = LogisticRegression(C=50000,penalty="l2")
lrcv.fit(X_train,y_train)


# In[ ]:


lrcv.score(X_test,y_test)


# In[ ]:


param1 = {
 'penalty':["l1","l2"]
}
gsearch = GridSearchCV(estimator = LogisticRegression(C=5, class_weight=None, dual=False,
                                                      fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l1', random_state=42, solver='warn',
          tol=0.0001, verbose=0, warm_start=False), 
 param_grid = param1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch.fit(X,y)
gsearch.best_params_,gsearch.best_score_


# In[ ]:


param2 = {
 'C':[i/1000 if i!=0 else 1 for i in range(0,100000,10)]
}
gsearch2 = GridSearchCV(estimator = LogisticRegression(C=5, class_weight=None, dual=False,
                                                       fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=42, solver='warn',
          tol=0.0001, verbose=0, warm_start=False), 
 param_grid = param2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(X,y)
gsearch2.best_params_, gsearch2.best_score_


# In[ ]:


param3 = {
 'tol':[i/10000 if i!=0 else 1 for i in range(0,1000,1)]
}
gsearch3 = GridSearchCV(estimator = LogisticRegression(C=2.4, class_weight=None, 
                                                      dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=42, solver='warn',
          tol=0.0001, verbose=0, warm_start=False), 
 param_grid = param3, scoring='roc_auc',verbose=4,n_jobs=32,iid=False, cv=5)
gsearch3.fit(X,y)
gsearch3.best_params_, gsearch3.best_score_


# In[ ]:


param4 = {
 'max_iter':[i for i in range(8,400,1)]
}
gsearch4 = GridSearchCV(estimator = LogisticRegression(C=2.4, class_weight=None, 
                                                      dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=42, solver='warn',
          tol=0.0001, verbose=0, warm_start=False), 
 param_grid = param4, scoring='roc_auc',verbose=4,n_jobs=32,iid=False, cv=5)
gsearch4.fit(X,y)
gsearch4.best_params_, gsearch4.best_score_


# In[ ]:


param5 = {
 'class_weight':["balanced",None],
}
gsearch5 = GridSearchCV(estimator = LogisticRegression(C=2.4, class_weight=None, 
                                                      dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=8, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=42, solver='warn',
          tol=0.0001, verbose=0, warm_start=False), 
 param_grid = param5, scoring='roc_auc',verbose=4,n_jobs=-1,iid=False, cv=5)
gsearch5.fit(X,y)
gsearch5.best_params_, gsearch5.best_score_


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lr = LogisticRegression(C=2.4, class_weight="balanced", 
                                                      dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=8, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=42, solver='warn',
          tol=0.0001, warm_start=False)
lr.fit(X_train,y_train)


# In[ ]:


lr.score(X_test,y_test)


# In[ ]:


lr.fit(X,y)
y_pred = pd.DataFrame({"target":lr.predict_proba(test_df.iloc[:,1:])})
y_pred["id"] = test_df["id"] 
y_pred.to_csv("submission with tuned logistic regression.csv", index = False)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier 


# In[ ]:


rf = RandomForestClassifier(n_estimators=200, n_jobs=4, class_weight='balanced', max_depth=6)
rf.fit(X_train,y_train)
rf.score(X_test, y_test)


# ### We will try Recursive feature elimination with our lr object

# In[ ]:


from sklearn.feature_selection import RFE


# In[ ]:


lr_rfe = RFE(lr, 75, step=1)
lr_rfe.fit(X_train,y_train)
# scores_table(selector, 'selector_clf')
lr_rfe.score(X_test, y_test) 
y_pred = lr_rfe.predict_proba(test_df.iloc[:,1:])
s = pd.read_csv('../input/sample_submission.csv')
s["target"] = y_pred
s.to_csv("submission with RFE.csv", index = False)

