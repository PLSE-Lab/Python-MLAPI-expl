#!/usr/bin/env python
# coding: utf-8

# # 1. Data preprocessing and visualisations
# ## Importing libraries and datasets

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_df=pd.read_csv('../input/amazon-employee-access-challenge/train.csv')
train_df.head()


# ## Data description
# After eyeballing the above data, it is clear that the first column which is the action is our target variable. Let us check what each of the columns actually mean.
# 
# Column Name	Description
# 
# **ACTION**	ACTION is 1 if the resource was approved, 0 if the resource was not
# 
# **RESOURCE**	An ID for each resource
# 
# **MGR_ID**	The EMPLOYEE ID of the manager of the current EMPLOYEE ID record; an employee may have only one manager at a time
# 
# **ROLE_ROLLUP_1**	Company role grouping category id 1 (e.g. US Engineering)
# 
# **ROLE_ROLLUP_2**	Company role grouping category id 2 (e.g. US Retail)
# 
# **ROLE_DEPTNAME**	Company role department description (e.g. Retail)
# 
# **ROLE_TITLE**	Company role business title description (e.g. Senior Engineering Retail Manager)
# 
# **ROLE_FAMILY_DESC**	Company role family extended description (e.g. Retail Manager, Software Engineering)
# 
# **ROLE_FAMILY**	Company role family description (e.g. Retail Manager)
# 
# **ROLE_CODE**	Company role code; this code is unique to each role (e.g. Manager)

# Let us separate out the action column into another dataframe called the target_df.

# In[ ]:


target_df=pd.DataFrame(train_df['ACTION'],columns=['ACTION'])


# In[ ]:


target_df['ACTION'].value_counts()


# As we can see, majority of the entries had the resources approved. Let us visualise this using a barplot.

# In[ ]:


sns.catplot('ACTION',data=target_df,kind='count')
plt.title('Action distributions',size=20)


# Let us check the various data types in the dataframe presented to us.

# In[ ]:


train_df.dtypes


# As we can see, all the data is in integer form. This is extremely helpful for machine learning purpose as it doesn't require any further feature engineering or preprocessing to be done.
# 
# Infact, it could be said that most of the preprocessing was already done for us. This is because although we have many categorical features, the data is in integer form. This suggests that the categorical data that we have is already label or ordinal encoded for us.
# 
# 
# Let us check if we have any missing variables to be take care of.

# In[ ]:


train_df.isna().any()


# As we can see, we have no particular missing values at all in the dataframe. Hence, it can be said that the dataframe is already preprocessed with removal of all the missing values.
# 
# This means we can directly head towards machine learning.

# In[ ]:


train_df.drop('ACTION',axis=1,inplace=True)


# # 2. Machine Learning
# 
# ## KNN

# Initially, let us use the KNearestNeighbor classifier to see how it classifies the ACTION correctly. For our classification, we will use a range of K values from 1 to 12.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix,roc_curve,precision_recall_curve,auc
from sklearn.preprocessing import StandardScaler


# In[ ]:


param_grid={'n_neighbors':[3,5,7]}
knn=KNeighborsClassifier()


# In[ ]:


X=train_df.values
y=target_df['ACTION'].values
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=0)


# In[ ]:


grid_search=GridSearchCV(knn,param_grid,scoring='roc_auc')


# In[ ]:


grid_result=grid_search.fit(X_train,y_train)


# In[ ]:


grid_result.best_params_


# In[ ]:


grid_result.score(X_train,y_train)


# However, to dig deeper into what value of K will be good, let us make a graph of accuracy Vs K for better understanding.

# In[ ]:


scores=[]
for i in range(1,13):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    cv_scores=cross_val_score(knn,X,y,cv=15)
    scores.append(cv_scores.mean())
    


# In[ ]:


neighbors=np.arange(1,13)
plt.figure(figsize=(10,8))
sns.set(style='white')
plt.plot(neighbors,scores,color='b')
plt.axvline(7,color='g')
plt.axhline(0.94,color='red')
plt.title('Accuracy Vs Neighbors',size=20)
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy score')
ticks=np.arange(1,13)
plt.xticks(ticks)


# From the above graph, it is clear that at **N neighbors=7** , the accuracy is quite high. This was confirmed by the GridSearchCV aswell. We try not to exceed 7 neighbors even if the accuracy is slightly increasing. Too many neighbors lead to underfitting of the data.
# 
# Let us now make the predictions on the test dataset.

# In[ ]:


knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)
y_knn_pred=knn.predict(X_test)


# In[ ]:


knn_score=knn.score(X_test,y_test)
knn_score


# In[ ]:


conf_mat_knn=confusion_matrix(y_test,y_knn_pred)
sns.heatmap(conf_mat_knn,annot=True,fmt='g',cmap='gnuplot')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# ## SVM

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svm=SVC(gamma=1e-07,C=1e9)
svm.fit(X_train,y_train)


# In[ ]:


svm.score(X_train,y_train)


# In[ ]:


y_pred_svc=svm.predict(X_test)
svm.score(X_test,y_test)


# In[ ]:


conf_mat_svc=confusion_matrix(y_test,y_pred_svc)
sns.heatmap(conf_mat_svc,annot=True,fmt='g',cmap='summer')


# Let us check how the Reciever operating characteristic curve and Precision recall curve look like.

# In[ ]:


y_svc=svm.fit(X_train,y_train).decision_function(X_test)


# In[ ]:


fpr,tpr,_=roc_curve(y_test,y_svc)
plt.plot(fpr,tpr,color='indianred')
plt.plot([0,1],[0,1],color='blue',linestyle='--')
plt.xlabel('False postive rate')
plt.ylabel('True positive rate')
auc_svc=auc(fpr,tpr)
plt.title('ROC curve for SVC with AUC: {0:.2f}'.format(auc_svc))


# As we can see, the area under curve is only 0.5. This shows that the model is performing **quite average** .
# 
# 
# Let us see how the precision-recall curve looks like.

# In[ ]:


precision,recall,threshold=precision_recall_curve(y_test,y_svc)
closest_zero=np.argmin(np.abs(threshold))
closest_zero_p=precision[closest_zero]
closest_zero_r = recall[closest_zero]
plt.plot(precision,recall)
plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
plt.title('Precision-Recall curve with SVC')
plt.xlabel('Precision')
plt.ylabel('Recall')


# The red dot above shows the optimum precision and recall value balance.

# ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


reg_log=LogisticRegression()


# In[ ]:


reg_log.fit(X_train,y_train)


# As we can see, the model will implement a L2 regularisation (or penalty) for each incorrect prediction.

# In[ ]:


reg_log.score(X_train,y_train)


# In[ ]:


y_pred_log=reg_log.predict(X_test)
reg_log.score(X_test,y_test)


# In[ ]:


conf_mat_log=confusion_matrix(y_pred_log,y_test)
sns.heatmap(conf_mat_log,annot=True,fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# As we can see, the logistic regression could not capture any of the targets with 0 (or access denied). This clearly indicates that the model is only accurate at capturing the 1s.

# In[ ]:


y_log=reg_log.fit(X_train,y_train).decision_function(X_test)


# In[ ]:


fpr,tpr,_=roc_curve(y_test,y_log)
plt.plot(fpr,tpr,color='indianred')
plt.plot([0,1],[0,1],color='blue',linestyle='--')
plt.xlabel('False postive rate')
plt.ylabel('True positive rate')
auc_svc=auc(fpr,tpr)
plt.title('ROC curve for Logistic regression with AUC: {0:.2f}'.format(auc_svc))


# This is another extremely poor model as shown by the above ROC curve. This was already indicated by the confusion matrix.

# ## Stochastic gradient descent

# In[ ]:


from sklearn.linear_model import SGDClassifier


# In[ ]:


sgd=SGDClassifier()


# In[ ]:


sgd.fit(X_train,y_train)


# In[ ]:


sgd.score(X_train,y_train)


# In[ ]:


y_pred_sgd=sgd.predict(X_test)
sgd.score(X_test,y_test)


# In[ ]:


conf_mat_sgd=confusion_matrix(y_pred_sgd,y_test)
sns.heatmap(conf_mat_sgd,annot=True,fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# As we can see, the results are exactly same as the logistic regression model. Hence, no improvements could be seen using SGD.

# In[ ]:


y_sgd=sgd.fit(X_train,y_train).decision_function(X_test)
fpr,tpr,_=roc_curve(y_test,y_sgd)
plt.plot(fpr,tpr,color='indianred')
plt.plot([0,1],[0,1],color='blue',linestyle='--')
plt.xlabel('False postive rate')
plt.ylabel('True positive rate')
auc_svc=auc(fpr,tpr)
plt.title('ROC curve for SGD with AUC: {0:.2f}'.format(auc_svc))


# From the ROC, it is clearly visible how poor the model is predicting. In fact, it has performed almost as poor as the  Logistic regression model.

# ## Decision tree

# We have already seen that the linear models and svc  performed poorly. Let us now go ahead with ensemble models to check how we well they can predict.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc.score(X_train,y_train)


# In[ ]:


y_pred_dtc=dtc.predict(X_test)
dtc.score(X_test,y_test)


# In[ ]:


conf_mat_dtc=confusion_matrix(y_pred_dtc,y_test)
sns.heatmap(conf_mat_dtc,annot=True,fmt='g',cmap='winter')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# As we can see, the decision tree has relatively done a better job than the other models since it could capture the 0s correctly as well.

# ## Random forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
param_grid={'n_estimators':[3,5,7,9],'max_depth':[5,7,9]}
grid_search=GridSearchCV(rfc,param_grid,scoring='roc_auc')


# In[ ]:


grid_search.fit(X_train,y_train)


# In[ ]:


grid_search.best_params_


# In[ ]:


grid_search.score(X_train,y_train)


# In[ ]:


y_pred_rfc=grid_search.predict(X_test)


# In[ ]:


grid_search.score(X_test,y_test)


# In[ ]:


conf_mat_rfc=confusion_matrix(y_pred_rfc,y_test)
sns.heatmap(conf_mat_rfc,annot=True,fmt='g',cmap='gnuplot')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# As we can see, the model made quite a few errors in predicting the 0s correctly. This has been a clear problem with most of the models we have dealt so far.

# ## Gradient boosted Decision Trees

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


gbdt=GradientBoostingClassifier()


# In[ ]:


params={'max_depth':[6,7,8,10,12]}
grid_search=GridSearchCV(gbdt,params,scoring='roc_auc')


# In[ ]:


grid_search.fit(X_train,y_train)


# In[ ]:


print('Best parameter:{}'.format(grid_search.best_params_))
print('Best cross validated score: {:.2f}'.format(grid_search.best_score_))


# In[ ]:


grid_search.score(X_train,y_train)


# In[ ]:


y_pred_gbdt=grid_search.predict(X_test)
grid_search.score(X_test,y_test)


# In[ ]:


conf_mat_gbdt=confusion_matrix(y_pred_gbdt,y_test)
sns.heatmap(conf_mat_gbdt,annot=True,fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# It is seen that for low max_depth of tree, the model couldn't predict the 0s correctly. As the max_depth was increased, it was seen that number of correct 0 predictions increased. Hence, we tried to tune the max_depth hyperparameter using a grid search technique. According to the grid search, the best max_depth was found to be 8. The correct 0 detection is however not satisfactory.

# ## XGBoost classification

# In[ ]:


import xgboost as xgb


# In[ ]:


xgb_class=xgb.XGBClassifier(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


# In[ ]:


xgb_class.fit(X_train,y_train)


# In[ ]:


xgb_class.score(X_train,y_train)


# In[ ]:


y_pred_xgb=xgb_class.predict(X_test)


# In[ ]:


xgb_class.score(X_test,y_test)


# In[ ]:


conf_mat_xgb=confusion_matrix(y_pred_xgb,y_test)


# In[ ]:


sns.heatmap(conf_mat_xgb,annot=True,fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# ## LightGBM classification

# In[ ]:


from lightgbm import LGBMClassifier


# In[ ]:


lgbm=LGBMClassifier(num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 30, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# In[ ]:


lgbm.fit(X_train,y_train)


# In[ ]:


lgbm.score(X_train,y_train)


# In[ ]:


y_pred_lgbm=lgbm.predict(X_test)


# In[ ]:


lgbm.score(X_test,y_test)


# In[ ]:


conf_mat_lgbm=confusion_matrix(y_pred_lgbm,y_test)
sns.heatmap(conf_mat_lgbm,annot=True,fmt='g')


# As we can see, the model again failed to correctly predict the 0s.

# Upon checking the various models, we shall take the following models under consideration for our test dataset:
# 
# * KNN
# * Decision Tree
# * GBDT
# * XGBoost

# # 3. Testing phase

# Let us import the test data and use standard scaling for the input data.

# In[ ]:


test_df=pd.read_csv('../input/amazon-employee-access-challenge/test.csv')
test_df.head()


# In[ ]:


test_id=pd.DataFrame(test_df.iloc[:,0],columns=['id'])
test_id.head()


# In[ ]:


test_df.drop('id',axis=1,inplace=True)


# In[ ]:


X_test=scaler.fit_transform(test_df)
X_test


# In[ ]:


y_final_knn=knn.predict(X_test)


# ### KNN predictions

# In[ ]:


knn_df=pd.DataFrame(columns=['Id','Action'])
knn_df['Action']=y_final_knn
knn_df['Id']=test_id['id']
knn_df.head()


# ### Decision tree predictions

# In[ ]:


y_final_dtc=dtc.predict(X_test)
dtc_df=pd.DataFrame(columns=['Id','Action'])
dtc_df['Action']=y_final_dtc
dtc_df['Id']=test_id['id']
dtc_df.head()


# In[ ]:


dtc_df.to_csv('DTC_predictions.csv',index=False)


# In[ ]:




