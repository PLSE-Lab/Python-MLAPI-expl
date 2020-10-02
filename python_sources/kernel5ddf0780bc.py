#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Clear Ipython memory
get_ipython().run_line_magic('reset', '-f')

# 1.1 Data manipulation and plotting modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1.2 Data pre-processing
#     z = (x-mean)/stdev
from sklearn.preprocessing import StandardScaler as ss

# 1.3 Dimensionality reduction
from sklearn.decomposition import PCA

# 1.4 Data splitting and model parameter search
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# 1.5 Modeling modules
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBClassifier

# 1.6 Model pipelining
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

# 1.7 Model evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

# 1.8
import matplotlib.pyplot as plt
from xgboost import plot_importance

# 1.9 Needed for Bayes optimization
from sklearn.model_selection import cross_val_score

# 1.10 Install as: pip install bayesian-optimization
from bayes_opt import BayesianOptimization

# 1.11 Find feature importance of ANY BLACK BOX estimator
import eli5
from eli5.sklearn import PermutationImportance

# 1.12 Misc
import time
import os
import gc
import random
from scipy.stats import uniform


# In[ ]:


# 1.13 Set option to dislay many rows
pd.set_option('display.max_columns', 100)


# In[ ]:


# 2.1 Set working directory

#os.chdir("F:\\Practice\\Machine Learning and Deep Learning\\Classes\\Assignment\\Kaggle\\3rd")
os.chdir("../input") 
os.listdir()


# In[ ]:


#Data
tr_f = "winequalityN.csv"

data = pd.read_csv(
         tr_f,
         header=0)


# DATA EXPLORATION

# In[ ]:


data.isnull().sum()


# In[ ]:


data = data.dropna(axis=0)


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.describe()


# In[ ]:


data.corr()


# In[ ]:


sns.countplot(x = data.quality, data=data, hue='type')


# In[ ]:


plt.figure(figsize=(14,14))
sns.heatmap(data.iloc[:,0:13].corr(), cbar = True,  square = True, annot=True, cmap= 'BuGn_r')


# In[ ]:


fig = plt.figure(figsize=(12,12))
ax1 = fig.add_subplot(1,4,1)
ax1.set_xticklabels(labels = 'Type', rotation=90)

sns.boxenplot(x='type',y='quality',data=data)
ax1 = fig.add_subplot(1,4,2)
sns.boxenplot(x='type',y='fixed acidity',data=data)
ax1 = fig.add_subplot(1,4,3)
sns.boxenplot(x='type',y='volatile acidity',data=data)
ax1 = fig.add_subplot(1,4,4)
sns.boxenplot(x='type',y='citric acid',data=data)

fig2 = plt.figure(figsize=(12,12))
ax2 = fig2.add_subplot(1,4,1)
sns.boxenplot(x='type',y='residual sugar',data=data)
ax2 = fig2.add_subplot(1,4,2)
sns.boxenplot(x='type',y='chlorides',data=data)
ax2 = fig2.add_subplot(1,4,3)
sns.boxenplot(x='type',y='free sulfur dioxide',data=data)
ax2 = fig2.add_subplot(1,4,4)
sns.boxenplot(x='type',y='total sulfur dioxide',data=data)

fig3 = plt.figure(figsize=(12,12))
ax3 = fig3.add_subplot(1,4,1)
sns.boxenplot(x='type',y='density',data=data)
ax3 = fig3.add_subplot(1,4,2)
sns.boxenplot(x='type',y='pH',data=data)
ax3 = fig3.add_subplot(1,4,3)
sns.boxenplot(x='type',y='sulphates',data=data)
ax3 = fig3.add_subplot(1,4,4)
sns.boxenplot(x='type',y='alcohol',data=data)


# In[ ]:


fig = plt.figure(figsize=(24,10))
features = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide","density","pH","sulphates","alcohol","quality"]

for i in range(12):
    ax1 = fig.add_subplot(3,4,i+1)
    sns.barplot(x='quality', y=features[i],data=data, hue='type')


# predictors and target

# In[ ]:


X = data.iloc[ :, 1:13]
X.head(3)


# In[ ]:


y = data.iloc[ : , 0]
y.head()


# In[ ]:


data.type.unique()


# In[ ]:


y = y.map({'white':1, 'red' : 0})
y.dtype


# In[ ]:


colnames = X.columns.tolist()
colnames


# Split dataset into train and validation parts

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.35,
                                                    shuffle = True
                                                    )


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


y_train.shape


# In[ ]:


y_test.shape


# Creating Pipeline

# In[ ]:


steps_xg = [('sts', ss() ),
            ('pca', PCA()),
            ('xg',  XGBClassifier(silent = False,
                                  n_jobs=2))]


# In[ ]:


pipe_xg = Pipeline(steps_xg)


# Parameter tuning using Grid Search, Random Search and Bayesian Optimization

# In[ ]:


parameters = {'xg__learning_rate':  [0, 1],
              'xg__n_estimators':   [50,  100],  
              'xg__max_depth':      [3,5],
              'pca__n_components' : [5,7]}


# In[ ]:


#Grid Search
clf = GridSearchCV(pipe_xg,
                   parameters,
                   n_jobs = 2,
                   cv =5 ,
                   verbose =2,
                   scoring = ['accuracy', 'roc_auc'], 
                   refit = 'roc_auc')


# In[ ]:


# Start fitting data to pipeline
start = time.time()
clf.fit(X_train, y_train)
end = time.time()
(end - start)/60


# In[ ]:


f"Best score: {clf.best_score_} "


# In[ ]:


f"Best parameter set {clf.best_params_}"


# In[ ]:


y_pred_gs = clf.predict(X_test)


# In[ ]:


# Accuracy
accuracy_gs = accuracy_score(y_test, y_pred_gs)
f"Accuracy: {accuracy_gs * 100.0}"


# In[ ]:


plt.bar(clf.best_params_.keys(), clf.best_params_.values())
plt.xticks(rotation=10)


# Random Search

# In[ ]:


parameters = {'xg__learning_rate':  uniform(0, 1),
              'xg__n_estimators':   range(50,100),
              'xg__max_depth':      range(3,5),
              'pca__n_components' : range(5,7)}


# In[ ]:


rs = RandomizedSearchCV(pipe_xg,
                        param_distributions=parameters,
                        scoring= ['roc_auc', 'accuracy'],
                        n_iter=20, 
                        verbose = 3,
                        refit = 'roc_auc',n_jobs = 2,
                        cv = 5)


# In[ ]:


start = time.time()
rs.fit(X_train, y_train)
end = time.time()
(end - start)/60


# In[ ]:


f"Best score: {rs.best_score_} "


# In[ ]:


f"Best parameter set: {rs.best_params_} "


# In[ ]:


plt.bar(rs.best_params_.keys(), rs.best_params_.values())
plt.xticks(rotation=10)


# In[ ]:


y_pred_rs = rs.predict(X_test)


# In[ ]:


accuracy_rs = accuracy_score(y_test, y_pred_rs)
f"Accuracy: {accuracy_rs * 100.0}"


# Bayesian Optimization

# In[ ]:


para_set = {
           'learning_rate':  (0, 1),                 
           'n_estimators':   (50,100),               
           'max_depth':      (3,5),                 
           'n_components' :  (5,7)          
            }


# In[ ]:


def xg_eval(learning_rate,n_estimators, max_depth,n_components):
    #  Make pipeline. Pass parameters directly here
    pipe_xg1 = make_pipeline (ss(),                    
                              PCA(n_components=int(round(n_components))),
                              XGBClassifier(
                                           silent = False,
                                           n_jobs=2,
                                           learning_rate=learning_rate,
                                           max_depth=int(round(max_depth)),
                                           n_estimators=int(round(n_estimators))
                                           )
                             )

    # Now fit the pipeline and evaluate
    cv_result = cross_val_score(estimator = pipe_xg1,
                                X= X_train,
                                y = y_train,
                                cv = 2,
                                n_jobs = 2,
                                scoring = 'f1'
                                ).mean()            


    #  Finally return maximum/average value of result
    return cv_result


# In[ ]:


xgBO = BayesianOptimization(
                             xg_eval, 
                             para_set 
                             )


# In[ ]:


start = time.time()
xgBO.maximize(init_points=5,    
               n_iter=25,        
               )
end = time.time()
(end-start)/60


# In[ ]:


f"Best parameter set: {xgBO.max} "


# In[ ]:


xgBO.max.values()


# In[ ]:


for features in xgBO.max.values():
        print(features)


# In[ ]:


features


# In[ ]:


plt.bar(features.keys(), features.values())
plt.xticks(rotation=10)


# Model fitting using best parameters from above tuning techniques.

# In[ ]:


# Grid Search
xg_gs = XGBClassifier(learning_rate = clf.best_params_['xg__learning_rate'],
                    max_depth = clf.best_params_['xg__max_depth'],
                    n_estimators=clf.best_params_['xg__n_estimators'])
#Randomized search
xg_rs = XGBClassifier(learning_rate = rs.best_params_['xg__learning_rate'],
                    max_depth = rs.best_params_['xg__max_depth'],
                    n_estimators=rs.best_params_['xg__n_estimators'])
#Bayes Optimization
xg_bo = XGBClassifier(learning_rate = xgBO.max['params']['learning_rate'],
                    max_depth = int(xgBO.max['params']['max_depth']),
                    n_estimators= int(xgBO.max['params']['n_estimators']))


# In[ ]:


start = time.time()
xg_gs.fit(X_train, y_train)
xg_rs.fit(X_train, y_train)
xg_bo.fit(X_train, y_train)


# In[ ]:


#Fit the data using X_Train and y_train
xg_gs1 = xg_gs.fit(X_train,y_train)
xg_rs1 = xg_rs.fit(X_train,y_train)
xg_bo1 = xg_bo.fit(X_train,y_train)


# In[ ]:


y_pred_xg_gs = xg_gs1.predict(X_test)
y_pred_xg_rs = xg_rs1.predict(X_test)
y_pred_xg_bo = xg_bo1.predict(X_test)


# In[ ]:


y_pred_xg_gs_prob = xg_gs1.predict_proba(X_test)
y_pred_xg_rs_prob = xg_rs1.predict_proba(X_test)
y_pred_xg_bo_prob = xg_bo1.predict_proba(X_test)


# In[ ]:


print (accuracy_score(y_test,y_pred_xg_gs))
print (accuracy_score(y_test,y_pred_xg_rs))
print (accuracy_score(y_test,y_pred_xg_bo))


# In[ ]:


confusion_matrix(y_test,y_pred_xg_gs)
confusion_matrix(y_test,y_pred_xg_rs)
confusion_matrix(y_test,y_pred_xg_bo)


# ROC GRAPH

# In[ ]:


fpr_xg_gs, tpr_xg_gs, thresholds = roc_curve(y_test,
                                 y_pred_xg_gs_prob[: , 1],
                                 pos_label= 1
                                 )
fpr_xg_rs, tpr_xg_rs, thresholds = roc_curve(y_test,
                                 y_pred_xg_rs_prob[: , 1],
                                 pos_label= 1
                                 )
fpr_xg_bo, tpr_xg_bo, thresholds = roc_curve(y_test,
                                 y_pred_xg_bo_prob[: , 1],
                                 pos_label= 1
                                 )


# Calculate the Precision, Recall and F1 Score

# In[ ]:


p_xg_gs,r_xg_gs,f_xg_gs,_ = precision_recall_fscore_support(y_test,y_pred_xg_gs)
p_xg_rs,r_xg_rs,f_xg_rs,_ = precision_recall_fscore_support(y_test,y_pred_xg_rs)
p_xg_bo,r_xg_bo,f_xg_bo,_ = precision_recall_fscore_support(y_test,y_pred_xg_bo)


# Calculate the AUC

# In[ ]:


print (auc(fpr_xg_gs,tpr_xg_gs))
print (auc(fpr_xg_rs,tpr_xg_rs))
print (auc(fpr_xg_bo,tpr_xg_bo))


# ROC curve

# In[ ]:


fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)
ax.plot([0, 1], [0, 1], ls="--")
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve for multiple tuning methods')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])


ax.plot(fpr_xg_gs, tpr_xg_gs, label = "xg_gs")
ax.plot(fpr_xg_rs, tpr_xg_rs, label = "xg_rs")
ax.plot(fpr_xg_bo, tpr_xg_bo, label = "xg_bo")

ax.legend(loc="lower right")
plt.show()


# Completed

# In[ ]:




