#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler as ss
from sklearn.decomposition import PCA

# Data splitting and model parameter search
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV                 
from sklearn.model_selection import RandomizedSearchCV         


# In[2]:


from xgboost.sklearn import XGBClassifier


# Model pipelining
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline


# Model evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from xgboost import plot_importance

# Needed for Bayes optimization

from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization

import eli5
from eli5.sklearn import PermutationImportance

import time
import gc
import random
from scipy.stats import uniform


# In[3]:


# Set option to dislay many rows
pd.set_option('display.max_columns', 100)


# In[4]:


#Reading file 
df= pd.read_csv("../input/winequalityN.csv")


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.shape


# In[8]:


df.columns.values


# In[9]:


df.describe()


# In[10]:


df.isnull().any()


# In[11]:


df.dropna(axis=0,inplace=True)


# In[12]:


df.shape


# In[13]:


import seaborn as sns
sns.countplot(x = df.quality, data=df, hue='type')


# In[14]:


plt.figure(figsize=(14,14))
sns.heatmap(df.iloc[:,0:13].corr(), cbar = True,  square = True, annot=True, cmap= 'ocean_r')


# In[15]:


fig = plt.figure(figsize=(22,10))
features = ["total sulfur dioxide", "residual sugar", "volatile acidity", "total sulfur dioxide", "chlorides", "fixed acidity", "density","sulphates"]

for i in range(8):
    ax1 = fig.add_subplot(2,4,i+1)
    sns.boxplot(x="type", y=features[i],data=df, palette="rocket");


# In[16]:


fig = plt.figure(figsize=(24,10))
features = ["total sulfur dioxide", "residual sugar", "volatile acidity", "total sulfur dioxide", "chlorides", "fixed acidity", "citric acid","sulphates"]

for i in range(8):
    ax1 = fig.add_subplot(2,4,i+1)
    sns.barplot(x='quality', y=features[i],data=df, hue='type', palette='rocket')


# In[17]:


#Splitting data into predictors and target
X = df.iloc[ :, 1:13]
y = df.iloc[ : , 0]


# In[18]:


X.head()


# In[19]:


y.head()


# In[20]:


#  Map Target data to '1' and '0'
y = y.map({'white':1, 'red' : 0})
y.dtype 


# In[21]:


colnames = X.columns.tolist()
colnames


# In[22]:


# Split dataset into train and test parts
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.30,
                                                    shuffle = True
                                                    )


# In[23]:


X_train.shape 


# In[24]:


X_test.shape 


# In[25]:


y_train.shape


# In[26]:


y_test.shape


# In[27]:


#Creating  pipelines

#### Pipe using XGBoost and instantiating it.

steps_xg = [('sts', ss() ),
            ('pca', PCA()),
            ('xg',  XGBClassifier(silent = False,
                                  n_jobs=2)        # Specify other parameters here
            )
            ]

pipe_xg = Pipeline(steps_xg)


# Grid Search

# In[28]:


parameters = {'xg__learning_rate':  [0.05, 0.4],
              'xg__n_estimators':   [50,  80],
              'xg__max_depth':      [3,5],
              'pca__n_components' : [5,8]
              }          


# In[29]:


clf = GridSearchCV(pipe_xg,            # pipeline object
                   parameters,         # possible parameters
                   n_jobs = 2,         # USe parallel cpu threads
                   cv =2 ,             # No of folds
                   verbose =2,         # Higher the value, more the verbosity
                   scoring = ['accuracy', 'roc_auc'],  # Metrics for performance
                   refit = 'roc_auc'   # Refitting final model on what parameters?
                                       # Those which maximise auc
                   )


# In[30]:


start = time.time()
clf.fit(X_train, y_train)
end = time.time()
(end - start)/60


# In[31]:


f"Best Score: {clf.best_score_} "


# In[32]:


f"Best Parameter set {clf.best_params_}"


# In[33]:


y_pred_gs = clf.predict(X_test)


# In[34]:


# Accuracy
accuracy_gs = accuracy_score(y_test, y_pred_gs)
f"Accuracy: {accuracy_gs * 100.0}"


# In[35]:


plt.bar(clf.best_params_.keys(), clf.best_params_.values())
plt.xticks(rotation=70)


# In[36]:


# Instantiate the importance object
perm = PermutationImportance(
                            clf,
                            random_state=1
                            )

# fit data & learn
start = time.time()
perm.fit(X_test, y_test)
end = time.time()
(end - start)/60


# In[37]:


eli5.show_weights(
                  perm,
                  feature_names = colnames      # X_test.columns.tolist()
                  )


# In[38]:


fw = eli5.explain_weights_df(
                  perm,
                  feature_names = colnames      # X_test.columns.tolist()
                  )

fw


# Random Search

# In[39]:


parameters = {'xg__learning_rate':  uniform(0, 1),
              'xg__n_estimators':   range(50,80),
              'xg__max_depth':      range(3,5),
              'pca__n_components' : range(5,7)}


# In[40]:


rs = RandomizedSearchCV(pipe_xg,
                        param_distributions=parameters,
                        scoring= ['roc_auc', 'accuracy'],
                        n_iter=12,          # Max combination of
                                            # parameter to try. Default = 10
                        verbose = 3,
                        refit = 'roc_auc',
                        n_jobs = 2,          # Use parallel cpu threads
                        cv = 2               # No of folds.
                                             # So n_iter * cv combinations
                        )


# In[41]:


start = time.time()
rs.fit(X_train, y_train)
end = time.time()
(end - start)/60


# In[42]:


f"Best Score: {rs.best_score_} "


# In[43]:


f"Best Parameter set: {rs.best_params_} "


# In[44]:


# Make predictions
y_pred_rs = rs.predict(X_test)


# In[45]:


# Accuracy
accuracy_rs = accuracy_score(y_test, y_pred_rs)
f"Accuracy: {accuracy_rs * 100.0}"


# In[47]:


plt.bar(rs.best_params_.keys(), rs.best_params_.values())
plt.xticks(rotation=50)


# Bayesian Optimization

# In[48]:


para_set = {
           'learning_rate':  (0.3, 0.9),                 
           'n_estimators':   (60,90),               
           'max_depth':      (3,5),                 
           'n_components' :  (5,7)          
            }


# In[50]:


def xg_eval(learning_rate,n_estimators, max_depth,n_components):
    #Pipeling for Bayesian Optimization
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
    #Fitting into pipeline 
    cv_result = cross_val_score(estimator = pipe_xg1,
                                X= X_train,
                                y = y_train,
                                cv = 2,
                                n_jobs = 2,
                                scoring = 'f1'
                                ).mean()             # taking mean of all results

    return cv_result       #Returning final mean of all results of cross val score


# In[51]:


xgBO = BayesianOptimization(
                             xg_eval, 
                             para_set 
                             )


# In[52]:


start = time.time()
xgBO.maximize(init_points=5,    
               n_iter=25,        
               )
end = time.time()
(end-start)/60


# In[54]:


#Results
xgBO.res


# In[56]:


#Best parametrs in maximizing the objective:
xgBO.max


# In[87]:


for features in xgBO.max.values():
        print(features)


# In[88]:


plt.bar(features.keys(), features.values())
plt.xticks(rotation=50)


# Fitting parameters in model

# In[63]:


# Model with parameters of grid search
model_gs = XGBClassifier(
                    learning_rate = clf.best_params_['xg__learning_rate'],
                    max_depth = clf.best_params_['xg__max_depth'],
                    n_estimators=clf.best_params_['xg__n_estimators']
                    )

#  Model with parameters of random search
model_rs = XGBClassifier(
                    learning_rate = rs.best_params_['xg__learning_rate'],
                    max_depth = rs.best_params_['xg__max_depth'],
                    n_estimators=rs.best_params_['xg__n_estimators']
                    )

#  Model with parameters of Bayesian Optimization
model_bo = XGBClassifier(
                    learning_rate = xgBO.max['params']['learning_rate'],
                    max_depth = int(xgBO.max['params']['max_depth']),
                    n_estimators= int(xgBO.max['params']['n_estimators'])
                    )


# In[64]:


start = time.time()
model_gs.fit(X_train, y_train)
model_rs.fit(X_train, y_train)
model_bo.fit(X_train, y_train)
end = time.time()
(end - start)/60


# In[65]:


# Predictions with all the models
y_pred_gs = model_gs.predict(X_test)
y_pred_rs = model_rs.predict(X_test)
y_pred_bo = model_bo.predict(X_test)


# In[66]:


# Accuracy of all the models
accuracy_gs = accuracy_score(y_test, y_pred_gs)
accuracy_rs = accuracy_score(y_test, y_pred_rs)
accuracy_bo = accuracy_score(y_test, y_pred_bo)
print("Accuracy by Grid Search           = ",accuracy_gs)
print("Accuracy by Random Search         = ",accuracy_rs)
print("Accuracy by Bayesian Optimization = ",accuracy_bo)


# In[67]:


# Get feature importances from all the models
model_gs.feature_importances_
model_rs.feature_importances_
model_bo.feature_importances_
plot_importance(model_gs)
plot_importance(model_rs)
plot_importance(model_bo)


# # Confusion matrix for all the models

# In[69]:


#Confusion Matrix for Grid Search model
confusion_matrix(y_test,y_pred_gs)


# In[70]:


#Confusion Matrix for Random Search model
confusion_matrix(y_test,y_pred_rs)


# In[71]:


#Confusion Matrix for Bayesian Optimization model
confusion_matrix(y_test,y_pred_bo)


# In[72]:


# Get probability of occurrence of each class
y_pred_prob_gs = model_gs.predict_proba(X_test)
y_pred_prob_rs = model_rs.predict_proba(X_test)
y_pred_prob_bo = model_bo.predict_proba(X_test)


# # Draw ROC curve

# In[77]:


# calculate fpr, tpr values
fpr_gs, tpr_gs, thresholds = roc_curve(y_test,
                                 y_pred_prob_gs[: , 1],
                                 pos_label= 1
                                 )

fpr_rs, tpr_rs, thresholds = roc_curve(y_test,
                                 y_pred_prob_rs[: , 1],
                                 pos_label= 1
                                 )

fpr_bo, tpr_bo, thresholds = roc_curve(y_test,
                                 y_pred_prob_bo[: , 1],
                                 pos_label= 1
                                 )


# In[78]:


fig = plt.figure(figsize=(12,10))  
ax = fig.add_subplot(111)   # Create axes

#Connect diagonals
ax.plot([0, 1], [0, 1], ls="--")  

#Labels 
ax.set_xlabel('False Positive Rate')  
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve for models')

#Set graph limits
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])

#Plot each graph now
ax.plot(fpr_gs, tpr_gs, label = "gs")
ax.plot(fpr_rs, tpr_rs, label = "rs")
ax.plot(fpr_bo, tpr_bo, label = "bo")


#Set legend and show plot
ax.legend(loc="lower right")
plt.show()


# In[79]:


# Calculate AUC
auc_gs = auc(fpr_gs,tpr_gs)
auc_rs = auc(fpr_rs,tpr_rs)
auc_bo = auc(fpr_bo,tpr_bo)


# In[82]:


#Calculate Precision, Recall and F1 Score
from sklearn.metrics import precision_recall_fscore_support
precision_gs,recall_gs,f1_gs,_ = precision_recall_fscore_support(y_test,y_pred_gs)
precision_rs,recall_rs,f1_rs,_ = precision_recall_fscore_support(y_test,y_pred_rs)
precision_bo,recall_bo,f1_bo,_ = precision_recall_fscore_support(y_test,y_pred_bo)


# # Performance Comparision of all models

# In[86]:


pc = pd.DataFrame({ "Classifiers":["Grid Search","Random Search",'Bayesian Optimization'],
                             "Accuracy": [accuracy_gs,accuracy_rs,accuracy_bo],
                             "Precision": [precision_gs,precision_rs,precision_bo],
                             "Recall":[recall_gs,recall_rs,recall_bo],
                             "f1_score":[f1_gs,f1_rs,f1_bo],
                             "AUC":[auc_gs,auc_rs,auc_bo]})
pc

