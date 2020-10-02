#!/usr/bin/env python
# coding: utf-8

# In[ ]:


################### AA. Call libraries #################


# In[1]:


get_ipython().run_line_magic('reset', '-f')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler as ss

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from xgboost.sklearn import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from xgboost import plot_importance

# 1.9 Needed for Bayes optimization
from sklearn.model_selection import cross_val_score

from bayes_opt import BayesianOptimization

import eli5
from eli5.sklearn import PermutationImportance

import time
import os
import gc
import random
from scipy.stats import uniform


# In[ ]:


pd.set_option('display.max_columns', 100)


# In[ ]:


print(os.listdir("../input")


# In[ ]:





# In[ ]:


data = pd.read_csv("../input/winequalityN.csv")


# In[ ]:


data.isnull().values.any()
data.isnull().sum()
dataset = data.dropna()

dataset.shape                
dataset.columns.values       # 
dataset.dtypes.value_counts()  # 
dataset.isnull().values.any()


# In[ ]:


dataset.head(50)


# In[ ]:


dataset.describe()


# In[ ]:


dataset.type.value_counts()


# In[ ]:


dataset.corr()


# In[ ]:


#Plotting Heap Map
corr=dataset.corr()
corr

plt.figure(figsize=(14,6))
g = sns.heatmap(corr,annot=True)


# In[ ]:


# Plotting the Countplot graph

sns.countplot(x='quality',data=dataset,hue='type')


# In[ ]:


# Plotting the Jointplot graph

sns.jointplot(x='alcohol',y='fixed acidity',data=dataset)


# In[ ]:


# Plotting the Boxplot Graph

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(1,4,1)
ax1.set_xticklabels(labels = 'quality', rotation=90)

sns.boxenplot(x='type',y='quality',data=dataset)
ax1 = fig.add_subplot(1,4,2)
sns.boxenplot(x='fixed acidity',y='quality',data=dataset)
ax1 = fig.add_subplot(1,4,3)
sns.boxenplot(x='type',y='density',data=dataset)
ax1 = fig.add_subplot(1,4,4)
sns.boxenplot(x='type',y='pH',data=dataset)


# In[ ]:


#Plotting in Pair Plot
sns.set(style="ticks", color_codes=True)
g = sns.pairplot(dataset, hue='type', diag_kind = 'kde')


# In[ ]:


#  Divide data into predictors and target
#     
X = dataset.iloc[:, 1:13]
X.head(2)


# In[ ]:


y = dataset.iloc[ : , 0]
y.head()


# In[ ]:


# Transform label data to '1' and '0'

dataset.type[dataset.type == 'white'] = 1
dataset.type[dataset.type == 'red'] = 0
print(dataset)
y = dataset['type']
y=y.astype('int')
y


# In[ ]:


y.dtype


# In[ ]:


# Store column names somewhere
#     for use in feature importance

column_names = X.columns.tolist()
column_names


# In[ ]:


# Split dataset into train and validation parts
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.30,
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


# In[ ]:


## Create the Pipeline ##

xg = [('sts', ss() ),
       ('pca', PCA()),
        ('xg',  XGBClassifier(silent = False,
                                  n_jobs=2))]


# In[ ]:


#  Instantiate Pipeline object
pipe_xg = Pipeline(xg)


# In[ ]:


# Tune parameters using Grid search
# Hyperparameters to tune and their ranges

parameters = {'xg__learning_rate':  [0.05, 0.4], 
              'xg__n_estimators':   [50,  100],
              'xg__max_depth':      [3,5],
              'pca__n_components' : [5,8]}


# In[ ]:


clf = GridSearchCV(pipe_xg,   
                   parameters,         
                   n_jobs = 2,    
                   cv =2 ,  
                   verbose =2,
                   scoring = ['accuracy', 'roc_auc'],  
                   refit = 'roc_auc')


# In[ ]:


## Start fitting data to pipeline ##

start = time.time()
clf.fit(X_train, y_train)
end = time.time()
(end - start)/60 


# In[ ]:


f"Best score: {clf.best_score_} "


# In[ ]:


f"Best parameter set {clf.best_params_}"


# In[ ]:


# Make predictions
y_pred = clf.predict(X_test)
y_pred


# In[ ]:


# Accuracy
accuracy = accuracy_score(y_test, y_pred)
f"Accuracy: {accuracy * 100.0}"


# In[ ]:


plt.xticks(rotation=45)
plt.bar(clf.best_params_.keys(), clf.best_params_.values())


# ##################### EE. Randomized Search #################

# In[ ]:


parameters1 = {'xg__learning_rate':  uniform(0, 1),
              'xg__n_estimators':   range(50,100),
              'xg__max_depth':      range(3,5),
              'pca__n_components' : range(5,7)}


# Tune parameters using random search

# In[ ]:


rs = RandomizedSearchCV(pipe_xg,
                        param_distributions=parameters1,
                        scoring= ['roc_auc', 'accuracy'],
                        n_iter=15,          
                        verbose = 3,
                        refit = 'roc_auc',
                        n_jobs = 2,          # Use parallel cpu threads
                        cv = 2               # No of folds.
                        )


# In[ ]:


start = time.time()
rs.fit(X_train, y_train)
end = time.time()
(end - start)/60


# In[ ]:


# Instantiate the importance object
perm = PermutationImportance(
                            rs,
                            random_state=1
                            )


# In[ ]:


# Fit data & learn
start = time.time()
perm.fit(X_test, y_test)
end = time.time()
(end - start)/60


# In[ ]:


eli5.show_weights(
                  perm,
                  feature_names = column_names      # X_test.columns.tolist()
                  )


# In[ ]:


df = eli5.explain_weights_df(
                  perm,
                  feature_names = column_names      # X_test.columns.tolist()
                  )


# In[ ]:


df


# ###### Bayesian Optimization ##########

# In[ ]:


para_set = {
           'learning_rate':  (0.5, 0.9),                 # any value between 0 and 1
           'n_estimators':   (50,100),               # any number between 50 to 300
           'max_depth':      (3,5),                 # any depth between 3 to 10
           'n_components' :  (5,7)                 # any number between 20 to 30
            }


# In[ ]:


def xg_eval(learning_rate,n_estimators, max_depth,n_components):
    # 12.1 Make pipeline. Pass parameters directly here
    pipe_xg1 = make_pipeline (ss(),                        # Why repeat this here for each evaluation?
                              PCA(n_components=int(round(n_components))),
                              XGBClassifier(
                                           silent = False,
                                           n_jobs=2,
                                           learning_rate=learning_rate,
                                           max_depth=int(round(max_depth)),
                                           n_estimators=int(round(n_estimators))
                                           )
                             )

    # 12.2 Now fit the pipeline and evaluate
    cv_result = cross_val_score(estimator = pipe_xg1,
                                X= X_train,
                                y = y_train,
                                cv = 2,
                                n_jobs = 2,
                                scoring = 'f1'
                                ).mean()             # take the average of all results
    return cv_result


# In[ ]:


xgBO = BayesianOptimization(
                             xg_eval,     # Function to evaluate performance.
                             para_set     # Parameter set from where parameters will be selected
                             )


# In[ ]:


xgBO


# In[ ]:


start = time.time()
xgBO.maximize(init_points=5,    # Number of randomly chosen points to
                                 # sample the target function before
                                 #  fitting the gaussian Process (gp)
                                 #  or gaussian graph
               n_iter=20,        # Total number of times the
               #acq="ucb",       # ucb: upper confidence bound
                                 #   process is to be repeated
                                 # ei: Expected improvement
               # kappa = 1.0     # kappa=1 : prefer exploitation; kappa=10, prefer exploration
#              **gp_params
               )
end = time.time()
(end-start)/60


# Get values of parameters that maximise the objective

# In[ ]:


xgBO.res


# In[ ]:


xgBO.max


# In[ ]:


for features in xgBO.max.values():
        print(features)


# In[ ]:


plt.bar(features.keys(), features.values())
plt.xticks(rotation=45)


# ############### FF. Fitting parameters in our model ##############

# In[ ]:


# Model with parameters of grid search
model_gs = XGBClassifier(
                    learning_rate = clf.best_params_['xg__learning_rate'],
                    max_depth = clf.best_params_['xg__max_depth'],
                    n_estimators=clf.best_params_['xg__n_estimators']
                    )


# In[ ]:


#  Model with parameters of random search
model_rs = XGBClassifier(
                    learning_rate = rs.best_params_['xg__learning_rate'],
                    max_depth = rs.best_params_['xg__max_depth'],
                    n_estimators=rs.best_params_['xg__n_estimators']
                    )


# In[ ]:


#  Model with parameters of Bayesian Optimization
model_bo = XGBClassifier(
                    learning_rate = xgBO.max['params']['learning_rate'],
                    max_depth = int(xgBO.max['params']['max_depth']),
                    n_estimators= int(xgBO.max['params']['n_estimators'])
                    )


# In[ ]:


start = time.time()
model_gs.fit(X_train, y_train)
model_rs.fit(X_train, y_train)
model_bo.fit(X_train, y_train)
end = time.time()
(end - start)/60


# In[ ]:


# Predictions with all models
y_pred_gs = model_gs.predict(X_test)
y_pred_rs = model_rs.predict(X_test)
y_pred_bo = model_bo.predict(X_test)


# In[ ]:


# Accuracy from all models
accuracy_gs = accuracy_score(y_test, y_pred_gs)
accuracy_rs = accuracy_score(y_test, y_pred_rs)
accuracy_bo = accuracy_score(y_test, y_pred_bo)


# In[ ]:


print("Grid Search Accuracy           = ",accuracy_gs)
print("Random Search Accuracy         = ",accuracy_rs)
print("Bayesian Optimization Accuracy = ",accuracy_bo)


# Get feature importances from all models

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'qt5')
model_gs.feature_importances_
model_rs.feature_importances_
model_bo.feature_importances_
plot_importance(model_gs)
plot_importance(model_rs)
plot_importance(model_bo)
plt.show()


# In[ ]:


# Confusion matrix
confusion_matrix( y_test,y_pred_gs)

confusion_matrix(y_test,y_pred_rs)

confusion_matrix(y_test,y_pred_bo)

y_pred_prob_gs = model_gs.predict_proba(X_test)
y_pred_prob_rs = model_rs.predict_proba(X_test)
y_pred_prob_bo = model_bo.predict_proba(X_test)


# In[ ]:


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


# In[ ]:


# Get probbaility of occurrence of each class
y_pred_prob = clf.predict_proba(X_test)

y_pred_prob.shape      # (34887, 2)
y_pred_prob


# Probability values in y_pred_prob are ordered
#     column-wise, as:
clf.classes_    # array([0, 1]) => Ist col for prob(class = 0)


# In[ ]:


# Draw ROC curve
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


# In[ ]:


## Computing PCA ##

pca = PCA()
principleComp = pca.fit_transform(X)
principleComp.shape
pca.explained_variance_ratio_
X = pca.explained_variance_ratio_.cumsum()
X


# In[ ]:





# In[ ]:




