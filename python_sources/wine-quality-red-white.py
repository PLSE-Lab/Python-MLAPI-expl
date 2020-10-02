#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Data manipulation and plotting modules
import numpy as np
import pandas as pd
import seaborn as sns

# Data pre-processing
from sklearn.preprocessing import StandardScaler as ss

# Dimensionality reduction
from sklearn.decomposition import PCA

# Data splitting and model parameter search
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# Modeling modules

from xgboost.sklearn import XGBClassifier


# Model pipelining
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline


# Model evaluation metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

# for Plotting
import matplotlib.pyplot as plt
from xgboost import plot_importance
get_ipython().run_line_magic('matplotlib', 'inline')

# Needed for Cross-validation in Bayes optimization

from sklearn.model_selection import cross_val_score

# Bayesian Optimization
from bayes_opt import BayesianOptimization


# Find feature importance of ANY BLACK BOX estimator

import eli5
from eli5.sklearn import PermutationImportance


# Misc
import time
import os
import gc
import random
from scipy.stats import uniform


# In[7]:


# Set option to dislay many rows
pd.set_option('display.max_columns', 100)


# In[8]:


# Read the data to be analysed


# In[9]:


#Read Dataset and check for any missing variables across columns
data = pd.read_csv("../input/winequalityN.csv")


# In[10]:


# Know the shape
data.shape


# In[11]:


# list of column names (attributes)
data.columns.values


# In[12]:


data.dtypes.value_counts()


# In[13]:


data.head(3)


# In[14]:


# statistical summary of numerical attributes
data.describe()


# In[15]:


## Look for and drop rows that contain null values
data.isnull().sum()


# In[16]:


data=data.dropna()


# In[17]:


data.isnull().sum()


# In[18]:


data.shape


# In[19]:


# Visualize the Data for what it is, to look for possible correlation among attributes
# and with wine type (Target)

plt.figure(figsize=(9, 9))
sns.heatmap(data.corr(),linewidths=0.6,annot=True,cmap='ocean_r',cbar=True)


# In[20]:


# Looking at above Correlation matrix, one can see the Positive, Negative or the lack of correlation among attributes.
# Out of 12*12 matrix above, below is only a subset corelation pair plot (for better visibility) showing the
# correlation between the chemicals vs Acidity

sns.pairplot(data,hue = 'type',x_vars = ['free sulfur dioxide','total sulfur dioxide','chlorides','sulphates'],y_vars = ['fixed acidity','volatile acidity','pH','citric acid'])


# In[21]:


# From the Correlation Matrix, "free sulfur dioxide" is a good example for statistical representation of 
# +ve Correlation (with total sulfur dioxide)
# no correlation (with density)
# -ve correlation (with volatile acidity)
sns.pairplot(data,hue = 'type',x_vars = ['free sulfur dioxide'],y_vars = ['volatile acidity','density','total sulfur dioxide'])


# In[22]:


# Countplot for Wine quality for both types
sns.countplot(x = data['quality'], data=data, hue='type')


# In[23]:


# Barplot for quality vs alcohol
sns.barplot(x='quality', y = 'alcohol', hue = 'type' , data = data)


# In[24]:


# pH value across both wine types vs Alcohol levels. Well concentrated as expected.
sns.jointplot(x='pH', y = 'alcohol',kind='kde',data = data)


# In[25]:


###Seperating explanatory(X) and target(y) variables from the dataset, 
##and perform test/train
X = data.iloc[ :, 1:13]
X.head(2)


# In[26]:


y = data.iloc[ : , 0]
y.head()


# In[27]:


#y = y.map({'white' :1, 'red' : 0})
y = y.map({'white':1, 'red' : 0})
y.dtype
y.head()


# In[28]:


##store column names of X to be used for feature importance
colnames = X.columns.tolist()


# In[29]:


### Split dataset into train and validation parts
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.35,
                                                    shuffle = True
                                                    )


# In[30]:


X_train.shape


# In[31]:


X_test.shape


# In[32]:


y_train.shape


# In[33]:


y_test.shape


# In[34]:


#### Pipe using XGBoost

steps_xg = [('sts', ss() ),
            ('pca', PCA()),
            ('xg',  XGBClassifier(silent = False,
                                  n_jobs=2)        # Specify other parameters here
            )
            ]

# 5.1  Instantiate Pipeline object
pipe_xg = Pipeline(steps_xg)


# In[35]:


##################### Grid Search #################

parameters = {'xg__learning_rate':  [0, 1],  
              'xg__n_estimators':   [50,  100],
              'xg__max_depth':      [3,5],
              'pca__n_components' : [5,7]
              } 


# In[36]:


#    Create Grid Search object first with all necessary Specs

clf = GridSearchCV(pipe_xg,
                   parameters,
                   n_jobs = 2,
                   cv =2 ,
                   verbose =2,
                   scoring = ['accuracy', 'roc_auc'],
                   refit = 'roc_auc'
                   )


# In[37]:


# 7.2. Start fitting data to pipeline
start = time.time()
clf.fit(X_train, y_train)
end = time.time()
(end - start)/60 


# In[38]:


f"Best score: {clf.best_score_} "


# In[39]:


f"Best parameter set {clf.best_params_}"


# In[40]:


#####Instantiate##########
perm_grid = PermutationImportance(
                            clf,
                            random_state=1
                            )


# In[41]:


# fit data & learn

start = time.time()
perm_grid.fit(X_test, y_test)
end = time.time()
(end - start)/60


# In[42]:


## Get feature weights
eli5.show_weights(
                  perm_grid,
                  feature_names = colnames
                  )


# In[43]:


fw_grid = eli5.explain_weights_df(
                  perm_grid,
                  feature_names = colnames
                  )


# In[44]:


# Print importance
fw_grid


# In[45]:


##################### Randomized Search parameter tuning #################

parameters = {'xg__learning_rate':  uniform(0, 1),
              'xg__n_estimators':   range(50,100),
              'xg__max_depth':      range(3,5),
              'pca__n_components' : range(5,7)}


# In[46]:


# Tune parameters using random search

rs = RandomizedSearchCV(pipe_xg,
                        param_distributions=parameters,
                        scoring= ['roc_auc', 'accuracy'],
                        n_iter=10,
                        verbose = 3,
                        refit = 'roc_auc',
                        n_jobs = 2,
                        cv = 2
                        )


# In[47]:


## fit data and learn
start = time.time()
rs.fit(X_train, y_train)
end = time.time()
(end - start)/60


# In[48]:


## Instantiate the importance object
perm_random = PermutationImportance(
                           rs,
                           random_state=1
                           )


# In[49]:


#fit data & learn

start = time.time()
perm_random.fit(X_test, y_test)
end = time.time()
(end - start)/60


# In[50]:


# Get feature weights
eli5.show_weights(
                  perm_random,
                  feature_names = colnames      # X_test.columns.tolist()
                  )


# In[51]:


fw_random = eli5.explain_weights_df(
                  perm_random,
                  feature_names = colnames      # X_test.columns.tolist()
                  )


# In[52]:


# Print importance
fw_random


# In[53]:


#####################   Bayesian  Optimization parameter tuning ###### 
para_set = {
           'learning_rate':  (0, 1),
           'n_estimators':   (50,100),
           'max_depth':      (3,5),
           'n_components' :  (5,7)
            }


# In[54]:


# Create a function that when passed some parameters
#    evaluates results using cross-validation
#    This function is used by BayesianOptimization() object

def xg_eval(learning_rate,n_estimators, max_depth,n_components):
    # passing parameters to make pipeline
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

    # fit the pipeline and evaluate
    cv_result = cross_val_score(estimator = pipe_xg1,
                                X= X_train,
                                y = y_train,
                                cv = 4,
                                n_jobs = 2,
                                scoring = 'f1'
                                ).mean()             # take the average of all results


    # return maximum/average value of result
    return cv_result


# In[55]:


xgBO = BayesianOptimization(
                             xg_eval,
                             para_set
                             )


# In[56]:


start = time.time()
xgBO.maximize(init_points=5, 
               n_iter=5,
               )
end = time.time()
(end-start)/60


# In[57]:


xgBO.res


# In[58]:


#Get values of parameters that maximise the objective
xgBO.max


# In[59]:


############### Fitting the best parameters in our model ##############

# Model with best parameters of grid search
model_gs = XGBClassifier(
                    learning_rate = clf.best_params_['xg__learning_rate'],
                    max_depth = clf.best_params_['xg__max_depth'],
                    n_estimators=clf.best_params_['xg__max_depth']
                    )

# Model with best parameters of random search
model_rs = XGBClassifier(
                    learning_rate = rs.best_params_['xg__learning_rate'],
                    max_depth = rs.best_params_['xg__max_depth'],
                    n_estimators=rs.best_params_['xg__max_depth']
                    )

# Model with best parameters of Bayesian Optimization
model_bo = XGBClassifier(
                    learning_rate = xgBO.max['params']['learning_rate'],
                    max_depth = int(xgBO.max['params']['max_depth']),
                    n_estimators= int(xgBO.max['params']['n_estimators'])
                    )


# In[60]:


# fitting with all 3 models
start = time.time()
model_gs.fit(X_train, y_train)
model_rs.fit(X_train, y_train)
model_bo.fit(X_train, y_train)
end = time.time()
(end - start)/60


# In[61]:


# Predictions with all 3 models
y_pred_gs = model_gs.predict(X_test)
y_pred_rs = model_rs.predict(X_test)
y_pred_bo = model_bo.predict(X_test)


# In[62]:


# Accuracy from 3 models
accuracy_gs = accuracy_score(y_test, y_pred_gs)
accuracy_rs = accuracy_score(y_test, y_pred_rs)
accuracy_bo = accuracy_score(y_test, y_pred_bo)


# In[63]:


accuracy_gs


# In[64]:


accuracy_rs


# In[65]:


accuracy_bo


# In[66]:


# Get feature importances from all 3 models

model_gs.feature_importances_
model_rs.feature_importances_
model_bo.feature_importances_
plot_importance(model_gs)
plot_importance(model_rs)
plot_importance(model_bo)


# In[67]:


# Confusion matrix for all the models


f"Confusion_matrix for Grid Search: {confusion_matrix(y_test,y_pred_gs)}"


# In[68]:


f"Confusion_matrix for Random Search: {confusion_matrix(y_test,y_pred_rs)}"


# In[69]:


f"Confusion_matrix for Bayes Opt: {confusion_matrix(y_test,y_pred_bo)}"


# In[70]:


# Get probability of occurrence of each class
y_pred_prob_gs = model_gs.predict_proba(X_test)
y_pred_prob_rs = model_rs.predict_proba(X_test)
y_pred_prob_bo = model_bo.predict_proba(X_test)


# In[71]:


# Draw ROC curve
fpr_gs, tpr_gs, thresholds = roc_curve(y_test,
                                 y_pred_prob_gs[: , 0],
                                 pos_label= 0
                                 )

fpr_rs, tpr_rs, thresholds = roc_curve(y_test,
                                 y_pred_prob_rs[: , 0],
                                 pos_label= 0
                                 )

fpr_bo, tpr_bo, thresholds = roc_curve(y_test,
                                 y_pred_prob_bo[: , 0],
                                 pos_label= 0
                                 )


# In[72]:


# AUC
auc_gs = auc(fpr_gs,tpr_gs)
auc_rs = auc(fpr_rs,tpr_rs)
auc_bo = auc(fpr_bo,tpr_bo)


# In[73]:


######## Plotting the ROC curves

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)

# Connect diagonals
ax.plot([0, 1], [0, 1], ls="--")

# Labels 
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve for models')

# Set graph limits
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])

# Plot each graph now
ax.plot(fpr_gs, tpr_gs, label = "Grid Search Model")
ax.plot(fpr_rs, tpr_rs, label = "Random Search Model")
ax.plot(fpr_bo, tpr_bo, label = "Bayesian Optimization")


# Set legend and show plot
ax.legend(loc="lower right")


# In[74]:


### Grid Search Perfromance Indicators
f"Grid Search: Accuracy = {accuracy_score(y_test,y_pred_gs)}, Precison = {precision_score(y_test,y_pred_gs)}, Recall = {recall_score(y_test,y_pred_gs)}, f1_score = {f1_score(y_test,y_pred_gs)}, AUC = {auc_gs}"


# In[75]:


### Random Search Perfromance Indicators
f"Random Search: Accuracy = {accuracy_score(y_test,y_pred_rs)}, Precison = {precision_score(y_test,y_pred_rs)}, Recall = {recall_score(y_test,y_pred_rs)}, f1_score = {f1_score(y_test,y_pred_rs)}, AUC = {auc_rs}"


# In[76]:


### Bayesian Optimization Perfromance Indicators
f"Bayesian Opt: Accuracy = {accuracy_score(y_test,y_pred_bo)}, Precison = {precision_score(y_test,y_pred_bo)}, Recall = {recall_score(y_test,y_pred_bo)}, f1_score = {f1_score(y_test,y_pred_bo)}, AUC = {auc_bo}"


# In[ ]:




