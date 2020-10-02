#!/usr/bin/env python
# coding: utf-8

# I am starting with all of this ML stuff, having done a few Coursera trainings this is my first kaggle kernel.
# 
# Any comment or suggestion is appreciated. I am trying to make a classification model to know if a shot will be made or not

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import plot_tree
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, mean_squared_error,precision_score
from sklearn.cross_validation import KFold, train_test_split
from sklearn.grid_search import GridSearchCV
import math

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png' #set 'png' here when working on notebook")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


dataset =  pd.read_csv('../input/shot_logs.csv', header=0)


# In[ ]:


dataset.describe()


# # Cleaning

# In[ ]:


#only positives TOUCH_TIME
dataset=dataset[dataset['TOUCH_TIME']>=0]
#shot_dist too far
dataset=dataset[dataset['SHOT_DIST']<40]

#REMOVE SHOT_CLOCK NAN
nan=float('nan')
dataset=dataset[~np.isnan(dataset['SHOT_CLOCK'])]

#REMOVE FGM NAN
nan=float('nan')
dataset=dataset[~np.isnan(dataset['FGM'])]

#CLose def dist
dataset=dataset[dataset['CLOSE_DEF_DIST']<30]


# In[ ]:


#target dataset
datasettarget = dataset['FGM']


# In[ ]:


datasetwithouttarget = dataset[['SHOT_DIST','TOUCH_TIME','FINAL_MARGIN','PERIOD','SHOT_CLOCK','DRIBBLES','CLOSE_DEF_DIST','PTS']]


# In[ ]:


print(dataset['FGM'].shape)
print(datasettarget.shape)
print(datasetwithouttarget.shape)


# # Looking for most predictive features

# In[ ]:


model = XGBClassifier()
model.fit(datasetwithouttarget,datasettarget)
# plot feature importance
plot_importance(model)
pyplot.show()


# **(OBVIOUS) PTS looks like it is totally correlated with FGM, let's remove it.**

# In[ ]:


datasetwithouttarget = dataset[['SHOT_DIST','TOUCH_TIME','FINAL_MARGIN','PERIOD','SHOT_CLOCK','DRIBBLES','CLOSE_DEF_DIST']]


# In[ ]:


model = XGBClassifier()
model.fit(datasetwithouttarget,datasettarget)
# plot feature importance
plot_importance(model, importance_type ='weight')
pyplot.show()


# *this looks better, we have the features that are correlated but not directly as can be PTS or FGM*

# # Exploring most important features

# ## Shot distance

# In[ ]:


distances = [0,5,10,20,30,40,50]

shot_made = [(dataset[np.logical_and(np.logical_and(dataset['SHOT_DIST']>distances[i-1],dataset['SHOT_DIST']<distances[i] ), dataset['FGM']==1)  ].size/dataset[np.logical_and(dataset['SHOT_DIST']>distances[i-1],dataset['SHOT_DIST']<distances[i] )  ].size)     for i in range(1,len(distances))]

lambda_results = pd.Series(shot_made, index =  distances[1:len(distances)])
lambda_results.plot(title = "Exploring - Shot distance")
plt.xlabel("Shot distance")
plt.ylabel("%")


# ## Close defense distance

# In[ ]:


distances = [0,5,10,20,30,40,50]

shot_made = [(dataset[np.logical_and(np.logical_and(dataset['CLOSE_DEF_DIST']>distances[i-1],dataset['CLOSE_DEF_DIST']<distances[i] ), dataset['FGM']==1)  ].size/dataset[np.logical_and(dataset['CLOSE_DEF_DIST']>distances[i-1],dataset['CLOSE_DEF_DIST']<distances[i] )  ].size)     for i in range(1,len(distances))]

lambda_results = pd.Series(shot_made, index = distances[1:len(distances)])
lambda_results.plot(title = "Exploring - Close defense distance")
plt.xlabel("Close defense distance")
plt.ylabel("%")


# # Set target: be better than the most common value

# In[ ]:


print ('shots made',np.count_nonzero(datasettarget))
print ('shots missed',datasettarget.size-np.count_nonzero(datasettarget))
print ('total shots',datasettarget.size)
print ('we must at least have better precision than ',(datasettarget.size-np.count_nonzero(datasettarget))/datasettarget.size)


# # Predict with every feature

# In[ ]:


#thanks to https://github.com/dmlc/xgboost/blob/master/demo/guide-python/sklearn_examples.py

print("every feature")
print(datasettarget.shape)


X_train, X_test, y_train, y_test = train_test_split( datasetwithouttarget[['SHOT_DIST','TOUCH_TIME','FINAL_MARGIN','PERIOD','SHOT_CLOCK','DRIBBLES','CLOSE_DEF_DIST']], datasettarget, test_size=0.50, random_state=42)
   
xgb_model = XGBClassifier().fit(X_train,y_train)

predictions = xgb_model.predict(X_test)

actuals = y_test

print(confusion_matrix(actuals, predictions))
print(precision_score(actuals, predictions) )
    
   


# # Predict with the 4 important features

# In[ ]:


print("4 most important features")

X_train, X_test, y_train, y_test = train_test_split( datasetwithouttarget[['SHOT_DIST','TOUCH_TIME','CLOSE_DEF_DIST','SHOT_CLOCK']], datasettarget, test_size=0.50, random_state=42)
y = np.array(datasettarget)
X = datasetwithouttarget[['SHOT_DIST','TOUCH_TIME','CLOSE_DEF_DIST','SHOT_CLOCK']].as_matrix()

   
xgb_model = XGBClassifier().fit(X_train,y_train)

predictions = xgb_model.predict(X_test)

actuals = y_test

print(confusion_matrix(actuals, predictions))
print(precision_score(actuals, predictions) )


# # RESULT
# Most common value precision: 55%
# 
# **My model: 65%**
# 
# I will try to improve it tuning the XGB parameters :)

# # Find XGB better configuration
# 
#  1. Split the dataset in training (50%), validation (25%) and testing (25%) sets
#  2. Train the model using training set and test with different parameters using validation set. 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split( datasetwithouttarget[['SHOT_DIST','TOUCH_TIME','CLOSE_DEF_DIST','SHOT_CLOCK']], datasettarget, test_size=0.50, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split( X_test, y_test, test_size=0.50, random_state=42)


# In[ ]:


parameters_for_testing = {
    'min_child_weight':[0.0001,0.001,0.01],
    'learning_rate':[0.00001,0.0001,0.001],
    'n_estimators':[1,3,5,10],
    'max_depth':[3,4]
}

xgb_model = XGBClassifier()

gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, scoring='precision')
gsearch1.fit(X_train[['SHOT_DIST','TOUCH_TIME','CLOSE_DEF_DIST','SHOT_CLOCK']],y_train)

print('best params')
print (gsearch1.best_params_)
print('best score')
print (gsearch1.best_score_)            


# ## Analyze alpha

# In[ ]:


def ExecuteWithAlpha(x_validation,y_validation, alpha):
    xgb_model = XGBClassifier(reg_alpha=alpha,min_child_weight=0.0001,learning_rate=1e-05,
                              n_estimators=1,max_depth=3).fit(X_train,y_train)
    predictions = xgb_model.predict(x_validation)
    actuals = y_validation        
    precision=precision_score(actuals, predictions)
    return precision

alphas = [0, 1, 5, 10, 15, 30, 50, 75,100,200]
cv_xgb = [ExecuteWithAlpha(X_validation,y_validation,alpha)
            for alpha in alphas]

alpha_results = pd.Series(cv_xgb, index = alphas)
alpha_results.plot(title = "Validation - alpha")
plt.xlabel("alpha")
plt.ylabel("precision")


# ## Analyze lambda

# In[ ]:


def ExecuteWithLambda(x_validation,y_validation, lamb):
    xgb_model = XGBClassifier(reg_lambda=lamb,min_child_weight=0.0001,learning_rate=1e-05,
                              n_estimators=1,max_depth=3).fit(X_train,y_train)
    predictions = xgb_model.predict(x_validation)
    actuals = y_validation        
    precision=precision_score(actuals, predictions)
    return precision

lambs = [0, 1, 5, 10, 15, 30, 50, 75,100,200]
cv_xgb = [ExecuteWithLambda(X_validation,y_validation,lamb)
            for lamb in lambs]

lambda_results = pd.Series(cv_xgb, index = lambs)
lambda_results.plot(title = "Validation - lambda")
plt.xlabel("lambda")
plt.ylabel("precision")


# Ok, 0 for lambda and alpha.

# # Execute final model against test set.

# In[ ]:


xgb_model = XGBClassifier(min_child_weight=0.0001,learning_rate=1e-05,
                              n_estimators=1,max_depth=3).fit(X_train,y_train)
predictions = xgb_model.predict(X_test)
actuals = y_test        
precision=precision_score(actuals, predictions)
print(precision)


# # RESULTS AFTER PARAMETER OPTIMIZATION
# 
# Most common value precision: 55%
# 
# My model without parameter optimization: 65%
# 
# **My model with parameter optimization: 68%**
