#!/usr/bin/env python
# coding: utf-8

# **Bank Marketing Campaign Analysis**
# 
# **Introduction**
# This dataset describes Portugal Bank marketing campaign results. The campaigns were conducted mostly on direct phone calls to offer clients a term deposit in the bank. If clients agreed, the result is marked as 'yes' or else 'no'.
# Client specific information is gathered like job, age, education, marital status, if there was a previous effort etc.
# 
# Task: Predict if a customer will be willing to open a term deposit given certain information about the client. This way, we can target certain people who can be potential customers.
# 
# Approach:
# 1. Initially, load the dataset and do some EDA
# 2. Perform encoding on categorical data
# 3. Fit a basic logistic regression model
# 4. Perform Grid Search CV and check if there is any score improvement.
# 5. Handle imbalanced classes by oversampling using SMOTE
# 6. Check if normalization or scaling is required
# 7. Plot AUC curve and check confusion matrix to look at TP and FP
# 
# Now, to improve score, lets try feature engineering, ensemble techniques. As the classes are imbalanced (we have more 'no' values then 'yes'), lets consider oversampling for 'yes' class and see if score improved.
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().system('python3 -m pip install -U scikit-learn')
get_ipython().system('pip install imblearn')
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from imblearn.over_sampling import SMOTE

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        inp= os.path.join(dirname, filename)

# Any results you write to the current directory are saved as output.

inp_file = pd.read_csv(inp, sep=';')

for i in ['job', 'marital', 'education', 'contact']:
    plt.figure(figsize=(10,4))
    sns.countplot(x=i,hue='y', data=inp_file)
    
corr = inp_file.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True)

inp_file = pd.get_dummies(inp_file, columns=['job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'poutcome'], drop_first=True)
labels = inp_file['y'].unique().tolist()
mapping = dict( zip(labels,range(len(labels))) )
inp_file.replace({'y': mapping},inplace=True)

inp_file=inp_file.drop(columns={'job_unknown', 'marital_unknown', 'education_unknown', 'default_unknown', 'housing_unknown', 'loan_unknown'})


# In[ ]:


#Split the dataset into train and test and use stratified split by 'y' variable as classes are imbalanced.

train, test = train_test_split(inp_file, test_size=0.2, random_state=0, stratify=inp_file['y'])
train_x=train.drop(columns={'y'})
train_y=train['y']
test_x=test.drop(columns={'y'})
test_y=test['y']

#####Base model with all features################
basemodel = LogisticRegression(solver='lbfgs',max_iter=10000)
basemodel.fit(train_x, train_y)
predictions_bm=basemodel.predict(test_x)
score_bm = basemodel.score(test_x, test['y'])

print("Base training model accuracy score: "+str(basemodel.score(train_x, train['y'])))
print("Score of base model on test data is:"+str(score_bm))
y_probas = basemodel.predict_proba(test_x)
skplt.metrics.plot_roc(test_y, y_probas)
plt.show()
cm_bm = metrics.confusion_matrix(test_y, predictions_bm, [0,1])
print("Confusion Matrix of base model:")
print(cm_bm)


# The accuracy for base model for training data is 91% and for test data is almost 91%. Looks like data is generalizing well on test data.
# But, we have many false positives and false negatives. So lets try checking if there is any multicollinearity.

# In[ ]:


#####Base model after dropping highly correlated features########
##Calculating VIF####

cc = np.corrcoef(train_x, rowvar=False)
VIF = np.linalg.inv(cc)
a=list(VIF.diagonal())

print("IVF values:")
for i in a:
    if i>=5:
        print(train_x.columns.values[a.index(i)]+':'+str(i))


# In[ ]:


#Drop the features which have high VIF

train_x=train_x.drop(columns=[ 'emp.var.rate','cons.price.idx', 'poutcome_success', 'euribor3m', 'nr.employed'])
test_x=test_x.drop(columns=[  'emp.var.rate','cons.price.idx', 'poutcome_success', 'euribor3m', 'nr.employed'])

basemodel = LogisticRegression(solver='lbfgs',max_iter=10000)
basemodel.fit(train_x, train_y)
predictions_bm=basemodel.predict(test_x)
score_bm = basemodel.score(test_x, test['y'])

print("Base training model accuracy score: "+str(basemodel.score(train_x, train['y'])))
print("Score of base model is:"+str(score_bm))
y_probas = basemodel.predict_proba(test_x)
skplt.metrics.plot_roc(test_y, y_probas)
plt.show()
cm_bm = metrics.confusion_matrix(test_y, predictions_bm, [0,1])
print("Confusion Matrix of base model:")
print(cm_bm)


# We see there is a tiny drop in the score after dropping highly correlated features. It looks like our model was overfitting earlier.

# In[ ]:


#####Base model after dropping highly correlated features########
##Calculating VIF####

cc = np.corrcoef(train_x, rowvar=False)
VIF = np.linalg.inv(cc)
a=list(VIF.diagonal())

print("IVF after dropping some columns:")
for i in a:
    if i>=5:
        print(train_x.columns.values[a.index(i)]+':'+str(i))


# In[ ]:


# Apply grid search cv and run logistic reg
#Dropped correlated features.

gridmodel = LogisticRegression(solver='lbfgs')
penalty = ['l2']
max_iter=[10000]    
# Create regularization hyperparameter space
C = [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5]
# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty, max_iter=max_iter)
model_gs = GridSearchCV(gridmodel, hyperparameters, cv=5, verbose=0)
best_model = model_gs.fit(train_x,train_y)
print('Best C:', best_model.best_estimator_.get_params()['C'])

print(best_model.score(train_x, train_y))
score_best = best_model.score(test_x, test_y)
print("Best model score:"+str(score_best))
predictions_best=best_model.predict(test_x)
y_probas = best_model.predict_proba(test_x)
skplt.metrics.plot_roc(test_y, y_probas)
plt.show()
cm_best = metrics.confusion_matrix(test_y, predictions_best, [0,1])
print("Confusion matrix of best model:")
print(cm_best)


# In the above step, we tried cross validation using grid search and picked the best parameters for the model. The accuracy score for training data is 90.7% and for test data is 90.5%. They are pretty close.

# In[ ]:


#Perform oversampling as true positve rate is low for 'yes' and then fit basic log reg
#Dropped correlated features.

X_resampled, y_resampled = SMOTE().fit_resample(train_x, train_y)
basemodel_resampled = LogisticRegression(solver='lbfgs',max_iter=10000)
basemodel_resampled.fit(X_resampled, y_resampled)
predictions_bm=basemodel_resampled.predict(test_x)
score_bm = basemodel_resampled.score(test_x, test['y'])

print(basemodel_resampled.score(X_resampled, y_resampled))
print("Score of base model after oversampling is:"+str(score_bm))
y_probas = basemodel_resampled.predict_proba(test_x)
skplt.metrics.plot_roc(test_y, y_probas)
plt.show()
cm_bm = metrics.confusion_matrix(test_y, predictions_bm, [0,1])
print("Confusion Matrix of base model after oversampling:")
print(cm_bm)


# Accuracy for test rate dropped, but improved for training set after oversampling. 

# In[ ]:


#Resampling, hyperparameter tuning with grid search cv to pick best model.

grid_resampled = LogisticRegression(solver='lbfgs')
penalty = ['l2']
max_iter=[10000]    
# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)
# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty, max_iter=max_iter)
model_gs = GridSearchCV(grid_resampled, hyperparameters, cv=5, verbose=0)
best_model = model_gs.fit(X_resampled,y_resampled)
print('Best C:', best_model.best_estimator_.get_params()['C'])
score_best = best_model.score(test_x, test_y)

print(best_model.score(X_resampled, y_resampled))
print("Best model score:"+str(score_best))
predictions_best=best_model.predict(test_x)
y_probas = best_model.predict_proba(test_x)
skplt.metrics.plot_roc(test_y, y_probas)
plt.show()
cm_best = metrics.confusion_matrix(test_y, predictions_best, [0,1])
print("Confusion matrix of best model:")
print(cm_best)


# In[ ]:




