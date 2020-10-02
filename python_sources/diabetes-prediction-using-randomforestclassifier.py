#!/usr/bin/env python
# coding: utf-8

# **A regression ensemble** works by combining several simple regressors and the final prediction of an ensemble model is a simple average of predictions from all the simple regression models that make up this ensemble.
# 
# **For an ensemble classifier**, the final prediction is a majority vote out of all the predictions made by the constituent classifiers that make up this ensemble.
# 
# Theoretically, **an ensemble model** can be built by combining any set of simple models. But tree models are a more popular choice.
# 
# **Another characteristic of tree based ensemble models** is that while training these models, only a subset of total data is used by each base learner, the way this data set is fed into each of the base learners is based on a data sampling scheme.
# 
# Different sampling schemes give rise to different types of tree based ensembles.
# 
# There are 3 popular tree based ensemble models :
# (i) Bagged Trees
# (ii) Random Forest
# (iii) Boosted Trees
# 
# I am implementing Random Forest in this notebook using Pima Indians Diabetes dataset.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# Importing required libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score,roc_curve,make_scorer
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Reading the dataset

data = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")


# In[ ]:


# Returning first few observations

data.head()


# Here, Outcome is a target/dependent variable and others are predictor/independent variables.

# In[ ]:


# Returning dimensions of dataframe

data.shape


# In[ ]:


#  List of column names

data.columns.tolist()


# In[ ]:


# Returning dtypes of each column

data.dtypes


# In[ ]:


# Returning no. of missing values in each column

data.isnull().sum()


# In[ ]:


# Summary of dataframe

data.info()


# In[ ]:


# Generating descriptive statistics

data.describe()


# As you can see above some of the variables (Glucose,BloodPressure,SkinThickness,Insulin,BMI) have 0 as minimum value and that is not possible.
# 
# These variables have missing values as 0 present in this dataset. So, marking them as missing values by replacing it by NaN.

# In[ ]:


# Replacing 0 by NaN

data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)


# In[ ]:


# Again, getting no. of missing values in each column 

data.isnull().sum()


# As you can see there are 5 variables having missing values.
# Now, filling missing values in these variables by specific value.

# In[ ]:


# Filling NaN values 

data['Glucose'].fillna(data['Glucose'].median(), inplace = True)
data['BloodPressure'].fillna(data['BloodPressure'].median(), inplace = True)
data['SkinThickness'].fillna(data['SkinThickness'].median(), inplace = True)
data['Insulin'].fillna(data['Insulin'].median(), inplace = True)
data['BMI'].fillna(data['BMI'].mean(), inplace = True)


# In[ ]:


# Checking missing values correctly replaced

data.isnull().sum()


# You can see missing values are now filled by specified methods.

# In[ ]:


# Getting correlations of each features in dataframe

corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (15,20))

# Plotting heat map

g = sns.heatmap(data[top_corr_features].corr(),annot = True,cmap = "RdYlGn")


# In[ ]:


# Finding pairwise correlation of all columns

data.corr()


# In[ ]:


# Getting unique values 

data['Pregnancies'].unique()


# In[ ]:


# Finding counts of unique values and sorting it in ascending order

data['Pregnancies'].value_counts().sort_values()


# In[ ]:


# Grouping predictor variables by target variable

data.groupby("Outcome")[["Pregnancies","Glucose","BloodPressure"]].agg(['max','min','mean'])


# In[ ]:


data.groupby("Outcome")[["SkinThickness","Insulin","BMI","Age"]].agg(['max','min','mean'])


# In[ ]:


# Finding counts of unique values 

data['Outcome'].value_counts()


# In[ ]:


# Plotting histogram of dataframe

p = data.hist(figsize = (15,20))


# In[ ]:


# Creating Predictor Matrix

X = data.drop('Outcome',axis = 1)


# In[ ]:


# Getting first few observations of predictor matrix

X.head()


# In[ ]:


# Target variable

y = data['Outcome']


# In[ ]:


# Getting first few observations of target variable

y.head()


# In[ ]:


# Splitting the matrices into random train & test subsets where test data contains 25% data and rest considered as training data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 200)


# In[ ]:


# Getting dimensions of train & test subsets

X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[ ]:


# Instantiating random forest classifier

clf = RandomForestClassifier(oob_score = True,n_jobs = -1,random_state = 100)
clf


# Cross validation score should between 0 and 1 and as high as possible.
# Here cross validation has been performed to find how well model is performing in terms of F1 score.

# In[ ]:


# Performing K-fold cross validation with 5 folds 

scores = cross_val_score(clf,X_train,y_train,cv = 5,scoring = "f1_macro")
scores.mean()


# In[ ]:


# Building a forest of trees from training set

clf.fit(X_train,y_train)


# In[ ]:


# Predicting on classifier created

train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test)


# In[ ]:


# Finding F1 score of training and testing sets 

print("The training F1 score is: ",f1_score(train_pred,y_train))
print("The testing F1 score is :",f1_score(test_pred,y_test))


# Training F1 score is high but testing F1 score is low. It shows that the model is overfitting.
# Model should have high training as well as high testing accuracy.
# Generally training score close to 1 means model is overfitting. But if you have low training accuracy but equally weighted testing accuracy then it's good generalized model

# Now, hyperparameter tuning needs to be done and looking for high F1 score that is why scorer variable is defined.

# In[ ]:


#  Tuning hyperparameters

parameters = {
             "max_depth":[2,3,4],
             "n_estimators":[100,104,106],
             "min_samples_split":[3,4,5],
             "min_samples_leaf":[4,8,9]
             }

scorer = make_scorer(f1_score)


# In[ ]:


# Using Randomized Search CV to find best optimal hyperparameter that best describe a classifier

clf1 = RandomizedSearchCV(clf,parameters,scoring = scorer)

# Fitting the model

clf1.fit(X_train,y_train)

# Getting best estimator having high score

best_clf_random = clf1.best_estimator_
best_clf_random


# In[ ]:


# Again, finding cross validation score

scores = cross_val_score(best_clf_random,X_train,y_train,cv = 5,scoring = "f1_macro")
scores.mean()


# As you can see cross validation score has decreased as compared to earlier score.
# It should increase and for that you have to try changing hyperparameter values so that better cross validation score can be achieved.

# In[ ]:


# Fitting the best estimator

best_clf_random.fit(X_train,y_train)


# In[ ]:


# Getting first estimator

best_clf_random.estimators_[0]


# Using above way you can get specific estimators / decision trees that combined up to form a random forest classifier.

# In[ ]:


# Predicting on best estimator

train_pred = best_clf_random.predict(X_train)
test_pred = best_clf_random.predict(X_test)


# In[ ]:


# Finding the F1 score of training & testing sets

print("The training F1 score is: ",f1_score(train_pred,y_train))
print("The testing F1 score is :",f1_score(test_pred,y_test))


# As you can notice that testing F1 score is more than training score. 
# You can try tuning hyperparameters to achieve more better score.

# In[ ]:


# Getting accuracy score 

accuracy_score(y_test,test_pred)


# Accuracy score is 81% and this is really a good score.

# In[ ]:


# Computing ROC AUC from prediction scores

roc_auc_score(y_test,best_clf_random.predict_proba(X_test)[:,1])


# In[ ]:


# Plotting ROC curve

fpr,tpr,thresholds = roc_curve(y_test,best_clf_random.predict_proba(X_test)[:,1])

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()


# Having high Roc curve shows model is performing well.

# In[ ]:


# Computing confusion matrix

pd.crosstab(y_test,test_pred,rownames = ['True'],colnames = ['Predicted'],margins = True)


# **True Positive** is when your actual value is 1 and your classifier predicts as 1.
# **True Negative** is when your actual value is 0 and your predicted value is 0.
# **False positive** is when your actual value is 0 and your predicted value is 1.
# **False Negative** is when your actual value is 1 and your predicted value is 0.

# In[ ]:


# Plotting confusion matrix

cnf_matrix = confusion_matrix(y_test,test_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix),annot = True,cmap = "YlGnBu",fmt = 'g')
plt.title("Confusion Matrix",y = 1.1)
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')


# Precision and recall also should be as high as possible.
# 
# **Precision** is out of all the samples that my classifier labeled as a positive sample, what fraction is actually correct.
# 
# **Recall** is out of all the positive samples that I have, what fraction is my classifier pick up.

# In[ ]:


# Computing the precision

precision_score(y_test,test_pred)


# In[ ]:


# Computing the recall

recall_score(y_test,test_pred)


# In[ ]:


# Getting feature importances

imp_features = pd.Series(best_clf_random.feature_importances_,index = X.columns)
imp_features.sort_values(ascending = False)


# In[ ]:


# Plotting feature importances 

imp_features.sort_values(ascending = False).plot(kind = "bar")

