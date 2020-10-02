#!/usr/bin/env python
# coding: utf-8

# ## Breast Cancer Wisconsin Data Set
# 
# Breast Cancer Wisconsin data set:
# https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
# 
# Attribute Information:
# 
# 1) ID number 
# 2) Diagnosis (M = malignant, B = benign)  
# 
# Ten real-valued features are computed for each cell nucleus:
# 
# a) radius (mean of distances from center to points on the perimeter) 
# b) texture (standard deviation of gray-scale values) 
# c) perimeter 
# d) area 
# e) smoothness (local variation in radius lengths) 
# f) compactness (perimeter^2 / area - 1.0) 
# g) concavity (severity of concave portions of the contour) 
# h) concave points (number of concave portions of the contour) 
# i) symmetry 
# j) fractal dimension ("coastline approximation" - 1)
# 
# The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.
# 
# All feature values are recoded with four significant digits.
# 
# Missing attribute values: none
# 
# Class distribution: 357 benign, 212 malignant

# ## Import Libraries

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import sem


# ## Import and Interrrogate Data

# In[ ]:


data =  pd.read_csv("../input/data.csv",header = 0)

# list columns
print(data.columns)


# We see that we have an 'Unnamed' column that appears to blank so we shall remove it. We also remove the id field in our data prep. We shall convert to numpy arrays

# In[ ]:


# Create Target

y = np.array(data.diagnosis)
labels = LabelEncoder()
target = labels.fit_transform(y)


# In[ ]:


# Create features normalise the data

cols = data.columns[(data.columns != 'id') & (data.columns != 'diagnosis') & (data.columns != 'Unnamed: 32')]
features = data[cols]
features = (features - features.mean()) / (features.std())


# In[ ]:


# Look at correlation between features using seaborn

corr = features.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True


with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(25, 20))
    sns.heatmap(corr, mask=mask, vmax=1, square=True, annot=True)


# As expected perimeter mean and area mean are very highly correlated to radius mean we shall remove them. Further we notice that radius worst, perimiter worst and area worst are also highly correlated so again we shall remove these also. Following this we shall take another look at our correlation matrix to ensure we have not missed any additional significant correlations.

# In[ ]:


features = features.drop(labels=['perimeter_mean','area_mean','radius_worst','perimeter_worst', 'area_worst'], axis=1)


# In[ ]:


# Look at correlation between features using seaborn

corr = features.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True


with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(25, 20))
    sns.heatmap(corr, mask=mask, vmax=1, square=True, annot=True)


# ## SVC for Cancer Diagnosis

# In[ ]:


# Split our data into training and test data

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=123)


# In[ ]:


# Use grid search to tune hyperparameters for our SVC on training data

svc = svm.SVC()
parameters = [{'kernel':['linear'], 'C': np.logspace(-2, 2, 25)},
             {'kernel':['poly'], 'C': np.logspace(-2,2,25), 'gamma': np.logspace(-4,0,25), 'degree': [2, 3, 4, 5, 6, 7]},
             {'kernel':['rbf', 'sigmoid'], 'C': np.logspace(-2, 2, 25), 'gamma': np.logspace(-4,0,25)}]
clf = GridSearchCV(svc, parameters)
clf.fit(X_train, y_train)


# In[ ]:


# Print out hyperparameter selections and set-up new classifier with these

print('Best Parameters:', clf.best_params_)
clf = svm.SVC(**clf.best_params_)
clf.fit(X_train, y_train)


# In[ ]:


# Test our calibrated model accuracy on our test data

print('Accuracy on training data:')
print(clf.score(X_train, y_train))
print('Accuracy on test data:')
print(clf.score(X_test, y_test))

y_pred = clf.predict(X_test)

print('Classification report:')
print(metrics.classification_report(y_test, y_pred))
print('Confusion matrix:')
print(metrics.confusion_matrix(y_test, y_pred))


# In[ ]:


# Cross validation of support vector classifier

cv = KFold(5, shuffle=True, random_state=123)
scores = cross_val_score(clf, features, target, cv=cv)
print(scores)
print("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))


# ## Observations on SVC:
# The accuracy on test data is fairly high and there does not appear to be significant overfitting to the training data.

# ## Finding Feature Importance Using a Random Forest

# One 'problem' with the support vector machine classifier above is it is fairly 'black box', we shall now use a random forest classifier in order to deduce feature importance

# In[ ]:


clf_rf = RandomForestClassifier(n_estimators=2500, max_features=None, criterion='entropy')
clf_rf.fit(X_train, y_train)


# In[ ]:


# Test our calibrated model accuracy on our test data

print('Accuracy on training data:')
print(clf_rf.score(X_train, y_train))
print('Accuracy on test data:')
print(clf_rf.score(X_test, y_test))

y_pred = clf_rf.predict(X_test)

print('Classification report:')
print(metrics.classification_report(y_test, y_pred))
print('Confusion matrix:')
print(metrics.confusion_matrix(y_test, y_pred))


# In[ ]:


# Cross validation of random forest

cv = KFold(5, shuffle=True, random_state=123)
scores = cross_val_score(clf_rf, X_train, y_train, cv=cv)
print(scores)
print("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))


# Showing slightly weaker performance than the support vector machine method, but only intending to use this for feature importance so for this purpose it is fine.

# In[ ]:


importances = clf_rf.feature_importances_
importances = pd.DataFrame(importances, index=features.columns, columns=["Importance"])
importances = importances.sort_values(by=['Importance'],ascending=False)


# In[ ]:


importances.plot(kind='bar')


# From this analysis we can see that the concave points (mean and worst) are the most important features followed by radius_mean and texture_worst. 

# ## Conclusions
# From the analysis above we can deduce the following:
# - The support vector machine classifier has a test data accuracy of 98.9% and a mean cross validation of 97.9%. There does not appear to be significant over-fitting. However the confusion matrix suggests that we have some false negatives (that is: we classify malignant examples as benign.) In a practical situation this would be a problem and we would wish to reduce this if possible.
# - From the random forest classifier we deduce that the most important features are: concave points_mean, concave points_worst, radius_mean and texture_worst.

# In[ ]:




