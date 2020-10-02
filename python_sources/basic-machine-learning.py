#!/usr/bin/env python
# coding: utf-8

# **This is my first attempt on Kaggle and also my first kernel/notebook** . Would love to hear from you. I wrote this after spending 2-3 months now on learning Python, Statistics, Data Science and Machine Learning.
# 
# > Machine Learning > Supervised Learning > Classification
# 
# * This code is for beginners who want to get their hands dirty with machine learning
# * This is a case for supervised learning and classification in particular
# * Here we teach computer to learn from previous available *classifications* done in the past and then *predict* on newly available data using best performing algorithm

# In[ ]:


# Importing modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, Normalizer, OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import accuracy_score, make_scorer, classification_report, confusion_matrix


# In[ ]:


# load data

data = load_iris()


# In[ ]:


# understand dataset description

print(data.DESCR)


# In[ ]:


# prepare features and labels

X = data.data
y = data.target
df_X = pd.DataFrame(X)
df_y = pd.DataFrame(y)
df = pd.concat([df_X,df_y], axis=1)
df.columns = ['sl','sw','pl','pw','class']


# In[ ]:


# lets understand the spread of each feature

sns.boxplot(data=X)
plt.show()


# In[ ]:


# lets understand how each feature and label relate to each other

sns.pairplot(data=df, hue='class')
plt.show()


# In[ ]:


# lets understand the data numerically

df.describe()


# In[ ]:


# figure out correlation to check multi collinearity(high correlated between features) exists

sns.heatmap(df.corr())
plt.show()


# In[ ]:


# set the random state to compare and discuss results

rs = 7


# In[ ]:


# split dataset for training and testing

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=rs)


# In[ ]:


# prepare fold for cross validation

kfold = KFold(n_splits=10, random_state=rs)


# In[ ]:


# Compare models

results = cross_val_score(LogisticRegression(), X_train, y_train, cv=kfold, scoring='accuracy')
print(results.mean(), results.std())


# In[ ]:


# Fit the model or train the model and predict for test dataset, then judge performance

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
predictions = logreg.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# In[ ]:


# lets do it with support vector classifier

logreg = SVC()
logreg.fit(X_train, y_train)
predictions = logreg.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# In[ ]:


# how about LDA

logreg = LinearDiscriminantAnalysis()
logreg.fit(X_train, y_train)
predictions = logreg.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# In[ ]:


# how about KNN Classifier

logreg = KNeighborsClassifier()
logreg.fit(X_train, y_train)
predictions = logreg.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# In[ ]:


# how about gradient boosting classifier

model = GradientBoostingClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# In[ ]:


# lets try with ensembled models

ensembles = []
ensembles.append(('AB', AdaBoostClassifier()))
ensembles.append(('GBM', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier()))
ensembles.append(('ET', ExtraTreesClassifier()))
results = []
names = []
for name, model in ensembles:
        kfold = KFold(n_splits=10, random_state=rs)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# **97% is good accuracy to stop iterating. 
# 
# **Please do share your feedback, Upvote, comment**

# In[ ]:




