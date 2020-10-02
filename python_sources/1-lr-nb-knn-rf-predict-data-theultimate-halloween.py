#!/usr/bin/env python
# coding: utf-8

# # Logistic regression (predicting a categorical value, often with two categories):
# 
# **Question:**
# - The Ultimate Halloween Candy Power Ranking: Can you predict if a candy is chocolate or not based on its other features?

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from warnings import filterwarnings
filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


candy_data = pd.read_csv("../input/the-ultimate-halloween-candy-power-ranking/candy-data.csv")


# # 1.EDA (Exploratory Data Analysis)

# In[ ]:


candy_data.head(2)


# In[ ]:


candy_data.info()


# In[ ]:


candy_data.describe().T


# In[ ]:


# Lets see 0, 1 numbers of chocolate as bar
candy_data['chocolate'].value_counts().plot.barh();


# In[ ]:


# "competitorname" feature we dont need and lets drop it
candy_data.drop("competitorname", inplace = True, axis=1)

y = candy_data.chocolate.values
X = candy_data.drop(["chocolate"], axis = 1)


# In[ ]:


# see how many null values we have then we dont need to normalize

candy_data.isnull().sum()


# # 2.Logistic Regression
# ## 2.1.Set Model

# ## 2.1.1.Scikit-learn

# In[ ]:


loj = LogisticRegression(solver = "liblinear")
loj_model = loj.fit(X,y)
loj_model


# In[ ]:


loj_model.intercept_      # constant value
loj_model.coef_           # independent values


# ## Predict & Model Tuning

# In[ ]:


y_pred = loj_model.predict(X)        # predict
confusion_matrix(y, y_pred)          # confussion matrix


# In[ ]:


accuracy_score(y, y_pred)


# In[ ]:


print(classification_report(y, y_pred))


# In[ ]:


# Model predict
loj_model.predict(X)[0:20]


# In[ ]:


loj_model.predict_proba(X)[0:10][:,0:2]


# In[ ]:


# Now lets try model 'predict_proba' probability

y_probs = loj_model.predict_proba(X)
y_probs = y_probs[:,1]
y_probs[0:20]


# In[ ]:


# giving limit for values

y_pred = [1 if i > 0.5 else 0 for i in y_probs]


# In[ ]:


# and compare with above you can see what happened
y_pred[0:20]


# In[ ]:


confusion_matrix(y, y_pred)


# In[ ]:


accuracy_score(y, y_pred)


# In[ ]:


print(classification_report(y, y_pred))


# In[ ]:


logit_roc_auc = roc_auc_score(y, loj_model.predict(X))

fpr, tpr, thresholds = roc_curve(y, loj_model.predict_proba(X)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()

# blue line: which we set our model
# red line: if we dont do it what can we take result


# In[ ]:


# lets split test train set

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.30, 
                                                    random_state = 42)


# In[ ]:


# set model

loj = LogisticRegression(solver = "liblinear")
loj_model = loj.fit(X_train,y_train)
loj_model


# In[ ]:


accuracy_score(y_test, loj_model.predict(X_test))


# In[ ]:


# with cross validation 

cross_val_score(loj_model, X_test, y_test, cv = 10).mean()


# # 3.Gaussian Naive Bayes

# In[ ]:


nb = GaussianNB()
nb_model = nb.fit(X_train, y_train)
nb_model


# In[ ]:


nb_model.predict(X_test)[0:10]


# In[ ]:


nb_model.predict_proba(X_test)[0:10]


# In[ ]:


# predict
y_pred = nb_model.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


cross_val_score(nb_model, X_test, y_test, cv = 10).mean()


# # 4.KNN
# ## 4.1.Predict & Model

# In[ ]:


knn = KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train)
knn_model


# In[ ]:


y_pred = knn_model.predict(X_test)
accuracy_score(y_test, y_pred)


# In[ ]:


# get detail print

print(classification_report(y_test, y_pred))


# ## 4.2.Model Tunning

# In[ ]:


# find KNN parameters
knn_params = {"n_neighbors": np.arange(1,50)}


# In[ ]:


# fit model classification & CV

knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, knn_params, cv=10)
knn_cv.fit(X_train, y_train)


# In[ ]:


# this is only observation

print("Best score:" + str(knn_cv.best_score_))
print("Best parameters: " + str(knn_cv.best_params_))


# In[ ]:


knn = KNeighborsClassifier(3)
knn_tuned = knn.fit(X_train, y_train)


# In[ ]:


knn_tuned.score(X_test, y_test)


# In[ ]:


y_pred = knn_tuned.predict(X_test)
accuracy_score(y_test, y_pred)


# # 5.Random Forest

# In[ ]:


rf_model = RandomForestClassifier().fit(X_train, y_train)
rf_model


# In[ ]:


y_pred = rf_model.predict(X_test)
accuracy_score(y_test, y_pred)


# ## Model Tuning

# In[ ]:


rf_params = {"max_depth": [2,5,8,10],
            "max_features": [2,5,8],
            "n_estimators": [10,500,1000],
            "min_samples_split": [2,5,10]}


# In[ ]:


rf_model = RandomForestClassifier()

rf_cv_model = GridSearchCV(rf_model, 
                           rf_params, 
                           cv = 10, 
                           n_jobs = -1, 
                           verbose = 2) 


# In[ ]:


rf_cv_model.fit(X_train, y_train)


# In[ ]:


print("Best Parameters: " + str(rf_cv_model.best_params_))


# In[ ]:


# using given parameters then create final model

rf_tuned = RandomForestClassifier(max_depth = 2, 
                                  max_features = 5, 
                                  min_samples_split = 10,
                                  n_estimators = 10)

rf_tuned.fit(X_train, y_train)


# In[ ]:


# tunned test model predict accuracy score

y_pred = rf_tuned.predict(X_test)
accuracy_score(y_test, y_pred)


# In[ ]:




