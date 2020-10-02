#!/usr/bin/env python
# coding: utf-8

# # EXPLORING SKLEARN WITH FINANCIAL DISTRESS DATA

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import r2_score, median_absolute_error
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score, learning_curve, GridSearchCV, train_test_split, learning_curve
from sklearn.feature_selection import RFECV


# **Import data**

# In[ ]:


fdata = pd.read_csv('../input/Financial Distress.csv')


# ****DISPLAYING SAMPLE DATAFRAME****

# In[ ]:


fdata.head(n= 10)


# PLOT OF COMPANY NUMBER VS. FINANCIAL DISTRESS

# In[ ]:


f, ax = plt.subplots(figsize= (12,12))
fdata.groupby(['Company'])['Financial Distress'].mean().plot()
plt.show()


# In[ ]:


correlation = fdata.drop(labels= ['Time', 'Company'], axis =1).corr()


# Correlation Heatmap Of Features with Each other and With Financial Distress

# In[ ]:


f, ax = plt.subplots(figsize= (25,25))
sns.heatmap(correlation)
plt.xticks(fontsize = 23)
plt.yticks(fontsize = 23)
plt.show()


# In[ ]:


correlation['Financial Distress'].plot()
plt.show()


# Bar Plot of Correlation Matrix

# In[ ]:


f, ax = plt.subplots(figsize= (25,25))
sns.barplot(x = correlation['Financial Distress'], y = correlation.index)
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 23)
plt.ylabel('FEATURES', fontsize= 25)
plt.xlabel('MEAN FINANCIAL DISTRESS', fontsize= 25)
plt.title('FEATURE CORRELATION', fontsize =30)
plt.show()


# In[ ]:


Y = fdata['Financial Distress']
X = fdata.drop(['Company', 'Time','Financial Distress'], axis = 1)


# In[ ]:


X.head()


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=20, test_size= 0.25)


# Applying Simple SVM Model 

# In[ ]:


svreg = svm.SVR(epsilon = 0.1, C = 80)
svreg.fit(X_train,Y_train)


# In[ ]:


svreg.score(X_train, Y_train)


# In[ ]:


cv = ShuffleSplit(n_splits = 3, test_size=0.25, random_state=0)


# In[ ]:


median_absolute_error(svreg.predict(X_test), Y_test)


# In[ ]:


mse(svreg.predict(X_test), Y_test)


# Applying Complex Boosting Ensemble i.e. Gradient Boosting Regressor

# In[ ]:


gbr = GradientBoostingRegressor()
gbr.fit(X_train, Y_train)


# Creating Feature Ranking DataFrame By Using Recursive Feature Elimination from sklearn

# In[ ]:


selector = RFECV(gbr, cv= cv)
selector.fit(X_train, Y_train)


# In[ ]:


f, ax = plt.subplots(figsize= (25,25))
sns.barplot(x = (1/selector.ranking_), y = X_train.columns)
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 23)
plt.xlabel('1/rank', fontsize= 25)
plt.ylabel('FEATURES', fontsize= 25)
plt.title('FEATURE RANKING', fontsize =30)
plt.show()


# In[ ]:


sr = pd.DataFrame(selector.ranking_)
rd = (pd.DataFrame(X_train.columns),sr)
pd.concat(rd, axis = 1 )


# In[ ]:


my_params= { 
             'learning_rate' : [0.1, 0.01],
             'max_depth' : [6, 4, 5],
            'n_estimators' : [150, 160, 140]
             
}


# Applying Exhaustive Hyper-Parameter Search for finding Out best Suitable parameters for GBR model 

# In[ ]:


gbrgs = GridSearchCV(gbr, my_params, cv =cv)
gbrgs.fit(X_train, Y_train)


# In[ ]:


gbrgs.best_params_


# In[ ]:


gbrgs.best_score_


# In[ ]:


gbrgs.best_estimator_.score(X_train, Y_train)


# In[ ]:


gbr1 = GradientBoostingRegressor(learning_rate = 0.01, max_depth = 7, n_estimators = 210)
gbr1.fit(X_train, Y_train)


# Training Score Of GBR Model

# In[ ]:


gbr1.score(X_train, Y_train)


# Median Absolute Error of GBR Model

# In[ ]:


median_absolute_error(gbr1.predict(X_test),Y_test)


# Mean Squared Error of GBR Model

# In[ ]:


mse(gbr1.predict(X_test),Y_test)


# In[ ]:


tra_sizes, tra_scores, test_scores = learning_curve(gbr1, X, Y,cv = cv)


# In[ ]:


tra_smean = np.mean(tra_scores, axis=1)
test_smean = np.mean(test_scores, axis=1)


# Learning Curve Of the model shows whether the model overfits or underfits. In this case, the model highly overfits but the predictions are satisfying

# In[ ]:


plt.figure(figsize=(12,10))
plt.title("Learning Curve Of Gradient Boosting -Regressor Model", fontsize = 25)
plt.xlabel("Training examples", fontsize= 25)
plt.ylabel("Score", fontsize = 25)
plt.plot(tra_sizes[1:5], tra_smean[1:5], 'o-', color = 'r', label = "training set")
plt.plot(tra_sizes[1:5], test_smean[1:5], 'o-', color = 'g', label = "cv set")
plt.legend(fontsize = 20)
plt.show()


# Plot of Predicted Values vs. Real Values of Financial Distress

# In[ ]:


plt.figure(figsize=(12,10))
plt.title("Prediction Curve", fontsize = 25)
plt.xlabel("Training examples",fontsize = 25)
plt.ylabel("Prediction", fontsize = 25)
plt.plot(X_test[2:20].reset_index().index, Y_test[2:20], 'o-', color = 'r', label = 'real')
plt.plot(X_test[2:20].reset_index().index, gbr1.predict(X_test)[2:20] , 'o-', color = 'g', label = "predicted")
plt.legend(fontsize = 25)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




