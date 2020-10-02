#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

from sklearn.metrics import mean_squared_error
from math import sqrt

import lightgbm as lgb


# In[ ]:


admission = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')


# In[ ]:


admission.info()


# In[ ]:


admission.head()


# __Admission Chance (Target Variable) Distribution__

# In[ ]:


admission.columns = [c.replace(' ', '_') for c in admission.columns]


# In[ ]:


sns.distplot(admission.Chance_of_Admit_, rug = True)
plt.title("Distribution of Admission Chances")
plt.xlabel("Admission Chance")
plt.ylabel("Count of student")
plt.show()


# In[ ]:


admission.Chance_of_Admit_.describe()


# The distribution is not that normal, we can try to log transform the data to see how it perform in this situation

# In[ ]:


admission['logadchange'] = np.log1p(admission["Chance_of_Admit_"])
sns.distplot(admission['logadchange'], color = "r")
plt.title("Log Transformed Distribution of Admission Chances")
plt.xlabel("Admission Chance")
plt.ylabel("Count of student")
plt.show()


# In[ ]:


admission.logadchange.describe()


# __Check GRE & TOEFL Score Vs Admission Chance__

# In[ ]:


admission.GRE_Score.describe()


# In[ ]:


sns.jointplot(x = "GRE_Score", y = "Chance_of_Admit_", data = admission, height = 8, ratio = 4, color = "b")
plt.show()


# In[ ]:


admission.TOEFL_Score.describe()


# In[ ]:


sns.jointplot(x = "GRE_Score", y = "Chance_of_Admit_", data = admission, height = 8, ratio = 4, color = "r")
plt.show()


# These 2 scoares are pretty good measurement of the admission rates. 

# __Check University Rating / Statement of Purpose / Letter of Recommendation Vs Admission Chance__

# In[ ]:


admission.University_Rating.describe()


# In[ ]:


plt.figure(figsize = (8, 6))
sns.countplot(admission['University_Rating'].sort_values())
plt.title("University Rating Count", fontsize = 15)
plt.xticks(fontsize = 14)
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize = (8, 6))
fig = sns.boxplot(x = admission['University_Rating'], y = admission['Chance_of_Admit_'], data = admission)
fig.axis(ymin = 0, ymax = 1.0)
plt.xlabel("University Ratings")
plt.ylabel("Admission Chance")
plt.show()


# In[ ]:


admission.SOP.describe()


# In[ ]:


plt.figure(figsize = (8, 6))
sns.countplot(admission['SOP'].sort_values())
plt.title("Statement of Purpose Strength Level Count", fontsize = 15)
plt.xticks(fontsize = 14)
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize = (8, 6))
fig = sns.boxplot(x = admission['SOP'], y = admission['Chance_of_Admit_'], data = admission)
fig.axis(ymin = 0, ymax = 1.0)
plt.xlabel("Statement of Purpose Strength Level")
plt.ylabel("Admission Chance")
plt.show()


# In[ ]:


admission.LOR_.describe()


# In[ ]:


plt.figure(figsize = (8, 6))
sns.countplot(admission['LOR_'].sort_values())
plt.title("Letter of Recommendation Strength Level Count", fontsize = 15)
plt.xticks(fontsize = 14)
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize = (8, 6))
fig = sns.boxplot(x = admission['LOR_'], y = admission['Chance_of_Admit_'], data = admission)
fig.axis(ymin = 0, ymax = 1.0)
plt.xlabel("Letter of Recommendation Strength Level")
plt.ylabel("Admission Chance")
plt.show()


# __Check Undergraduate GPA Vs Admission Chance__

# In[ ]:


admission.CGPA.describe()


# In[ ]:


sns.jointplot(x = "CGPA", y = "Chance_of_Admit_", data = admission, height = 8, ratio = 4, color = "b")
plt.show()


# __Correlation Matrix__

# In[ ]:


admission = admission.iloc[:, admission.columns != 'Serial_No.']
admission_expect = admission.iloc[:, admission.columns != 'logadchange']
cor = admission_expect.corr()
f, ax = plt.subplots(figsize = (8, 6))
sns.heatmap(cor, vmax = 0.9, annot = True, square = True, fmt = '.2f')


# __The takeaways from this heatmaps:__
#     - All variables have significant relationship with our predictor admission chance
#     - TOEFL score, CGPA, GRE score, SOP and University Rating is strongly correlated with each other

# In[ ]:


#Split the X and the y
X = admission.iloc[:, admission.columns != 'Chance_of_Admit_']
y = admission.iloc[:, admission.columns == 'Chance_of_Admit_']


# In[ ]:


X.shape


# In[ ]:


y.shape


# We choose to use Principle Component Analysis to reduce our dimensionality while preserve the highest variance up to 95% of the data

# In[ ]:


from sklearn.decomposition import PCA
from sklearn import preprocessing

pca = PCA()
pca.fit(X)
pca.data = pca.transform(X)


#Percentage variance of each pca component stands for
per_var = np.round(pca.explained_variance_ratio_*100, decimals = 1)
#Create labels for the scree plot
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

#Plot the data
plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label = labels)
plt.ylabel('percentage of Explained Variance')
plt.xlabel('Principle Component')
plt.title('Scree plot')
plt.show()


# In[ ]:


print('PC1 + PC2 add up to ' +  str(sum(per_var[:2])) + ' % of the variance')


# In[ ]:


#Extract the PC1 and PC2 information
newdata = pd.DataFrame(data=pca.data[:,:2], columns = ['PC1', 'PC2'])


# In[ ]:


newdata.head()


# In[ ]:


newdata.shape #Checking shape


# In[ ]:


newdata['PC1'].describe()


# In[ ]:


#train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(newdata, y,
                                                   test_size = 0.25, random_state = 2019)


# Perform SGD Regression (Might not be a good model for this example since it is less than 100k sample)

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)  # Don't cheat - fit only on training data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  # apply same transformation to test data


# In[ ]:


#Tunning parameters using our gridsearch
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV

params = {
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'tol': [0.001, 0.1, 1],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
}

sgd_clf = SGDRegressor(max_iter=1000)
sgd_grid_clf = GridSearchCV(sgd_clf, params, cv = 5)
sgd_grid_clf.fit(X_train, y_train.values.ravel())


# In[ ]:


#Calculating the RMSE
sgd_pred = sgd_grid_clf.predict(X_test)
sgd_rms = sqrt(mean_squared_error(y_test, sgd_pred))
sgd_rms


# __VS Randomsearch__

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

params_rand = {
    'alpha': uniform(),
    'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'tol': [0.001, 0.1, 1],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
}


# In[ ]:


sgd_rand_clf = RandomizedSearchCV(sgd_clf, params_rand, cv = 5, random_state = 2019)
sgd_rand_clf.fit(X_train, y_train.values.ravel())


# In[ ]:


#Calculating the RMSE
sgd_rand_pred = sgd_rand_clf.predict(X_test)
sgd_rand_rms = sqrt(mean_squared_error(y_test, sgd_rand_pred))
sgd_rand_rms


# __Train on Lasso Regressor__

# In[ ]:


from sklearn.linear_model import Lasso

lasso_params = {
    'alpha': [0.001, 0.01, 0.1, 1.0],
    'tol': [0.001, 0.1, 1]
}

lasso_clf = Lasso(max_iter=1000)
lasso_grid_clf = GridSearchCV(lasso_clf, lasso_params, cv = 5)
lasso_grid_clf.fit(X_train, y_train.values.ravel())


# In[ ]:


#Calculating the RMSE
lasso_pred = lasso_grid_clf.predict(X_test)
lasso_rms = sqrt(mean_squared_error(y_test, lasso_pred))
lasso_rms


# __VS RandomizedSearch__

# In[ ]:


lasso_rand_params = {
    'alpha': uniform(),
    'tol': [0.001, 0.1, 1]
}


# In[ ]:


lasso_rand_clf = RandomizedSearchCV(lasso_clf, lasso_rand_params, cv = 5,  n_iter = 100, random_state = 2019)
lasso_rand_clf.fit(X_train, y_train.values.ravel())


# In[ ]:


#Calculating the RMSE
lasso_rand_pred = lasso_rand_clf.predict(X_test)
lasso_rms = sqrt(mean_squared_error(y_test, lasso_rand_pred))
lasso_rms


# In[ ]:





# ### __LightGBM__

# In[ ]:


# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)


# In[ ]:


#Tunning in hyperparameters
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': {'l2', 'l1'},
    'max_depth': 6,
    'learning_rate': 0.06,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'max_bin': 255,
    'verbose': 0
}


# In[ ]:


#Train the model
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=120,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)


# In[ ]:


#Making predictions
gbm_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
print('The rmse of prediction is:', mean_squared_error(y_test, gbm_pred) ** 0.5)


# In[ ]:




