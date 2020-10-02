#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('/kaggle/input/learn-together/train.csv')
test = pd.read_csv('/kaggle/input/learn-together/test.csv')


# # Read Data

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


#calculate cardinalities of each categorical variables
train.apply(pd.Series.nunique)


#           wow! There are 9 features with over 100 unique values.

# ## Observe any inbalance in target feature.

# In[ ]:


train['Cover_Type'].value_counts()


# In[ ]:


plt.title('Class Distribution')
sns.countplot(data = train, x = 'Cover_Type')
plt.show;


# ## Selecting the best features

# ### Univariate method

# In[ ]:


from sklearn.feature_selection import SelectKBest, f_classif
X = train.drop(['Cover_Type', 'Id'], axis = 1)
y = train['Cover_Type']
#select 10 best features using SelectKBest class
bestfeatures = SelectKBest(f_classif, k = 10)
bestfeatures.fit(X,y)
train_scores = pd.DataFrame(bestfeatures.scores_) 
train_cols = pd.DataFrame(X.columns)
#concantenate the two dataframes for better visualization
best_features_scores = pd.concat([train_cols, train_scores], axis = 1)
best_features_scores.columns = ['feature', 'score']
#best_features_scores.sort_values('score', ascending = False)
best_20_features = best_features_scores.nlargest(20, ['score'])
best_20_features


# In[ ]:


plt.figure(figsize=(6,10))
sns.barplot(x = 'score', y = 'feature', data = best_20_features[:10]);


# ### Feature Importance Method

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier

selector = ExtraTreesClassifier()
selector.fit(X,y)
important_features = pd.Series(selector.feature_importances_, index=X.columns)
plt.figure(figsize=(6,10))
important_features.nlargest(10).plot(kind = 'barh');


# ### Correlation Matrix with Heatmap

# In[ ]:


plt.figure(figsize=(12,8))
best_20_cols = list(best_20_features['feature'][:10])
sns.heatmap(train[best_20_cols].corr(),annot = True, linewidth = 1.0)


# # Train The Model

# In[ ]:


# Separate training features from target feature
X_reduced = train[best_20_cols]
test_reduced = test[best_20_cols]
y = train['Cover_Type']
test_id = test['Id']


# In[ ]:


X_reduced.head()


# In[ ]:


test_reduced.head()


# In[ ]:


# import important libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[ ]:


# Split data into training and validation data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 22)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[ ]:


rFModel_1 = RandomForestClassifier(n_estimators=1000)
rFModel_1.fit(X_train, y_train)


# In[ ]:


#print training accuracy
print('Training accuracy is :', round(rFModel_1.score(X_train, y_train),2)*100,'%')


# In[ ]:


#print validation accuracy
pred = rFModel_1.predict(X_test)
print("The validation accuracy :", round(accuracy_score(y_test, pred),2)*100,'%')


# **A clear case of overfitting as as the validation model is 14% short of training model.**

# In[ ]:


print('Get the list of hyperparameters used by the current model:\n')
rFModel_1.get_params()


# ## Hyperparameters tuning with RandomSearch Grid

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
rFModel_2 = RandomForestClassifier(random_state = 22)
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# In[ ]:


rf_random = RandomizedSearchCV(estimator = rFModel_2, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, 
                               random_state=22, n_jobs = -1)

# Fit the random search model
rf_random.fit(X, y)


# In[ ]:


pred_grid = rf_random.predict(X_test)
print("The validation accuracy for grid search :", round(accuracy_score(y_test, pred_grid),2)*100,'%')


# In[ ]:


test.head()


# # Predictions

# In[ ]:


#get the test data predictions
test = test.drop(['Id'], axis = 1)
pred_test = rf_random.predict(test)
pred_test_output = pd.DataFrame({'Id' : test_id, 'Cover_Type' : pred_test})
pred_test_output.to_csv('submission1.csv', index = False)


# In[ ]:




