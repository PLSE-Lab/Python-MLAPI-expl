#!/usr/bin/env python
# coding: utf-8

# # Predicting students' math performances through the data set
# 
# In this kernel, I am just focusing on predicting the math scores. I reach quite a low RMSE with XGBoost (~5 ish) which is not too shabby. 

# First, we load in the data and inspect it. 

# In[ ]:


import os

import numpy as np
import pandas as pd

data = pd.read_csv('../input/StudentsPerformance.csv')

# Convert to categorical!
for i in range(data.shape[1]):
    if i not in [5,6,7]:
        data.iloc[:, i] = data.iloc[:,i].astype("category")

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

for i in range(data.shape[1]):
    if i not in [1,2,5,6,7]:
        data.iloc[:, i] = label_encoder.fit_transform(data.iloc[:,i])



data = pd.get_dummies(data)
data.head()


# So this is the data. What do we do with it now? We could try a logistic regression analysis, since we have many classes. The classes are mostly binary or tertiary, apart from parental level of education. Random Forest and XGBoost are also great options, so I will implement these with cross-validation. For each score, I will create one of each model mentioned above and compute their MSE scores. But first, we divide into training and test data. 

# In[ ]:



from sklearn.model_selection import train_test_split 

train, test = train_test_split(data, random_state = 123)

print(train.shape)
print(test.shape)

col_names = list(data.columns)


# ## Predicting Math score
# 
# First, we predict the math score. 

# In[ ]:


math_pred_feats = [x for i, x in enumerate(col_names) if i in list(range(0,data.shape[1])) and i not in [3]]
math_pred_label = col_names[3]
train_math = train[math_pred_feats]
y_train_math = train[math_pred_label]
test_math = test[math_pred_feats]
y_test_math = test[math_pred_label]



from xgboost import XGBRegressor
import xgboost as xgb


xgb_model = XGBRegressor(max_depth = 5,
                n_estimators=500,
                n_jobs=4,
                subsample=1.0,
                colsample_bytree=0.7,
                random_state=1302)
xgb_params = xgb_model.get_xgb_params()

xgb_model.fit(train_math, y_train_math, verbose = True)

train_math.head()


# In[ ]:


test_math.head()


# Now, we have a model. Now, we use that to predict. 

# In[ ]:


preds = xgb_model.predict(test_math)
from sklearn.metrics import mean_squared_error

print("RMSE: ",np.sqrt(mean_squared_error(y_test_math,preds)))


# Quite a nice RMSE. Let's see if we can reduce this by performing a randomized search. Randomized searches are quicker and thus more convenient, and usually performs more or less as good as exhaustive grid searches. 

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform, randint


param_dist = {
    'colsample_bytree':uniform(0.1,0.9),
    'gamma':reciprocal(1e-5,1),
    'min_child_weight':[1,3],
    'learning_rate':reciprocal(1e-4,1),
    'max_depth':randint(2,6),
    'n_estimators':randint(100,1000),
    'reg_alpha':[1e-5, 0.1],
    'reg_lambda':[1e-5, 0.1],
    'subsample':[0.8]
}

rand_search = RandomizedSearchCV(estimator = xgb_model, param_distributions = param_dist, n_iter = 3, n_jobs=3, iid=False,verbose=True, scoring = 'neg_mean_squared_error', random_state = 123)
print("Fitting model...")
rand_search.fit(train_math, y_train_math)
print("Model fitted")
print("Best score: ")
print(rand_search.best_score_)
print("Best model: ")
print(rand_search.best_params_)


# Now use the best model to fit, and check how much the MSE has been reduced. 

# In[ ]:


best_model = XGBRegressor(**rand_search.best_params_)
best_model.fit(train_math, y_train_math)

y_preds = best_model.predict(test_math)
print("RMSE: ",np.sqrt(mean_squared_error(y_test_math,y_preds)))


# Nice. Now, let's try randomforest. For speed, let's just do a quick Randomized search. 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor()

from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    "max_depth":randint(3,5),
    "max_features":randint(3,5),
    "bootstrap":[True,False],
    "min_samples_split":randint(2,7),
    'n_estimators':randint(10,500)
}

rand_search = RandomizedSearchCV(rf_model, param_distributions = param_dist, n_jobs=5, cv = 5, verbose = True, n_iter=90, random_state = 123)
rand_search.fit(train_math, y_train_math)


# In[ ]:


print(rand_search.best_score_)
print(rand_search.best_params_)


# Now use the best model!

# In[ ]:


best_rf_model = RandomForestRegressor(**rand_search.best_params_)
best_rf_model.fit(train_math,y_train_math)

rf_preds = best_rf_model.predict(test_math)
print("MSE:", mean_squared_error(y_test_math,y_preds))


# Not very good either. Better can be achieved. 
# 
# ## Adaboost with decision stumps
# 
# Let's try decision stumps, and see if they work well. 

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
param_dist = {'criterion':['mse'], 
              'max_depth':[1], 
              'max_features':[None],
           'max_leaf_nodes':[None], 
              'min_impurity_decrease':uniform(0,1),
           'min_impurity_split':[None], 
              'min_samples_leaf':uniform(1e-9,(0.5-1e-9)),
           'min_samples_split':randint(2,5), 
              'min_weight_fraction_leaf':reciprocal(1e-7,0.5),
           'presort':[False], 
              'random_state':[123], 
              'splitter':['best']
             }

rand_search = RandomizedSearchCV(DecisionTreeRegressor(), param_distributions = param_dist, n_jobs=5, cv = 5, verbose = True, n_iter=90)
rand_search.fit(train_math, y_train_math)

adaboost_model = AdaBoostRegressor(DecisionTreeRegressor(**rand_search.best_params_), n_estimators=50).fit(train_math, y_train_math)

print(adaboost_model)
y_preds = adaboost_model.predict(test_math)
print("MSE:", mean_squared_error(y_test_math, y_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test_math, y_preds)))


# Okay, these decision stumps did not work that well. Let's try something else.  

# ## Gaussian process regression
# 
# How about Gaussian Process regression? We have very few dimensions, which definitely makes it suitable. We just want to scale the data of course, to ensure a mean of 0 on all variables which is needed for Gaussian Processes. Of course, it probably won't be a mean of 0 as it is unbalanced between -1 and 1, but let's try. 

# In[ ]:


data.head()


# In[ ]:


from sklearn.preprocessing import scale
math_score_mean = np.mean(data['math score'])
math_score_sd = np.std(data['math score'])
data.iloc[:,0:3] = data.iloc[:,0:3].replace({1:1, 0:-1})
data.iloc[:,3:6] = data.iloc[:,3:6].apply(lambda x: (x - np.mean(x)) / np.std(x), axis=0)
data.iloc[:,6:17] = data.iloc[:,6:17].replace({1:1, 0:-1})


train, test = train_test_split(data, random_state = 123)
data.head()

train_math_gp = train.drop(['math score','reading score','writing score'], axis = 1)
y_train_math_gp = train['math score']

test_math_gp = test.drop(['math score','reading score', 'writing score'], axis = 1)
y_test_math_gp = test['math score']


# In[ ]:


print(test_math_gp.shape)
test_math_gp.head()


# In[ ]:


print(train_math_gp.shape)
train_math_gp.head()


# In[ ]:


from sklearn.gaussian_process import GaussianProcessRegressor

gp_model = GaussianProcessRegressor(n_restarts_optimizer = 50)
gp_model.fit(train_math_gp, y_train_math_gp)
y_preds = gp_model.predict(test_math_gp)
print("RMSE: ", np.sqrt(mean_squared_error(y_test_math_gp, y_preds)))
print("MSE: ", mean_squared_error(y_test_math_gp, y_preds))

from sklearn.model_selection import cross_val_score
print(cross_val_score(gp_model,cv=5,X = data.drop(['math score','reading score','writing score'], axis = 1), y = data['math score'], scoring = 'neg_mean_squared_error'))

# Rescale it. 

MSE = mean_squared_error(y_test_math_gp, y_preds)*math_score_sd + math_score_mean

print("MSE", MSE)


# Okay, this was not that nice to be honest - not a particularly good score. It seems like XGBoost was the best in this case, achieving quite a low MSE.
