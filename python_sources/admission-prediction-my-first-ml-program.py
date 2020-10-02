#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

Admission = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')

Admission.info()


# In[ ]:


Admission.describe()


# In[ ]:


Admission.head()


# In[ ]:


import matplotlib.pyplot as plt
Admission.hist(bins=50, figsize=(20,15))


# In[ ]:


num_vars = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'CGPA']
cat_vars = ['Research']
label_var = 'Chance of Admit '


# In[ ]:


admission_set = Admission.drop([label_var] + ['Serial No.'], axis=1)
label_set = Admission[label_var].copy()


# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(admission_set, label_set, test_size = 0.2, random_state = 42)
                                           
print('train size: {}'.format(x_train.count()))
print('test size: {}'.format(x_test.count()))


# In[ ]:


corr_matrix = Admission.corr()
print(corr_matrix)


# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_vars)),
    ('min_max_scaler', MinMaxScaler()),
])

from sklearn.preprocessing import OneHotEncoder

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_vars)),
    ('one_hot_encoder', OneHotEncoder()),
])
            


# In[ ]:


from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)
])

x_train_prepared = full_pipeline.fit_transform(x_train)
x_train_prepared


# In[ ]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x_train_prepared, y_train)


# In[ ]:


from sklearn.metrics import mean_squared_error
import numpy as np

x_test_prepared = full_pipeline.fit_transform(x_test)

prediction_train = lin_reg.predict(x_train_prepared)
prediction = lin_reg.predict(x_test_prepared)
lin_mse = mean_squared_error(prediction, y_test)
lin_mse_train = mean_squared_error(prediction_train, y_train)
lin_rmse = np.sqrt(lin_mse)
lin_rmse_train = np.sqrt(lin_mse_train)
print('on training set: {}'.format(lin_rmse_train))
print('on test set: {}'.format(lin_rmse))


# In[ ]:


def testNewAlgo(algo):
    algo.fit(x_train_prepared, y_train)
    prediction_train = algo.predict(x_train_prepared)
    prediction = algo.predict(x_test_prepared)
    lin_mse = mean_squared_error(prediction, y_test)
    lin_mse_train = mean_squared_error(prediction_train, y_train)
    lin_rmse = np.sqrt(lin_mse)
    lin_rmse_train = np.sqrt(lin_mse_train)
    print('on training set: {}'.format(lin_rmse_train))
    print('on test set: {}'.format(lin_rmse))
    
from sklearn.neighbors import KNeighborsRegressor
Knr = KNeighborsRegressor()
print('KneighborsRegressor')
testNewAlgo(Knr)

from sklearn.ensemble import RandomForestRegressor
print('RandomForestRegressor')
Rfr = RandomForestRegressor()

testNewAlgo(Rfr)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
import scipy as sp

param_distributions={'n_estimators': sp.stats.randint(3, 30),
                     'max_features': sp.stats.randint(2, 7),
                     #'bootstrap': [True, False],
}

search = RandomizedSearchCV(Rfr, param_distributions, cv = 10, n_iter=40, scoring = 'neg_mean_squared_error')

search.fit(x_train_prepared, y_train)


# In[ ]:


print(search.best_estimator_)
print(search.best_score_) # should be ~ -0.004
print(search.best_params_)


# In[ ]:


prediction_final = search.best_estimator_.predict(x_test_prepared)
lin_mse_final = mean_squared_error(prediction_final, y_test)
lin_rmse_final = np.sqrt(lin_mse_final)
print('The final score on the test set is: {}'.format(lin_rmse_final))

