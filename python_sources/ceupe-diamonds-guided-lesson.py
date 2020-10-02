#!/usr/bin/env python
# coding: utf-8

# ## 0. python imports & setup

# for learning purposes, libraries will be imported inside its corresponding usage section...

# ## 1. data loading

# In[ ]:


import pandas as pd


# * diamonds: labeled data we can use for training and testing
# * diamonds_predict: diamonds to predict its price and upload result to Kaggle

# In[ ]:


diamonds = pd.read_csv('../input/ceupe-big-data-analytics/diamonds_train.csv')
diamonds_predict = pd.read_csv('../input/ceupe-big-data-analytics/diamonds_test.csv')


# In[ ]:


diamonds.head().T


# as you can see, there are both categorical and numerical columns...

# ## 2. eda

# this section is up to you! this guided lesson is about a machine learning pipeline...

# ## 3. ml preprocessing

# in this section I will teach how to use scikit-learn's Pipiline and ColumnTransformer, one of the best practices for composing preprocessing and modeling in a single and elegand class... pay attention as it is hard to understand...

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# * https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
# * https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer
# * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

# let's identify numerical and categorical features...

# In[ ]:


NUM_FEATS = ['carat', 'depth', 'table', 'x', 'y', 'z']
CAT_FEATS = ['cut', 'color', 'clarity']
FEATS = NUM_FEATS + CAT_FEATS
TARGET = 'price'


# let's define a preprocessing transformer for numerical columns...

# In[ ]:


numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), 
                ('scaler', StandardScaler())])


# let's define a preprocessing transformer for categorical columns...

# In[ ]:


categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])


# let's join these transformers using a `ColumnTransformer`:

# In[ ]:


preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, NUM_FEATS),
                                ('cat', categorical_transformer, CAT_FEATS)])


# inspecting the full preprocessor:

# In[ ]:


preprocessor


# how does this preprocessing looks like?

# at least in this case, it is at the cost of interpretability of transformed DataFrame...

# In[ ]:


pd.DataFrame(data=preprocessor.fit_transform(diamonds)).head()


# ## 4. train a simple model

# first, lets train a simple model using holdout, train - test split...

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


diamonds_train, diamonds_test = train_test_split(diamonds)


# In[ ]:


print(diamonds_train.shape)
print(diamonds_test.shape)


# let's choose a model from scikit-learn cheatsheet: https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

model= Pipeline(steps=[('preprocessor', preprocessor),
                       ('regressor', RandomForestRegressor())])


# In[ ]:


model.fit(diamonds_train[FEATS], diamonds_train[TARGET]);


# ## 5. check model performance on test and train data

# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


y_test = model.predict(diamonds_test[FEATS])
y_train = model.predict(diamonds_train[FEATS])


# In[ ]:


print(f"test error: {mean_squared_error(y_pred=y_test, y_true=diamonds_test[TARGET], squared=False)}")
print(f"train error: {mean_squared_error(y_pred=y_train, y_true=diamonds_train[TARGET], squared=False)}")


# ## 6. check model performance using cross validation

# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


scores = cross_val_score(model, 
                         diamonds[FEATS], 
                         diamonds[TARGET], 
                         scoring='neg_root_mean_squared_error', 
                         cv=5, n_jobs=-1)


# In[ ]:


import numpy as np
np.mean(-scores)


# ## 7. optimize model using grid search

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'regressor__n_estimators': [16, 32, 64, 128, 256, 512],
    'regressor__max_depth': [2, 4, 8, 16],
}

grid_search = RandomizedSearchCV(model, 
                                 param_grid, 
                                 cv=5, 
                                 verbose=10, 
                                 scoring='neg_root_mean_squared_error', 
                                 n_jobs=-1,
                                 n_iter=32)

grid_search.fit(diamonds[FEATS], diamonds[TARGET])


# In[ ]:


grid_search.best_params_


# In[ ]:


grid_search.best_score_


# ## 8. prepare submission

# In[ ]:


y_pred = grid_search.predict(diamonds_predict[FEATS])


# In[ ]:


submission_df = pd.DataFrame({'id': diamonds_predict['id'], 'price': y_pred})


# In[ ]:


submission_df.head()


# In[ ]:


submission_df.describe()


# In[ ]:


submission_df.price.clip(0, 20000, inplace=True)


# In[ ]:


submission_df.to_csv('diamonds_rf.csv', index=False)


# ## 9. let's try more models...
