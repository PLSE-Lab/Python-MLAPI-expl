#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


# Read the data and seperate the predictor variables from the predicted
data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
X = data.drop(['SalePrice'], axis=1)
y = data.SalePrice
X_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


X.head()


# ### Missing Values
# Drop columns that have more than 10 percent missing values. Impute the other columns with median strategy

# In[ ]:


total = X.isnull().sum().sort_values(ascending = False)
percent = (X.isnull().sum()/X.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total, percent], axis =1, keys=['Total', 'Percentage'])
missing_data[(missing_data.Percentage > 0)]


# In[ ]:


X = X.drop(missing_data[missing_data.Percentage > 0.1].index, axis = 1)


# In[ ]:


X_test = X_test.drop(missing_data[missing_data.Percentage > 0.1].index, axis = 1)


# In[ ]:


X = X.drop(['SaleType','SaleCondition','Id','MoSold', 'YrSold'], axis = 1)


# In[ ]:


X_test = X_test.drop(['SaleType','SaleCondition','Id','MoSold', 'YrSold'], axis = 1)


# In[ ]:


numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
categorical_cols = [cname for cname in X.columns if X[cname].nunique() < 10 and 
                    X[cname].dtype == "object"]
my_cols = categorical_cols + numerical_cols
print(categorical_cols,numerical_cols)


# ### Preprocess
# Imputation of missing values & encoding of catergorical variables

# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median')
# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# In[ ]:


X = X[numerical_cols]
X_test = X_test[numerical_cols]


# In[ ]:


X = numerical_transformer.fit_transform(X)
X_test = numerical_transformer.fit_transform(X_test)


# In[ ]:


print('X shape: ' + str(X.shape))
print('X_test shape:  ' + str(X_test.shape))


# In[ ]:


from sklearn.feature_selection import SelectKBest, chi2


# In[ ]:


Classifier = SelectKBest(chi2, k=10)
Classifier.fit(X, y)


# ### Find out if you only need to transform X_ext and how this goes to X_test

# In[ ]:


mask = Classifier.get_support(indices = True)
X_extracted = X[:,mask]


# In[ ]:


X_extracted.shape


# In[ ]:


X_extracted_test = X_test[:,mask]


# ### Model
# Defining of a model.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print('Random grid created')
print(random_grid)


# In[ ]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
model = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
model_hyper = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=10, random_state=42, 
                                 n_jobs = -1)


# In[ ]:


model_hyper.fit(X_extracted ,y)


# In[ ]:


PARAMS = model_hyper.best_params_


# In[ ]:


model = RandomForestRegressor(**PARAMS)


# In[ ]:


model.fit(X_extracted, y)


# In[ ]:


predictions = model.predict(X_extracted_test)


# ### Creating of the pipeline
# Inserting the preprocessor, the model and assessing the quality using cross validation.

# In[ ]:


pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])


# In[ ]:


X_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


output = pd.DataFrame({'Id': X_test.Id,
                       'SalePrice': predictions})
output.to_csv('submission.csv', index=False)

