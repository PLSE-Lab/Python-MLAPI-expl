#!/usr/bin/env python
# coding: utf-8

# I was able to get a score of `0.79198` on the public leaderboards using an `extraTrees` classifier in R and wanted to see if I could do better with SciKit. My first attempt at using `ExtraTreesClassifier` yielded a score of `0.75771` on the public leaderboards.

# In[ ]:


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier as Classifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


# Define target column as well as the ones we want to drop. All other columns will be preserved in the pipeline. Categorical columns are already one-hot encoded (one column per category) so we don't need to worry about transforming any of the columns.

# In[ ]:


target = 'Cover_Type'
cols_to_drop = ["Soil_Type7", "Soil_Type15"]


# Set up a preprocessor that will do data transformations on data sources before using it in a model. For example, each time data is split during cross validation or before the test data is used for final predictions.
# 
# For now, it'll only be used to remove unwanted columns. Later, it can be used for feature engineering.

# In[ ]:


preprocessor = ColumnTransformer(
    remainder='passthrough',                  # keep all columns
    transformers=[
        ('drop', 'drop', cols_to_drop),       # except these
        # Could possibly use `FunctionTransformer` (as many as needed ) for feature engineering
    ])


# # Tuning
# ## The Pipeline
# 
# First, set up a pipeline that will preprocess the data set and then feed it into the model. We'll set up a moel to use all available cores and some `random_state` for reproducible output. The rest of the parameters will be fed in via `GridSearchCV`

# In[ ]:


model = Classifier(n_jobs=-1, random_state=0)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])


# ## Load Training Data
# The `Id` column will be identified as the `index_col`. As such, it won't be used for training the model. Also, the data won't be split into train/validation sets as the `GridSearchCV` will do this (several times, stratified) for cross validation.

# In[ ]:


train = pd.read_csv('../input/learn-together/train.csv', index_col='Id')  # 15120 records
X, y = train.drop([target], axis=1), train[target]


# ## GridSearchCV

# `GridSearchCV` uses "brute force" to exhaustively try all combinations of configured parameters. It can also do cross validation, so I'll set up a cross validator.
# 
# By default, `GridSearchCV` would use a `StratifiedKFold` CV anyway, since the target column is multi-class (contains more than two discrete values, is not a sequence of sequences, and is 1d or a column vector) but I want to be able to control its `random_state` which by default is `None`. I'll use `3` splits for now since the training data is not that large (15120 observtions). `5` might also be reasonable.

# In[ ]:


cv = StratifiedKFold(n_splits=3, random_state=0)
param_grid = {
    "model__random_state": [0],   # [0, 1, 2, 3, 4],
    "model__n_estimators": [360], # [320, 340, 360, 380, 400],
    "model__max_depth": [32]      # [25, 30, 32, 34, 38, 45]
}
searchCV = GridSearchCV(estimator=pipeline, scoring='accuracy', cv=cv, param_grid=param_grid, verbose=True)

# WARNING: This could take some time to run.
searchCV.fit(X, y)

print('Best index:', searchCV.best_index_)
print('Best score:', searchCV.best_score_)
print('Best params:', searchCV.best_params_)


# # Predict
# 
# The model has been automatically refitted with the best parameters from the grid search

# In[ ]:


X_test = pd.read_csv('../input/learn-together/test.csv', index_col='Id')  # 565892 records
test_preds = searchCV.predict(X_test)
output = pd.DataFrame({'Id': X_test.index, 'Cover_Type': test_preds})
output.to_csv('submission.csv', index=False)

