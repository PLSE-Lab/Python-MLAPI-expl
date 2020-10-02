#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/IowaHouses_train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/IowaHouses_test.csv', index_col='Id')


# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)


# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)


# Select categorical columns with relatively low cardinality (convenient but arbitrary for One-Hot encoding)
categorical_cols = [col for col in X_train_full.columns if
                    X_train_full[col].nunique() < 10 and 
                    X_train_full[col].dtype == "object"]

# Select numerical columns
numerical_cols = [col for col in X_train_full.columns if 
                X_train_full[col].dtype in ['int64', 'float64']]



# Keep selected columns only
all_relevant_cols = categorical_cols + numerical_cols
X_train = X_train_full[all_relevant_cols].copy()
X_valid = X_valid_full[all_relevant_cols].copy()
X_test = X_test_full[all_relevant_cols].copy()


# get indexes of columns
categorical_cols_index = [X_train.columns.get_loc(col) for col in categorical_cols]   # Get indices of column names
                                                                                      # Also make sure to use X_train
                                                                                      # instead of X_train_full !!!!
numerical_cols_index = [X_train.columns.get_loc(col) for col in numerical_cols]


# In[ ]:


X_test.shape


# In[ ]:


print(categorical_cols_index)
print(numerical_cols_index)

# As you can see, the categorical columns come first in the data frame,
# followed by the numerical columns


#  

# In[ ]:


# Visually verify to see if index values correspond to associated numerical / categorical columns
X_train.head()


#  
#  

# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


#  

# # Create your data transformations and a data transformation pipeline
# 
# A great resource for more information is given by the following link:
# https://machinelearningmastery.com/columntransformer-for-numerical-and-categorical-data/

#  

# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

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

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Preprocessing of training data, fit model 
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))


# ## vs. my way
#  

#  

# In[ ]:


# Define your model
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)

# Define your imputer for NUMERICAL DATA
num_imputer = SimpleImputer(strategy='constant')     # can also play around with setting strategy to 'median'

# Define your imputer for CATEGORICAL DATA
cat_imputer = SimpleImputer(strategy='most_frequent')   # must use this strategy when working with strings

# Define your encoder for CATEGORICAL DATA
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

# Define your transformation lists
impute_transformation_list = [('impute_num', num_imputer, numerical_cols_index),    # NOTE:  we want the indexes,
                              ('impute_cat', cat_imputer, categorical_cols_index)]  #        NOT the column names
encode_transformation_list = [('encode_cat', OH_encoder, categorical_cols_index)]

# Define your transformations
impute_transformation = ColumnTransformer(transformers= impute_transformation_list)
encode_transformation = ColumnTransformer(transformers= encode_transformation_list)

# Define your pipeline
pipeline = Pipeline(steps= [('impute', impute_transformation), 
                            ('encode', encode_transformation), 
                            ('model', rf_model)])

# Fit your model on the transformed data using the pipeline
pipeline.fit(X_train, y_train)

# Predict using your model
predictions = pipeline.predict(X_valid)

# Evaluate your model using MAE
print('MAE:', mean_absolute_error(y_valid, predictions))


# ## Question:  Why does my code yield *considerably* worse results than the provided Kaggle code when it seems like they do the same thing?
# 
# I'm just a beginner, but my guess would be that the order of the workflow is the underlying factor behind the vast difference in performance because:
# 
# - the provided Kaggle code bundles the categorical column transformation steps into a pipeline so that missing column values are imputed and immediately followed up with One-Hot encoding.
# 
# vs. 
# 
# - my code does the imputation of missing numerical values --> imputation of missing categorical values --> encoding of relevant categorical values.
# 
# 
# HOWEVER, if this is in fact the reason behind the difference in performance, I struggle to see why this difference matters / makes a difference.
# 
# 
# Since I am a beginner and learning all of this from scratch, any help / advice would be greatly appreciated!!!!!
# 
# 
# Thank you so much in advance,
# Ray
# 
# 

# In[ ]:




