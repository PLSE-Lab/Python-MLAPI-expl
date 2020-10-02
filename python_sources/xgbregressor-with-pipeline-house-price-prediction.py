#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing the training data
X_full=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')


# In[ ]:


#selecting the target,in this case the SalePrice of the house
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)


# In[ ]:


#import the tools from sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
train_X_full,val_X_full,train_y,val_y=train_test_split(X_full, y, random_state=1000,test_size=0.25)


# In[ ]:


#choosing only categorical variables with a cardinality(the number of unique values) less than 10
categorical_cols = [cname for cname in train_X_full.columns if
                    train_X_full[cname].nunique() < 10 and 
                    train_X_full[cname].dtype == "object"]


# In[ ]:


#selecting all the numerical data
numerical_cols = [cname for cname in train_X_full.columns if 
                train_X_full[cname].dtype in ['int64', 'float64']]


# In[ ]:


# selecting the features that I want and using that data for the model.
my_cols = categorical_cols + numerical_cols
X_train = train_X_full[my_cols].copy()
X_valid = val_X_full[my_cols].copy()
X=X_full[my_cols].copy()


# In[ ]:


#importing all the libraries for preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error


# In[ ]:


#creating a pipeline for the purpose of preprocessing of categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# In[ ]:


# Bundling the precprocessing of both numerical and categorical data.
num_transformer = SimpleImputer(strategy='constant')
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# In[ ]:


from xgboost import XGBRegressor
numerical_transformer =SimpleImputer(strategy='constant') 

# Preprocessing for categorical variables
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

# Define the  model(here I'm using XGBoost)
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.0095, n_jobs=4)


# In[ ]:


#creating a pipeline with the XGBoost model
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model',my_model)
                             ])

# since I'm using a pipline the preprocessing and fitting the model is one line 
my_pipeline.fit(X_train,train_y)

#Same thing applies for validation data, getting predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(val_y, preds)
print('MAE:', score)


# In[ ]:


#for a better indicator of model performance let's use cross validation 
from sklearn.model_selection import cross_val_score
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')
print("Average MAE score (across all the experiments):")
print(scores.mean())


# In[ ]:


# now for for the test data
X_test_full = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


#using the same features for the training data
X_test = X_test_full[my_cols].copy()


# In[ ]:


#get the predictions
preds_test=my_pipeline.predict(X_test)


# In[ ]:



output = pd.DataFrame({'Id': X_test.Id,
                       'SalePrice': preds_test})

