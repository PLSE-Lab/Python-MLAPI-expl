#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Prediction

# This is my second version of Titanic prediction.

# **Libraries**

# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pandas as pd
import numpy as np


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test  = pd.read_csv('../input/test.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_train.info()


# In[ ]:


df_train.isna().sum()/df_train.shape[0]


# ## Missing Values

# As you can see, we should have 1309 records. Columns with the missing values are **Age**, **Embarked**, **Fare** and **Cabin**.
# 
# Three commonly used approaches for numerical variables:
# 
# 1. Drop columns with Missing Values
# 2. Imputation
# 3. Imputation with additional missing indicator columns
# 
# Three common approaches for catogerical variables:
# 
# 1. Drop Categorical Variables
# 2. Label Encoding
# 3. One-Hot Encoding or Dummies

# In[ ]:


y=df_train.Survived
X=df_train.drop(['PassengerId','Survived'], axis=1)

X_train_full, X_valid_full,y_train,y_valid = train_test_split(X,y, test_size=0.2,random_state=0)


# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns 
                  if X_train_full[cname].dtype in ['int64', 'float64']]


# Select categorical columns
categorical_cols = [cname for cname in X_train_full.columns 
                    if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]


# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = df_test[my_cols].copy()


# In[ ]:


X_train.head()


# # Step 1:Define Preprocessing steps

# In[ ]:


# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy ='median')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

#Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# # Step 2: Define the Model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state = 0, class_weight='balanced')

model_xgb = XGBClassifier(n_estimators = 50, learning_rate=0.01, n_jobs=4)


# # Step 3: Create and Evaluate the Pipeline

# In[ ]:


# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor',preprocessor),
                             ('model', model_xgb)])

my_pipeline.fit(X_train,y_train)

# Prediction
preds = my_pipeline.predict(X_valid)

print (accuracy_score(y_valid, preds))


# # Cross Validation

# In[ ]:


scores = cross_val_score(my_pipeline, X,y, cv=5,scoring='accuracy')

#Random Forest: 0.803
#XGB Classifier:0.818
print ("Average accuracy scores:\n", scores.mean())


# # Prediction

# In[ ]:


preds_test = my_pipeline.predict(X_test)


output= pd.DataFrame({'PassengerId': df_test.PassengerId,
                     'Survived': preds_test})

output.to_csv('titanic.csv', index=False)

