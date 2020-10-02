#!/usr/bin/env python
# coding: utf-8

# ### Import Relevant Libraries

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import random 
random.seed(50)


# In[ ]:


input_file = '../input/titanic3.csv'
data = pd.read_csv(input_file)


# In[ ]:


data['PassengerId'] = data.index+1


# In[ ]:


data.head()


# ### Data Preprocessing 

# In[ ]:


data.describe()


# #### Lets create a list with numeric and categorical features

# In[ ]:


numeric_features = ['age','fare']
categorical_features = ['embarked','sex','pclass']


# In[ ]:


data['embarked'].value_counts()


# #### One Hot Encoding
# 
# Lets create a pipleline to handle missing values. 

# In[ ]:


numeric_pipeline = Pipeline(steps=[('miss', SimpleImputer(strategy='mean')),('scale', StandardScaler())])
category_pipeline = Pipeline(steps=[('miss',SimpleImputer(strategy='constant',fill_value='missing')),('hot',OneHotEncoder(handle_unknown='ignore'))])


# In[ ]:


preprocess = ColumnTransformer(transformers = [
    ('numeric', numeric_pipeline,numeric_features),
    ('categorical', category_pipeline,categorical_features)
])


# ### Model Fitting/Evaluation - Logistic Regression
# 
# Given all features, lets predict the survived variable using logistic regression

# In[ ]:


model_pipeline = Pipeline(steps=[
    ('prep', preprocess),
    ('model',LogisticRegression(solver='lbfgs'))
])


# In[ ]:


x = data.drop('survived', axis=1)
y = data['survived']


# In[ ]:


x_train = x.head(621)
y_train = y.head(621)
x_test = x.tail(418)
y_test = y.tail(418)


# In[ ]:


y_test.shape


# In[ ]:


model_pipeline.fit(x_train,y_train)
model_pipeline.score(x_test,y_test)


# ### Model Fitting/Evaluation - RandomForestClassifier
# 
# Given all features, lets predict the survived variable using a Random Forest Classifier

# In[ ]:


model_pipeline = Pipeline(steps=[
    ('prep', preprocess),
    ('model', RandomForestClassifier())
])


# In[ ]:


model_pipeline.fit(x_train,y_train)
model_pipeline.score(x_test,y_test)


# In[ ]:


df = data.tail(418)
df = df.drop(['pclass','name','sex','age','sibsp','parch','ticket','fare','cabin','embarked','boat','body','home.dest','survived'], axis=1)


# In[ ]:


df['Survived'] = model_pipeline.predict(x_test)
df.tail()


# In[ ]:


df.to_csv('titanic.csv',index=False)

