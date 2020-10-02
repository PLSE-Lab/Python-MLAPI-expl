#!/usr/bin/env python
# coding: utf-8

# Visualizations created during the development of the model herein can be found here: https://www.kaggle.com/db102291/titanic-competition-visualization-w-seaborn

# In[ ]:


from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
import category_encoders as ce
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 


# In[ ]:


#Import training and testing data
train_data = pd.read_csv("../input/titanic/train.csv")
test_data = pd.read_csv("../input/titanic/test.csv")


# In[ ]:


#Which columns have missing values?
display(train_data.isnull().sum().sort_values(ascending=False))
display(test_data.isnull().sum().sort_values(ascending=False))


# In[ ]:


#Create new Cabin variable
train_data['Cabin_new'] = train_data['Cabin'].str[0]
test_data['Cabin_new'] = train_data['Cabin'].str[0]

#Create title variable
train_data['Title']=train_data.Name.str.extract('([A-Za-z]+)\.')
test_data['Title']=test_data.Name.str.extract('([A-Za-z]+)\.')

#Create Fam_size variable
train_data['Fam_size'] = train_data['SibSp'] + train_data['Parch']
test_data['Fam_size'] = test_data['SibSp'] + test_data['Parch']


# In[ ]:


train_data


# In[ ]:


#Preprocessing numerical data
#Fare
train_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)
test_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)

#Age
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(train_data['Age'].mean(), inplace=True)


# In[ ]:


features = ["Pclass", "Age", "Fam_size", "Fare", "Sex", "Embarked"]
cat_cols = ['Sex', 'Embarked', "Pclass"]
num_cols = ['Age', 'Fam_size', 'Fare']

y = train_data["Survived"]
X = train_data[features]
X_test = test_data[features]


# In[ ]:


#Preprocessing for categorical data
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

#Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(transformers=[('cat', cat_transformer, cat_cols)])


# In[ ]:


#Model
model = XGBClassifier(random_state=126)

#Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocess', preprocessor), 
                              ('model', model)])

param_grid = {
    'model__n_estimators': [20, 40, 60],
    'model__learning_rate': [0.05, 0.07, 0.08],
    'model__max_depth': [4, 6, 8]}


search = GridSearchCV(my_pipeline, param_grid, n_jobs=-1, verbose=10, cv=10)
search.fit(X, y)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)


# In[ ]:


predictions = search.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

