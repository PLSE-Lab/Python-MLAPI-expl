#!/usr/bin/env python
# coding: utf-8

# ## What is Rose's PassengerId?

# ### Description

# In[ ]:


#In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive.
#In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.


# ### Getting the Data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")


# ### Understanding the Data

# In[ ]:


train_data.head()


# In[ ]:


train_data.info()


# ### Gain Insights

# #### Viz

# In[ ]:


train_data.describe()


# In[ ]:


train_data.drop(columns=["PassengerId"]).hist(bins=40, figsize=(15,12))


# In[ ]:


train_data.Embarked.value_counts()


# #### Correlations

# In[ ]:


corr_matrix = train_data.corr()
corr_matrix["Survived"].sort_values(ascending=False)


# ### Preparing the Data / Pipeline

# #### Data Cleaning [ Selector/Imputer ]

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin

# A class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]


# In[ ]:


from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy="median")


# In[ ]:


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X], index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


# #### Text and Categories attributes handling [ Encoding ]

# In[ ]:


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)


# #### Feature Scaling [ Normalization/Standarization ]

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


# #### Pipeline

# In[ ]:


from sklearn.pipeline import Pipeline

num_attribs = ["Age", "SibSp", "Parch", "Fare"]

num_pipeline = Pipeline([
    ('select_num', DataFrameSelector(num_attribs)),
    ('imputer', num_imputer),
    ('scaler', scaler)
])


# In[ ]:


cat_attribs = ["Pclass", "Sex", "Embarked"]

cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(cat_attribs)),
        ("imputer", MostFrequentImputer()),
        ('encoder', encoder)
    ])


# In[ ]:


from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)
])


# ### Selecting & Training the Model

# In[ ]:


X_train = full_pipeline.fit_transform(train_data)
y_train = train_data['Survived']


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(random_state=11)
rf_clf.fit(X_train, y_train)
cross_val_score(rf_clf, X_train, y_train, cv=10).mean()


# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = { 
    'n_estimators': [100, 200, 300, 400],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [2, 4, 6, 8],
    'criterion' :['gini', 'entropy']
}

grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)


# In[ ]:


grid_search.best_params_


# In[ ]:


grid_search.best_score_


# In[ ]:


from sklearn.metrics import accuracy_score

X_test = full_pipeline.fit_transform(test_data)
PassengerId = test_data['PassengerId']

y_pred = grid_search.predict(X_test)
submission = pd.DataFrame({'PassengerId' : PassengerId,
                          'Survived' : y_pred})
submission.head()

