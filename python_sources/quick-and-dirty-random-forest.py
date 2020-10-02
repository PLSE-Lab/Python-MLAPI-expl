#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


train_data = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')
test_data = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')


# In[ ]:


y = train_data['Survived'] # Setting target
train_data.drop(['Survived'], axis=1, inplace=True) # Dropping target from features data


# In[ ]:


numerical_cols = [col for col in train_data.columns
                    if train_data[col].dtype in ['int64', 'float64']]
print(numerical_cols)


# In[ ]:


categorical_cols = [col for col in train_data.columns
                        if train_data[col].dtype == 'object'
                        and train_data[col].nunique() < 10]
print(categorical_cols)


# In[ ]:


# Selecting columns
my_cols = numerical_cols + categorical_cols
X = train_data[my_cols]
X_test = test_data[my_cols]


# ### Preprocessors

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


# In[ ]:


num_transformer = SimpleImputer()


# In[ ]:


cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# In[ ]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', cat_transformer, categorical_cols)
    ])


# ### Model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


model = RandomForestClassifier(n_estimators=50, max_features=6, n_jobs=4)


# ### Pipeline

# In[ ]:


my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])


# ### Cross Validation

# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


cv_scores = cross_val_score(my_pipeline, X, y, cv=5, scoring='accuracy', n_jobs=4)


# In[ ]:


print('Mean Accuracy: {:.3f}'.format(cv_scores.mean()))


# ### Training Model

# In[ ]:


my_pipeline.fit(X, y)


# ### Test Predictions

# In[ ]:


test_preds = my_pipeline.predict(X_test)


# In[ ]:


output = pd.DataFrame({
    'PassengerId': X_test.index,
    'Survived': test_preds
})


# In[ ]:


output.to_csv('submission.csv', index=False)

