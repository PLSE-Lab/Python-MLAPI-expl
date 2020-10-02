#!/usr/bin/env python
# coding: utf-8

# # Initialization

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')


# In[ ]:


data.head()


# In[ ]:


data.info()


# The **Embarked** feature has missing values, but only two, we can drop the two examples without worries.

# In[ ]:


data.dropna(subset=['Embarked'], inplace=True)


# # Data Visualization

# Do some data visualization so that we can do better data preprocessing later.

# In[ ]:


my_data = data.copy()


# In[ ]:


# Extract the first character of each value of the 'Cabin' feature
my_data['Cabin'] = my_data['Cabin'].map(lambda x: x[0] if type(x) is str else 'Unknown')

sns.catplot('Cabin', hue='Survived', data=my_data, kind='count', aspect=1.5)


# In[ ]:


# Combine the categories of the 'Cabin' feature
my_data['Cabin'] = my_data['Cabin'].map(lambda x: x if x == 'Unknown' else 'Known')

sns.catplot('Cabin', hue='Survived', data=my_data, kind='count')


# In[ ]:


# Combine the 'SibSp' and 'Parch' features into a new feature 'Family'
my_data['Family'] = my_data['SibSp'] + my_data['Parch']

sns.catplot('Family', hue='Survived', data=my_data, kind='count', aspect=1.5)


# In[ ]:


# Combine the categories of the 'Family' feature
my_data['Family'] = my_data['Family'].map(lambda x: 'No' if x == 0
                                               else 'Few' if x in [1, 2, 3]
                                               else 'Many')

sns.catplot('Family', hue='Survived', data=my_data, kind='count')


# In[ ]:


plt.figure(figsize=(10, 5))
sns.distplot(my_data['Fare'], label='Fare')
sns.distplot(my_data['Age'], label='Age')
plt.xlabel('')
plt.legend()


# # Data Preprocessing

# Split the data into X and y, then split the X and y into training set and test set.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = data.drop(columns=['Survived'])
y = data['Survived'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# The **Name** and **Ticket** features seem to be difficult to extract infomation. Let's ignore them this time.

# In[ ]:


invalid_feats = ['Name', 'Ticket']


# The **Cabin**, **SibSp** and **Parch** features can do a little feature engineering (*Reference from [Basic Feature Engineering with the Titanic Data](https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/)*).

# In[ ]:


def extract_feats(X):
    "Extract the 'Cabin' and 'Family' features."
    X['Cabin'] = X['Cabin'].map(lambda x: 'Known' if type(x) is str else 'Unknown')
    
    X['Family'] = X['SibSp'] + X['Parch']
    X['Family'] = X['Family'].map(lambda x: 'No' if x == 0
                                       else 'Few' if x in [1, 2, 3]
                                       else 'Many')
    return X.drop(columns=['SibSp', 'Parch'])


# Combine all the estimators into one using **Pipeline** with **ColumnTransformer**.

# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# In[ ]:


column_trans = make_column_transformer(
                    ('drop', invalid_feats),
                    (make_pipeline( # Extract the features and then encode them
                        FunctionTransformer(extract_feats),
                        OneHotEncoder()), ['Cabin', 'SibSp', 'Parch']),
                    (OneHotEncoder(), ['Pclass', 'Sex', 'Embarked']),
                    remainder=StandardScaler()) # Scale the continuous variables


# In[ ]:


preprocessor = make_pipeline(column_trans,
                             IterativeImputer(random_state=0)) # Impute missing values


# Get all the feature names of the preprocessed X.

# In[ ]:


my_X = preprocessor.fit_transform(X)

transfs = preprocessor[0].transformers_

extracted_feats = transfs[1][1][1].get_feature_names(['Cabin', 'Family'])
encoded_feats = transfs[2][1].get_feature_names(['Pclass', 'Sex', 'Embarked'])
remainder_feats = ['Age', 'Fare']


# In[ ]:


all_feats = list(extracted_feats) + list(encoded_feats) + remainder_feats


# In[ ]:


my_X = pd.DataFrame(my_X, X.index, all_feats)
my_X.head()


# # Evaluation Model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


default_model = make_pipeline(preprocessor,
                              RandomForestClassifier(random_state=0))


# **Search** the optimal **hyperparameters**.

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


param_grid = {'randomforestclassifier__n_estimators':[50, 100, 110, 120, 200],
              'randomforestclassifier__max_depth':[5, 10, 15, 50, None]}
GridSearchCV(default_model, param_grid, n_jobs=-1, verbose=1).fit(X_train, y_train).best_params_


# **Compare** the **accuracy** between the default model and the optimized model.

# In[ ]:


default_acc = default_model.fit(X_train, y_train).score(X_test, y_test)

opted_model = make_pipeline(preprocessor,
                            RandomForestClassifier(110, max_depth=10, random_state=0))
opted_acc = opted_model.fit(X_train, y_train).score(X_test, y_test)

print(f'Default accuracy: {default_acc:.2%}')
print(f'Optimized accuracy: {opted_acc:.2%}')


# # Explaining Model

# In[ ]:


opted_model.fit(X, y)
my_preprocessor = opted_model[0]


# In[ ]:


my_X = pd.DataFrame(my_preprocessor.transform(X), X.index, all_feats)
my_model = opted_model[1]


# In[ ]:


import shap
shap.initjs()


# In[ ]:


explainer = shap.TreeExplainer(my_model)
shap_values = explainer.shap_values(my_X)


# In[ ]:


# The index of 'shap_values' is 1 meaning that y > 0 represents a positive result 
shap.summary_plot(shap_values[1], my_X)


# In[ ]:


shap.dependence_plot('Age', shap_values[1], my_X)


# # Predicting New Data

# In[ ]:


model = make_pipeline(preprocessor,
                      RandomForestClassifier(110, max_depth=10, random_state=0))
model = model.fit(X, y)


# In[ ]:


test_data = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')

preds = model.predict(test_data)
output = pd.DataFrame({'PassengerId':test_data.index, 'Survived':preds})


# In[ ]:


output.to_csv('titanic_preds.csv', index=False)


# # Saving and Loading Model

# In[ ]:


from joblib import dump, load


# In[ ]:


titanic_joblib = dump(model, 'titanic.joblib')[0]


# In[ ]:


del model

model = load(titanic_joblib)
model.predict(X)[:10]

