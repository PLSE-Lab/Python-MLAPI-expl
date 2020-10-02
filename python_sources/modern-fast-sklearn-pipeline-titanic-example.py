#!/usr/bin/env python
# coding: utf-8

# Possibly not the best modeling, but fast & modern sklearn approach using pipeline for both preprocessing and model development.  I struggled to find easy ways to extract column features after transforming, if you were struggling to hope this clarifies.  

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer
from sklearn.compose import ColumnTransformer, make_column_selector as selector 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix, roc_auc_score, roc_curve


# In[ ]:


train_data = pd.read_csv("../input/titanic/train.csv")
test_data = pd.read_csv("../input/titanic/test.csv")


# In[ ]:


train_data.sample(10)


# In[ ]:


features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked', 'Name']

X = train_data[features]
y = train_data.Survived
X_test = test_data[features]


# In[ ]:


#hard coding specific data cleaning for title names, not ideal to include in pipeline.  Change for both training and test holdout.

#Title cleaning for training data
for dataset in [X, X_test]:
    dataset.loc[:,'Title'] = dataset.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
    dataset.loc[~dataset.Title.isin(['Mr', 'Miss', 'Mrs', 'Master']), 'Title'] = 'Other'
    dataset.loc[:,'Title'] = dataset.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
    dataset.loc[~dataset.Title.isin(['Mr', 'Miss', 'Mrs', 'Master']), 'Title'] = 'Other'


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for dataset in [X, X_test]:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

X.drop(['Name'], axis=1, inplace = True)
X_test.drop(['Name'], axis=1, inplace = True)


# Important step to flip a few features to categories for pipeline

# In[ ]:


cols = ['Sex', 'Embarked']
X.loc[:,cols] = X.loc[:,cols].astype('category')
X_test.loc[:,cols] = X.loc[:,cols].astype('category')
X.info()


# Create pipleine and train data.  The iterative imputer is based on MICE and takes into account other features when imputing , specifically for 'Age'

# In[ ]:


numeric_transformer = Pipeline(steps=[
    ('imp_num', IterativeImputer(max_iter=10, min_value=0)),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imp_cat', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first'))])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, selector(dtype_exclude="category")),
    ('cat', categorical_transformer, selector(dtype_include="category"))])


classifier = RandomForestClassifier(
                      max_depth=5,
                      n_estimators=500,
                      max_features='auto')

class_model = Pipeline([('preprocess', preprocessor),
                        ('classifier', classifier)])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

class_model.fit(X_train, y_train)

print("model score: %.3f" % class_model.score(X_val, y_val))


# In[ ]:


#cross validate the test
y_train_score = cross_val_score(class_model, X_val, y_val, cv=10)
print(y_train_score.mean())
print(y_train_score.std())


# Even though not that useful for this data set, I found it interesting to be able to inverse tranform the entire pipeline to a data frame to see output of the prepocessing step.  You can use the below to trace back to the pipleine steps above.

# In[ ]:


X_values = preprocessor.fit_transform(X_train)
onehot_columns = class_model.named_steps['preprocess'].named_transformers_['cat'].named_steps['onehot'].get_feature_names().tolist()
numerical_columns = X.columns[X.dtypes != 'category'].tolist()

df_pipeline = pd.DataFrame(X_values, columns = numerical_columns + list(onehot_columns) )
df_pipeline.head()


# Here again, using the transformed pipeline to extract feature importance, which is NOT STRAIGHTFORWARD!

# In[ ]:


#feature importance , extract from pipeline
import numpy as np
feature_importance = pd.Series(data= class_model.named_steps['classifier'].feature_importances_, index = np.array(numerical_columns + list(onehot_columns)))
feature_importance.sort_values(ascending=False)


# In[ ]:


# verify best hyper parmeters for the random forest
param_grid = {
    'classifier__max_depth': [3, 5, 8],
    'classifier__n_estimators': [100, 500],
    'classifier__max_features': [2, 7, 'auto']
     }

grid_search = GridSearchCV(class_model, param_grid, cv=8, verbose=0)
grid_search.fit(X_train, y_train)

print(("best random foreest classification from grid search: %.3f"
       % grid_search.score(X_val, y_val)))


# In[ ]:


grid_search.best_params_


# Below start to use the `yellowbrick` package for ML viz , very easy and effective.

# In[ ]:


from yellowbrick.classifier import ConfusionMatrix, ROCAUC

mapping = {0: "died", 1: "survived"}
fig, ax = plt.subplots(figsize=(6, 6))
cm_viz = ConfusionMatrix(
    class_model,
    classes=["died", "survived"],
    label_encoder=mapping,
)
cm_viz.score(X_val, y_val)
cm_viz.poof()


# In[ ]:


fig, ax = plt.subplots(figsize=(8, 6))
roc_viz = ROCAUC(class_model)
roc_viz.score(X_val, y_val)
roc_viz.poof()


# In[ ]:


from yellowbrick.model_selection import LearningCurve
import numpy as np

fig, ax = plt.subplots(figsize=(8, 6))
cv = StratifiedKFold(12)
sizes = np.linspace(0.3, 1.0, 10)
lc_viz = LearningCurve(
    class_model,
    cv=cv,
    train_sizes=sizes,
    scoring="f1_weighted",
    n_jobs=4,
    ax=ax,
)
lc_viz.fit(X, y)
lc_viz.poof()


# In[ ]:


predictions = class_model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output = output.astype(int)
output.to_csv('../output/my_submissionKAGGLE.csv', index=False)

