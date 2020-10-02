#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline

from xgboost import XGBClassifier
from sklearn.svm import LinearSVC

from sklearn.feature_selection import SelectFromModel


# ## Load data and create dataframes

# In[ ]:


root_df = pd.read_csv('../input/train.csv')
X = root_df.drop(['id', 'target'], axis=1)
y = root_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# ### Data is imbalanced!
# 
# So lets use the 'roc_auc' metric to account for this
# 

# In[ ]:


y.value_counts()


# ## Build a pipeline / classifier

# In[ ]:


pipe = Pipeline(steps=[
    ('pre', None),
    ('feature_selection', None),
    ('clf', LogisticRegression(solver='liblinear')),
    ]
)

params = {  
    'pre': [
        None, StandardScaler(), MinMaxScaler(),
    ],
    'feature_selection': [
        None, 
        SelectFromModel(LogisticRegression(solver='liblinear')),
        SelectFromModel(XGBClassifier(n_estimators=500, max_depth=3)),
    ],
    'clf__penalty': ['l1', 'l2'],
    'clf__C': [000.1, 00.1, 0.1, 1, 10],
    'clf__class_weight': [None, 'balanced']
}
clf = GridSearchCV(pipe, param_grid=params, scoring='roc_auc', cv=8, n_jobs=-1)


# ## Fit the model

# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


clf.best_estimator_


# ## Score the model

# In[ ]:


roc_auc_score(y_test, clf.predict(X_test))


# ## Export Data

# In[ ]:


test_df = pd.read_csv('../input/test.csv')
X = test_df.drop(['id'], axis=1)


# In[ ]:


predictions = clf.predict_proba(X)[:,1]
submission = {
    "id": test_df['id'],
    "target": predictions
}
submission = pd.DataFrame(submission)


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




