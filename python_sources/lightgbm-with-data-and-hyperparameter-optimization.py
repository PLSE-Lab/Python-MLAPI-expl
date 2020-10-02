#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline


# # prepare data

# In[ ]:


town_data = pd.read_csv('/kaggle/input/town-data/town_data.csv', header=[1])
town_data = town_data.replace('*', np.nan)
town_data = town_data.replace('X', np.nan)


# In[ ]:


train_data = pd.read_csv('/kaggle/input/machine-learning-lab-cas-data-science-fs-20/houses_train.csv', index_col=0)


# In[ ]:


train_data = train_data.join(town_data.set_index('Gemeindename'), on='municipality_name')


# In[ ]:


train_data = train_data.fillna(-1)


# In[ ]:


X_data = train_data.drop(columns='price')
y_data = train_data['price']


# In[ ]:


X_train, X_dev, y_train, y_dev = train_test_split(X_data, y_data, stratify=X_data['object_type_name'], test_size=0.1)


# # define and train model

# In[ ]:


pipeline = Pipeline([
    ('pre', make_column_transformer((OneHotEncoder(handle_unknown='ignore'), ['zipcode', 'municipality_name', 'object_type_name']), remainder='passthrough')),
    ('clf', LGBMRegressor(
        num_leaves=50,
        #max_depth=5,
        learning_rate=0.05,
        n_estimators=200
    ))
])


# In[ ]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# In[ ]:


parameters = {
    'clf__num_leaves': [10,20,30,40,50,60,70,80,90,100], 
    'clf__max_depth': [3,4,5,6,7,8,9,10],
    'clf__n_estimators': [100,150,200,500] 
}
clf = RandomizedSearchCV(pipeline, parameters, cv=5, n_iter=40, verbose=1, n_jobs=-1)


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


est = clf.best_estimator_


# # Predict and evaluate prices for dev set

# In[ ]:


y_dev_pred = est.predict(X_dev)


# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[ ]:


mean_absolute_percentage_error(y_dev, y_dev_pred)


# # Train with all data

# In[ ]:


est.fit(X_data, y_data)


# # Predict prices for test set

# In[ ]:


X_test = pd.read_csv('/kaggle/input/machine-learning-lab-cas-data-science-fs-20/houses_test.csv', index_col=0)


# In[ ]:


X_test = X_test.join(town_data.set_index('Gemeindename'), on='municipality_name')
X_test = X_test.fillna(-1)


# In[ ]:


y_test_pred = est.predict(X_test)


# In[ ]:


X_test_submission = pd.DataFrame(index=X_test.index)


# In[ ]:


X_test_submission['price'] = y_test_pred


# In[ ]:


X_test_submission.to_csv('lightgbm_submission_all.csv', header=True, index_label='id')


# In[ ]:




