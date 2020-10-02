#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# DATA_DIR = Path('..', 'data', 'final', 'public')

# DATA_DIR = Path('..', 'data', 'final', 'public')

# In[ ]:


# for training our model
train_values = pd.read_csv('../input/warm-up-machine-learning-with-a-heart/train_values.csv', index_col='patient_id')
train_labels = pd.read_csv('../input/warm-up-machine-learning-with-a-heart/train_labels.csv', index_col='patient_id')


# In[ ]:


train_values.head()


# #reference http://drivendata.co/blog/machine-learning-with-a-heart-benchmark/

# In[ ]:


train_values.dtypes


# In[ ]:


train_labels.head()


# In[ ]:


train_labels.heart_disease_present.value_counts().plot.bar(title='Number with Heart Disease')


# In[ ]:


selected_features = ['age', 
                     'sex', 
                     'max_heart_rate_achieved', 
                     'resting_blood_pressure']
train_values_subset = train_values[selected_features]


# In[ ]:


sns.pairplot(train_values.join(train_labels), 
             hue='heart_disease_present', 
             vars=selected_features)


# In[ ]:


# for preprocessing the data
from sklearn.preprocessing import StandardScaler

# the model
from sklearn.linear_model import LogisticRegression

# for combining the preprocess with model training
from sklearn.pipeline import Pipeline

# for optimizing parameters of the pipeline
from sklearn.model_selection import GridSearchCV


# In[ ]:


pipe = Pipeline(steps=[('scale', StandardScaler()), 
                       ('logistic', LogisticRegression())])
pipe


# In[ ]:


param_grid = {'logistic__C': [0.0001, 0.001, 0.01, 1, 10], 
              'logistic__penalty': ['l1', 'l2']}
gs = GridSearchCV(estimator=pipe, 
                  param_grid=param_grid, 
                  cv=3)


# In[ ]:


gs.fit(train_values_subset, train_labels.heart_disease_present)


# In[ ]:


gs.best_params_


# In[ ]:


from sklearn.metrics import log_loss

in_sample_preds = gs.predict_proba(train_values[selected_features])
log_loss(train_labels.heart_disease_present, in_sample_preds)


# In[ ]:


test_values = pd.read_csv('../input/warm-up-machine-learning-with-a-heart/test_values.csv', index_col='patient_id')


# In[ ]:


test_values_subset = test_values[selected_features]


# In[ ]:


predictions = gs.predict_proba(test_values_subset)[:, 1]


# In[ ]:


submission_format = pd.read_csv('../input/format/submission_format.csv', index_col='patient_id')


# In[ ]:


my_submission = pd.DataFrame(data=predictions,
                             columns=submission_format.columns,
                             index=submission_format.index)


# In[ ]:


my_submission.head()


# In[ ]:


my_submission.to_csv('../input/solution.csv')

