#!/usr/bin/env python
# coding: utf-8

# # Titanic (2/2): Prediction
# - In the [first Kernel](https://www.kaggle.com/hamzael1/titanic-1-2-exploration-pre-processing), I cleaned the data, and did some Feature Selection & Engineering. This is the Second Kernel which will use the Output of the first one to make predictions using Random Forests..

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

IN_CLOUD = True
INPUT_DIR = '../input/titanic-1-2-exploration-pre-processing' if IN_CLOUD else './data'


# ## Import Cleaned Data

# In[ ]:


train_clean_df = pd.read_csv(f'{INPUT_DIR}/train_clean.csv')
test_clean_df  = pd.read_csv(f'{INPUT_DIR}/test_clean.csv' )


# In[ ]:


train_clean_df.sample(3)


# #### Cut the Passenger Ids from the Test Dataset (because it's not a feature). We will append them later after we do our predictions:

# In[ ]:


test_passenger_ids = test_clean_df.PassengerId
test_clean_df = test_clean_df.drop('PassengerId', axis=1)


# #### Do some Feature Selection and experiment with different features: 

# In[ ]:


drop_cols = ['NbrRelatives', 'Age']

train_clean_df.drop(drop_cols, axis=1, inplace=True)
test_clean_df.drop(drop_cols, axis=1, inplace=True)


# In[ ]:


train_clean_df.sample(3)


# In[ ]:


test_clean_df.sample(3)


# ## Modeling

# In[ ]:


train_y = train_clean_df.Survived
train_x = train_clean_df.drop('Survived', axis=1)

train_x.sample(3)


# #### Imports related to modeling:

# In[ ]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn import metrics


# In[ ]:


GRID_SEARCH = False
if GRID_SEARCH:
    param_grid={
        'n_estimators': [x for x in range(50, 400, 50)],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4,5,6,7],
        'criterion': ['gini', 'entropy']
    }
    rfc = RandomForestClassifier()
    model = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    print('Fitting ... ')
    model.fit(train_x, train_y)
    print('Best Params: ', model.best_params_)
    print('CV results: ', model.cv_results_)


# In[ ]:


model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=5, max_features='auto')
CROSS_VAL = True
if CROSS_VAL:
    scores = cross_val_score(model, train_x, train_y, cv=5)
    print(scores.mean())


# In[ ]:


PREDICT = not GRID_SEARCH
if PREDICT:
    model.fit(train_x, train_y)
    predictions = model.predict(test_clean_df)


# In[ ]:


if PREDICT:
    OUTPUT_DIR = '' if IN_CLOUD else './data/'
    OUTPUT = True
    submission = pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': predictions})
    if OUTPUT:
        submission.to_csv(f'{OUTPUT_DIR}submission.csv', index=False)
        print('Done exporting !')
    print(submission.sample(5))


# In[ ]:




