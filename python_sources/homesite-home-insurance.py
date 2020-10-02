#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Loading libraries
import pandas as pd
import numpy as np
import xgboost as xgb


# In[ ]:


# Loading data
train = pd.read_csv("../input/train.csv", parse_dates=['Original_Quote_Date'])
test = pd.read_csv("../input/test.csv", parse_dates=['Original_Quote_Date'])


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


train.head(n=5)


# In[ ]:


test.head(n=5)


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


# Removing unnecessary columns
train = train.drop(['QuoteNumber'], axis = 1)


# In[ ]:


train_nulls = train.isnull().sum()
train_null_cols = list(train_nulls[train_nulls > 0].index)
print(train_null_cols)


# In[ ]:


test_nulls = test.isnull().sum()
test_null_cols = list(test_nulls[test_nulls > 0].index)
print(test_null_cols)


# In[ ]:


# Filling missing values with mode
train[train_null_cols] = train[train_null_cols].fillna(train[train_null_cols].mode().iloc[0])


# In[ ]:


# Filling missing values with mode
test[test_null_cols] = test[test_null_cols].fillna(test[test_null_cols].mode().iloc[0])


# In[ ]:


def feature_engineering(data):
    data['Original_Quote_Date_year'] = data['Original_Quote_Date'].dt.year
    data['Original_Quote_Date_month'] = data['Original_Quote_Date'].dt.month
    data['Original_Quote_Date_weekday'] = data['Original_Quote_Date'].dt.weekday
    return data


# In[ ]:


train = feature_engineering(train)


# In[ ]:


test = feature_engineering(test)


# In[ ]:


train = train.drop('Original_Quote_Date', axis=1)
test = test.drop('Original_Quote_Date', axis=1)


# In[ ]:


# Encoding object columns
from sklearn.preprocessing import LabelEncoder

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(np.unique(list(train[c].values) + list(test[c].values)))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))


# In[ ]:


# Verifying data after encoding
train.describe().transpose()


# In[ ]:


#print(train.columns) # X = train.iloc[:,2:], y = train.iloc[:, 1]
#print(test.columns) # X = test.iloc[:, 2:]
x_cols = list(train.columns[2:])
print(x_cols)


# In[ ]:


from sklearn.grid_search import RandomizedSearchCV
clf = xgb.XGBClassifier()
params = {
 "learning_rate": [0.1, 0.001],
 "max_depth": [3, 6],
 "n_estimators": [100, 200, 300],
 "objective": ["binary:logistic"],
 "nthread": 2
 }
random_search = RandomizedSearchCV(estimator=clf, 
                                   param_distributions=params, 
                                   scoring='roc_auc',
                                   refit=True,
                                   verbose=1)
#clf.get_params()
random_search


# In[ ]:


X_train = train[x_cols].values
y_train = train['QuoteConversion_Flag'].values


# In[ ]:


random_search.fit(X_train, y_train)


# In[ ]:


print(random_search.best_params_)
print(random_search.best_score_)
print(random_search.best_estimator_)


# In[ ]:


# Generating predictions
predictions = random_search.predict(test[x_cols])


# In[ ]:


# Generating submission dataframe
submission = pd.DataFrame()
submission['QuoteNumber'] = test['QuoteNumber']
submission['QuoteConversion_Flag'] = predictions


# In[ ]:


import time
PREDICTIONS_FILENAME_PREFIX = 'predictions_'
PREDICTIONS_FILENAME = PREDICTIONS_FILENAME_PREFIX + time.strftime('%Y%m%d-%H%M%S') + '.csv'

print(PREDICTIONS_FILENAME)

submission.to_csv(PREDICTIONS_FILENAME, index = False)

