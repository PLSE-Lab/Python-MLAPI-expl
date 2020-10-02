#!/usr/bin/env python
# coding: utf-8

# https://www.kaggle.com/c/titanic/notebooks

# In[ ]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
random.seed(2020)

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRFClassifier


# In[ ]:


# Read in the datasets
train_raw = pd.read_csv('/kaggle/input/titanic/train.csv')
test_raw = pd.read_csv('/kaggle/input/titanic/test.csv')

train_raw.shape, test_raw.shape


# ### Some simple EDA

# In[ ]:


# Look at the training data
train_raw.head()


# In[ ]:


# Check variable types
train_raw.info()


# In[ ]:


# Check missing values in the raw training set
train_raw.isna().mean()


# * There're 3 variabels having missing values. 
# * There's over 77% of data missing in `Cabin`. I will simply drop it in my first try. If I have more time, I will do more research on `Cabin` to find out if I can impute those large number of missing values in `Cabin`.
# * There's around 20% of data missing in `Age`. I will look into its distribution and impute those missing values.
# * Only 2 data points are missing in `Embarked`. I will impute them with mode value for `Embarked`.

# In[ ]:


# The distribution of Age
train_raw.Age.hist()


# The `Age` is approximatelly normally distributed.

# In[ ]:


# Check the mean value and std value of the Age distribution
train_raw.Age.mean(), train_raw.Age.std()


# ### Feature engineering

# In[ ]:


class Transformer:
    def fit(self, X, y = None):
        df = X.copy()
        self.age_mean = int(df.Age.mean()) + 1
        self.age_std = int(df.Age.std())
        self.embark_mod = df.Embarked.mode()
        self.ticket_mod = df.Ticket.map(lambda x: x.split(' ')[-1]).mode()[0]
        self.fare_mean = df.Fare.mean()
    
    def transform(self, X, y = None):
        df = X.copy()
        df.Age = df.Age.map(lambda x: int(random.gauss(self.age_mean, self.age_std)) if type(x) == float else x)
        df.Ticket = df.Ticket.map(lambda x: x.split(' ')[-1])
        df.Ticket[df.Ticket == 'LINE'] = self.ticket_mod
        df.Ticket = df.Ticket.map(lambda x: int(x))
        df.Fare[df.Fare.isnull()] = self.fare_mean
        df.Embarked[df.Embarked.isnull()] = self.embark_mod
        df.drop(['PassengerId', 'Name', 'Cabin'], axis = 1, inplace = True)
        df = pd.get_dummies(df, drop_first=True)
   
        return df
    
    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.transform(X)


# In[ ]:


# How's the dataframe look like after feature engineering
Transformer().fit_transform(train_raw).head()


# In[ ]:


# Transform the training set and test set
transformer = Transformer()
train = transformer.fit_transform(train_raw)
test = transformer.transform(test_raw)

train.shape, test.shape


# In[ ]:


# Extract response variable from training data
xtrain = train.drop('Survived', axis = 1)
ytrain = train.Survived.values


# In[ ]:


# Split the training data into training set and validation set using a 80 vs. 20 split
X_train, X_valid, y_train, y_valid = train_test_split(xtrain, ytrain, stratify = ytrain, 
                                                      test_size = 0.2, random_state = 2020)


# In[ ]:


X_train.shape


# ### Model 1: Logistic Regression

# In[ ]:


# Build pipelines to wrap data rescale and model together
step = [('scale', MinMaxScaler()), 
       ('lr', LogisticRegressionCV(random_state=2020))]
mod = Pipeline(step)
mod.fit(X_train, y_train)


# In[ ]:


# Make predictions
pred_train = mod.predict(X_train)
pred_valid = mod.predict(X_valid)

prob_train = mod.predict_proba(X_train)[:,1]
prob_valid = mod.predict_proba(X_valid)[:,1]


# In[ ]:


print('Train Accuracy: {:.3f}'.format(accuracy_score(y_train, pred_train)))
print('Train AUC: {:.3f}'.format(roc_auc_score(y_train, prob_train)), '\n')

print('Valid Accuracy: {:.3f}'.format(accuracy_score(y_valid, pred_valid)))
print('Valid AUC: {:.3f}'.format(roc_auc_score(y_valid, prob_valid)))


# ### Model 2: Random Forest

# In[ ]:


rf = RandomForestClassifier(random_state=2020)
rf.fit(X_train, y_train)

pred_train = rf.predict(X_train)
pred_valid = rf.predict(X_valid)

prob_train = rf.predict_proba(X_train)[:,1]
prob_valid = rf.predict_proba(X_valid)[:,1]


# In[ ]:


print('Train Accuracy: {:.3f}'.format(accuracy_score(y_train, pred_train)))
print('Train AUC: {:.3f}'.format(roc_auc_score(y_train, prob_train)), '\n')

print('Valid Accuracy: {:.3f}'.format(accuracy_score(y_valid, pred_valid)))
print('Valid AUC: {:.3f}'.format(roc_auc_score(y_valid, prob_valid)))


# The accuracy is better than that from logistic regression. However, there's definitely overfitting problem exist here.

# ### Model 3: XGBoost

# In[ ]:


xgb = XGBRFClassifier(random_state=2020)
xgb.fit(X_train, y_train)

pred_train = xgb.predict(X_train)
pred_valid = xgb.predict(X_valid)

prob_train = xgb.predict_proba(X_train)[:,1]
prob_valid = xgb.predict_proba(X_valid)[:,1]


# In[ ]:


print('Train Accuracy: {:.3f}'.format(accuracy_score(y_train, pred_train)))
print('Train AUC: {:.3f}'.format(roc_auc_score(y_train, prob_train)), '\n')

print('Valid Accuracy: {:.3f}'.format(accuracy_score(y_valid, pred_valid)))
print('Valid AUC: {:.3f}'.format(roc_auc_score(y_valid, prob_valid)))


# Less overfitting than the random forest.

# ### Without model tuning, use the predictions from the XGBoost model.

# In[ ]:


pred_test = xgb.predict(test)
pred_test


# In[ ]:


test_result = pd.concat([test_raw['PassengerId'], pd.DataFrame(pred_test, columns=['Survived'])], axis = 1)
test_result.shape


# In[ ]:


test_result.to_csv('titanic_prediction.csv', header=True, index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




