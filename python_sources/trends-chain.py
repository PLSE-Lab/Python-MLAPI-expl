#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


# # Dataset

# In[ ]:


DATA_PATH = '/kaggle/input/trends-assessment-prediction/'


# In[ ]:


loading = pd.read_csv(DATA_PATH + 'loading.csv')
loading.head()


# In[ ]:


fnc = pd.read_csv(DATA_PATH + 'fnc.csv')
fnc[fnc.columns[1:]] /= 600 
fnc.head()


# In[ ]:


dataset = loading.merge(fnc, on='Id')
dataset.head()


# In[ ]:


y_train = pd.read_csv(DATA_PATH + 'train_scores.csv')
print('Number of training samples: {}'.format(len(y_train)))
y_train.head()


# In[ ]:


y_train = y_train.fillna(y_train.mean()) #interpolate(method='nearest')
y_train.head()


# In[ ]:


x_train = dataset.loc[dataset['Id'].isin(y_train['Id'])]
x_train.head()


# In[ ]:


x_test = dataset.loc[~dataset['Id'].isin(y_train['Id'])]
test_ids = x_test['Id'] # Needed for submission
print('Number of test samples: {}'.format(len(x_test)))
x_test.head()


# In[ ]:


x_train = x_train.drop('Id', axis=1).values
x_test = x_test.drop('Id', axis=1).values
y_train = y_train.drop('Id', axis=1).values


# In[ ]:


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# # Model

# In[ ]:


model = RegressorChain(BayesianRidge(alpha_init=1e-4))


# # Training and Evaluation

# In[ ]:


def score(y_pred, y_true):
     return sum(list(map(lambda w, s: w * s, [.3, .175, .175, .175, .175], np.sum(np.abs(y_true - y_pred), axis=0) / np.sum(y_true, axis=0))))


# In[ ]:


n = 10
y_test = np.zeros((len(x_test), 5, n))
scores = np.zeros(n)
for i, (train_indexes, valid_indexes) in enumerate(KFold(n).split(x_train)):
    print('Split {} of {} ...'.format(i + 1, n))
    x, x_valid = x_train[train_indexes], x_train[valid_indexes]
    y, y_valid = y_train[train_indexes], y_train[valid_indexes]
    
    model.fit(x, y)

    y_pred = model.predict(x_valid)
    scores[i] = score(y_pred, y_valid)
    print('Score = {}'.format(scores[i]))
    print('Predicted : {}'.format(y_pred[0]))
    print('Expected  : {}\n'.format(y_valid[0]))
    
    y_test[:,:,i] = model.predict(x_test)


# In[ ]:


print('Average score = {}'.format(scores.mean()))


# # Prediction

# In[ ]:


y_test = y_test.mean(axis=2)
print(y_test)


# # Submission

# In[ ]:


outputs = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
ids = ['{}_{}'.format(int(id_), output)  for id_ in test_ids for output in outputs]
predicted = y_test.reshape(5 * len(y_test))

assert len(predicted) == 29385
submission = pd.DataFrame({'Id': ids, 'Predicted': predicted})
submission.head(10)


# In[ ]:


submission.to_csv('submission.csv', index = False)
get_ipython().system('head submission.csv')

