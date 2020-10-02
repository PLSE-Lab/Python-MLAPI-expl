#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from sklearn.utils import class_weight


# In[ ]:


df = pd.read_csv('../input/train.csv')


# In[ ]:


df.describe()


# As you can see the mean of target is 0.1 which means that the class 1 just 10% of the data which makes this data unbalanced.

# In[ ]:


fcol = ['var_' + str(x) for x in range(200)]
X = df[fcol].values
y = df['target'].values
mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
rng = maxs - mins
X = 1 - ((maxs - X) / rng)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)


# I select all the 200 features and make prediction from it. Since the neural network need a lot of data, I think sample the input to 5% of the test data is enough. Unlike the boosting algorithm, neural network will work better if the input is normalized.

# In[ ]:


#To handle imbalance data
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)


# The total line of data is only 200k, so maybe it's better to make the neural network shallow one. The SGD optimizer is good for shallow network so I'll be using that.

# In[ ]:


model = Sequential()
model.add(Dense(32, input_dim=X.shape[1], activation='relu'))
model.add(Dense(16, input_dim=X.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(
    X_train, 
    y_train, 
    epochs=150, 
    batch_size=10, 
    callbacks=[EarlyStopping(monitor='acc')],
    validation_data=(X_val, y_val),
    class_weight=class_weights
)


# In[ ]:


train_predict = model.predict_proba(X_train)
train_roc = roc_auc_score(y_train, train_predict)
print('Train AUC: {}'.format(train_roc))

val_predict = model.predict_proba(X_val)
val_roc = roc_auc_score(y_val, val_predict)
print('Val AUC: {}'.format(val_roc))


# In[ ]:


test_df = pd.read_csv('../input/test.csv')


# In[ ]:


X_test = test_df[fcol].values
X_test = 1 - ((maxs - X_test) / rng)


# In[ ]:


prediction = model.predict_proba(X_test)
submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] = prediction
submission.to_csv('submit.csv', index=False)


# In[ ]:




