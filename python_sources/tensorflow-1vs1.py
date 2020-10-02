#!/usr/bin/env python
# coding: utf-8

# In[ ]:


id = 1


# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import tensorflow as tf

import tflearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# In[ ]:


#sns.pairplot(train.drop("id", axis=1), hue="type", markers=["o", "s", "D"], diag_kind="kde")


# In[ ]:


X_train = train.drop(["id","type", "color"], axis=1)
y_train = pd.get_dummies(train["type"])

X_test = test.drop(["id", "color"], axis=1)


# # 1: Neural network

# In[ ]:


if id == 1:
    with tf.Graph().as_default():
        net = tflearn.input_data([None, 4])

        net = tflearn.fully_connected(net, 256, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
        net = tflearn.dropout(net, 0.6)
        
        net = tflearn.fully_connected(net, 256, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
        net = tflearn.dropout(net, 0.6)
        
        net = tflearn.fully_connected(net, 3, activation='softmax')

        net = tflearn.regression(net)

        model = tflearn.DNN(net)

        model.fit(X_train.values, 
                  y_train.values, 
                  n_epoch=150,
                  snapshot_epoch=False,
                  snapshot_step=50, 
                  show_metric=True)


# In[ ]:


y_pred = model.predict(X_train)
y_pred = np.argmax(y_pred, axis=1)

y_true = np.argmax(y_train.values, axis=1)

print(classification_report(y_pred, y_true))


# In[ ]:


if id == 1:
    y_submission = model.predict(X_test)
    y_submission = np.argmax(y_submission, axis=1)

    Y = pd.DataFrame()
    Y["id"] = test["id"]
    Y["type"] = y_submission

    Y.loc[Y['type'] == 0, "type"] = "Ghost"
    Y.loc[Y['type'] == 1, "type"] = "Ghoul"
    Y.loc[Y['type'] == 2, "type"] = "Goblin"

    Y.to_csv("submission.csv",index=False)


# In[ ]:




