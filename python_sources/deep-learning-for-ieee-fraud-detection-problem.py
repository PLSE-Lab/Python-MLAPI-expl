#!/usr/bin/env python
# coding: utf-8

# Thise code is a slightly updated version of xhululu's kernel. Enjoy!
# > Xhulu's kernel: https://www.kaggle.com/xhlulu/ieee-fraud-xgboost-with-gpu-fit-in-40s

# In[ ]:


import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')\ntest_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')\n\ntrain_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')\ntest_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')\n\nsample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')\n\ntrain = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)\ntest = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)\n\nprint(train.shape)\nprint(test.shape)\n\ny_train = train['isFraud'].copy()\ndel train_transaction, train_identity, test_transaction, test_identity\n\n# Drop target, fill in NaNs\nX_train = train.drop('isFraud', axis=1)\nX_test = test.copy()\n\ndel train, test\n\n# Label Encoding\nfor f in X_train.columns:\n    if X_train[f].dtype=='object' or X_test[f].dtype=='object': \n        lbl = preprocessing.LabelEncoder()\n        lbl.fit(list(X_train[f].values) + list(X_test[f].values))\n        X_train[f] = lbl.transform(list(X_train[f].values))\n        X_test[f] = lbl.transform(list(X_test[f].values))   ")


# In[ ]:


get_ipython().run_cell_magic('time', '', '# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage\n# WARNING! THIS CAN DAMAGE THE DATA \ndef reduce_mem_usage(df):\n    """ iterate through all the columns of a dataframe and modify the data type\n        to reduce memory usage.        \n    """\n    start_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage of dataframe is {:.2f} MB\'.format(start_mem))\n    \n    for col in df.columns:\n        col_type = df[col].dtype\n        \n        if col_type != object:\n            c_min = df[col].min()\n            c_max = df[col].max()\n            if str(col_type)[:3] == \'int\':\n                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:\n                    df[col] = df[col].astype(np.int8)\n                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:\n                    df[col] = df[col].astype(np.int16)\n                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:\n                    df[col] = df[col].astype(np.int32)\n                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:\n                    df[col] = df[col].astype(np.int64)  \n            else:\n                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:\n                    df[col] = df[col].astype(np.float16)\n                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:\n                    df[col] = df[col].astype(np.float32)\n                else:\n                    df[col] = df[col].astype(np.float64)\n        else:\n            df[col] = df[col].astype(\'category\')\n\n    end_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage after optimization is: {:.2f} MB\'.format(end_mem))\n    print(\'Decreased by {:.1f}%\'.format(100 * (start_mem - end_mem) / start_mem))\n    \n    return df\nX_train = reduce_mem_usage(X_train)\nX_test = reduce_mem_usage(X_test)')


# In[ ]:


df1 = X_train.head(n=490540)
df2 = X_train.tail(n=100000)

X1 = df1
Y1 = y_train[:490540]
X1 = np.array(X1).astype(float)  
Y1 = np.array(Y1).astype(np.int32)

# Modeling
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=432))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='nadam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])     
model.fit(X1,Y1,verbose=1,shuffle=True, nb_epoch=10,batch_size=100,validation_split=0.3)


# In[ ]:


df2 = X_train.tail(n=100000)

X2 = df2
Y2 = y_train[490540:]
X2 = np.array(X2).astype(float)  
Y2 = np.array(Y2).astype(np.int32)

score = model.evaluate(X2,Y2, batch_size=16)
print("LOSS")
print(score[0])
print("precision")
print(score[1])


# In[ ]:


y_pred = model.predict(X_test)[:,1]


# In[ ]:


y_pred[:10]


# In[ ]:


sample_submission['isFraud'] = y_pred
sample_submission.to_csv('simple_deep_learning.csv')


# In[ ]:




