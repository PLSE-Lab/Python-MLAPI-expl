#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
submit =  pd.read_csv("../input/sample_submission.csv")


# In[ ]:


test.head()


# In[ ]:


labels = train.target
train_df = train.drop(columns=['ID_code','target'])
test_df = test.drop(columns=['ID_code'])


# In[ ]:


feat_cols= []
for col in train_df.columns:
    feat_cols.append(tf.feature_column.numeric_column(col))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_df,labels,test_size=0.33, random_state=101)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()


# In[ ]:


scaler.fit(X_train)


# In[ ]:


X_train = pd.DataFrame(data= scaler.transform(X_train),columns=X_train.columns,index=X_train.index)
X_test = pd.DataFrame(data= scaler.transform(X_test),columns=X_test.columns,index=X_test.index)


# In[ ]:


input_func = tf.estimator.inputs.pandas_input_fn(x= X_train,y=y_train,batch_size=100,num_epochs=1000,shuffle=False)


# In[ ]:


#model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)


# In[ ]:


#model.train(input_fn=input_func,steps=1000)


# In[ ]:


eval_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      y=y_test,
      batch_size=100,
      num_epochs=1,
      shuffle=False)


# In[ ]:


#model.evaluate(eval_input_func)


# Nh = Ns/ (a * (Ni + No))
# 
# Nh - number of hidden neurons
# Ns- number of samples in training data set
# Ni - nnumber of input neurons
# No - Number of output neurons 
# a - scaling factor varies from  2 -10 

# In[ ]:


135000/(4* 201)


# In[ ]:


dnn_model = tf.estimator.DNNClassifier(hidden_units=[336,168,84,42],feature_columns=feat_cols,n_classes=2)


# In[ ]:


dnn_model.train(input_fn= input_func,steps=10000)


# In[ ]:


dnn_model.evaluate(eval_input_func)


# In[ ]:


test_df = pd.DataFrame(data= scaler.transform(test_df),columns=test_df.columns,index=test_df.index)


# In[ ]:


predict_input_func = tf.estimator.inputs.pandas_input_fn(x=test_df,batch_size=100,num_epochs=1,shuffle=False)


# In[ ]:


predictions = dnn_model.predict(input_fn=predict_input_func)


# In[ ]:


results = list(predictions)


# In[ ]:


results[1:10]


# In[ ]:


prediction= [i['probabilities'][1] for i in results]


# In[ ]:


submit['target'] = prediction
submit.to_csv("DNN2.csv",index=False)
submit.head()


# In[ ]:


submit[submit.target>0.5].head(20)

