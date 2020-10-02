#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


bank = pd.read_csv('/kaggle/input/bank-notes/bank_note_data.csv')


# In[ ]:


bank.head()


# In[ ]:


bank.info()


# In[ ]:


sns.countplot(x='Class', data=bank)


# In[ ]:


sns.pairplot(bank, hue='Class')


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()


# In[ ]:


scaler.fit(bank.drop('Class', axis=1))


# In[ ]:


scaled_features = scaler.transform(bank.drop('Class', axis=1))


# In[ ]:


df_feat = pd.DataFrame(scaled_features, columns = bank.columns[:-1])
df_feat.head()


# In[ ]:


X = df_feat
y = bank['Class']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[ ]:


import tensorflow as tf


# In[ ]:


df_feat.columns


# In[ ]:


feat_cols = []

for col in df_feat.columns:
    feat_cols.append(tf.feature_column.numeric_column(col))


# In[ ]:


feat_cols


# In[ ]:


classifier = tf.estimator.DNNClassifier(hidden_units=[10,20,10], n_classes=2, feature_columns=feat_cols)


# In[ ]:


input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=20, shuffle=True)


# In[ ]:


classifier.train(input_fn=input_func, steps = 500)


# In[ ]:


pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(X_test), shuffle=False)


# In[ ]:


note_predictions = list(classifier.predict(input_fn=pred_fn))


# In[ ]:


note_predictions[0]


# In[ ]:


final_preds = []

for pred in note_predictions:
    final_preds.append(pred['class_ids'][0])


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


print(confusion_matrix(y_test, final_preds))


# In[ ]:


print(classification_report(y_test,final_preds))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier(n_estimators=200)


# In[ ]:


rfc.fit(X_train, y_train)


# In[ ]:


rfc_preds =rfc.predict(X_test)


# In[ ]:


print(classification_report(y_test,rfc_preds))


# In[ ]:


print(confusion_matrix(y_test,rfc_preds))


# In[ ]:




