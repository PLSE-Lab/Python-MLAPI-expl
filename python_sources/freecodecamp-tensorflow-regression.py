#!/usr/bin/env python
# coding: utf-8

# # Linear Regression Algorithm

# ## Import Dependencies

# In[ ]:


from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib


# In[ ]:


import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf


# ## Data

# In[ ]:


dftrain = pd.read_csv("/kaggle/input/train.csv")
dfeval = pd.read_csv("/kaggle/input/eval.csv")


# In[ ]:


dftrain.head()


# In[ ]:


dftrain.shape


# In[ ]:


y_train = dftrain.pop("survived")
y_eval = dfeval.pop("survived")

y_train.head()


# In[ ]:


dftrain.age.hist(bins = 20)


# In[ ]:


dftrain.sex.value_counts().plot(kind = "barh")


# In[ ]:


dftrain["class"].value_counts().plot(kind = "barh")


# In[ ]:


survival_distribution = pd.concat([dftrain, y_train], axis=1).groupby("sex").survived.mean() * 100
survival_distribution.plot(kind = "barh").set_xlabel("% survived")


# ## Generate Feature Columns for Tensorflow

# In[ ]:


dftrain.dtypes


# In[ ]:


CATEGORICAL_COLUMNS = ["sex", "n_siblings_spouses", "parch", "class", "deck", "embark_town", "alone"]
NUMERIC_COLUMNS = ["age", "fare"]

feature_columns = []

for feature in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature, vocabulary))
    
for feature in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature, dtype = tf.float64))
    
feature_columns


# ## Input Function Generator for Generating `tf.data.Dataset`

# In[ ]:


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_fn():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_fn


# In[ ]:


train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)


# ## Creating the Linear Classifier Model

# In[ ]:


linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)


# ## Training the Model

# In[ ]:


linear_est.train(train_input_fn)


# ## Evaluating the Model

# In[ ]:


result = linear_est.evaluate(eval_input_fn)
result


# ## Using Model to Make Predictions

# In[ ]:


predictions = list(linear_est.predict(eval_input_fn))
survival_probabilities = pd.Series([pred["probabilities"][1] for pred in predictions])


# In[ ]:


for i in range(len(dfeval.head())):
    print(dfeval.loc[i])
    print("survived: {}".format("yes" if (y_eval.loc[i] == 1) else "no"))
    print(f"predicted survival probability: {survival_probabilities[i]}")


# In[ ]:


survival_probabilities.plot(kind="hist", bins=20, title="Survival Probabilities")

