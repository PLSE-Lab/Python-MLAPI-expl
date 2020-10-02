#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -U tensorflow==2.0.0-alpha0')


# Let's kernel restart after execute "!pip install -U tensorflow==2.0.0-alpha0".
# 
# How to "kernel restart" here.
# 
# > Ctrl+shift+p and select "confirm restart kernel" This will restart the Jupyter Kernel instance and reload the installed libraries.
# 
# https://github.com/Kaggle/docker-python/issues/340
# 

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
tf.random.set_seed(123)


# In[ ]:


tf.__version__


# https://www.tensorflow.org/alpha/tutorials/estimators/boosted_trees
# https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/estimators/boosted_trees.ipynb

# In[ ]:


train = pd.read_csv("../input/train.csv")


# In[ ]:


train.head()


# In[ ]:


train.columns


# In[ ]:


train.info()


# In[ ]:


train.describe()


# In[ ]:


train.Age[train.Age.isnull()] = train.Age.mean


# In[ ]:


train.Embarked.fillna("unknown", inplace=True)


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


dftrain, dfeval = train_test_split(train, test_size=0.2, random_state=42)


# In[ ]:


y_train = dftrain.Survived
dftrain.drop(columns=["Survived", "Cabin", "Ticket", 'Age', 'Fare'], inplace=True)
y_eval = dfeval.Survived
dfeval.drop(columns=["Survived", "Cabin", "Ticket", 'Age', 'Fare'], inplace=True)


# In[ ]:


dftrain.head()


# In[ ]:


dftrain.info()


# In[ ]:


fc = tf.feature_column
CATEGORICAL_COLUMNS = ['Sex', 'Parch', 'Pclass', 'SibSp', "Embarked"]
# NUMERIC_COLUMNS = ['Age', 'Fare']


# In[ ]:


def one_hot_cat_column(feature_name, vocab):
    return tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))


# In[ ]:


feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    # Need to one-hot encode categorical features.
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(one_hot_cat_column(feature_name, vocabulary))


# In[ ]:


# TODO: not solved feature_columns add num_columns error
#   for feature_name in NUMERIC_COLUMNS:
#     feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


# In[ ]:


feature_columns


# In[ ]:


example = dict(dftrain.head(1))
class_fc = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('Pclass', (1, 2, 3)))
print('Feature value: "{}"'.format(example['Pclass'].iloc[0]))
print('One-hot encoded: ', tf.keras.layers.DenseFeatures([class_fc])(example).numpy())


# In[ ]:


tf.keras.layers.DenseFeatures(feature_columns)(example).numpy()


# In[ ]:


# Use entire batch since this is such a small dataset.
NUM_EXAMPLES = len(y_train)

def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
            dataset = dataset.shuffle(NUM_EXAMPLES)
        # For training, cycle thru dataset as many times as need (n_epochs=None).    
        dataset = dataset.repeat(n_epochs)
        # In memory training doesn't use batching.
        dataset = dataset.batch(NUM_EXAMPLES)
        return dataset
    return input_fn

# Training and evaluation input functions.
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, n_epochs=1)


# In[ ]:


linear_est = tf.estimator.LinearClassifier(feature_columns)

# Train model.
linear_est.train(train_input_fn, max_steps=100)

# Evaluation.
result = linear_est.evaluate(eval_input_fn)
# clear_output()
print(pd.Series(result))


# In[ ]:


# Since data fits into memory, use entire dataset per layer. It will be faster.
# Above one batch is defined as the entire dataset. 
n_batches = 1
est = tf.estimator.BoostedTreesClassifier(feature_columns,
                                          n_batches_per_layer=n_batches)

# The model will stop training once the specified number of trees is built, not 
# based on the number of steps.
est.train(train_input_fn, max_steps=100)

# Eval.
result = est.evaluate(eval_input_fn)
# clear_output()
print(pd.Series(result))


# In[ ]:




