#!/usr/bin/env python
# coding: utf-8

# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

import math
import pprint
pp = pprint.PrettyPrinter(indent=2)

np.random.seed(72341)


# In[20]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[41]:


data = pd.read_csv("../input/mushrooms.csv")
data.head(6)


# In[22]:


data.describe()


# In[42]:


# Shuffle and split the dataset

data = data.sample(frac=1).reset_index(drop=True)
data_total_len = data[data.columns[0]].size

data_train_frac = 0.6
split_index = math.floor(data_total_len*data_train_frac)

train_data = data.iloc[:split_index]
eval_data = data.iloc[split_index:]


# In[24]:


train_data.describe()


# In[25]:


eval_data.describe()


# In[26]:


# This is used later in feature columns

X = data.iloc[:,1:23]  # all rows, all the features and no labels
y = data.iloc[:, 0]  # all rows, label only

y = y.apply(lambda x: "p" in x).astype(int) # 1 for poisonous, 0 for edible

X.head()
y.head()


# In[27]:


def gen_input_fn(data, batch_size=100, epochs=1, shuffle=True):
    """An input function for training"""

    # all rows, all the features and no labels
    features = data.iloc[:,1:23]  
    # all rows, label only
    # 1 for poisonous, 0 for edible
    labels = data.iloc[:, 0].apply(lambda x: "p" in x).astype(int) 
    
    def _input_fn():
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffle, repeat, and batch the examples.
        if shuffle:
            dataset = dataset.shuffle(1000)
        dataset = dataset.repeat(epochs).batch(batch_size)

        # Return the dataset.
        return dataset
    return _input_fn

gen_input_fn(train_data)()


# In[28]:


# Test dataset

with tf.Session() as sess:
    ds = gen_input_fn(eval_data)()
    iterator = ds.make_one_shot_iterator()
    next_element = iterator.get_next()

    element = sess.run(next_element)
    print(len(element)) # 2, one for features, one for label
    print(element)


# ## List unique values in each column

# In[29]:


i = 0
for col_name in data.columns: 
    print("{}. {}: {}".format(i, col_name, data[col_name].unique()))
    i+=1
    


# In[30]:


def get_feature_columns(data):
    feature_columns = [
        tf.feature_column.categorical_column_with_vocabulary_list(
            key=col_name,
            vocabulary_list=data[col_name].unique()
        ) for col_name in data.columns
    ]
    return feature_columns

feature_cols = get_feature_columns(X)
pp.pprint(feature_cols)


# In[31]:


linear_model = tf.estimator.LinearClassifier(
    feature_columns=get_feature_columns(X),
    n_classes=2)


# In[32]:


def train_and_eval(model, train_data=train_data, eval_data=eval_data):
    model.train(input_fn=gen_input_fn(train_data))
    model.evaluate(input_fn=gen_input_fn(eval_data))


# In[33]:


train_and_eval(linear_model)


# ## Train a DNN
# Wrap wide columns in indicator columns to facilitate DNN. Let's see if it can do feature extraction.

# In[34]:


deep_features = [tf.feature_column.indicator_column(col) for col in get_feature_columns(X)]

deep_model = tf.estimator.DNNClassifier(
    feature_columns=deep_features,
    hidden_units=[30,20,10],
    n_classes=2)


# In[35]:


train_and_eval(deep_model)


# ## Make some Predictions

# In[52]:


predict_data = eval_data[200:205]
predictions = deep_model.predict(input_fn=gen_input_fn(predict_data, shuffle=False))

for i, prediction in enumerate(predictions):
    print("Predictions:    {} with probabilities {}\nTrue answer: {}\n".format(prediction["classes"], prediction["probabilities"], predict_data["class"].iloc[i]))


# ## Which ones did we get wrong?
# Let's check the full evaluation set to see which specific mushroom was wrongly categorized.

# In[54]:


predict_data = eval_data
predictions = deep_model.predict(input_fn=gen_input_fn(predict_data, shuffle=False))

for i, prediction in enumerate(predictions):
    if int(prediction["classes"]): # if it's a 1, it should be 'p'
        if predict_data["class"].iloc[i] in 'e': # so if it shows 'e', then it's wrong
            print("[WRONG] Predictions:    {} with probabilities {}\nTrue answer: {}\n".format(prediction["classes"], prediction["probabilities"], predict_data["class"].iloc[i]))       


# In[ ]:




