#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

import math
import pprint
pp = pprint.PrettyPrinter(indent=2)
tf.logging.set_verbosity(tf.logging.WARN)
np.random.seed(72341)


# In[2]:


get_ipython().system('ls -l')


# In[6]:


data = pd.read_csv("../input/zoo.csv")
data.head(6)


# In[7]:


data.describe()


# In[8]:


# Shuffle and split the dataset

data = data.sample(frac=1).reset_index(drop=True)
data_total_len = data[data.columns[0]].size

data_train_frac = 0.6
split_index = math.floor(data_total_len*data_train_frac)

train_data = data.iloc[:split_index]
eval_data = data.iloc[split_index:]


# In[9]:


train_data.describe()


# In[10]:


eval_data.describe()


# In[11]:


data.iloc[:,16:18]


# In[ ]:


def preprocess(data):
  X = data.iloc[:, 1:17]  # all rows, all the features and no labels
  y = data.iloc[:, 17]  # all rows, label only
  y = y-1 # shift value range from 1-7 to be 0-6
  return X, y


# In[13]:


# This is used later in feature columns

# X = data.iloc[:,1:17]  # all rows, all the features and no labels
# y = data.iloc[:, 17]  # all rows, label only

X, y = preprocess(data)

X.head()
y.head()


# In[50]:


def gen_input_fn(data, batch_size=32, epochs=1, shuffle=True):
    """An input function for training"""

    features, labels = preprocess(data)
    
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


# In[51]:


# Test dataset

with tf.Session() as sess:
    ds = gen_input_fn(eval_data)()
    iterator = ds.make_one_shot_iterator()
    next_element = iterator.get_next()

    element = sess.run(next_element)
    print(len(element)) # 2, one for features, one for label
    print(element)


# ## List unique values in each column

# In[17]:


i = 0
for col_name in data.columns: 
    print("{}. {}: {}".format(i, col_name, data[col_name].unique()))
    i+=1
    


# In[52]:


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


# In[72]:


linear_model = tf.estimator.LinearClassifier(
    feature_columns=get_feature_columns(X),
    n_classes=7)


# In[ ]:


def train_and_eval(model, train_data=train_data, eval_data=eval_data, epochs=1):
    model.train(input_fn=gen_input_fn(train_data, epochs=epochs))
    return model.evaluate(input_fn=gen_input_fn(eval_data, shuffle=False))


# In[74]:


train_and_eval(linear_model, epochs=1)


# ## Train a DNN
# Wrap wide columns in indicator columns to facilitate DNN. Let's see if it can do feature extraction.

# In[75]:


deep_features = [tf.feature_column.indicator_column(col) for col in get_feature_columns(X)]

deep_model = tf.estimator.DNNClassifier(
    feature_columns=deep_features,
    hidden_units=[30,20,10],
    n_classes=7)


# In[77]:


train_and_eval(deep_model, epochs=5)
# train_and_eval(deep_model)


# ## Make some Predictions

# In[ ]:


# 1-7 is Mammal, Bird, Reptile, Fish, Amphibian, Bug and Invertebrate
animal_type = ['Mammal', 'Bird', 'Reptile', 'Fish', 'Amphibian', 'Bug', 'Invertebrate']


# In[78]:


predict_data = eval_data[10:15]
predictions = deep_model.predict(input_fn=gen_input_fn(predict_data, shuffle=False))

for i, prediction in enumerate(predictions):
  predicted_animal = animal_type[int(prediction["classes"][0].decode("utf8"))]
  correct_animal = animal_type[predict_data["class_type"].iloc[i]-1]
  print("Prediction:   {} \nTrue answer:   {}\n".format(
        predicted_animal, correct_animal))


# ## Which ones did we get wrong?
# Let's check the full evaluation set to see which specific mushroom was wrongly categorized.

# In[ ]:


def show_wrong_predictions(model, predict_data):
  predictions = model.predict(input_fn=gen_input_fn(predict_data, shuffle=False))

  for i, prediction in enumerate(predictions):
      if int(prediction["classes"]) != int(predict_data["class_type"].iloc[i]-1):
#           print("[WRONG] Predictions:   {} with probabilities {}\nTrue answer:   {}\n".format(
#               prediction["classes"][0].decode("utf8"), prediction["probabilities"], predict_data["class_type"].iloc[i]-1))
          predicted_animal = animal_type[int(prediction["classes"][0].decode("utf8"))]
          correct_animal = animal_type[predict_data["class_type"].iloc[i]-1]
          print("Prediction:   {} \nTrue answer:   {}\n".format(
                predicted_animal, correct_animal))


# In[47]:


show_wrong_predictions(linear_model, eval_data)


# In[46]:


show_wrong_predictions(deep_model, eval_data)


# In[ ]:




