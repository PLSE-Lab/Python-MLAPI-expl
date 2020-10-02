#!/usr/bin/env python
# coding: utf-8

# I'd like to thank the author of the notebook "Tensorflow- Deep Learning to Solve Titanic", from which this code is adapted.

# In[ ]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print(tf.__version__)


# In[ ]:


# load data
train_data = pd.read_csv(r"../input/titanic/train.csv")
test_data = pd.read_csv(r"../input/titanic/test.csv")


# In[ ]:


train_data


# In[ ]:


test_data


# In[ ]:


# Feature Engineering

from sklearn.impute import SimpleImputer

def nan_padding(data, columns):
    for column in columns:
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        data[column]=imputer.fit_transform(data[column].values.reshape(-1,1))
    return data


nan_columns = ["Age", "SibSp", "Parch"]

train_data = nan_padding(train_data, nan_columns)
test_data = nan_padding(test_data, nan_columns)


# In[ ]:


train_data


# In[ ]:


#save PassengerId for evaluation
test_passenger_id=test_data["PassengerId"]


# In[ ]:


def drop_not_concerned(data, columns):
    return data.drop(columns, axis=1)

not_concerned_columns = ["PassengerId","Name", "Ticket", "Fare", "Cabin", "Embarked"]
train_data = drop_not_concerned(train_data, not_concerned_columns)
test_data = drop_not_concerned(test_data, not_concerned_columns)


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


def dummy_data(data, columns):
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        data = data.drop(column, axis=1)
    return data


dummy_columns = ["Pclass"]
train_data=dummy_data(train_data, dummy_columns)
test_data=dummy_data(test_data, dummy_columns)


# In[ ]:


test_data.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
def sex_to_int(data):
    le = LabelEncoder()
    le.fit(["male","female"])
    data["Sex"]=le.transform(data["Sex"]) 
    return data

train_data = sex_to_int(train_data)
test_data = sex_to_int(test_data)
train_data.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

def normalize_age(data):
    scaler = MinMaxScaler()
    data["Age"] = scaler.fit_transform(data["Age"].values.reshape(-1,1))
    return data
train_data = normalize_age(train_data)
test_data = normalize_age(test_data)
train_data.head()


# In[ ]:


from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def split_valid_test_data(data, fraction=(1 - 0.8)):
    data_y = data["Survived"]
    lb = LabelBinarizer()
    data_y = lb.fit_transform(data_y)

    data_x = data.drop(["Survived"], axis=1)

    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=fraction)

    return train_x.values, train_y, valid_x, valid_y

train_x, train_y, valid_x, valid_y = split_valid_test_data(train_data)
print("train_x:{}".format(train_x.shape))
print("train_y:{}".format(train_y.shape))
print("train_y content:{}".format(train_y[:3]))

print("valid_x:{}".format(valid_x.shape))
print("valid_y:{}".format(valid_y.shape))


# Okay, so the things that we need for the model are:
# 
# inputs, labels, learning_rate, is_training, logits, cost, optimizer, predicted, and accuracy

# In[ ]:


train_x.shape[1]


# In[ ]:


get_ipython().system('pip install -U keras-tuner')


# In[ ]:


# Build Neural Network
import kerastuner

def build_neural_network(hp):    
    initializer = tf.keras.initializers.GlorotUniform()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units = hp.Int('units', min_value = 5, max_value = 500, step = 25),
                                    activation=None,kernel_initializer=initializer,
                                    input_shape = ( train_x.shape[1],)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1, activation = 'relu'))
    
    model.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.Adam(hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])), metrics = ['accuracy'])

    return model

tuner = kerastuner.tuners.RandomSearch(
    build_neural_network,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3)

#model = build_neural_network(learning_rate_value)


# In[ ]:


epochs = 50
tuner.search(train_x, train_y, epochs = 200,validation_data = (valid_x, valid_y))


# In[ ]:


tuner.results_summary()


# In[ ]:


op_model = tuner.get_best_models()[0]
op_model.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.Adam(0.001), metrics = ['accuracy'])


# In[ ]:



history = op_model.fit(train_x, train_y, epochs = 200, validation_data = (valid_x, valid_y))


# In[ ]:


## this code was stolen from udacity colabs
import matplotlib.pyplot as plt
plt.style.use("dark_background")

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")


# In[ ]:


len(test_data)


# In[ ]:


test_data


# In[ ]:


passenger_id=test_passenger_id.copy()
evaluation=passenger_id.to_frame()
evaluation["Survived"]=[int(bool(i)) for i in op_model.predict(test_data)]
evaluation[:10]


# In[ ]:


evaluation.to_csv("evaluation_submission.csv",index=False)


# In[ ]:


answers = pd.read_csv(r"../input/titanic/gender_submission.csv")


# In[ ]:


from sklearn.metrics import confusion_matrix
c = confusion_matrix(answers['Survived'], evaluation['Survived'], normalize = 'true')


# In[ ]:


c


# In[ ]:




