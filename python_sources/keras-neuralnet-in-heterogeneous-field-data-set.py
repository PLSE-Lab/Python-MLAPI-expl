#!/usr/bin/env python
# coding: utf-8

# # Scope and Goals
# This notebook is an illustrative example for using Keras Neural Networks for regression on a data sets with heterogeneous variables.
# 
# The data used is from [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques), a competition data set for data science students who have completed an online course in machine learning and are looking to expand their skill set before trying a featured competition.
# 
# One of the most pivotal part to build an efficient model for machine learning is analysing the data and its features. We are leaving this analysis out of this notebook, and we focus mainly on building the model. Therefore, we leave out of the scope: feature exploration, analysis of the correlation of the features, data formating (e.g. transforming data categorical data collected as numbers in the proper format) and data condensation.
# 
# Additionally, we show an automated process to tweak the parameters of the model. This process is very simple and could be substituted by more interesting exploratory searches. A* returns complete searches, however, we rather suggest to be more practical and use local (neighborhood) searches, like Tabu search, Simulated annealing, ACO or genetic algorithms.

# First, we load the libraries that we are using in the notebook

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.keras.models import load_model


# We define the constants that are specific of the data model, such as the target column, and the names and paths of the files

# In[ ]:


# Input
file_train = '../input/house-prices-advanced-regression-techniques/train.csv'
file_test = '../input/house-prices-advanced-regression-techniques/test.csv'

# Output
model_output_file = 'NN_1D_CSV_model.h5'

# Model specific
target = 'SalePrice'
index_col = 'Id'

# Validation Split, to separate testing and validation data
VALIDATION_SPLIT_SIZE = 0.2


# # Loading data
# We are using the following function to load raw data

# In[ ]:


def load_raw_data(file_path):
    return pd.read_csv(file_path, index_col = index_col)


# # Cleaning Data
# Within a given dataset, we ofter find missing data. Furthermore, the regression is done based on logic or numeric features, therefore, we want to convert any discrete feature into a categorical representation. We are using [one_hot](https://en.wikipedia.org/wiki/One-hot) for this, but we are excluding features with an elevate number of discrete values. Furthermore, as we are not doing a proper analysis on the data, we are not converting categorical values currently represented in a numeric format.

# In[ ]:


NA_THRESHOLD = 200
NA_CATEGORICAL_SUBSTITUTION = 'NULL'
NA_NUMERICAL_SUBSTITUTION = 0

def feature_extract(data):
        feature_columns = []
        # Columns with high missing values
        highna = [cname for cname in data.columns if data[cname].isna().sum() > NA_THRESHOLD]
        lowna = [cname for cname in data.columns if data[cname].isna().sum() <= NA_THRESHOLD]
        
        # Dropping columns with high number of missing values
        data = data.drop(highna, axis=1)

        # Low cardinality cols only
        categorical_cols = [cname for cname in data.columns
                                if data[cname].nunique() < 10 and data[cname].dtype == "object"]
        numeric_cols = [cname for cname in data.columns if data[cname].dtype in ['int64', 'float64']]
        if target in categorical_cols: categorical_cols.remove(target)
        if target in numeric_cols: numeric_cols.remove(target)

        return feature_columns, categorical_cols, numeric_cols
    


# # Preprocessing Data
# We use the following function to clean and give format to the raw data. We assign a constant to the missing values and normalize the data. Note that the normalization has to be done using the stats from the trained (known) data.

# In[ ]:


def prep_features(data, categorical_cols, numeric_cols, x_train = None):
    empty_fill = {}
    for feature in categorical_cols:
        empty_fill[feature] = NA_CATEGORICAL_SUBSTITUTION
    for feature in numeric_cols:
        empty_fill[feature] = NA_NUMERICAL_SUBSTITUTION
    data = data.fillna(empty_fill)

    data = data[categorical_cols + numeric_cols]

    # One Hot Numeric values
    x = pd.get_dummies(data, dummy_na=True)
    if x_train is not None:
        x_train, x = x_train.align(x, join='left', axis=1)
        x = x.fillna(0)
    else:
        x_train = x

    # Normalization on numerical values
    normed_x = norm(x, x_train)

    return normed_x
    
def norm(x, x_train):
    train_stats = x_train.describe()
    train_stats = train_stats.transpose()
    normalized =  (x - train_stats['mean']) / train_stats['std']
    return normalized.fillna(0)


# # Model
# We have build a function to parametrize building the neural network. We have 5 layers: the input, a layer for Normalization, a layer to prevent leaky ReLU, a Dropout and the output. The normalization layer and the leaky ReLU will prevent to converge to NA values. The dropout prevent overfiting, randomly discarting samples on each iteration.

# In[ ]:


def generate_model (feature_len, optimizer, dropout = 0.3, units = 64, leaky = 0.05):
    model = Sequential([
        layers.Dense(units, activation='relu', input_shape=feature_len),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=leaky),
        layers.Dropout(dropout),
        layers.Dense(1)
    ])
    model.compile(loss='mse',
                       optimizer=optimizer,
                       metrics=['mae', 'mse'])
    return model


# # Evaluating the model
# The next step is to build the model. We want to evalute the performance of the model with different parameters. This process is very simple and could be substituted by more interesting exploratory searches. A* returns complete searches, however, we rather suggest to be more practical and use local (neighborhood) searches, like Tabu search, Simulated annealing, ACO or genetic algorithms.

# In[ ]:


def evaluate_model (x_train, y_train, x_val, y_val, optimizer, dropout = 0.3, units = 64, leaky = 0.05, epochs = 10):
    model = generate_model(feature_len = [len(x_train.keys())],
                           optimizer = optimizer,
                           dropout = dropout,
                           units = units,
                           leaky = leaky)
    model.fit(x = x_train,
              y = y_train,
              epochs = epochs,
              validation_split = 0.2,
              verbose = 0)
    loss, mae, mse = model.evaluate(x_val, y_val, verbose=0)
    return mae
    
def evaluate_model_parameters (x_train, y_train, x_val, y_val):
    evaluating_optimizers = [tf.keras.optimizers.RMSprop(0.001),
                             tf.keras.optimizers.Adagrad(learning_rate=0.001),
                             tf.keras.optimizers.Adadelta(0.001),
                             tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
                             ]
    evaluating_dropouts = [0.9, 0.8, 0.7, 0.6, 0.5,0.25,0.1]
    evaluating_units = [32, 64, 128]
    evaluating_leak = [0.05, 0.1, 0.01]
    evaluating_epochs = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    # Basic Search
    best_score = float("inf")
    best_optimizer = evaluating_optimizers[0]
    for optimizer in evaluating_optimizers:
        score = evaluate_model (x_train, y_train, x_val, y_val, optimizer)
        if best_score > score:
            best_score = score
            best_optimizer = optimizer

    best_score = float("inf")
    best_dropout = evaluating_dropouts[0]
    for dropout in evaluating_dropouts:
        score = evaluate_model (x_train, y_train, x_val, y_val, best_optimizer,
                                dropout = dropout)
        if best_score > score:
            best_score = score
            best_dropout = dropout

    best_score = float("inf")
    best_units = evaluating_units[0]
    for units in evaluating_units:
        score = evaluate_model (x_train, y_train, x_val, y_val, best_optimizer,
                                dropout = best_dropout,
                                units = units)
        if best_score > score:
            best_score = score
            best_units = units

    best_score = float("inf")
    best_epochs = evaluating_epochs[0]
    for epochs in evaluating_epochs:
        score = evaluate_model (x_train, y_train, x_val, y_val, best_optimizer,
                                dropout = best_dropout,
                                units = best_units,
                                epochs = epochs)
        if best_score > score:
            best_score = score
            best_epochs = epochs

    best_score = float("inf")
    best_leak = evaluating_leak[0]
    for leaky in evaluating_leak:
        score = evaluate_model (x_train, y_train, x_val, y_val, best_optimizer,
                                dropout = best_dropout,
                                units = best_units,
                                leaky = leaky,
                                epochs = epochs)
        if best_score > score:
            best_score = score
            best_leak = leaky

    return best_optimizer, best_dropout, best_units, best_epochs, best_leak


# # Main script
# Here we call to our functions to load the data, prepocessing the training data, evaluate the model and training it.

# In[ ]:


# Load training data
raw_train = load_raw_data(file_train)
raw_train, raw_val = train_test_split(raw_train, test_size = VALIDATION_SPLIT_SIZE)

# Prepocessing Training Data
feature_columns, categorical_cols, numeric_cols = feature_extract(raw_train)

x_train = prep_features(raw_train, categorical_cols, numeric_cols)        
x_val = prep_features(raw_val, categorical_cols, numeric_cols, x_train)

y_train = raw_train[target]
y_val = raw_val[target]

Evaluating the parameters of the model: optimizer, dropout factor, units in the dense layer, leak factor and epochs in the training.
# In[ ]:


best_optimizer, best_dropout, best_units, best_epochs, best_leak = evaluate_model_parameters(x_train,
                                                                                             y_train,
                                                                                             x_val,
                                                                                             y_val)

print ('Best optimizer: ')
print (best_optimizer)
print ('Best dropout: ' + str(best_dropout))
print ('Best units: ' + str(best_units))
print ('Best leak factor: ' + str(best_leak))
print ('Best epochs: ' + str(best_epochs))


# Trainig the model with the parameters returned from the evaluation

# In[ ]:


model = generate_model(feature_len = [len(x_train.keys())],
                       optimizer = best_optimizer,
                       dropout = best_dropout,
                       units = best_units,
                       leaky = best_leak)
model.fit(x = x_train,
          y = y_train,
          epochs = best_epochs,
          validation_split = 0.2,
          verbose = 0)
loss, mae, mse = model.evaluate(x_val, y_val, verbose=2)
print ('MAE: ' + str(mae))
print ('MSE: ' + str(mse))
print ('Loss: ' + str(loss))


# # Testing the model
# 

# In[ ]:


raw_test = load_raw_data(file_test)
x_test = prep_features(raw_test, categorical_cols, numeric_cols, x_train)
preds = model.predict(x_test).flatten()


# # Saving the output

# In[ ]:


output = pd.DataFrame({'Id': x_test.index,
                       target: preds})
output = output.fillna(y_train.mean())
output.to_csv('submission.csv', index=False)


# # Exporting the model

# In[ ]:


## Saving the model
model.save(model_output_file)

## saving x_train meta data: categorical_cols, numeric_cols
x_train_json = json.dumps({"categorical": categorical_cols, "numerical": numeric_cols}, indent = 4) 
  
# Writing to sample.json 
with open("x_train.json", "w") as outfile: 
    outfile.write(json_object) 


# ## Verifying exported model

# In[ ]:


## Loading the model
model = load_model(model_output_file)

with open('x_train.json', 'r') as openfile: 
    x_train_meta = json.load(openfile) 

preds = model.predict(x=x_test, batch_size=100, verbose=0)
print("Predictions:")
print(preds)

