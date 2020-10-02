#!/usr/bin/env python
# coding: utf-8

# ## Keras Binary Classification
# 
# Use Keras with the standard sonar dataset.
# 
# Dataset describes
# sonar chirp returns bouncing off different surfaces. The 60 input variables are the strength of
# the returns at different angles. It is a binary classification problem that requires a model to
# differentiate rocks from metal cylinders.
# All of the variables are continuous and generally in the
# range of 0 to 1. The output variable is a string M for mine and R for rock, which will need to be
# converted to integers 1 and 0. The dataset contains 208 observations.

# In[ ]:


import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# In[ ]:


# load dataset
dataframe = read_csv("../input/sonar.all-data.csv", header=None)
dataset = dataframe.values
# split into input and output variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
dataframe.head()


# In[ ]:


#one hot encode the targets
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)


# In[ ]:


encoded_Y


# ### Define and evaluate the model
# The weights are initialized using a small Gaussian random number. The Rectifier activation
# function is used. The output layer contains a single neuron in order to make predictions. Use the sigmoid activation function in order to produce a probability output in the range of
# 0 to 1 that can easily and automatically be converted to crisp class values. Use the logarithmic loss function (binary crossentropy) during training, the preferred loss
# function for binary classification problems.

# In[ ]:


# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[ ]:


# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=10, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# ### Data Preperation
# Standardize the data - rescale such that the mean value for each attribute is 0 and the standard
# deviation is 1. This preserves Gaussian and Gaussian-like distributions whilst normalizing the
# central tendencies for each attribute.
# 
# Rather than performing the standardization on the entire dataset, it is
# good practice to train the standardization procedure on the training data within the pass of a
# cross-validation run and to use the trained standardization instance to prepare the unseen test makes standardization a step in model preparation in the cross-validation process
# and it prevents the algorithm having knowledge of unseen data during evaluation, knowledge
# that might be passed from the data preparation scheme like a crisper distribution
# Can use the Pipeline class to define a StandardScaler followed by neural network model.

# In[ ]:


# evaluate baseline model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=10,batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# Standardisation has produced a small lift in performance

# ### Tuning of Network
# #### Try a smaller network
# Take baseline model with 60 neurons above and reduce to 30 - force type of feature extraction by restricting the representational space in the first hidden layer.

# In[ ]:


# baseline model
def create_smaller():
    # create model
    model = Sequential()
    model.add(Dense(30, input_dim=60, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# evaluate baseline model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, epochs=10,batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# Small increase in performance with smaller network, lets try making bigger network

# #### Try a larger network
# Add one new layer to the network - another hidden layer with 30 neurons:
# 60 inputs --> [60 -->30] --> 1 output
# Give the network the opportunity to model all the input variables before being bottlenecked

# In[ ]:


# larger model
def create_larger():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_larger, epochs=10, batch_size=5,verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# Larger network sees increase in performance 

# If you liked this kernel then please upvote

# In[ ]:




