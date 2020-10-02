#!/usr/bin/env python
# coding: utf-8

# **Load Data**

# In[26]:


from keras.models import Sequential
from keras.layers import Dense 
from keras.models import model_from_json
import numpy
import os
import pandas 

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


data = pandas.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
array = data.values
X = array[:,0:8]
Y = array[:,8]

data.head(5)


# **Load Saved Model Architecture and Weights**

# In[27]:


#load json and create model
json_file = open('../input/runmila-diabetes-prediction/diabetes_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

#load weights into new model
loaded_model.load_weights("../input/runmila-diabetes-prediction/diabetes_model.h5")

loaded_model.summary()


# **Use Trained Model to Make Predictions and Estimations**

# In[34]:


# Calculate Predictions
predictions = loaded_model.predict(X)


# round predictions
rounded = [round(x[0],1) for x in predictions]

#Not Rounded For Non-Linear Outputs as Probability
print(rounded)

