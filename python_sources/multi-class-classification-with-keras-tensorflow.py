#!/usr/bin/env python
# coding: utf-8

# **Dataset Information**: Dataset is extracted from the electric current drive signals. The drive contains intact as well as defective components. Therefore, dataset has 11 classes based on the condition of the components. Aim is, to predict the correct component condition based on the input variables using **Deep Learning** technique. Tools used: **Keras TensorFlow** 

# **Dataset Rights**: This dataset has been taken from "University of California Irvine Machine Learning Repository" for the knowledge purpose and all the rights for this dataset are reserved by them. For more details like content of the dataset, owner of the dataset and reference research paper, please refer the following link: https://archive.ics.uci.edu/ml/datasets/Dataset+for+Sensorless+Drive+Diagnosis

# In[1]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
#from keras.utils import np_utils
#from sklearn.preprocessing import LabelEncoder


# Step 2. Import and print data

# In[3]:


# Random seed for reproducibility
seed = 10
np.random.seed(seed)
# Import data
df = pd.read_csv('../input/Sensorless_drive_diagnosis.csv', sep = ' ', header = None)
# Print first 10 samples
print(df.head(10))


# Column indices 0 to 47 are input variables (total 48 columns). Column index 48 is target column that contains 11 different classes (1 column). 

# Step 3. Data pre-processing

# In[4]:


# Check missing values
print(df.isna().sum())


# No missing values. 

# In[5]:


# Remove missing values IF AVAILABLE and print first 10 samples
# df = df.dropna()
# print(df.head(10))
# print(df.shape)


# In[7]:


# Divide data into features X and target (Classes) Y
X = df.loc[:,0:47]
Y = df.loc[:,48]
print(X.shape)
print(Y.shape)


# In[8]:


# Statistical summary of the variables
print(X.describe())


# Scale of all the variables is different. Therefore, feature scaling is important.    

# In[9]:


# Check for class imbalance
print(df.groupby(Y).size())


# Since all the classes have same sample size, there is no class imbalance. 

# In[11]:


# Normalize features within range 0 (minimum) and 1 (maximum)
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
X = pd.DataFrame(X)


# To train the Neural Network, single target column must be converted into one hot encoded fomat. For more details, visit this link: https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/

# In[13]:


# Convert target Y to one hot encoded Y for Neural Network
Y = pd.get_dummies(Y)
# If target is in string form, use following code:
# First encode target values as integers from string
# Then perform one hot encoding
# encoder = LabelEncoder()
# encoder.fit(Y)
# Y = encoder.transform(Y)
# Y = np_utils.to_categorical(Y)


# In[14]:


# For Keras, convert dataframe to array values (Inbuilt requirement of Keras)
X = X.values
Y = Y.values


# Step 4. Define Neural Network Model 

# Two hidden layers are defined with "Rectified Linear Unit" (relu) and 15 neurons each. Furthermore, this is a multi-class classification problem and there are total 11 target clsses, therefore "softmax" activation function and 11 neurons are used in the output layer. For hidden layers, the number of neurons should be in between the input data dimension and the output data dimension. In this case, the input data has 48 variable columns and output classes are 11. Therefore, the number of neurons for the hidden layer should be in between 11 and 48. You can try different values for the number of neurons as well as different number of hidden layers.  

# In[19]:


# First define baseline model. Then use it in Keras Classifier for the training
def baseline_model():
    # Create model here
    model = Sequential()
    model.add(Dense(15, input_dim = 48, activation = 'relu')) # Rectified Linear Unit Activation Function
    model.add(Dense(15, activation = 'relu'))
    model.add(Dense(11, activation = 'softmax')) # Softmax for multi-class classification
    # Compile model here
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model


# Input dimension (input_dim) is 48, because the input variable columns are 48. It will change as per the dimension of the input variables.

# **Note**: If you use only one hidden layer, then it will be the case of simple Neural Network problem. But, if you use more than one hidden layers for example 3, it will be considered as the deep learning problem.  

# In[21]:


# Create Keras Classifier and use predefined baseline model
estimator = KerasClassifier(build_fn = baseline_model, epochs = 100, batch_size = 10, verbose = 0)
# Try different values for epoch and batch size


# Step 5. Define cross-validation and train pre-defined model 

# In[24]:


# KFold Cross Validation
kfold = KFold(n_splits = 5, shuffle = True, random_state = seed)
# Try different values of splits e.g., 10


# In[25]:


# Object to describe the result
results = cross_val_score(estimator, X, Y, cv = kfold)
# Result
print("Result: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# From the result above, the accuracy is 94% and it can be improved by techniques like feature extraction, selection and feature engineering. 
