#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<iframe width="560" height="315" src="https://www.youtube.com/embed/anRHgbYy7PE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')


# 
# 
# ****WINES****
# 
# ****Problem Question****
# Is it possible to predict if the Wine type in terms of if it is Red or White by just known about the quantity components?
# 
# ****Objective****
# 
# Build a NN model using Keras to predict the type of wine(Red or White) using 12 features to feed the model.

# In[ ]:


import numpy as np                # linear algebra
import pandas as pd               # data frames
import seaborn as sns             # visualizations
import matplotlib.pyplot as plt   # visualizations
import scipy.stats                # statistics
import sklearn.cluster as sk
import plotly as plotly
from sklearn import preprocessing


import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv("../input/wine.csv",sep=";")

# Print the head of df
print(df.head())

# Print the info of df
print(df.info())

# Print the shape of df
print(df.shape)


# In[ ]:


df.describe()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(df[['fixed acidity', 'volatile acidity', 'citric acid',      
        'residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH',
        'sulphates','alcohol','quality']])
plt.show()


# In[ ]:


# Correlation matrix
corr=df.iloc[:,0:13].corr()

# Mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Diverging colormap
cmap = sns.diverging_palette(50, 80, as_cmap=True)

# Heatmap with the mask and correct aspect ratio
sns.heatmap(corr, annot=True, mask=mask, vmax=0.5,linewidths=0.1)


# In[ ]:


#Scaling 
df_scale = df.copy()
scaler = preprocessing.StandardScaler()
columns =df.columns[0:13]
df_scale[columns] = scaler.fit_transform(df_scale[columns])
df_scale.head()
df_scale = df_scale.iloc[:,0:13]


# In[ ]:


#Sampling
sample = np.random.choice(df_scale.index, size=int(len(df_scale)*0.8), replace=False)
train_data, test_data = df_scale.iloc[sample], df_scale.drop(sample)

print("Number of training samples is", len(train_data))
print("Number of testing samples is", len(test_data))
print(train_data[:13])
print(test_data[:13])


# In[ ]:


#Features and Target
features = train_data.drop('Wine', axis=1)
targets = train_data['Wine']
targets = targets > 0.5
features_test = test_data.drop('Wine', axis=1)
targets_test = test_data['Wine']
targets_test = targets_test > 0.5


# In[ ]:


# Activation (sigmoid) function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))
def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)
def error_term_formula(y, output):
    return (y-output) * output * (1 - output)


# In[ ]:


# Neural Network hyperparameters
epochs = 9000
learnrate = 0.3

# Training function
def train_nn(features, targets, epochs, learnrate):
    
    # Use to same seed to make debugging easier
    np.random.seed(42)

    n_records, n_features = features.shape
    last_loss = None

    # Initialize weights
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
    print(weights.shape)

    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features.values, targets):
            # Loop through all records, x is the input, y is the target

            # Activation of the output unit
            #   Notice we multiply the inputs and the weights here 
            #   rather than storing h as a separate variable 
            output = sigmoid(np.dot(x, weights))

            # The error, the target minus the network output
            error = error_formula(y, output)

            # The error term
            #   Notice we calulate f'(h) here instead of defining a separate
            #   sigmoid_prime function. This just makes it faster because we
            #   can re-use the result of the sigmoid function stored in
            #   the output variable
            error_term = error_term_formula(y, output)

            # The gradient descent step, the error times the gradient times the inputs
            del_w += error_term * x

        # Update the weights here. The learning rate times the 
        # change in weights, divided by the number of records to average
        weights += learnrate * del_w / n_records

        # Printing out the mean square error on the training set
        if e % (epochs / 10) == 0:
            out = sigmoid(np.dot(features, weights))
            loss = np.mean((out - targets) ** 2)
            print("Epoch:", e)
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            print("=========")
    print("Finished training!")
    return weights
    
weights = train_nn(features, targets, epochs, learnrate)


# In[ ]:


# Calculate accuracy on test data
tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
#predictions = tes_out
accuracy = np.mean((predictions == targets_test))
print("Prediction accuracy: {:.3f}".format(accuracy))


# In[ ]:


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

# Building the model
model = Sequential()
model.add(Dense(500, activation='relu', input_shape=(12,)))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='softmax'))


# In[ ]:


# Compiling the model
model.compile(loss = 'binary_crossentropy', optimizer='sgd', metrics=['accuracy'])


# In[ ]:


# Training the model
model.fit(features, targets, epochs=20, batch_size=1, verbose=1)


# In[ ]:


score= model.evaluate(features_test, targets_test,verbose=1)
score

