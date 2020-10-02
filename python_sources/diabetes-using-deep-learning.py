#!/usr/bin/env python
# coding: utf-8

# # Pima Indians Diabetes

# # Breakdown of this notebook:
# 1. **Importing Libraries**
# 2. **Loading the dataset:** Load the data and import the libraries
# 3. **Data Cleaning:** <br>
#  - Deleting redundant columns.
#  - Renaming the columns.
#  - Dropping duplicates.
#  - Cleaning individual columns.
#  - Remove the NaN values from the dataset
#  - Some Transformations
# 3. **Traininig the Model**
# 4. **Generate a Classification Report**

# <h3>Importing Libraries</h3>

# In[ ]:


import sys
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import keras

print('Python: {}'.format(sys.version))
print('Pandas: {}'.format(pd.__version__))
print('Numpy: {}'.format(np.__version__))
print('Sklearn: {}'.format(sklearn.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))


import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# **Loading the Dataset**

# In[ ]:


df=pd.read_csv("../input/diabetes.csv")


# In[ ]:


#Describe the dataset
df.describe()


# **DataFrame where the Glucose concentration of a patient is pulled to 0**

# In[ ]:


df[df['Glucose'] == 0]


# In[ ]:


df.info()


# **Removing the Duplicates**

# In[ ]:


df.duplicated().sum()
df.drop_duplicates(inplace=True)


# In[ ]:


df.info()


# **Preprocess the data, mark zero values as NaN and drop**

# In[ ]:


columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in columns:
    df[col].replace(0, np.NaN, inplace=True)
    
df.describe()


# In[ ]:


df.info()


# **Drop rows with missing values**

# In[ ]:


df.dropna(inplace=True)

# summarize the number of rows and columns in df
df.describe()


# In[ ]:


df.info()


# ## Training the Model

# **Convert dataframe to numpy array**

# In[ ]:


dataset = df.values
print(dataset.shape)


# **Split into input (X) and an output (Y)**

# In[ ]:


X = dataset[:,0:8]
Y = dataset[:, 8].astype(int)


# In[ ]:


print(X.shape)
print(Y.shape)
print(Y[:5])


# **Normalize the data using sklearn StandardScaler**

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)


# In[ ]:


print(scaler)


# **Transform and display the training data**

# In[ ]:


X_standardized = scaler.transform(X)

data = pd.DataFrame(X_standardized)
data.describe()


# **Import necessary sklearn and keras packages**

# In[ ]:


from sklearn.model_selection import GridSearchCV, KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam


# **Do a grid search for the optimal batch size and number of epochs**

# In[ ]:


# Define a random seed
seed = 6
np.random.seed(seed)

# Start defining the model
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim = 8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, input_dim = 8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # compile the model
    adam = Adam(lr = 0.01)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

# create the model
model = KerasClassifier(build_fn = create_model, verbose = 1)

# define the grid search parameters
batch_size = [10, 20, 40]
epochs = [10, 50, 100]

# make a dictionary of the grid search parameters
param_grid = dict(batch_size=batch_size, epochs=epochs)

# build and fit the GridSearchCV
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=seed), verbose = 10)
grid_results = grid.fit(X_standardized, Y)

# summarize the results
print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))


# **Do a grid search to find the optimal number of neurons in each hidden layer**

# In[ ]:


# import necessary packages

# Define a random seed
seed = 6
np.random.seed(seed)

# Start defining the model
def create_model(neuron1, neuron2):
    # create model
    model = Sequential()
    model.add(Dense(neuron1, input_dim = 8, kernel_initializer= 'uniform', activation= 'linear'))
    model.add(Dense(neuron2, input_dim = neuron1, kernel_initializer= 'uniform', activation= 'linear'))
    model.add(Dense(1, activation='sigmoid'))
    
    # compile the model
    adam = Adam(lr = 0.001)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

# create the model
model = KerasClassifier(build_fn = create_model, epochs = 100, batch_size = 20, verbose = 0)

# define the grid search parameters
neuron1 = [4, 8, 16]
neuron2 = [2, 4, 8]

# make a dictionary of the grid search parameters
param_grid = dict(neuron1 = neuron1, neuron2 = neuron2)

# build and fit the GridSearchCV
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=seed), refit = True, verbose = 10)
grid_results = grid.fit(X_standardized, Y)

# summarize the results
print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))


# **Generate predictions with optimal hyperparameters**

# In[ ]:


y_pred = grid.predict(X_standardized)


# In[ ]:


print(y_pred.shape)


# In[ ]:


print(y_pred[:5])


# <h2>Generate a classification report</h2>

# In[ ]:


from sklearn.metrics import classification_report, accuracy_score

print(accuracy_score(Y, y_pred))
print(classification_report(Y, y_pred))


#  # <font color='green'>Please upvote if you found this helpful :)</font>
