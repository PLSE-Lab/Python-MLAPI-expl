#!/usr/bin/env python
# coding: utf-8

# #Iris Datset

# In[1]:


import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

print('Python: {}'.format(sys.version))
print('scipy: {}'.format(scipy.__version__))
print('numpy: {}'.format(numpy.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('pandas: {}'.format(pandas.__version__))
print('sklearn: {}'.format(sklearn.__version__))


# ## 1. Import libraries
# 
# Import all of the modules, functions, and objects we will use in this tutorial.

# In[2]:


from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# ## 2. Load the Dataset
# 
# We will be using the iris flowers dataset, which contains 150 observations of iris flowers. There are four columns of measurements and the species of flower observed.  Only three species are present in this dataset.
# 
# The data can be loaded directly from the UCI Machine Learning Repository

# In[8]:


# Load Dataset
dataset=pandas.read_csv('../input/Iris.csv')


# ## 2.1 Dataset Properties
# 
# Lets take a look at the dataset by observing its dimensions, the first few rows of data, a statistical summary of the attributes, and a breakdown of the data by the class variable.

# In[17]:


del dataset['Id']


# In[18]:


# Shape
print(dataset.shape)


# In[19]:


# Head
print(dataset.head(20))


# In[20]:


# descriptions
print(dataset.describe())


# In[21]:


# class distribution
print(dataset.groupby('Species').mean())


# In[22]:


print(dataset.groupby('Species').size())


# In[23]:


print(dataset.groupby('Species').median())


# ## 2.2 Data Visualizations
# 
# Lets visualize the data so we can understand the distribution of the input attributes. We will use histograms of each attribute, as well as some multivariate plots so that we can view the interactions between variables.

# In[24]:


# histograms
dataset.hist(color='green')
plt.show()


# In[25]:


# scatter plot matrix
scatter_matrix(dataset)
plt.show()


# ## 3. Evaluate Algorithms
# 
# Lets create some models of the data and estimate their accuracy on unseen data.
# 
# We are going to,
# 
# * Create a validation dataset
# * Set-up cross validation
# * Build three different models to predict species from flower measurement
# * Select the best model
# 
# ## 3.1 Create Validation Dataset
# 
# Lets split the loaded dataset into two.  80% of the data will be used for training, while 20% will be used for validation.

# In[27]:


# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.30
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)


# ## 3.2 10-fold Cross Validation
# 
# This will split our dataset into 10 parts, train on 9 and test on 1 and repeate for all combinations of train-test splits

# In[28]:


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'


# ## 3.3 Build Models
# 
# Lets evaluate three models:
# 
# * Logistic Regression (LR)
# * K-Nearest Neighbors (KNN)
# * Support Vector Machine (SVM)

# In[29]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# ## 4. Make Predictions
# 
# Lets test the model on the validation set to make sure that our algorithms can generalize to new data.  Otherwise, we may be overfitting the training data.  

# In[30]:


# Make predictions on validation dataset

for name, model in models:
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    print(name)
    print(accuracy_score(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))


# In[31]:


# if u find it helpful please upvote the notebook

