#!/usr/bin/env python
# coding: utf-8

# # Using ML/DL to predict Pulsar Star
# 
# In this notebook, I'll explore the dataset of pulsar stars and use multiple Machine and Deep learning models to classify between pulsar and other stars.

# ## Import libraries

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.preprocessing import StandardScaler


# ## Import dataset
# 
# I'll import the dataset and take a quick view about the details of the dataset.

# In[ ]:


dataset = pd.read_csv('../input/predicting-a-pulsar-star/pulsar_stars.csv')
dataset.head(5)


# In[ ]:


dataset.isnull().sum()


# There are no null values in the dataset, thus I can proceed with direcly working with the complete dataset.

# In[ ]:


dataset.describe().T


# As is clear from the table above, the values in each column vary across different ranges and thus, using scaling on the dataset would really help in model training and prediction without bias towards any specific feature.

# ## Exploratory Data Analysis
# 
# I'll next take a look at the dataset and its various features.

# In[ ]:


plt.figure(figsize = (12, 8))
plot = sns.countplot(dataset['target_class'])
plot.set_title("Target Class count")
for p in plot.patches:
    plot.annotate('{}'.format(p.get_height()), xy = (p.get_x() + 0.35, p.get_height() + 40))


# The dataset is highly imbalanced. The number of non-pulsar data points is approcimately 10 times the number of pulsar data points.

# In[ ]:


plt.figure(figsize = (12, 8))
sns.heatmap(dataset.corr(), annot = True, fmt = ".2f")


# From the heatmap, we can see that `Mean of the integrated profile`, `Excess kurtosis of the integrated profile` and `Skewness of the integrated profile` are highly correlated features to the target class.

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(dataset.iloc[:, :-1])
principalDf = pd.DataFrame(data = principalComponents, columns = ['Principal component 1', 'Principal component 2'])
principalDf = pd.concat([principalDf, dataset.iloc[:, -1]], axis = 1)


# In[ ]:


plt.figure(figsize = (20, 12))
sns.scatterplot(x = 'Principal component 1', 
                y = 'Principal component 2', 
                data = principalDf,
               hue = 'target_class')


# barring a few outliers, the PCA analysis reveals that the data is quite separable when we consider just two principal features.

# ## Machine and Deep Learning
# 
# I'll now explore two machine learning models and one Artificial Neural Network to classify between the stars. But first, I'll split the data into training (70%) and testing (30%) data.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:, :-1], dataset.iloc[:, -1], random_state = 0, test_size = 0.3)


# I'll define a `metrics` method which would allow us to easily get the accuracy and confusion matrix.

# In[ ]:


def metrics(model, y_true, y_pred):
    print("The accuracy of the model {} is: {:.2f}%".format(model, accuracy_score(y_true, y_pred)*100))
    print("Confusion matrix for {}".format(model))
    print(confusion_matrix(y_true, y_pred))
    print("-"*40)


# Given the size and range of each column is varied, it's always a good practice to scale the data.

# In[ ]:


standardScaler = StandardScaler()
X_train = standardScaler.fit_transform(X_train)
X_test = standardScaler.transform(X_test)


# ### Model creation and training
# 
# I'll create a Support Vector Classifier, Random Forest Classifier and an Artificial Neural Network.

# In[ ]:


# Support Vector Classifier
supportVectorClassifier = SVC(kernel = 'rbf')
supportVectorClassifier.fit(X_train, y_train)

# Random Forest Classifier
randomForestClassifier = RandomForestClassifier(n_estimators = 100)
randomForestClassifier.fit(X_train, y_train)

# Artificial Neural Network
artificialNeuralNetwork = Sequential()
artificialNeuralNetwork.add(Dense(units = 32, activation = 'relu', input_dim = 8))
artificialNeuralNetwork.add(Dropout(0.5))
artificialNeuralNetwork.add(Dense(units = 64, activation = 'relu'))
artificialNeuralNetwork.add(Dropout(0.5))
artificialNeuralNetwork.add(Dense(units = 128, activation = 'relu'))
artificialNeuralNetwork.add(Dropout(0.5))
artificialNeuralNetwork.add(Dense(units = 1, activation = 'sigmoid'))
artificialNeuralNetwork.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
artificialNeuralNetwork.fit(X_train, y_train, epochs = 50, shuffle = False, validation_split = 0.1, verbose = 0)


# ### Training and results
# 
# Let's now test the models and see the accuracy and confusion matrices.

# In[ ]:


# Support Vector Classifier
metrics("Support Vector Classifier", y_test, supportVectorClassifier.predict(X_test)) 

# Random Forest Classifier
metrics("Random Forest Classifier", y_test, randomForestClassifier.predict(X_test)) 

# Artificial Neural Network
metrics("Artificial Neural Network", y_test, (artificialNeuralNetwork.predict(X_test) > 0.5))


# Taking a look at the results above we can see that **Random Forest Classifier** performs the best classification. It's amazing how it's able to classify both pulsar and non-pulsar stars with greater accuracy as can be seen from the confusion matrix.
