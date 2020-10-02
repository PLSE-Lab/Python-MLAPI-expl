#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# setting the font size
import matplotlib
matplotlib.rcParams.update({'font.size': 15})

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Data preprocessing

# ## Import data

# In[ ]:


iris = pd.read_csv('/kaggle/input/iris/Iris.csv')
iris.head()


# The **Id** column won't be useful, we will remove it

# In[ ]:


iris.drop('Id', axis=1, inplace=True)
iris.columns


# In[ ]:


target = 'Species'
features = [val for val in iris.columns if val != target]


# ## Perform classification algorithms

# Split the dataset into training and testing datasets

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = iris[features]
y = iris[target]
print(X.shape)
print(y.shape)


# In[ ]:


X = iris[features]
y = iris[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, random_state=12)
for data in ['X_train', 'X_test', 'y_train', 'y_test']:
    print(f'{data}: {eval(data).shape}')


# # Model testing
# 
# We are going to test a few well-known models of classification

# ## Support Vector Machine

# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[ ]:


print(f"Accuracy SVM: {accuracy_score(y_pred, y_test)}")


# ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy Logitic Regression: {accuracy_score(y_pred, y_test)}")


# ## Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy Decision Tree: {accuracy_score(y_pred, y_test)}")


# ## K Nearest Neighbors

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy KNN: {accuracy_score(y_pred, y_test)}")


# In[ ]:


accuracies = []
for n_neighors in range(1, 100, 5):
    model = KNeighborsClassifier(n_neighbors=n_neighors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy {n_neighors} neighbors: {accuracy_score(y_pred, y_test)}")
    accuracies.append(accuracy_score(y_pred, y_test))


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(15, 20)


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(15, 10)

plt.grid()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.plot(accuracies)


# We can see that after a value of **n_neighbors=11**, the accuracy drops 

# We are going to save the accuracy scoring for a specific model in a function called ***calculate_accuracy***

# In[ ]:


def calculate_accuracy(model_name, X_train, y_train, X_test, y_test, print_result=True, **hyperparameters):
    """
    model_name: the name of the model that we want to train and test
    X_train: The training dataframe of variables
    y_train: The training series: the target
    X_test: The testing dataframe of variables
    y_test: The testing series: the target
    **hyperparameters: potential hyperparameters of the model
    """
    model = eval(model_name)(**hyperparameters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    if print_result:
        print(f"Accuracy {model_name} with {hyperparameters}: \n{accuracy}")
    return accuracy


# Let's test it

# In[ ]:


_ = calculate_accuracy('KNeighborsClassifier', 
                       X_train,
                       y_train, 
                       X_test, 
                       y_test, 
                       n_neighbors=2)


# ## Sub-datasets
# 
# The variables can be divided between those of:
# - petals
# - sepals
# 
# Let's see the performances of the two sub-datasets respectively with the variables:
# - PetalLengthCm and PetalWidthCm
# 
# and
# 
# - SepalLengthCm and SepalWidthCm
# 

# In[ ]:


sepal_features = ['SepalLengthCm', 'SepalWidthCm']
petal_features = ['PetalLengthCm', 'PetalWidthCm']


sepal = iris[sepal_features]
petal = iris[petal_features]
y = iris[target]


# In[ ]:


sepal.head()


# In[ ]:


petal.head()


# Let's split both of the dataframes into training and testing sets.
# 
# ***Note:***
# *We set the same seed for both of them*

# In[ ]:


sepal_train, sepal_test, y_train, y_test = train_test_split(sepal, y, 
                                                    test_size=0.3, random_state=12)
petal_train, petal_test, y_train, y_test = train_test_split(petal, y, 
                                                    test_size=0.3, random_state=12)


# Let's now perform the three models that we saw before with Petals and Sepals:
# - Support Vector Machine
# - Logistic Regression
# - K nearest neighbors
# 

# In[ ]:


for model_name in ['SVC', 'DecisionTreeClassifier', 'KNeighborsClassifier']:
    print(model_name)
    print('==================================')
    print('\n\nFor petals:')
    _ = calculate_accuracy(model_name, 
                           X_train=petal_train,
                          X_test=petal_test,
                          y_train=y_train,
                          y_test=y_test)
    print('\nFor sepeals:')   
    _ = calculate_accuracy(model_name, 
                       X_train=sepal_train,
                      X_test=sepal_test,
                      y_train=y_train,
                      y_test=y_test)
    print('\n')


# ## Observation
# 
# - Training the dataset with Petal variables rather than Sepals variables gives a better accuracy
# - Let's only keep the petals variables thereafter

# # Cross-validation

# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


# Looping over the three models
for model_name in ['SVC', 'DecisionTreeClassifier', 'KNeighborsClassifier']:
    print(model_name)
    print('==================================')
    clf = eval(model_name)()
    accuracies = cross_val_score(clf, petal, y, cv=10)
    print('\nResults:')
    mean = sum(accuracies)/len(accuracies)
    print(f'Average accuracy: {mean}')
    var = sum((i - mean) ** 2 for i in accuracies) / len(accuracies)
    print(f'Variance accuracy: {var}')
    print('\n')


# ## Observations
# 
# - The three of these models seem to perform really well
# - For all the models the variance is insignificant
# - KNN seems to be the best

# Let's test different values of neighbors with K-Fold cross-validation with the hole dataset iris

# In[ ]:


avg_accuracies = {}

for n_neighbors in range(1, 100):
    clf= KNeighborsClassifier(n_neighbors=n_neighbors)
    accuracies_kfold = cross_val_score(clf, X, y, cv=10)
    # Calculate the average accuracy over the k-folds
    mean_accuracy = sum(accuracies_kfold)/len(accuracies_kfold)
    avg_accuracies[n_neighbors] = mean_accuracy


# In[ ]:


avg_accuracies_list = sorted(avg_accuracies.items())

n_neighbors,accuracy = zip(*avg_accuracies_list) # unpack a list of pairs into two tuples

fig = plt.gcf()
fig.set_size_inches(15, 10)

plt.grid()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.plot(n_neighbors,accuracy)


# In[ ]:


best_n_neighbors = max(avg_accuracies, key=avg_accuracies.get)
print('The value n_neighbors with the maximum accuracy is: ')
print(f'Max_n_neighbors: {best_n_neighbors} --> Max_accuracy: {avg_accuracies[best_n_neighbors]}')      

