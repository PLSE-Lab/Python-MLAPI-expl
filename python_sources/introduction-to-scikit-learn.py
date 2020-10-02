#!/usr/bin/env python
# coding: utf-8

# <h1><center> Introduction to Scikit-Learn and Pandas </center></h1>
# <center>**November 2018**</center>
# <br><br>
# + Scikit-learn <br>
# [Scikit-learn](http://scikit-learn.org/stable/) (formerly scikits.learn) is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.
# <br>
# + Matplotlib <br>
# [Matplotlib](https://matplotlib.org/) is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits like Tkinter, wxPython, Qt, or GTK+. There is also a procedural "pylab" interface based on a state machine (like OpenGL), designed to closely resemble that of MATLAB, though its use is discouraged.
# <br>
# + Pandas <br>
# [Pandas](https://pandas.pydata.org/) (Python Data Analysis Library) is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language. pandas is a NumFOCUS sponsored project. This will help ensure the success of development of pandas as a world-class open-source project, and makes it possible to donate to the project.

# # Import Modules

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# for splitting data to train and test partition
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

# Classifier for modelling
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Evaluates model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Sample data provided by sklearn
from sklearn.datasets import load_iris


# # Load, Preprocessing and Preview Data

# In[ ]:


# load data from scikit learn datasets
iris = load_iris()
print(iris.DESCR)
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
df.head()


# In[ ]:


# observe data shape and target unique values
print("Data frame shape :", df.shape)
print("target unique values :", df['target'].unique())


# In[ ]:


# checking null value
for attribute in df.columns:
    print("column", attribute, "null \t:", df[attribute].isnull().sum())


# In[ ]:


# load data from external file
tennis = pd.read_csv('../input/weather.nominal.csv')
tennis.head()


# # Modelling

# In this notebook, we will use 3 modelling method: 
# * Full Training : all of the sample data will be used as training data. Prediction is not a part of this method
# * Hold-out : split the data sample to two parts (training data and test data)
# * Cross-validation : the data will be split randomly into *k* group, one group will be used as train data, and the rest are used as test data

# ## 1. Full training

# ### Naive Bayes

# In[ ]:


# Gaussian
clf_gnb = GaussianNB()

clf_gnb.fit(iris.data, iris.target)


# ### Decision Tree

# In[ ]:


# fitting
clf_tree = tree.DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=6)

clf_tree.fit(iris.data, iris.target)


# In[ ]:


# visualize model
import graphviz
dot_data = tree.export_graphviz(clf_tree, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data) 
graph
# graph.render("iris") 


# ### kNN

# In[ ]:


clf_neigh = KNeighborsClassifier(n_neighbors=3)

clf_neigh.fit(iris.data, iris.target)


# ### Neural Network (MLP)

# In[ ]:


clf_neuron = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=42, max_iter=1000, solver='lbfgs')

clf_neuron.fit(iris.data, iris.target)


# ## 2. Hold-Out

# In[ ]:


# Spltting
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1, random_state=42)
class_names = iris.target_names


# In[ ]:


# plotting confusion matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# ### Naive Bayes

# In[ ]:


# Gaussian
clf_gnb = GaussianNB()

clf_gnb.fit(X_train, y_train)

pred_gnb = clf_gnb.predict(X_test)
print("accuracy score \t=", accuracy_score(y_test, pred_gnb, normalize=True))
print("recall score \t=", recall_score(y_test, pred_gnb, average='micro'))
print("f1 score \t=", f1_score(y_test, pred_gnb, average='micro'))


# In[ ]:


cnf_matrix_gnb = confusion_matrix(y_test, pred_gnb)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix_gnb, classes=class_names,normalize=True,
                      title='GaussianNB Confusion Matrix')

plt.show()


# ### Decision Tree

# In[ ]:


# fitting
clf_tree = tree.DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=3)

clf_tree.fit(X_train, y_train)

pred_tree = clf_tree.predict(X_test)
print("accuracy score =", accuracy_score(y_test, pred_tree, normalize=True))
print("recall score \t=", recall_score(y_test, pred_tree, average='micro'))
print("f1 score \t=", f1_score(y_test, pred_tree, average='micro'))


# In[ ]:


cnf_matrix_tree = confusion_matrix(y_test, pred_tree)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix_tree, classes=class_names,normalize=True,
                      title='DecisionTree Confusion Matrix')

plt.show()


# ### kNN

# In[ ]:


clf_neigh = KNeighborsClassifier(n_neighbors=3)

clf_neigh.fit(X_train, y_train)

pred_neigh = clf_neigh.predict(X_test)
print("accuracy score \t=", accuracy_score(y_test, pred_neigh, normalize=True))
print("recall score \t=", recall_score(y_test, pred_neigh, average='micro'))
print("f1 score \t=", f1_score(y_test, pred_neigh, average='micro'))


# In[ ]:


cnf_matrix_neigh = confusion_matrix(y_test, pred_neigh)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix_neigh, classes=class_names,normalize=True,
                      title='kNN Confusion Matrix')

plt.show()


# ### Neural Network (MLP)

# In[ ]:


clf_neuron = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=42, max_iter=1000, solver='lbfgs')

clf_neuron.fit(X_train, y_train)

pred_neuron = clf_neuron.predict(X_test)
print("accuracy score \t=", accuracy_score(y_test, pred_neuron, normalize=True))
print("recall score \t=", recall_score(y_test, pred_gnb, average='micro'))
print("f1 score \t=", f1_score(y_test, pred_gnb, average='micro'))


# In[ ]:


cnf_matrix_neuron = confusion_matrix(y_test, pred_neuron)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix_neuron, classes=class_names,normalize=True,
                      title='NeuralNetwork Confusion Matrix')

plt.show()


# ## 3. Cross-Validation

# In[ ]:


## splitting and cross validate
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, cross_validate

X = iris.data
y = iris.target
kf = KFold(n_splits=10, random_state=False)
print(kf.get_n_splits())

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# ### Naive Bayes

# In[ ]:


# Gaussian
clf_gnb = GaussianNB()

clf_gnb.fit(X_train, y_train)

gnb_cv_score = cross_val_score(clf_gnb, X_train, y_train, cv = 10)
print("cross validate score =", gnb_cv_score)
pred_gnb = cross_val_predict(clf_gnb, X_test, y_test, cv=10)
print("accuracy score =", accuracy_score(y_test, pred_gnb, normalize=True))


# ### Decision Tree

# In[ ]:


# fitting
clf_tree = tree.DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=6)

clf_tree.fit(X_train, y_train)

tree_cv_score = cross_val_score(clf_tree, X_train, y_train, cv = 10)
print("cross validate score =", tree_cv_score)
pred_tree = cross_val_predict(clf_tree, X_test, y_test, cv=10)
print("accuracy score =", accuracy_score(y_test, pred_tree, normalize=True))


# ### k-Nearest Neighbors (kNN)

# In[ ]:


clf_neigh = KNeighborsClassifier(n_neighbors=3)

clf_neigh.fit(X_train, y_train)

neigh_cv_score = cross_val_score(clf_neigh, X_train, y_train, cv = 10)
print("cross validate score =", neigh_cv_score)
pred_neigh = cross_val_predict(clf_neigh, X_test, y_test, cv=10)
print("accuracy score =", accuracy_score(y_test, pred_neigh, normalize=True))


# ### Neural Network (MLP)

# In[ ]:


clf_neuron = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=42, max_iter=1000, solver='lbfgs')

clf_neuron.fit(X_train, y_train)

neuron_cv_score = cross_val_score(clf_neuron, X_train, y_train, cv = 10)
print("cross validate score =", neuron_cv_score)
pred_neuron = cross_val_predict(clf_neuron, X_test, y_test, cv=10)
print("accuracy score =", accuracy_score(y_test, pred_neuron, normalize=True))


# ## Save the Model

# In[ ]:


# Save the model using Pickle Library
import pickle

models = []
models.append(clf_gnb)
models.append(clf_tree)
models.append(clf_neigh)
models.append(clf_neuron)

pkl_filename = 'model-iris.pkl'
with open(pkl_filename, 'wb') as file:  
    for model in models:
        pickle.dump(model, file)


# ## Load Model

# In[ ]:


models = []
pkl_filename = 'model-iris.pkl'
with open(pkl_filename, 'rb') as file:
    while True:
        try:
            models.append(pickle.load(file))
        except EOFError:
            break


# In[ ]:


loaded_gnb = models[0]
loaded_tree = models[1]
loaded_neigh = models[2]
loaded_neuron = models[3]


# ## Adding an Instance

# In[ ]:


import random as rd

# get extreme value from each attribute, and make a random instance
max_v = np.amax(X, axis = 0)
min_v = np.amin(X, axis = 0)
new_instance = [round(rd.uniform(min_v[0],max_v[0]),1),
               round(rd.uniform(min_v[1],max_v[1]),1),
               round(rd.uniform(min_v[2],max_v[2]),1),
               round(rd.uniform(min_v[3],max_v[3]),1)]
print(new_instance)


# In[ ]:


# predicting the new instance 
new_i_pred1 = loaded_gnb.predict([new_instance])
print('new instance prediction using Naive bayes:', iris.target_names[new_i_pred1])

new_i_pred2 = loaded_tree.predict([new_instance])
print('new instance prediction using DecisionTree:', iris.target_names[new_i_pred2])

new_i_pred3 = loaded_neigh.predict([new_instance])
print('new instance prediction using kNN:', iris.target_names[new_i_pred3])

new_i_pred4 = loaded_neuron.predict([new_instance])
print('new instance prediction using NeuralNetwork:', iris.target_names[new_i_pred4])

