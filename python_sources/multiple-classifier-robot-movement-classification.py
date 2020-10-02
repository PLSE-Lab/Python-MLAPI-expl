#!/usr/bin/env python
# coding: utf-8

# # Multiple Classifier Robot Movement Classification
# 
# * by [Ayo Ayibiowu](https://thehapyone.com)

# ## Pattern Recognition
# 
# In this task, the goal is to be able to predict the position label of a moving robot based on the sensor readings from the robot during its movement.
# 
# The data is collected during the course of a robot navigating through a room following the wall in a clockwise direction, for 4 rounds. 
# 
# Also, I attempted in trying out several models. The dataset (sensor reading) is non-linear in nature

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Importing the libraries needed
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy
from sklearn.svm import SVC

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# importing the dataset
data = np.loadtxt("/kaggle/input/wall-following-robot/sensor_readings_24.csv", delimiter=',', dtype=np.str)

raw_data = pd.DataFrame(data[:,:24], dtype=np.float)
raw_data = pd.concat([raw_data, pd.DataFrame(data[:, 24], columns=['Label'])], axis=1)
                      
print("Data size - ", raw_data.shape)


# In[ ]:


print ("Sample raw data")
raw_data.head(n=10)


# In[ ]:


# Describe the dataset
raw_data.describe()
# from the nature of the data, it can be seen that we don't need to normalize it.


# In[ ]:


# Evaluating features (sensors) contribution towards the label
fig = plt.figure(figsize=(15,5))
ax = sns.countplot(x='Label',data=raw_data,alpha=0.5)


# # Observation
# From the plot above, it is evident of a class inblance happening. This inbalance might influence our result[](http://)

# In[ ]:


# label count
raw_data.groupby(['Label']).count()[0]


# In[ ]:


# pair plot of the data
colss = raw_data.columns[0:24]
sns.pairplot(raw_data, vars=colss, hue='Label')


# # Feature Selection
# The fisher score is a good discrimating function for finding features that contributes more towards the data. 
# The code below shows how to calculate it. You can find the latest fisher calculation code [here](https://gist.github.com/e55758727bce8c5acc7ca6785ad63a5f) - https://gist.github.com/e55758727bce8c5acc7ca6785ad63a5f

# In[ ]:


# helper function for evalating the fisher ndex
def fisher_index_calc(trainingSet, labelSet):
    (dim1_T, dim2_T) = trainingSet.shape
    (dim1_L, dim2_L) = labelSet.shape

    # create the fisher output variable - A vector of all the features
    fisher_ratios = np.zeros((1, dim2_T), dtype=float).flatten()
    # It's expected that the dim1_T and dim1_L be of the same size, else this input parameters is nulled.
    if dim1_L != dim1_T:
        return fisher_ratios

    # First extract out the number of features available.
    # grouped both data together, and create a pandas dataframe from it.
    train1 = pd.DataFrame(trainingSet)
    label1 = pd.DataFrame(labelSet, columns=['LABEL'])
    grouped = pd.concat([train1, label1], axis=1)

    # fetch the number of classes
    (no_classes, demo) = grouped.groupby('LABEL').count()[[0]].shape
    #print grouped

    # loop through all features
    for j in range(dim2_T):
        # the variance of the feature j
        j_variance = np.var(trainingSet[:,j])
        j_mean = np.mean(trainingSet[:,j])
        j_summation = 0
        for k in range(no_classes):
            output = grouped.groupby('LABEL').count()[[j]]
            k_feature_count = output.iloc[k,0]
            # mean for class k of feature j
            output = grouped.groupby('LABEL').mean()[[j]]
            k_feature_mean = output.iloc[k,0]
            currentSum = k_feature_count * np.square((k_feature_mean - j_mean))
            j_summation = j_summation + currentSum
        fisher_ratios[j] = j_summation / np.square(j_variance)

    return fisher_ratios


# In[ ]:


training_set = raw_data.iloc[:, :(raw_data.shape[1]-1)].values
label_set = raw_data.iloc[:, (raw_data.shape[1]-1):].values


# In[ ]:


# calculates the fisher score of the sensors features
fisher_scores = fisher_index_calc(training_set, label_set)

fig= plt.figure(figsize=(23, 10))
df = pd.DataFrame({'Fisher Ratio For All Features': fisher_scores})
ax = df.plot.bar(figsize=(20,10))
plt.show()


# # Feature Selection.
# Fisher Score is a good discrimintating attribute for features. Here, features with scores less than 300 are discarded. 
# > **Pro - Tip**: The fisher based score helps to improve the performance of the model.

# In[ ]:


# feature selection based on fisher score
# Fisher Index Ratio Filter - Remove features with low score
# indices of features to remove based on fisher ratios
to_remove = []
for i in range((len(fisher_scores))):
    if fisher_scores[i] < 300:
        # we mark for removal
        to_remove.append(i)

# remove features with low fisher score
training_set_fisher = np.delete(training_set, to_remove, 1)
training_set_fisher.shape
# ihave about 18 features left.
#print "fisher - ", fisher_ratios


# In[ ]:


# encoding the label set with a label encoder
from sklearn.preprocessing import LabelEncoder

labelEn = LabelEncoder()
encoded_labels = labelEn.fit_transform(raw_data.iloc[:, 24].values)
class_names = labelEn.classes_
class_names


# In[ ]:


# performaing PCA

from sklearn.decomposition import PCA

pca = PCA()

pca_sets = pca.fit_transform(training_set)
print (100 * np.sum(pca.explained_variance_ratio_))

x1 = np.arange(1, pca.explained_variance_ratio_.shape[0]+1, 1)
fig = plt.figure(figsize=(20,5))
sns.barplot(x = x1, y = 100 * pca.explained_variance_ratio_)


# In[ ]:


# normalizaling the data with standard scaler
### The data doesn't require to be normalize, it already looks normalized

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scalled_set = scaler.fit_transform(training_set)


# # Scalling and PCA Observation
# After testing several models based on the cross validation, doing PCA and Standard Scaler doesn't contribute any much difference to the model.
# Moreover, the dataset already looks scaled properly. 

# In[ ]:


# Splitting the data for Training and Testing
from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(training_set_fisher, encoded_labels, test_size=0.3, shuffle=False)


# # Class Inbalance
# The output class for this dataset is highly inbalance which has the chance of creating basis in the model. I created class weights to compensate for this inbalance. 
# This weight can be used in a model.

# In[ ]:


from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.testing import assert_almost_equal

# this function creats a class weight

def create_class_weight(y):
    classes = np.unique(y)
    cw = compute_class_weight("balanced", classes, y)
    # evaluate if weights are truly balanced
    # total effect of samples is preserved
    class_counts = np.bincount(y)
    # print (class_counts)
    assert_almost_equal(np.dot(cw, class_counts), y.shape[0])
    assert cw[0] < cw[1] < cw[2]   

    return cw

label_weights = create_class_weight(encoded_labels)
# Convert class_weights to a dictionary to pass it to class_weight in model.fit
label_weights = dict(enumerate(label_weights))


# In[ ]:


# function for confusion matrix

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# # Logistic Regression
# This is the first classifier used here. The performance is quite poor. The task is non-linear

# In[ ]:


# Cross validation with Logistic Regression
'''
# Set the parameters by cross-validation
c_range = np.arange(1,100,5)
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

# Hyper Paremters to be used for Model 1
tuned_parameters = {'C': c_range, 'solver': solver}

# defining the model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer

lg_model = LogisticRegression(max_iter = 1000, multi_class = 'auto', n_jobs = -1, random_state = 0, class_weight= label_weights)

# Creates a GridSearch Classifier using parameters for model 1
lg_grid = GridSearchCV(lg_model, tuned_parameters, cv=10, scoring=make_scorer(accuracy_score), n_jobs=-1)

# commence training - NOTE: It takes somes time to be complete
lg_grid.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(lg_grid.best_estimator_)
print ("Best Params CV1 - ", lg_grid.best_params_)
print ("Best score - ", lg_grid.best_score_)
print()

'''


# In[ ]:


# defining the model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer

lg_model = LogisticRegression(C= 86, solver='liblinear', max_iter = 1000, multi_class = 'auto', random_state = 0, class_weight= label_weights)

# commence training -
lg_model.fit(X_train, y_train)

# predict the result
y_pred = lg_model.predict(X_test)
print ("Logisitic Regression Result Considering Class Inbalance")
print ("Performance - " + str(100*accuracy_score(y_pred, y_test)) + "%")

### Without considering the class inbalance
lg_model = LogisticRegression(C= 86, solver='liblinear', max_iter = 1000, multi_class = 'auto', random_state = 0)

# commence training -
lg_model.fit(X_train, y_train)

# predict the result
y_pred = lg_model.predict(X_test)
print ("Logisitic Regression Result Without considering Class Inbalance")
print ("Performance - " + str(100*accuracy_score(y_pred, y_test)) + "%")


# In[ ]:


# Plot non-normalized confusion matrix

plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix For MLP Model')


# # Multi-layer Perceptron
# MLP can handle both linear and non-linear task. MLP performed best here and it was able to handle the class inbalance case even without supplying a class weight.

# In[ ]:


# Cross validation with MLP
'''
# Set the parameters by cross-validation
# number of neurons to use for the hidden layers
n_neurons_range = np.arange(5,80,5)
n_neurons = [(5,5), (10,10), (20,20), (30,30), (40,40), (50,50), (60,60), (65,65), (40,40,40)]
a_param_range = 10.0 ** -np.arange(1, 7)
# Hyper Paremters to be used for Model 1
tuned_parameters = {'solver': ['lbfgs', 'adam'], 'activation': ['tanh', 'logistic', 'relu'], 'alpha': a_param_range,
                     'hidden_layer_sizes': n_neurons_range}

# Creates the MLP Classifier
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(random_state=1, early_stopping=True, max_iter=1000)

# Creates a GridSearch Classifier using parameters for model 1
mlp_grid = GridSearchCV(mlp, tuned_parameters, cv=10, scoring=make_scorer(accuracy_score), n_jobs=-1)
# commence training - NOTE: It takes hours to be complete
mlp_grid.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(mlp_grid.best_estimator_)
print ("Best Params CV1 - ", mlp_grid.best_params_)
print ("Best score - ", mlp_grid.best_score_)
print()
'''


# In[ ]:


# Creates the MLP Classifier
from sklearn.neural_network import MLPClassifier

# After training with Cross validation, this was derived as the best model.
mlp = MLPClassifier(activation= 'logistic', alpha= 0.01, hidden_layer_sizes= (40,), solver= 'lbfgs', random_state=1, max_iter=1000)

# commence training -
mlp.fit(X_train, y_train)

# predict the result
y_pred = mlp.predict(X_test)
print ("MLP Result Without considering Class Inbalance")
print ("Performance - " + str(100*accuracy_score(y_pred, y_test)) + "%")


# In[ ]:


# Plot non-normalized confusion matrix

plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix For MLP Model')


# # Support Vector Machine
# SVM result improved after using the fisher based feature selection.

# In[ ]:


# Cross validation with Support Vector Machines
'''
# Set the parameters by cross-validation
# number of neurons to use for the hidden layers
c_range = [0.5, 1, 2, 5, 10, 15, 17, 19, 20, 22, 25, 30, 35, 40, 45, 60, 70, 100, 300]
#c_range = [0.1, 0.2, 0.5, 1]

kernel_sv = ['linear', 'poly', 'rbf']
degree_sv = np.arange(1, 5, 1)
# Hyper Paremters to be used for Model 1
tuned_parameters = {'C': c_range, 'kernel': kernel_sv, 'degree': degree_sv}

# defining the SVC model
from sklearn.metrics import accuracy_score, make_scorer

svc_model = SVC(gamma='auto')
# Creates a GridSearch Classifier using parameters for model 1
svc_grid = GridSearchCV(svc_model, tuned_parameters, cv=10, scoring=make_scorer(accuracy_score), n_jobs=-1)

# commence training - NOTE: It takes hours to be complete
svc_grid.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(svc_grid.best_estimator_)
print ("Best Params CV1 - ", svc_grid.best_params_)
print ("Best score - ", svc_grid.best_score_)
print()
'''


# In[ ]:


# defining the SVC model
from sklearn.metrics import accuracy_score

svc_model = SVC(C=26, degree=1, kernel='rbf', gamma='scale', class_weight=label_weights)

# commence training
svc_model.fit(X_train, y_train)

# predict the result
y_pred = svc_model.predict(X_test)
print ("Support Vector Machines Result Considering Class Inbalance")
print ("Performance - " + str(100*accuracy_score(y_pred, y_test)) + "%")

svc_model = SVC(C=26, degree=1, kernel='rbf', gamma='scale')

# commence training
svc_model.fit(X_train, y_train)

# predict the result
y_pred = svc_model.predict(X_test)
print ("Support Vector Machines Result Without considering Class Inbalance")
print ("Performance - " + str(100*accuracy_score(y_pred, y_test)) + "%")


# In[ ]:


# Plot non-normalized confusion matrix

plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix For MLP Model')

