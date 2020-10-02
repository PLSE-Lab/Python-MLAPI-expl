#!/usr/bin/env python
# coding: utf-8

# Hello everyone!!, this is my first kaggle and therefore any advice is welcome to learn and become betterThese are the libraries that i used in order to do my stuff.

# In[ ]:


import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve


# Load the training set and the test set, print some info about the training set

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.info()


# First i decided to adjust some features of my dataset. The problem with this dataset is that it is not plentiful and doesn't have many features so what we can try to do is clean my data and trying to add some features in order to get a better estimate. 
# First as i saw in another kernel, i decided to obtain the social title of people as probably those who had a higher title socially took precedence during the rescue

# In[ ]:


train['Title'] = train.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
test['Title'] = test.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
# normalize the titles
normalized_titles = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}

train.Title = train.Title.map(normalized_titles)
test.Title = train.Title.map(normalized_titles)

train.Title.replace(['Mr', 'Miss', 'Mrs', 'Master', 'Officer', 'Royalty'], [1, 2, 3, 4, 5, 6], inplace=True)
test.Title.replace(['Mr', 'Miss', 'Mrs', 'Master', 'Officer', 'Royalty'], [1, 2, 3, 4, 5, 6], inplace=True)


# Then, i decided obviusly to remove the name feature since is not useful in order to do a good prediction. I coded the Embarked and the Sex features and i decided to temporary drop the Ticket and the Cabin since are very complex to code 
# as there is a lot of missing data. Then i decided to fill the NaN values of the Age with the mean value as it is usually a strategy used. I removed the last two NaN values in the Embarked column

# In[ ]:


# Drop the name feature
train = train.drop('Name', 1)
test = test.drop('Name', 1)
# Encoding the features
train.Embarked.replace(['S', 'C', 'Q'], [1, 2, 3], inplace=True)
train.Sex.replace(['male', 'female'], [1, 2], inplace=True)
test.Embarked.replace(['S', 'C', 'Q'], [1, 2, 3], inplace=True)
test.Sex.replace(['male', 'female'], [1, 2], inplace=True)

# Temporary drop of some columns
train = train.drop('Ticket', 1)
train = train.drop('Cabin', 1)
test = test.drop('Ticket', 1)
test = test.drop('Cabin', 1)

#Fill NaN values of Age with the mean Age
train['Age'] = train['Age'].fillna(train['Age'].sum()/len(train))
test['Age'] = test['Age'].fillna(test['Age'].sum()/len(test))
#Drop other raw where there is a NaN value
train= train.dropna(how='any',axis=0)  
test = test.fillna(0)

train.info()


# So after some data tuning and adjustments this is the training set:

# In[ ]:


train


# So here we are, let's do some predictions! I decided to use SVM since it is a powerful classification tool. First i started with a simple SVM with a linear kernel  in order to see how this simple model predict well our data. Then to obtain a better accurancy i decided to use the Radial Basis Function Kernel, an amazing type of kernel that can bring our data in an infinite number of dimensions in order to find the best separating hyperplane. I tryed different parameters and i used GridSearch to find the best parameters that gave me the best score

# In[ ]:


Data_train = train.values
Data_test = test.values

# m = number of input samples
m_train = len(Data_train)
m_test = len(Data_test)
# prediction for training
Ytrain = Data_train[:m_train,1]
# features for training
Xtrain = Data_train[:m_train,2:]
# features for testing
Xtest = Data_test[:m_test,1:]

parameters = {'C': [1, 10, 50, 100,200,300, 1000],'gamma':[0.0001,0.001,0.01,0.1,1.]}
#run SVM with rbf kernel
rbf_SVM = SVC(kernel='rbf')
# ADD CODE: DO THE SAME AS ABOVE FOR RBF KERNEL
clf = GridSearchCV(rbf_SVM,parameters,cv=5)
clf.fit(Xtrain,Ytrain)

print ('\n RESULTS FOR rbf KERNEL \n')

print("Best parameters set found:")
best_param = clf.best_params_
value_best_param_rbf_gammma = best_param['gamma']
value_best_param_rbf_c = best_param['C']
estim_best = clf.best_estimator_
print("Best Estimator: ", estim_best)

#get training and test error for the best SVM model from CV
best_SVM = SVC(C = value_best_param_rbf_c, gamma = value_best_param_rbf_gammma, kernel='rbf')

best_SVM.fit(Xtrain,Ytrain)

training_error = 1. - best_SVM.score(Xtrain,Ytrain)
print("Training error: ", training_error)

Ytest_predicted = best_SVM.predict(Xtest)


# With this model i obtained an accurancy of 0.8279% in the training set, but, now i have to understand how this model perform with the test set, and check that not overfitted the training set. So i decided to plot the learning curves with a 5-fold cross validation since as i said, the dataset is not plentiful.

# In[ ]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes= np.linspace(.1, 1.0, 10)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

plot_learning_curve(estim_best,"SVC learning curves",Xtrain,Ytrain,cv=5)
plt.show()


# As we can see, increasing the number of samples the difference between the two errors is reduced and consequently we can assume not to be overfitting. I looked at other kernels of kaggle that used other features but did not get better results, so I can think that it is not a problem of the number of features, because the features extracted by the other users were linear combinations of the already existing features. I do not know how to improve performance and get a better score in the ranking of the challenge? any suggestions?
