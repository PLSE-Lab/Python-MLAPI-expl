#!/usr/bin/env python
# coding: utf-8

# Comparing different classifiers to predict the type of breast cancer 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#First import all the libraries needed

import numpy as np #for linear algebra
import pandas as pd #for chopping, processing
import csv #for opening csv files
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt #for plotting the graphs
from sklearn.linear_model import LogisticRegression #for logistic regression
from sklearn.pipeline import Pipeline #to assemble steps for cross validation
from sklearn.preprocessing import PolynomialFeatures #for all the polynomial features
from sklearn import svm #for Support Vector Machines
from sklearn.neighbors import NearestNeighbors #for nearest neighbor classifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier #for decision tree classifier
from sklearn.naive_bayes import GaussianNB  #for naive bayes classifier
from scipy import stats #for statistical info
from sklearn.model_selection import train_test_split # to split the data in train and test
from sklearn.model_selection import KFold # for cross validation
from sklearn.grid_search import GridSearchCV  # for tuning parameters
from sklearn.neighbors import KNeighborsClassifier  #for k-neighbor classifier
from sklearn import metrics  # for checking the accuracy 
from time import time


# In[ ]:


#load data
data = pd.read_csv("../input/data.csv")


# In[ ]:


#to know the number of cases and the number of variables
data.shape


# In[ ]:


#Description of the dataset

#how many cases are included in the dataset
length = len(data)
#how many features are in the dataset
features = data.shape[1]-1

# Number of malignant cases
malignant = len(data[data['diagnosis']=='M'])

#Number of benign cases
benign = len(data[data['diagnosis']=='B'])

#Rate of malignant tumors over all cases
rate = (float(malignant)/(length))*100

print ("There are "+ str(len(data))+" cases in this dataset")
print ("There are {}".format(features)+" features in this dataset")
print ("There are {}".format(malignant)+" cases diagnosed as malignant tumor")
print ("There are {}".format(benign)+" cases diagnosed as benign tumor")
print ("The percentage of malignant cases is: {:.4f}%".format(rate))


# In[ ]:


data.diagnosis.unique()


# In[ ]:


#drop ID because we do not need the ID number 

data.drop('id',axis=1,inplace=True)
#check that dropped
data.head(1)


# In[ ]:


# Extract feature columns where everything but the diagnosis is included.
# I am separating all the features that are helpful in determining the diagnosis
features = list(data.columns[1:30])
print (features)


# In[ ]:



#Our target is predicting the diagnosis in benign or malignant, so we need
#to extract this one as the dependent variable - the variable we will predict
target = data.columns[0:1]
print (target)


# In[ ]:


#Now we need to separate the data into feature data and target data
X = data[features] #our features that we will use to predict Y
Y = data[target] #our dependent variable, the one we are trying to predict from X


# In[ ]:


# X should have 29 variables and 569 cases
X.shape


# In[ ]:


# Y should have 1 variable - just the diagnosis and 569 cases
Y.shape


# In[ ]:


# Show the feature information by printing the first row
# Show the traget information by also printing the first row
print ("\nFeature values:")
print (X.head(1))
print ("\nTarget values:")
print (Y.head(1))


# In[ ]:


df=pd.DataFrame(data)


# In[ ]:


#Research shows that any variables that are highly correlated
#should be removed from further analysis. But, PCA takes care of multicollinearity, so maybe 
#I identify them which ones there are and let PCA to do its job.  
#Just in case let's see how two highly correlated variables look like
#using prettyplots
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1)
for i in range(1):
    x=df['perimeter_mean']
    y=df['area_worst']
    ax.scatter(x,y, label=str(i))
#ax.legend()
ax.set_title('Correlation of perimeter_mean and area_worst with correlation .99 or r-square= .81')
fig.savefig('scatter.png')


# In[ ]:


#Let's visualize another set of variables that are not correlated as highly as the first ones
#These have a correlation coefficient of .75 which means an r-squared score of approximately .49
fig, ax = plt.subplots(1)
for i in range(1):
    x=df['concavity_mean']
    y=df['compactness_worst']
    ax.scatter(x,y, label=str(i))
#ax.legend()
ax.set_title('Correlation of the mean of concavity and worst compactness')
fig.savefig('scatter.png')


# In[ ]:


def preprocess_features(X):
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all M/B malignant/benign values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['M', 'B'], [1, 0])
 
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

X = preprocess_features(X)
Y = preprocess_features(Y)
print ("Processed feature columns ({} total features):\n{}".format(len(X.columns), list(X.columns)))
print ("Target columns ({} total features):\n{}".format(len(Y.columns), list(Y.columns)))


# In[ ]:


# import cross_validation to split the train and testing
from sklearn.cross_validation import train_test_split
# Set the number of training points
nr_train = 300
# Set the number of testing points
nr_test = X.shape[0] - nr_train

# Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=nr_test, random_state=40)

# Show the results of the split
print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))


# In[ ]:


from sklearn.metrics import f1_score
def train_classifier(clf, X_train, Y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, Y_train)
    end = time()
    
    # Print the results
    print ("Trained model in {:.4f} seconds".format(end - start))

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    Y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print ("Made predictions in {:.4f} seconds.".format(end - start))
    return f1_score(target.values, Y_pred, pos_label=1)


def train_predict(clf, X_train, Y_train, X_test, Y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    
    # Train the classifier
    train_classifier(clf, X_train, Y_train)
    
    # Print the results of prediction for both training and testing
    print ("F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, Y_train)))
    print ("F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, Y_test)))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier


# In[ ]:


clf_A = KNeighborsClassifier()
clf_B = DecisionTreeClassifier(random_state=0)
clf_C = SVC()
clf_D = GaussianNB()
clf_E = RandomForestClassifier(n_estimators=10)
clf_G = AdaBoostClassifier()
clf_H = QuadraticDiscriminantAnalysis()
clf_I = MLPClassifier(alpha=1)


# In[ ]:


X_train_100 = X_train[:100]
Y_train_100 = Y_train[:100]

X_train_200 = X_train[:200]
Y_train_200 = Y_train[:200]

X_train_300 = X_train[:300]
Y_train_300 = Y_train[:300]

X_train_300 = X_train[:400]
Y_train_300 = Y_train[:400]


# In[ ]:


for clf in [clf_A, clf_B, clf_C, clf_D, clf_E, clf_G, clf_H, clf_I]:
    for size in [300, 400]:
        train_predict(clf, X_train[:size], Y_train[:size], X_test, Y_test)
        print ('/n')


# In[ ]:


#additional tuning of the best classifiers
clf_D = GaussianNB()
clf_G = AdaBoostClassifier(algorithm = 'SAMME')
clf_H = QuadraticDiscriminantAnalysis(reg_param = 0.001, store_covariances=True, tol = 0.01)


# In[ ]:


#trying with only 300 and 400 sampling sizes
X_train_300 = X_train[:300]
Y_train_300 = Y_train[:300]

X_train_300 = X_train[:400]
Y_train_300 = Y_train[:400]


# In[ ]:



for clf in [clf_D, clf_G, clf_H]:
    for size in [300, 400]:
        train_predict(clf, X_train[:size], Y_train[:size], X_test, Y_test)
        print ('/n')


# In[ ]:


from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn import metrics
import pandas as pd
from ggplot import *


# In[ ]:


# ROC curve for Naive Bayes
preds = clf_D.predict_proba(X_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(Y_test, preds)

df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
ggplot(df, aes(x='fpr', y='tpr')) +    geom_line() +    geom_abline(linetype='dashed')+    ggtitle ("ROC for Naive Bayes has an area under the curve of " + str(metrics.auc(fpr,tpr)))


# In[ ]:


# ROC curve for AdaBoost
preds = clf_G.predict_proba(X_test)[:,1]
fprA, tprA, _ = metrics.roc_curve(Y_test, preds)

df = pd.DataFrame(dict(fprA=fprA, tprA=tprA))
ggplot(df, aes(x='fprA', y='tprA')) +    geom_line() +    geom_abline(linetype='dashed')+    ggtitle ("ROC for AdaBoost has an area under the curve of " + str(metrics.auc(fprA,tprA)))


# In[ ]:


# ROC curve for QDA
preds = clf_H.predict_proba(X_test)[:,1]
fprH, tprH, _ = metrics.roc_curve(Y_test, preds)

df = pd.DataFrame(dict(fprH=fprH, tprH=tprH))
ggplot(df, aes(x='fprH', y='tprH')) +    geom_line() +    geom_abline(linetype='dashed')+    ggtitle ("ROC for QDA has an area under the curve of " + str(metrics.auc(fprH,tprH)))


# In[ ]:


# ROC curve for K-Neighbors
preds = clf_A.predict_proba(X_test)[:,1]
fprN, tprN, _ = metrics.roc_curve(Y_test, preds)

df = pd.DataFrame(dict(fprN=fprN, tprN=tprN))
ggplot(df, aes(x='fprN', y='tprN')) +    geom_line() +    geom_abline(linetype='dashed')+    ggtitle ("ROC for K-Neighbors has an area under the curve of " + str(metrics.auc(fprN,tprN)))


# In[ ]:


# ROC curve for Decision trees
preds = clf_B.predict_proba(X_test)[:,1]
fprD, tprD, _ = metrics.roc_curve(Y_test, preds)

df = pd.DataFrame(dict(fprD=fprD, tprD=tprD))
ggplot(df, aes(x='fprD', y='tprD')) +    geom_line() +    geom_abline(linetype='dashed')+    ggtitle ("ROC for Decision Trees has an area under the curve of " + str(metrics.auc(fprD,tprD)))


# In[ ]:


# Set the number of training points much smaller than before
nr_train = 50
# Set the number of testing points
nr_test = X.shape[0] - nr_train

# Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=nr_test, random_state=40)

# Show the results of the split
print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))


# In[ ]:


for clf in [clf_D, clf_G, clf_H]:
    for size in [50]:
        train_predict(clf, X_train[:size], Y_train[:size], X_test, Y_test)
        print ('/n')


# In[ ]:


Even for a small size of training set, these three perform very well with this data.  We need to remember that these dataset has no missing values at all and very few outliers. 


# In[ ]:




