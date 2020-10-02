#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
from subprocess import check_output


# In[ ]:


#Read data
df=pd.read_csv('../input/data.csv',index_col='id')
# Change diagnosis cell to int and keep it in 'type' column
df['type']=df['diagnosis'].map({'B':0,'M':1}).astype(int)
del df['diagnosis']

#There is an extra comma at the end of the csv file header. remove it
df=df[df.columns[~df.columns.str.contains('Unnamed:')]]

# get a list of the header names
header=list(df.columns.values)
header.remove('type')


# In[ ]:


#Check the ditribution of features in regard to the cancer type
df.groupby('type').hist(figsize=(6, 6))

#get correlation of features among eachother
#from pandas.tools.plotting import scatter_matrix
#scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')


# In[ ]:


# get the correlation of features to the type of tumor 
df.corr()['type'].plot(kind='bar')


# In[ ]:


#Scale values to plot them as boxplot and to cheak the outliers
from sklearn.preprocessing import StandardScaler,MinMaxScaler
df[header] = MinMaxScaler().fit_transform(df[header])
bar=df.boxplot(rot=90,column=header,return_type='axes')


# In[ ]:


# Create a test and training dataset and create a simple classifier
# Break data to training and test
# Generate the training set.  Set random_state to be able to replicate results.
train = df.sample(frac=0.8, random_state=1)
# Select anything not in the training set and put it in the testing set.
test = df.loc[~df.index.isin(train.index)]

train_features_value=train.ix[:,train.columns !='type']
test_features_value=test.ix[:,test.columns !='type']
h=train_features_value.columns.values

train_data=train_features_value.values
test_data=test_features_value.values

actual_test_prediction=test['type'].values

# Build a simple RandomForest Classifier using all features

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import cross_validation

clf = RandomForestClassifier(n_estimators=100)
scores = cross_validation.cross_val_score(clf, train_data,train['type'].values, cv=3)
print(scores)

#Or

# forest = RandomForestClassifier(n_estimators = 100)
#  Fit the training data to the Survived labels and create the decision trees
# forest = forest.fit(train_data,train['type'].values)
# # # Take the same decision trees and run it on the test data
# prediction_array = forest.predict(test_data)


# In[ ]:


#Is our simple calssifier doing better than majority vote class
def majority_class_classifier(actual):
    num_positive = ( actual == 1).sum()
    num_negative = ( actual== 0).sum()
    if num_positive>num_negative:
        majority_class= 1
    else:
        majority_class= 0

    # build an array where all the prediction is the majority vote class: e.g. a list of all +1
    class_prediction_array=[majority_class]*len(actual)
    # use this to calculate the accuracy
    acc_majority=calculate_accuracy(actual,class_prediction_array)

    print ("Majority_class_classifier_accuracy _is:",acc_majority)

def calculate_accuracy(actual,prediction):
    # print "actual data:",test_data['sentiment']
    actual_data_array=np.array(actual)
    prediction_array=np.array(prediction)
    # print actual_data_array

    accuracy= accuracy_score(actual_data_array, prediction_array)
    return accuracy

#Yes it does!
majority_class_classifier(actual_test_prediction)


# In[ ]:


# Use Random Forest to calculate feature importance 

from sklearn.ensemble import ExtraTreesClassifier
forest = ExtraTreesClassifier(n_estimators=100,random_state=0)
X=train_data
y=train['type'].values
forest.fit(X, y)
importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
indices = np.argsort(importances)[::-1]
lable_features_sorted=[]
for i in indices:
     lable_features_sorted.append(header[i])

plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]),lable_features_sorted ,rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

