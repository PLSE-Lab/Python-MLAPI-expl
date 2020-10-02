#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Preprocessing data
# 
# Import data from csv file and show 5 first lines as an example

# In[ ]:


data =  pd.read_csv('/kaggle/input/malicious-and-benign-websites/dataset.csv')
data.head()


# First of all,  we do some basic statistics to give an overview

# In[ ]:


data.describe(include='all')


# From the table, we can see that there are some columns which contain unique values, especially URL that is totally unique.
# 
# For example:
# * URL (100% unique)
# * WHOIS_REGDATE (50% unique) 
# * WHOIS_UPDATED_DATE (33% unique) 
# 
# 
# Those columns can make noises and decrease the accuracy because the difference between them will increase a distance in our modal without any reason.
# 
# Therefore we have to drop those column to make data more clean

# In[ ]:


data = data.drop(labels=['URL','WHOIS_REGDATE','WHOIS_UPDATED_DATE'], axis='columns')


# Next, we will check loss data in our data set by printing out all lines that contain any missing value

# In[ ]:


print(data.isnull().sum())

data[pd.isnull(data).any(axis='columns')]


# For missing values, there are 2 ways to treat them:
# 
# * Assign a specific value that you think it is reasonable such as NaN = 0 in `CONTENT_LENGTH`
# * Using [interpolate method](https://en.wikipedia.org/wiki/Interpolation) to fill missing values
# 
# I choose the second one to fill missing data

# In[ ]:


processed_data =  data.interpolate()
print(processed_data.isnull().sum())


# After the processing, there is still 1 missing value. The reason is: `interpolate()` function in `pd.dataframe` only support `linear` function that only apply in numerical data so for categorial data in `SERVER`column it cannot apply.
# 
# So our solution is we will fill the last one by the highest frequency value in `SERVER`

# In[ ]:


max_value = processed_data['SERVER'].value_counts().idxmax()

print('Highest frequency value:',max_value)

processed_data['SERVER'].fillna(max_value, inplace=True)

print(processed_data.isnull().sum())


# # k-NN
# The first algorithm we wanna try that is k-NN. For the numerical data, it is easy to calculate a distance by using Euclide measure. However, there is many columns containing categorical data so we cannot apply Euclide measure on those columns.
# 
# For solving this problem, we will apply a simple logic that is:
# * If value on x of 2 vector are not different -> distance between them will be 0
# * Else distance will be 1
# 
# An idea to conduct it that is we try to split a column A with n unique value to n columns with value is 0 or 1
# 
# For example:
# * A = {a1,a2,a3} 
# * Data: {id=1, A=a1}
# 
# Then we will have
# * Data: {id=1, A_a1=1, A_a2=0, A_a3=0}
# 
# After spliting, we can apply Euclide measure to calculate a distance for all columns.
# 

# In[ ]:


knn_data = pd.get_dummies(processed_data, prefix_sep="_")
knn_data.head()


# Next, we have to prepare a training set and test set

# In[ ]:


X = knn_data.drop(labels='Type', axis='columns')
y = knn_data['Type']
X.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print("Training size: %d" % len(y_train))
print("Test size    : %d" % len(y_test))


# Then we first try with K = 10 and Euclide measure

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
k=10

clf = KNeighborsClassifier(n_neighbors=k, p=2, weights='distance')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Print results for 30 test data points:")
print("Predicted labels: ", y_pred[30:60])
print("Ground truth    : ", y_test.to_numpy()[30:60])

print("Accuracy of %d NN: %.2f %%" % (k, 100 * accuracy_score(y_test.to_numpy(), y_pred)))
print('Classification Report:\n{}\n'.format(classification_report(y_test.to_numpy(),clf.predict(X_test))))


# Accuracy of k=10 is quite good (80%-90%).
# 
# Additional, we want to show a result in probability so that we can try to use exist function as follows

# In[ ]:


y_pred_proba = clf.predict_proba(X_test)
print(y_pred_proba[30:60]*100)


# # Naive Bayes
# 
# It becomes more complicated with Naive Bayes
# 
# First of all, we have to start with same data processing as k-NN 
# 

# In[ ]:


nb_data = pd.get_dummies(processed_data, prefix_sep="_")
nb_data.head()


# In[ ]:


X = nb_data.drop(labels='Type', axis='columns')
y = nb_data['Type']
X.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(nb_data, y, test_size=0.3)

print("Training size: %d" % len(y_train))
print("Test size    : %d" % len(y_test))


# First, I will try with Gauusian Naive Bayes modal

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb_clf= GaussianNB()
nb_clf.fit(X_train, y_train)

y_pred = nb_clf.predict(X_test)

print("Print results for 30 test data points:")
print("Predicted labels: ", y_pred[30:60])
print("Ground truth    : ", y_test.to_numpy()[30:60])

print("Accuracy of GNB: %.2f %%" % ( 100 * accuracy_score(y_test.to_numpy(), y_pred)))
print('Classification Report:\n{}\n'.format(classification_report(y_test.to_numpy(),nb_clf.predict(X_test))))


# The accuracy of this one is quite slow. 
# 
# So let do the second try with other modals: MultinomialNB and BernoulliNB

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
nb_clf= MultinomialNB()
nb_clf.fit(X_train, y_train)

y_pred = nb_clf.predict(X_test)

print("Print results for 30 test data points:")
print("Predicted labels: ", y_pred[30:60])
print("Ground truth    : ", y_test.to_numpy()[30:60])

print("Accuracy of MNB: %.2f %%" % ( 100 * accuracy_score(y_test.to_numpy(), y_pred)))
print('Classification Report:\n{}\n'.format(classification_report(y_test.to_numpy(),nb_clf.predict(X_test))))


# In[ ]:


from sklearn.naive_bayes import BernoulliNB
nb_clf= BernoulliNB()
nb_clf.fit(X_train, y_train)

y_pred = nb_clf.predict(X_test)

print("Print results for 30 test data points:")
print("Predicted labels: ", y_pred[30:60])
print("Ground truth    : ", y_test.to_numpy()[30:60])

print("Accuracy of BNB: %.2f %%" % ( 100 * accuracy_score(y_test.to_numpy(), y_pred)))
print('Classification Report:\n{}\n'.format(classification_report(y_test.to_numpy(),nb_clf.predict(X_test))))


# It seem BernoulliNB has the highest accuracy. I guess it because our data contains large of categorical data so Gaussian is not suiltable in this case 
# 
# Then we compare between 2 modals BernoulliNB and MultinomialNB:
# 
#  MultinomialNB care about how many time X value appear in our dataset while BernoulliNB only care about whether X value appear or not.
#  So I think some numerical columns make MultinomialNB accurracy decrease by the difference between their unit (for example character vs bytes)
# 
# So the idea to increase the perfomance of MultinomialNB is trying to scale all numerical columns into range (0,1) (normalization) and run MultinomialNB again

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data  = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.4)

print("Training size: %d" % len(y_train))
print("Test size    : %d" % len(y_test))


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
nb_clf= MultinomialNB()
nb_clf.fit(X_train, y_train)

y_pred = nb_clf.predict(X_test)

print("Print results for 30 test data points:")
print("Predicted labels: ", y_pred[30:60])
print("Ground truth    : ", y_test.to_numpy()[30:60])

print("Accuracy of MNB: %.2f %%" % ( 100 * accuracy_score(y_test.to_numpy(), y_pred)))
print('Classification Report:\n{}\n'.format(classification_report(y_test.to_numpy(),nb_clf.predict(X_test))))


# Let's see now MultinomialNB is also has accuracy similar to BernoulliNB :)
# 
# # Conclusion
# 
# Between 2 algorithm k-NN and Naive Bayes, NB has accuracy is higher than k-NN (especially BernoulliNB) but we have to try different model to choose the suitable one.
# 
# Below is comparison between modals

# In[ ]:


import matplotlib.pyplot as plt
import time
from sklearn.model_selection import KFold

models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors=k, p=2, weights='distance'), 0))
models.append(('GNB', GaussianNB(), 0))
models.append(('MNB', MultinomialNB(), 0))
models.append(('BNB', BernoulliNB(), 0))
# models with normalization on numerical columns
models.append(('KNN-S', KNeighborsClassifier(n_neighbors=k, p=2, weights='distance'), 1))
models.append(('GNB-S', GaussianNB(), 1))
models.append(('MNB-S', MultinomialNB(), 1))
models.append(('BNB-S', BernoulliNB(), 1))

results = []
names = []
run_times = []
scoring = 'accuracy'
for name, model, scaler in models:
    start = time.time()
    kfold = KFold(n_splits=10, random_state=7)
    if(scaler==1):
        scaler = MinMaxScaler()
        scaled_X  = scaler.fit_transform(X)
        cv_results = cross_val_score(model, scaled_X, y, cv=10, scoring=scoring)
    else:
        cv_results = cross_val_score(model, X, y, cv=10, scoring=scoring)
    stop= time.time()
    run_times.append(stop-start)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    

print( "Run times: %s" % (run_times))
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Accuracy Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


y_pos = np.arange(len(names))
plt.bar(y_pos, run_times, align='center', alpha=0.5)
plt.xticks(y_pos, names)
plt.title('Time Comparison')
plt.show()

