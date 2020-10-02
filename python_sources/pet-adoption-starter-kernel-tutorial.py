#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pylab as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Wlcome to my little Tutorial on the Pet adoption competition.**  
# I'm sorry for the bad formatting, I know there are many ways of making a kernel look fancy and cool,  
# but It's pretty late at night and I just wanted to add a few notes on my kernel and share with others who may be interested in the competition.
# As you can see the code isn't as high end as some other kernels you'll come across, but it's just how a starter coder tried to achieve results.  
# This is a little intro to the competition for anyone who may be interested.  
# So let's get started :)

# In[ ]:


train = pd.read_csv("../input/train/train.csv")
test = pd.read_csv("../input/test/test.csv")
color_labels = pd.read_csv("../input/color_labels.csv")
breed_labels = pd.read_csv("../input/breed_labels.csv")
state_labels = pd.read_csv("../input/state_labels.csv")
train.describe()


# We just took a little look at the data we're working with, and below we have the test array.

# In[ ]:


test.sample(5)


# Here we drop the columns that only mess up the data and don't contribute to the chance of the animal being adopted in anyway.

# In[ ]:


train_simple = pd.read_csv("../input/train/train.csv")
train_simple.drop(['Name', 'RescuerID', 'Description',"PetID"], axis=1, inplace=True)
test.drop(['Name', 'RescuerID', 'Description'], axis=1, inplace=True)
train.drop(['Name', 'RescuerID', 'Description'], axis=1, inplace=True)


# In[ ]:


train_simple.sample(5)


# In[ ]:


test.sample(5)


# In[ ]:


train_simple["NewState"] = train_simple["State"]-41323
train_simple.drop(["State"],inplace = True,axis=1)


# In[ ]:


test["NewState"] = test["State"]-41323
test.drop(["State"],inplace = True,axis=1)


# Just cleaning the state data,  
# the minimum value was 41322, so I simply used that value as a gauge value to simplify the state collumn a bit

# In[ ]:


train_simple.sample(3)


# I wanted to simplfy the data a tad bit further since it's a starter kernel, so I simply looked if the dog being adopted was a Mixed breed or a pure breed to see if that had any effect.

# In[ ]:


train_simple["Mixed"] = 0
test["Mixed"] = 0


# In[ ]:


train_simple.sample(5)


# In[ ]:


indexer = 0 
for x in train_simple["Breed2"]:
    if x > 0:
        train_simple.loc[[indexer],"Mixed"] = 1
        indexer +=1


# In[ ]:


indexer = 0 
for x in test["Breed2"]:
    if x > 0:
        test.loc[[indexer],"Mixed"] = 1
        indexer +=1


# The next two columns are the "PhotoAmt" and "VideoAmt"  
# These were just representation of how much coverage these puppies got on the website, so I almost dumbed down the array a bit to make the number crunching easier and to simply things.

# In[ ]:


train_simple["Exp"] = 0
for x in range(len(train["PhotoAmt"])):
    train_simple.loc[[x],"Exp"] = train_simple.loc[[x],"PhotoAmt"] + train_simple.loc[[x],"VideoAmt"]
train_simple.drop(["PhotoAmt","VideoAmt"],inplace=True,axis=1)


# In[ ]:


test["Exp"] = 0
for x in range(len(test["PhotoAmt"])):
    test.loc[[x],"Exp"] = test.loc[[x],"PhotoAmt"] + test.loc[[x],"VideoAmt"]
test.drop(["PhotoAmt","VideoAmt"],inplace=True,axis=1)


# So I noticed while going through the data on my own that the majority of the adoptions were free, and a few had absurd numbers like 3000, so I just made sure that the arrays are as simple as they can possibly be.

# In[ ]:


indexer = 0 
for x in train_simple["Fee"]:
    if x > 0:
        train_simple.loc[[indexer],"Fee"] = 1
        indexer +=1
    else:
        train_simple.loc[[indexer],"Fee"] = 0


# In[ ]:


indexer = 0 
for x in test["Fee"]:
    if x > 0:
        test.loc[[indexer],"Fee"] = 1
        indexer += 1
    else:
        test.loc[[indexer],"Fee"] = 0


# Now I prepare the data to be fit and read by the different models I managed to find online to see which one worked base for the model we had :)

# In[ ]:


from sklearn.model_selection import train_test_split

target = train_simple["AdoptionSpeed"]
predict = train_simple.drop(["AdoptionSpeed"],axis = 1)

x_train, x_val, y_train, y_val = train_test_split(predict, target, test_size = 0.22, random_state = 0)


# We fit all the data in the next blocks and print the values on a final board so we can see easily :)

# In[ ]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train,y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)


# In[ ]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)


# In[ ]:


# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)


# In[ ]:


# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)


# In[ ]:


# Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)


# In[ ]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)


# In[ ]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)


# In[ ]:


# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)


# In[ ]:


# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)


# In[ ]:


# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)


# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)


# Now that we cab see that the Gradient Boosting Classifier did the best work out of all of them, let's use that for the submission :)  
# But before that, I want to train it with the entire dataset to maximize our chances

# In[ ]:


full_train = train_simple.drop(["AdoptionSpeed"],axis = 1)
full_test = train_simple["AdoptionSpeed"]
gbk.fit(full_train, full_test)


# Just checking the format of the output file so we don't get a zero percent in our first submission

# In[ ]:


sample = pd.read_csv("../input/test/sample_submission.csv")
sample.sample(5)


# In[ ]:


#set ids as PassengerId and predict survival 
ids = test["PetID"]
predictions = gbk.predict(test.drop(["PetID"],axis = 1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({"PetID": ids ,"AdoptionSpeed": predictions})
output.to_csv('submission.csv', index=False)


# I know this is just scratching the surface of this competition but it's a nice little start in my opinion.  
# I plan on adding the image recognition and feature detection (Beginner level again of course) based on if the kernel is received well/viewed at all.  
# I hope you all have a great day, keep coding everyone :)
