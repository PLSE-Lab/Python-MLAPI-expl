#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv


# In[ ]:


def importData(filename):
    dataset = csv.reader(open(filename,'r'))
    dataset = list(dataset)
    header = dataset.pop(0)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset, header

heartData, header = importData('../input/heart.csv')
heartData = np.array(heartData)
print(heartData.shape)


# In[ ]:


X = heartData[:,0:13]
Y = heartData[:,-1]

print(X.shape)
print("-"*40)
print(Y.shape)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.33, random_state=88)
print(len(x_train), len(y_train), len(x_test), len(y_test))


# In[ ]:


### Voting Classifier

clf1=LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state=1)
clf2=RandomForestClassifier(n_estimators=50, random_state=1)
clf3=GaussianNB()

eclf1 = VotingClassifier(estimators=[('LR', clf1), ('RF', clf2), ('GNB', clf3)], voting='hard')
eclf1 = eclf1.fit(X, Y)


# In[ ]:


vc_predicted = eclf1.predict(x_test)
len(vc_predicted)


# In[ ]:


print("VC Accuracy:", accuracy_score(y_test, vc_predicted))

