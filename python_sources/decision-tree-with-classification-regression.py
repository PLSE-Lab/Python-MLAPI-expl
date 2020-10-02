#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os


# Loading the requird libraries

# In[ ]:


import numpy as np
import pandas as pd
import graphviz
import matplotlib.pyplot as plt
import seaborn
from sklearn.tree import DecisionTreeClassifier, export_graphviz,DecisionTreeRegressor
from sklearn.metrics import accuracy_score,mean_absolute_error
from sklearn import tree


# In[ ]:


test1 = np.array([1,0,1,0,1,0,1,0,1])
test2 = np.array([7.5,6,7,4,5,10,9.5,8.8,9])
x = np.array([test1, test2])

job = np.array([0,0,0,0,0,1,1,1,1])


# In[ ]:


plt.scatter(test1, job)
plt.xlabel('test1')
plt.ylabel("job")


# In[ ]:


plt.scatter(test2, job)
plt.xlabel("test2")
plt.ylabel("job")


# In[ ]:


dtc = DecisionTreeClassifier()


# In[ ]:


dtc.fit(X=x.T,y=job)


# In[ ]:


dtc.predict(X=x.T)


# In[ ]:


graphviz.Source(export_graphviz(dtc, out_file=None))


# *Decision Tree Classification*

# In[ ]:


import missingno as msno


# In[ ]:


bank1 = pd.read_csv("../input/new_bank.csv")


# In[ ]:


bank1.head()


# In[ ]:


bank1.tail()


# In[ ]:


bank1.describe()


# In[ ]:


msno.matrix(bank1)


# In[ ]:


bank2 = pd.get_dummies(bank1,columns=['marital','loan','contact','y'],drop_first=True)


# In[ ]:


bank2.head()


# In[ ]:


bank2.tail()


# In[ ]:


msno.matrix(bank2)


# In[ ]:


#Imputing train_test_split
from sklearn.model_selection import train_test_split


# In[ ]:



trainx,testx,trainy,testy = train_test_split(bank2.iloc[:,:-1],bank2.iloc[:,-1],                                             test_size=0.3,random_state=1)


# *DTC with no parameter tuning*

# In[ ]:


dtc = DecisionTreeClassifier()


# In[ ]:


dtc.fit(trainx,trainy)


# In[ ]:


predict_train = dtc.predict(trainx)
predict_test = dtc.predict(testx)


# In[ ]:


print("Accuracy on train is:",accuracy_score(trainy,predict_train))
print("Accuracy on test is:",accuracy_score(testy,predict_test))


# * DTC with maxdepth

# In[ ]:


dtc_2 = DecisionTreeClassifier(max_depth=2)


# In[ ]:


dtc_2.fit(trainx,trainy)


# In[ ]:


predict_train_2 = dtc_2.predict(trainx)
predict_test_2 = dtc_2.predict(testx)


# In[ ]:


print("Accuracy on train is:",accuracy_score(trainy,predict_train_2))
print("Accuracy on test is:",accuracy_score(testy,predict_test_2))


# In[ ]:


graphviz.Source(export_graphviz(dtc_2,feature_names=trainx.columns,                                filled=True,class_names=["0","1"],out_file=None))


# **Decision Tree Regression***

# In[ ]:


customer1 = pd.read_csv("../input/new_customer.csv")


# In[ ]:


customer1.head()


# In[ ]:


customer2 = pd.get_dummies(customer1, columns=["City","FavoriteGame"], drop_first = True)


# In[ ]:


customer2.head()


# In[ ]:


trainx,testx,trainy,testy = train_test_split(customer2.iloc[:,np.r_[0:4,5,6]],                                customer2.TotalRevenueGenerated,test_size=0.3,random_state=1)


# DTR with no paramater tuning

# In[ ]:


dtr = DecisionTreeRegressor()


# In[ ]:


dtr.fit(trainx,trainy)


# In[ ]:


predict_train = dtr.predict(trainx)
predict_test = dtr.predict(testx)


# In[ ]:


print("Train Error:", mean_absolute_error(trainy,predict_train))
print("Test Error:", mean_absolute_error(testy,predict_test))


# DTR with max depth

# In[ ]:


dtr_2 = DecisionTreeRegressor(max_depth = 2)


# In[ ]:


dtr_2.fit(trainx,trainy)


# In[ ]:


predict_train_2 = dtr_2.predict(trainx)
predict_test_2 = dtr_2.predict(testx)


# In[ ]:


print("Train Error:", mean_absolute_error(trainy,predict_train_2))


# In[ ]:


print("Train Error:", mean_absolute_error(testy,predict_test_2))

