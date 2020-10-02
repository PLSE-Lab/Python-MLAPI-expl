#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Please note that task 4 is done by hand as well, will be verified by code here
#First question 1 of task 4 is done


# In[ ]:


import pandas as pd
Task4First = pd.read_csv("../input/task4first/Task4First.csv")


# In[ ]:


# Importing the dataset
dataset = Task4First
Xtrain=dataset[['Home/Away','In/Out','Media']]
Ytrain=dataset[['Label']]


# In[ ]:


Xtrain


# In[ ]:


Ytrain


# In[ ]:


Xtrain = pd.get_dummies(Xtrain) #Alternative way to do one hot encoding (OHE)


# In[ ]:


Xtrain


# In[ ]:


from sklearn import tree
clf=tree.DecisionTreeClassifier(random_state=0,criterion="entropy")
clf.fit(Xtrain, Ytrain)


# In[ ]:


from graphviz import Source
#Source( tree.export_graphviz(clf, out_file=None))
Source( tree.export_graphviz(clf, out_file=None, feature_names=Xtrain.columns,class_names=['Lose','Win']))


# In[ ]:


#So far question 1 of task 4 is verified


# In[ ]:


#Now question 2 of task 4


# In[ ]:


import pandas as pd


# In[ ]:


dataset = pd.read_csv("../input/task4second/Task4.csv")


# In[ ]:


dataset


# In[ ]:


X=dataset[['Outlook','Temperature','Humidity','Windy']]
Y=dataset[['Play']]


# In[ ]:


XOHE = pd.get_dummies(X) #Alternative way to do one hot encoding (OHE)


# In[ ]:


XOHE


# In[ ]:


Xtrain=XOHE.head(14)
Xtest=XOHE.tail(1)
Xtrain


# In[ ]:


Ytrain=Y.head(14)
#Ytest=Y.tail(1)
Ytrain


# In[ ]:


from sklearn import tree
clf=tree.DecisionTreeClassifier(random_state=0,criterion="entropy")
clf.fit(Xtrain, Ytrain)


# In[ ]:


Ypred = clf.predict(Xtest)


# In[ ]:


Ypred[0]


# In[ ]:


from graphviz import Source
#Source( tree.export_graphviz(clf, out_file=None))
Source( tree.export_graphviz(clf, out_file=None, feature_names=Xtrain.columns,class_names=['No_Play','Play']))


# In[ ]:


#So far task 4 is completed, task 4 is performed by hand and code


# In[ ]:


#Task 5 now will be performed, task 5 is done using only code, please note that for CART, we should use gini index and for ID3 we use entropy


# In[ ]:


import pandas as pd
Task5 = pd.read_csv("../input/task5g/Task5.csv")


# In[ ]:


dataset=Task5
dataset


# In[ ]:


X=dataset[['Home/Away','In/Out','Media']]
Y=dataset[['Play']]


# In[ ]:


XOHE = pd.get_dummies(X) #Alternative way to do one hot encoding (OHE)


# In[ ]:


XOHE


# In[ ]:


Xtrain=XOHE.head(24)
Xtest=XOHE.tail(12)
Xtest


# In[ ]:


Ytrain=Y.head(24)
Ytrue=Y.tail(12)
Ytrain


# In[ ]:


from sklearn import tree
clf=tree.DecisionTreeClassifier(random_state=0,criterion="entropy")
clf.fit(Xtrain, Ytrain)


# In[ ]:


from graphviz import Source
#Source( tree.export_graphviz(clf, out_file=None))
Source( tree.export_graphviz(clf, out_file=None, feature_names=Xtrain.columns,class_names=['Lose','Win']))


# In[ ]:


Ypred = clf.predict(Xtest)


# In[ ]:


Ypred=pd.DataFrame(Ypred)


# In[ ]:


Ypred


# In[ ]:


Ytrue


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(Ytrue, Ypred)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
Ytrue = lb_make.fit_transform(Ytrue)
Ypred = lb_make.fit_transform(Ypred)


# In[ ]:


from sklearn.metrics import precision_score
precision_score(Ytrue, Ypred)


# In[ ]:


from sklearn.metrics import recall_score
recall_score(Ytrue, Ypred)


# In[ ]:


from sklearn.metrics import f1_score
f1_score(Ytrue, Ypred)


# In[ ]:




