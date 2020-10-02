#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import sklearn


# In[ ]:


train_data = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")


# In[ ]:


adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")


# In[ ]:


adult.shape


# In[ ]:


adult.head()


# In[ ]:


adult["Country"].value_counts()


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


adult["Age"].value_counts().plot(kind="bar")


# In[ ]:


adult["Sex"].value_counts().plot(kind="bar")


# In[ ]:


adult["Education"].value_counts().plot(kind="bar")


# In[ ]:


adult["Occupation"].value_counts().plot(kind="bar")


# In[ ]:


nadult = adult.dropna()


# In[ ]:


nadult


# In[ ]:


nadult.drop('Id',axis=0,inplace=True)
nadult


# In[ ]:


testAdult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")


# In[ ]:


nTestAdult = testAdult.dropna()


# In[ ]:


Xadult = nadult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]


# In[ ]:


Yadult = nadult.Target


# In[ ]:


XtestAdult = nTestAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]


# In[ ]:


YtestAdult = nTestAdult.Target


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=3)


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


scores = cross_val_score(knn, Xadult, Yadult, cv=10)


# In[ ]:


scores


# In[ ]:


knn.fit(Xadult,Yadult)
XtestAdult


# In[ ]:


XtestAdult.drop('Id',axis=0,inplace=True)
XtestAdult


# In[ ]:


YtestAdult.drop('Id',axis=0,inplace=True)
YtestAdult


# In[ ]:


YtestPred = knn.predict(XtestAdult)


# In[ ]:


YtestPred


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(YtestAdult,YtestPred)


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=30)


# In[ ]:


knn.fit(Xadult,Yadult)


# In[ ]:


scores = cross_val_score(knn, Xadult, Yadult, cv=10)


# In[ ]:


YtestPred = knn.predict(XtestAdult)


# In[ ]:


accuracy_score(YtestAdult,YtestPred)


# In[ ]:


from sklearn import preprocessing


# In[ ]:


numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)


# In[ ]:


numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)


# In[ ]:


Xadult = numAdult.iloc[:,0:14]


# In[ ]:


Yadult = numAdult.Target


# In[ ]:


XtestAdult = numTestAdult.iloc[:,0:14]


# In[ ]:


YtestAdult = numTestAdult.Target


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=30)


# In[ ]:


knn.fit(Xadult,Yadult)


# In[ ]:


YtestPred = knn.predict(XtestAdult)


# In[ ]:


accuracy_score(YtestAdult,YtestPred)


# In[ ]:


Xadult = numAdult[["Age", "Workclass", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"]]


# In[ ]:


XtestAdult = numTestAdult[["Age", "Workclass", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"]]


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=30)


# In[ ]:


knn.fit(Xadult,Yadult)


# In[ ]:


YtestPred = knn.predict(XtestAdult)


# In[ ]:


accuracy_score(YtestAdult,YtestPred)


# In[ ]:


Xadult = numAdult[["Age", "Workclass", "Education-Num", 
        "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week"]]


# In[ ]:


XtestAdult = numTestAdult[["Age", "Workclass", "Education-Num", 
        "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week"]]


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=30)


# In[ ]:


knn.fit(Xadult,Yadult)


# In[ ]:


YtestPred = knn.predict(XtestAdult)


# In[ ]:


accuracy_score(YtestAdult,YtestPred)


# In[ ]:


Index = []
for j in range( len(YtestPred)):
    Index.append(j)


# In[ ]:


Id = pd.DataFrame(Index)


# In[ ]:


Pred = pd.DataFrame(YtestPred)
Pred.columns = ['Income']
Pred.insert(0, 'Id', Id, True)
Pred


# In[ ]:


Pred.to_csv('Trabalho_Adult.csv',header = True,index = False)


# In[ ]:




