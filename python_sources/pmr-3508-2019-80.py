#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


adult = pd.read_csv('../input/adult-pmr3508/train_data.csv', names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")


# In[ ]:


adultTest = pd.read_csv('../input/adult-pmr3508/test_data.csv', names=["Id",
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")


# In[ ]:


nadult = adult.dropna()


# In[ ]:


Xadult = nadult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
Xadult.head()


# In[ ]:


Yadult = nadult.Target


# In[ ]:


XtestAdult = adultTest[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
XtestAdult.head()


# In[ ]:


for a in XtestAdult.columns:
    XtestAdult[a].fillna(XtestAdult[a].mode()[0],inplace=True)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


# In[ ]:


Xadult.drop(['Id'])


# In[ ]:


Xadult.to_csv('Xadult', header=False, index=False)
Xadult = pd.read_csv('Xadult', header=None)


# In[ ]:


Xadult = Xadult.drop([0])


# In[ ]:


Yadult = Yadult.drop(['Id'])


# In[ ]:


best = 0
media_anterior = 0
for i in range(1,30):
    knn = KNeighborsClassifier(n_neighbors = i)
    scores = cross_val_score(knn, Xadult, Yadult, cv=10)
    media = sum(scores)/len(scores)
    if media > media_anterior:
        media_anterior = media
        best = i
print(media_anterior)
print(best)


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = best)


# In[ ]:


knn.fit(Xadult,Yadult)


# In[ ]:


XtestAdult = XtestAdult.drop([0])


# In[ ]:


XtestAdult.to_csv('XtestAdult', header=False, index=False)
XtestAdult = pd.read_csv('XtestAdult', header=None)


# In[ ]:


XtestAdult


# In[ ]:


YtestPred = knn.predict(XtestAdult)


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


Pred.to_csv('predictionAdult.csv')

