#!/usr/bin/env python
# coding: utf-8

# Importando as bibliotecas

# In[ ]:


import pandas as pd
import numpy as np
import sklearn


# In[ ]:


adult = pd.read_csv("../input/adultdb/train_data.csv",
        names=[
        "Id","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")


# Analisando a base de dados

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


testAdult = pd.read_csv("../input/adultdb/test_data.csv",
        names=[
        "Id","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")


# In[ ]:


nTestAdult=testAdult


# In[ ]:


Xadult = nadult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]


# In[ ]:


Yadult = nadult.Target


# In[ ]:


XtestAdult = nTestAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
XtestAdult


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


# In[ ]:


YtestPred = knn.predict(XtestAdult)


# In[ ]:


YtestPred


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=30)


# In[ ]:


knn.fit(Xadult,Yadult)


# In[ ]:


scores = cross_val_score(knn, Xadult, Yadult, cv=10)


# In[ ]:


scores


# In[ ]:


YtestPred = knn.predict(XtestAdult)


# In[ ]:


YtestPred


# In[ ]:


result = np.vstack((nTestAdult["Id"], YtestPred)).T
x = ["id","income"]
Resultado = pd.DataFrame(columns = x, data = result)
Resultado.to_csv("results.csv", index = False)
Resultado


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




