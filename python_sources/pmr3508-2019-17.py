#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import sklearn
import numpy as np


# In[ ]:


train="../input/adult-pmr3508/train_data.csv"
test="../input/adult-pmr3508/test_data.csv"


# In[ ]:


test = pd.read_csv(test,
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")


# In[ ]:


train = pd.read_csv(train,
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country","Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")


# In[ ]:


train.drop(train.index[0],inplace=True)
test.drop(test.index[0],inplace=True)


# In[ ]:


test.head() 

 
# In[ ]:


train.head() 


# In[ ]:


train["Country"].value_counts()


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


train["Age"].value_counts().plot(kind="bar")


# In[ ]:


train["Sex"].value_counts().plot(kind="bar")


# In[ ]:


train["Education"].value_counts().plot(kind="bar")


# In[ ]:


train["Occupation"].value_counts().plot(kind="bar")


# In[ ]:


ntrain = train.dropna()


# In[ ]:


train.shape


# In[ ]:


#fazendo tudo igual para o test


# In[ ]:


ntest = test.dropna()


# In[ ]:


test.shape


# In[ ]:


Xtrain = ntrain[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]


# In[ ]:


Ytrain = ntrain.Target


# In[ ]:


Xtest = ntest[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
knn.fit(Xtrain,Ytrain)
YtestPred = knn.predict(Xtest) 
YtestPred


# In[ ]:


best_mean = 0
best_k = 0
for k in range(3,30):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
    mean = np.mean(scores)
    if (mean > best_mean):
        best_mean = mean
        best_k = k
        
    
    


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=k)
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
knn.fit(Xtrain,Ytrain)
YtestPred = knn.predict(Xtest) 
YtestPred


# In[ ]:


savepath = "YtestPred.csv"
prev = pd.DataFrame(YtestPred, columns = ["income"])
prev.to_csv(savepath, index_label="Id")
prev


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




