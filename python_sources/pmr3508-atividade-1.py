#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot
import numpy 


# In[ ]:


adultTrainData = pd.read_csv("../input/adultb/train_data.csv",
            names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        skiprows=[0],
        na_values="?")


# In[ ]:


adultTrainData.shape


# In[ ]:


adultTrainData.head(5)


# In[ ]:


cleanAdultTrainData = adultTrainData.dropna()
cleanAdultTrainData


# In[ ]:


from sklearn import preprocessing 
numCleanAdultTrainData = cleanAdultTrainData.apply(preprocessing.LabelEncoder().fit_transform)


# In[ ]:


adultTestData = pd.read_csv("../input/adultb/test_data.csv",
            names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"],
        sep=r'\s*,\s*',
        engine='python',
        skiprows=[0],
        na_values="?")


# In[ ]:


adultTestData.shape


# In[ ]:


adultTestData.head(5)


# In[ ]:


cleanAdultTestData = adultTestData.dropna()
cleanAdultTestData


# In[ ]:


from sklearn import preprocessing 
numCleanAdultTestData = cleanAdultTestData.apply(preprocessing.LabelEncoder().fit_transform)
numCleanAdultTestData


# In[ ]:


adultTrainData["Workclass"].value_counts().plot(kind = "bar")


# In[ ]:


adultTrainData["Age"].value_counts().plot(kind = "bar")


# In[ ]:


adultTrainData["Education"].value_counts().plot(kind = "bar")


# In[ ]:


adultTrainData["Capital Gain"].value_counts().plot(kind = "pie")


# In[ ]:


adultTrainData["Capital Loss"].value_counts().plot(kind = "pie")


# In[ ]:


adultTrainData["Hours per week"].value_counts().plot(kind = "pie")


# In[ ]:


adultTrainData["Race"].value_counts().plot(kind = "pie")


# In[ ]:


adultTrainData["Sex"].value_counts().plot(kind = "pie")


# In[ ]:


adultTrainData.describe()


# In[ ]:


Xadult = numCleanAdultTrainData[["Age", "Relationship","Race", "Martial Status", "Sex", "Country","Education-Num","Capital Gain", "Capital Loss"]]
XTestAdult = numCleanAdultTestData[["Age", "Relationship","Race", "Martial Status", "Sex", "Country","Education-Num","Capital Gain", "Capital Loss"]]
Yadult = cleanAdultTrainData.Target


# In[ ]:


n = 1;
knn = KNeighborsClassifier(n_neighbors=n)


# In[ ]:


scores = cross_val_score(knn, Xadult, Yadult, cv = 10) 
scoresAux = numpy.copy(scores)
while scores.mean() >= scoresAux.mean():
    scoresAux = numpy.copy(scores)
    n = n + 5
    knn = KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(knn, Xadult, Yadult, cv = 10)
scores = numpy.copy(scoresAux)
n = n - 5
while scores.mean() >= scoresAux.mean():
    scoresAux = numpy.copy(scores)
    n = n + 1
    knn = KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(knn, Xadult, Yadult, cv = 10)
scores = numpy.copy(scoresAux)
n = n - 1


# In[ ]:


print('n: ' + str(n))
print('Scores mean: ' + str(scores.mean()))
knn = KNeighborsClassifier(n_neighbors=n)


# In[ ]:


knn.fit(Xadult, Yadult)


# In[ ]:


YtestPred2 = knn.predict(XTestAdult)
YtestPred2


# In[ ]:


YtestPredFinal = []

count = 0
for i in range(YtestPred2.size):
    row = i - count
    dif =numCleanAdultTestData.index.values[row] - i 
    if ( dif == 0):
        YtestPredFinal.append(YtestPred2[row])
    else:
        for j in range(dif):
            YtestPredFinal.append(YtestPred2[row])
            count = count + 1
YtestPredFinal


# In[ ]:


Id = []
for i in range(0, YtestPred2.size):
    Id.append(i)


# In[ ]:


d = {'Id' : Id, 'Income' : YtestPredFinal}
my_df = pd.DataFrame(d) 
my_df.to_csv('prediction.csv',
             index=False, sep=',', line_terminator = '\n', header = ["Id", "income"])


# In[ ]:


my_df

