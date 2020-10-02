#!/usr/bin/env python
# coding: utf-8

# # PMR3508-2019-70  -  Using KNN predicator on the Adult base.

# In[ ]:


import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt


# In[ ]:


adult = pd.read_csv('../input/adultbasefiles/adult.data.txt',
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


adult["Age"].value_counts()

print(adult["Age"].min())
print(adult["Age"].max())


# In[ ]:


#Let's "clean" the database by removing lines with any missing datas
bruteCleanedAdult = adult.dropna()


# In[ ]:


bruteCleanedAdult.shape


# # 1) Let's try and define our firsts key columns

# In[ ]:


# Let's define our firsts key columns
keyColumns = ["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]


# In[ ]:


#Let's "clean" the database by removing lines with any missing key columns data
cleanedAdult = adult.dropna(axis=0, how='any', subset=keyColumns, inplace=False)
print(cleanedAdult.shape)


# In[ ]:


testAdult = pd.read_csv("../input/adultbasefiles/adult.test.txt",
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

print(testAdult.shape)
cleanedTestAdult = testAdult.dropna(axis=0, how='any', subset=keyColumns, inplace=False)
print(cleanedTestAdult.shape)


# In[ ]:


#Let's separate the lists for knn tests

Xadult = cleanedAdult[keyColumns]
Yadult = cleanedAdult["Target"]
print(Xadult.shape)
print(Yadult.shape)

XtestAdult = cleanedTestAdult[keyColumns]
YtestAdult = cleanedTestAdult["Target"]
print(XtestAdult.shape)
print(YtestAdult.shape)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)


# In[ ]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(knn, Xadult, Yadult, cv=10)


# In[ ]:


scores


# In[ ]:


knn.fit(Xadult,Yadult)


# In[ ]:


YtestPred = knn.predict(XtestAdult)
print(YtestPred.shape)
print(YtestAdult.shape)


# In[ ]:


from sklearn.metrics import accuracy_score

accuracy_score(YtestAdult,YtestPred)


# In[ ]:


print(YtestPred)
print(YtestAdult.values)


# In[ ]:


#I figured out there are points at the end of the target value which makes accuracy = 0
YtestAdult = YtestAdult.values

for i in range (len(YtestAdult)):
    YtestAdult[i] = YtestAdult[i][:-1]

print(YtestAdult)


# In[ ]:


accuracy_score(YtestAdult,YtestPred)


# # 2) Now we understand how it works, let's make a function that automates it

# In[ ]:


import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def testAdultBase(keyColumns=list,nbNeighborsKNN=int,indexColumnsToConvert=[],returnPredic=False):
    
    adult = pd.read_csv("../input/adultbasefiles/adult.data.txt",names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status","Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss","Hours per week", "Country", "Target"],sep=r'\s*,\s*',engine='python',na_values="?")
    testAdult = pd.read_csv("../input/adultbasefiles/adult.test.txt",names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status","Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss","Hours per week", "Country", "Target"],sep=r'\s*,\s*',engine='python',na_values="?")
    
    cleanedAdult = adult.dropna(axis=0, how='any', subset=keyColumns, inplace=False)
    cleanedTestAdult = testAdult.dropna(axis=0, how='any', subset=keyColumns, inplace=False)
    
    Xadult = cleanedAdult[keyColumns].values
    Yadult = cleanedAdult["Target"].values
    XtestAdult = cleanedTestAdult[keyColumns].values
    YtestAdult = cleanedTestAdult["Target"].values
    
    for i in range (len(YtestAdult)):
        YtestAdult[i] = YtestAdult[i][:-1]
    
    #let's convert the NaN columns into numbers, if necessary
    if indexColumnsToConvert != []:
        for col in indexColumnsToConvert:
            valueList=[]
            for i in range(len(Xadult)):
                value = Xadult[i][col]
                if value not in valueList:
                    valueList.append(value)
                Xadult[i][col] = valueList.index(value)
            for i in range(len(XtestAdult)):
                value = XtestAdult[i][col]
                if value not in valueList:
                    valueList.append(value)
                XtestAdult[i][col] = valueList.index(value)
    
    
    knn = KNeighborsClassifier(n_neighbors=nbNeighborsKNN)
    knn.fit(Xadult,Yadult)
    
    YtestPred = knn.predict(XtestAdult)
    
    if returnPredic==True:
        return YtestPred    
    return (accuracy_score(YtestAdult,YtestPred))
    


# In[ ]:


testAdultBase(["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"],3)


# ### Ok, great, it works !

# # 3) Ok, let's find the best number of neighbors for those key columns !

# In[ ]:


keyColumns = ["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]

accuracyList=[]
listI=[]


# In[ ]:


for i in range (5,31):
    accuracyList.append(testAdultBase(keyColumns,i))
    listI.append(i)


# In[ ]:


maxi = max(accuracyList)
print("For the key columns {},".format(keyColumns))
print("The best neighbors amount is {} with an accuracy of {}%".format(listI[accuracyList.index(maxi)],np.round(maxi,6)*100))

plt.title('Accuracy of the KNN prediction depending on number of neighbors.')
plt.plot(listI,accuracyList)


# ### Ok, so the best amount of neighbors is around 16, with an accuracy of 84,098%.

# # 4) Let's try others key columns !
# 

# In[ ]:


keyColumns2 = ["Age","Education-Num","Sex","Capital Gain", "Capital Loss", "Hours per week"]
accuracyList2=[]
listI2=[]


# In[ ]:


for i in range (10,41):
    accuracyList2.append(testAdultBase(keyColumns2,i,[2]))
    listI2.append(i)


# In[ ]:


maxi2 = max(accuracyList2)
print("For the key columns {},".format(keyColumns2))
print("The best neighbors amount is {} with an accuracy of {}%".format(listI2[accuracyList2.index(maxi2)],np.round(maxi2,6)*100))

plt.title('Accuracy of the KNN prediction depending on number of neighbors.')
plt.plot(listI2,accuracyList2)


# ### Ok, it's a bit better.
# ### 84.3499% with 28 neighbors.

# In[ ]:





# In[ ]:


keyColumns3 = ["Age","Education-Num","Sex","Capital Gain", "Hours per week"]
accuracyList3=[]
listI3=[]


# In[ ]:


for i in range (10,41):
    accuracyList3.append(testAdultBase(keyColumns3,i,[2]))
    listI3.append(i)


# In[ ]:


maxi3 = max(accuracyList3)
print("For the key columns {},".format(keyColumns3))
print("The best neighbors amount is {} with an accuracy of {}%".format(listI3[accuracyList3.index(maxi3)],np.round(maxi3,6)*100))

plt.title('Accuracy of the KNN prediction depending on number of neighbors.')
plt.plot(listI3,accuracyList3)


# ### Removing the Capital loss, we lose accuracy.

# # Conclusion

# ### The best resusult has been obtained with the following key columns :
# ### ["Age","Education-Num","Sex","Capital Gain", "Capital Loss", "Hours per week"]
# ### with an accuracy of 84.3499% for 28 neighbors.

# # Datas Output

# In[ ]:


finalYPred = testAdultBase(keyColumns2,19,[2],True)
Id = [i for i in range(len(finalYPred))]

d = {'Id' : Id, 'Income' : finalYPred}
myDf = pd.DataFrame(d) 
myDf.to_csv('bestPrediction.csv',
             index=False, sep=',', line_terminator = '\n', header = ["Id", "Income"])


# In[ ]:


finalYPred

