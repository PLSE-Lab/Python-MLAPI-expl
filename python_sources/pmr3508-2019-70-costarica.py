#!/usr/bin/env python
# coding: utf-8

# # PMR3508-2019-70 - KNN applied to Costa Rica household poverty base

# In[ ]:


import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


#HHI = pd.read_csv("../input/train.csv", sep=r'\s*,\s*', engine='python', na_values="")
HHI = pd.read_csv("../input/costa-rican-household-poverty-prediction/train.csv",sep=r'\s*,\s*', engine='python', na_values="?")
HHItest = pd.read_csv("../input/costa-rican-household-poverty-prediction/test.csv",sep=r'\s*,\s*', engine='python', na_values="?")


# In[ ]:


print(HHI.shape)
print(HHItest.shape)


# In[ ]:


HHI


# # 1) Let's clean the datas

# In[ ]:


missingDatas = pd.DataFrame(HHI.isnull().sum()).rename(columns = {0: 'totNull'})
missingDatas.sort_values('totNull', ascending = False).head(8)


# In[ ]:


missingDatasTest = pd.DataFrame(HHItest.isnull().sum()).rename(columns = {0: 'totNull'})
missingDatasTest.sort_values('totNull', ascending = False).head(8)


# #### So we have 5 columns with missing datas. In this 5, 3 are critically touched.
# #### So I'll remove those 3 columns and only remove the concerned lines for the two others columns.

# In[ ]:


HHI_old = HHI
HHItest_old = HHItest

dropList = ["rez_esc","v18q1","v2a1"]
for col in dropList:
    HHI = HHI.drop(columns = col)
    HHItest = HHItest.drop(columns = col)

HHI.dropna(axis=0, how='any', inplace=True)
HHItest.dropna(axis=0, how='any', inplace=True)
    
    
print(HHI.shape)
print(HHItest.shape)


# In[ ]:


missingDatas = pd.DataFrame(HHI.isnull().sum()).rename(columns = {0: 'totNull'})
missingDatas.sort_values('totNull', ascending = False).head(8)


# ### Now there is no more missing values, let's see what type of values there are in this datasheets.

# In[ ]:


print(HHI.info())
print(" ")
print(HHItest.info())


# In[ ]:


print(HHI.select_dtypes('object').head())
print(" ")
print(HHItest.select_dtypes('object').head())


# ### Ok, so there are 5 columns which are not numericals... 2 from them are the ids and 3 which are other stuffs.
# ### Reading the project, we can understand that no =0 and yes=1 so we can try to replace this in the 3 last columns, and drop the 2 ids columns. Let's try.

# In[ ]:


dropList = ["Id","idhogar"]
IdDF = pd.DataFrame(HHI["Id"])
IdDFtest = pd.DataFrame(HHItest["Id"])

for col in dropList:
    HHI = HHI.drop(columns = col)
    HHItest = HHItest.drop(columns = col)


mapping = {"yes": 1, "no": 0}

# Apply same operation to both train and test
for df in [HHI, HHItest]:
    # Fill in the values with the correct mapping
    df['dependency'] = df['dependency'].replace(mapping).astype(np.float64)
    df['edjefa'] = df['edjefa'].replace(mapping).astype(np.float64)
    df['edjefe'] = df['edjefe'].replace(mapping).astype(np.float64)


# In[ ]:


print(HHI.info())
print(" ")
print(HHItest.info())


# ### Ok, great, this is it, let's begin !

# # 2) Let's see a first KNN try on the whole base

# In[ ]:


XHHI = HHI.drop(columns = "Target")
YHHI = HHI["Target"]


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, XHHI, YHHI, cv=8)
print(scores)
print(scores.mean())


# ### Let's find out if what's the "best" amount of neighbors for this first test !

# In[ ]:


scoresMeanList=[]
kList=[]
for k in range(1,101):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, XHHI, YHHI, cv=8)
    scoresMeanList.append(scores.mean())
    kList.append(k)    


# In[ ]:


maxi = max(scoresMeanList)
print("The best amount of neighbors for this forst test is {}, with a score of {}".format(scoresMeanList.index(maxi),maxi))
plt.plot(kList,scoresMeanList)


# # 3) Let's see what will it predict

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=73)
knn.fit(XHHI,YHHI)
YHHIpred = knn.predict(HHItest)
YHHIpred = pd.DataFrame(data = YHHIpred)


# In[ ]:


print(YHHIpred[0].value_counts())
YHHIpred[0].value_counts().plot("bar")


# In[ ]:


YHHI = pd.DataFrame(data = YHHI)
print(YHHI["Target"].value_counts())
YHHI["Target"].value_counts().plot("bar")


# ### As we can see, in the train base, there is a huge maority of people that are in the 4th category. So logically, with a high number of neighbors, the predicator will prefers say almost everyone is in the 4th category.
# ### Let's try with a smaller number of neighbors ?

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(XHHI,YHHI)
YHHIpred = knn.predict(HHItest)
YHHIpred = pd.DataFrame(data = YHHIpred)

print(YHHIpred[0].value_counts())
YHHIpred[0].value_counts().plot("bar")


# ### 3 neighbors seems to be a acceptable compromise.

# # 4) Let's create a formated file to make an acceptable output

# In[ ]:


results =[]
Ids = IdDFtest.values
target = YHHIpred.values
for i in range(len(Ids)):
    results.append([Ids[i][0],target[i][0]])
results


# In[ ]:


results = pd.DataFrame(columns=["Id","Target"] ,data = results)
results.head(21)


# In[ ]:


results.to_csv('results.csv', index=False)

