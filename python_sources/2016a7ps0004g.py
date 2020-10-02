#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix  
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from math import sqrt
from sklearn.metrics import mean_squared_error


# In[ ]:


train = pd.read_csv("../input/eval-lab-1-f464-v2/train.csv")
train.dropna(axis = 0, inplace=True)
train["type"] = train.type.eq("new").mul(1)


# In[ ]:


train.head()


# In[ ]:


test = pd.read_csv("../input/eval-lab-1-f464-v2/test.csv")
test.fillna(test.mean(), inplace = True)
test["type"] = test.type.eq("new").mul(1)
testX = test.drop(["id",], axis = 1)


# In[ ]:


test.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X = train.drop(["id", "rating"], axis = 1)
Y = train["rating"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)


# In[ ]:


# from sklearn.preprocessing import StandardScaler

# numerical_feature = ["feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "feature10", "feature11"]

# x_train[numerical_feature] = StandardScaler().fit_transform(x_train[numerical_feature])
# x_test[numerical_feature] = StandardScaler().fit_transform(x_test[numerical_feature])
# x_train[numerical_feature].describe()


# In[ ]:


# from sklearn.preprocessing import RobustScaler

# numerical_feature = ["feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "type", "feature10", "feature11"]

# scaler = RobustScaler()
# x_train[numerical_feature] = scaler.fit_transform(x_train[numerical_feature])
# x_test[numerical_feature] = scaler.transform(x_test[numerical_feature])

# x_train[numerical_feature].head()


# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
clf=RandomForestRegressor(n_estimators=500, random_state=42, max_depth=50)
clf.fit(x_train, y_train)


# In[ ]:


rdfpredreg = clf.predict(x_test)


# In[ ]:


for i in range(len(rdfpredreg)):
    rdfpredreg[i] = round(rdfpredreg[i])


# In[ ]:


rdfpredreg = rdfpredreg.astype(np.int64)


# In[ ]:


accuracy_score(y_test, rdfpredreg)


# In[ ]:


sqrt(mean_squared_error(y_test, rdfpredreg))


# In[ ]:


clf=RandomForestRegressor(n_estimators=500, random_state=42, max_depth=50)
clf.fit(X, Y)


# In[ ]:


rdfpredregFinal = clf.predict(testX)


# In[ ]:


rdfpredregFinal


# In[ ]:


for i in range(len(rdfpredregFinal)):
    rdfpredregFinal[i] = round(rdfpredregFinal[i])


# In[ ]:


rdfpredregFinal = rdfpredregFinal.astype(np.int64)


# In[ ]:


testId = test["id"]
a = list(zip(["id",], ["rating",]))
a = a + (list(zip(testId, rdfpredregFinal)))
for i in range(len(a)):
    a[i] = list(a[i])


# In[ ]:


finaldf = pd.DataFrame(data=a[1:][0:], columns=a[0][0:])
finaldf.to_csv('rdfreg.csv', index = False)


# ## Event Trees Regressor

# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor
clf = ExtraTreesRegressor(n_estimators=500, random_state=42, max_depth=50)
clf.fit(x_train, y_train)


# In[ ]:


etrpred = clf.predict(x_test)


# In[ ]:


for i in range(len(etrpred)):
    etrpred[i] = round(etrpred[i])


# In[ ]:


accuracy_score(y_test, etrpred)


# In[ ]:


sqrt(mean_squared_error(y_test, etrpred))


# In[ ]:


# scaler = StandardScaler()
# X[numerical_feature] = scaler.fit_transform(X[numerical_feature])

# X[numerical_feature].head()


# In[ ]:


clf = ExtraTreesRegressor(n_estimators=500, random_state=42, max_depth=50)
clf.fit(X, Y)


# In[ ]:


# scaler = StandardScaler()
# testX[numerical_feature] = scaler.fit_transform(testX[numerical_feature])

# testX[numerical_feature].head()


# In[ ]:


etrFinal = clf.predict(testX)


# In[ ]:


len(etrFinal)


# In[ ]:


type(etrFinal[0])


# In[ ]:


for i in range(len(etrFinal)):
    etrFinal[i] = round(etrFinal[i])


# In[ ]:


etrFinal = etrFinal.astype(np.int64)


# In[ ]:


testId = test["id"]
a2 = list(zip(["id",], ["rating",]))
a2 = a2 + (list(zip(testId, etrFinal)))
for i in range(len(a2)):
    a2[i] = list(a2[i])


# In[ ]:


finaldf2 = pd.DataFrame(data=a2[1:][0:], columns=a2[0][0:])


# In[ ]:


finaldf2.to_csv('rdfETR.csv', index = False)


# In[ ]:




