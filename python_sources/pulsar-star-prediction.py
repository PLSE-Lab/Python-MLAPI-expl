#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.metrics import accuracy_score,cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# In[ ]:


data=pd.read_csv("../input/predicting-a-pulsar-star/pulsar_stars.csv")


# In[ ]:


data.head()


# In[ ]:


data.describe().T


# In[ ]:


data.isna().sum()


# In[ ]:


data.info()


# In[ ]:


corr=data.corr()


# In[ ]:


plt.figure(figsize=(10,7))
sns.heatmap(corr,annot=True)


# In[ ]:


k=sns.countplot(data["target_class"])

for b in k.patches:
    k.annotate(format(b.get_height(),'.0f'),(b.get_x()+b.get_width() / 2.,b.get_height()))


# #lets check mean of star is different for pulsar or not a pulsar
# sns.boxplot(x=" Mean of the integrated profile",y="target_class",data=data,hue="target_class")

# In[ ]:


#Model Building
x=data.drop("target_class",axis=1)
y=data.target_class


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3, random_state=100)


# In[ ]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()


# In[ ]:


dtree.fit(xtrain,ytrain)


# In[ ]:


model_tree=dtree.predict(xtest)


# In[ ]:


print("Decision Tree Acc Score",accuracy_score(ytest,model_tree))
print("Decision Tree Kappa score",cohen_kappa_score(ytest,model_tree))


# In[ ]:


#Random Forest
rf = RandomForestClassifier()


# In[ ]:


rf.fit(xtrain,ytrain)


# In[ ]:


model_forest=rf.predict(xtest)


# In[ ]:


print("Random Forest Acc Score",accuracy_score(ytest,model_forest))
print("Random Forest Kappa Score",cohen_kappa_score(ytest,model_forest))


# In[ ]:


#Naive Bayes
nb = GaussianNB()
nb.fit(xtrain, ytrain)
model_nb=nb.predict(xtest)


# In[ ]:


print("Naive Bayes Acc Score",accuracy_score(ytest,model_nb))
print("Naive Bayes Kappa Score",cohen_kappa_score(ytest,model_nb))


# In[ ]:


#K-Nearest Neigbors
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(xtrain, ytrain)
model_knn = knn.predict(xtest)


# In[ ]:


print("KNN Acc Score",accuracy_score(ytest,model_knn))
print("KNN Kappa Score",cohen_kappa_score(ytest,model_knn))


# In[ ]:




