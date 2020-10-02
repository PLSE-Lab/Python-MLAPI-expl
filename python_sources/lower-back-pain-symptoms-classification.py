#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 

import warnings
warnings.filterwarnings("ignore")

import seaborn as sns 
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


# In[ ]:


data = pd.read_csv("../input/Dataset_spine.csv")


# In[ ]:


data.info()


# In[ ]:


data.head()


# In[ ]:


data.drop("Unnamed: 13",axis=1,inplace=True)


# In[ ]:


data["Class_att"] = [1 if i =="Abnormal" else 0  for i in data.Class_att]


# In[ ]:


plt.figure(figsize = (15,15))
sns.heatmap(data = data.corr(), annot=True, linewidths=.3, fmt="1.2f")
plt.show()


# In[ ]:


data.describe()


# In[ ]:


c1 = data[data.Class_att == 1].describe()
c1


# In[ ]:


c2 = data[data.Class_att == 0].describe()
c2


# In[ ]:


plt.figure(figsize = (15,12))

plt.subplot(221)
sns.scatterplot(x=data.Col1,y=data.Col5,hue=data.Class_att)
plt.xlabel("Pelvic Incidence")
plt.ylabel("Pelvic Radius")

plt.subplot(222)
sns.scatterplot(x = data.Col1,y = data.Col4,hue=data.Class_att)
plt.xlabel("Pelvic Incidence")
plt.ylabel("Sacral Slope")

plt.subplot(223)
sns.scatterplot(x = data.Col4,y = data.Col5,hue=data.Class_att)
plt.xlabel("Sacral Slope")
plt.ylabel("Pelvic Radius")

plt.subplot(224)
sns.scatterplot(x = data.Col1,y = data.Col3,hue=data.Class_att)
plt.xlabel("Pelvic Incidence")
plt.ylabel("Lumbar Lordosis Angle")

plt.show()


# In[ ]:


plt.figure(figsize=(15,12))

plt.subplot(221)
sns.boxplot(x = data.Class_att,y = data.Col5)
plt.ylabel("Pelvic Radius")

plt.subplot(222)
sns.swarmplot(x = data.Class_att, y = data.Col6)
plt.ylabel("Degree Spondylolisthesis")
plt.show()


# In[ ]:


X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


# In[ ]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)


# In[ ]:


from sklearn.model_selection import train_test_split,cross_val_score
X_train,X_test,y_train,y_test = train_test_split(X,y ,test_size=0.3,random_state=47)


#  **Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,auc

lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)


# In[ ]:


cvs = cross_val_score(estimator = lr, X = X_train, y = y_train, cv = 10)

print(cvs)
print(50*"*")
print(cvs.mean())


# In[ ]:


cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=.3)
plt.show()

print(classification_report(y_test,y_pred))


# In[ ]:


import scikitplot as skplt

skplt.metrics.plot_roc_curve(y_test, lr.predict_proba(X_test),figsize=(6,6))
plt.show()

print("Auc Score: {}".format(roc_auc_score(y_test,y_pred)))


# **KNN Classification**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

l = []
for i in range(1,21):
    knn = KNeighborsClassifier(n_neighbors  = i)
    knn.fit(X_train,y_train)
    l.append(knn.score(X_test,y_test))

plt.figure(figsize=(10,5))    
sns.lineplot(x = range(1,21), y = l)
plt.show()


# In[ ]:


knn = KNeighborsClassifier(n_neighbors  = 17)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

plt.figure(figsize=(5,5))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,linewidths=.3)
plt.show()

print(classification_report(y_test,y_pred))


# **Support Vector Machine**

# In[ ]:


from sklearn.svm import SVC

l = []
for i in ["linear","poly","rbf"]:
    svc = SVC(kernel = i)
    svc.fit(X_train,y_train)
    y_pred = svc.predict(X_test)
    
    l.append(y_pred)


print("Linear score {}".format(classification_report(y_test,l[0])))
print("Poly score {}".format(classification_report(y_test,l[1])))
print("Rbf score {}".format(classification_report(y_test,l[2])))


# **Naive Bayes**

# In[ ]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train,y_train)
y_pred = nb.predict(X_test)

print(classification_report(y_test,y_pred))


# **Decision Tree**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

dt = DecisionTreeClassifier()

param_dist = {"max_depth": [3, None],
              "max_features": range(1, 9),
              "min_samples_leaf": range(1, 9),
              "criterion": ["gini", "entropy"]}

dt_cv =RandomizedSearchCV(estimator = dt,param_distributions=param_dist)
dt_cv.fit(X_train,y_train)
    
print(dt_cv.best_params_)
print(dt_cv.best_score_)


# In[ ]:


dt = DecisionTreeClassifier(criterion="gini",max_depth=  3,max_features=8,min_samples_leaf=7)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test) 
print(classification_report(y_test,y_pred))


# **Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

l = ["gini","entropy"]

grid = {"criterion":l,"n_estimators":range(1,20)}

rf = RandomForestClassifier()

rf_cv = GridSearchCV(estimator = rf , param_grid = grid , cv = 10)
rf_cv.fit(X_train,y_train)

print(rf_cv.best_params_)
print(rf_cv.best_score_)


# In[ ]:


rf = RandomForestClassifier(criterion="entropy",n_estimators = 10)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)

plt.figure(figsize=(5,5))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,linewidths=.3)
plt.show()

print(classification_report(y_test,y_pred))
print(rf.score(X_test,y_test))

