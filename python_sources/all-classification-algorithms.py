#!/usr/bin/env python
# coding: utf-8

# # KNN algorithm

# In[ ]:


import pandas as pd 
import numpy as np


# In[ ]:


data = pd.read_csv("../input/Iris.csv")


# In[ ]:


data.head()


# In[ ]:


data.drop(["Id"],axis=1)


# In[ ]:


data.info()


# In[ ]:


y=data.Species.values
x_data=data.drop(["Species"],axis=1).values


# Normalization of data

# In[ ]:


x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))


# Train and tet split

# In[ ]:


from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)


# In[ ]:


# NN- algorithm
from sklearn.neighbors import KNeighborsClassifier
knn =KNeighborsClassifier(n_neighbors =7)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)


# In[ ]:


print("{} nn score:{}".format(7,knn.score(x_test,y_test)))


# In[ ]:


import matplotlib.pyplot as plt
score_list=[]
for each in range(1,30):
    knn2=KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
plt.plot(range(1,30),score_list)
plt.xlabel("k values")
plt.ylabel=("accuracy")
plt.show()


# # Now , we will examine the dataset using logistic regression methods

# first step is read csv file using pandas 
# second step is analyse data  and  droping unnecessary columns 

# In[ ]:


import pandas as pd 
import numpy as np 

data = pd.read_csv("../input/Iris.csv")

data.info()


# Id is unnecessary  column. Therefore, we must drop from iris dataset

# In[ ]:


data.drop("Id",axis=1)


# In[ ]:


x_data = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# # Now , we make normalization to bring data between 1 and 0 do
# # normalization formula is (x - min(x))/(max(x)-min(x))

# In[ ]:


x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))


# # this step is train and test split

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=100)


# In[ ]:


# importing cross_validation functions

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[ ]:


# Training on linear model

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))
y_true = y_test
# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)


#  cm visualization
import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.show()


# # Now, we examine Naive bayas classification

# In[ ]:





# In[ ]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))

y_true = y_test
#%% confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)


# %% cm visualization
import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.show()


# # training on support vector machine
# 

# In[ ]:


from sklearn.svm import SVC

classifier = SVC()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))

y_true = y_test
#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)


#  cm visualization
import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.show()


# # Decision Tree's
# 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
print("score",dt.score(x_test,y_test))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# # Random Forest 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100,random_state = 1)
rf.fit(x_train,y_train)
print("random forest algo result: ",rf.score(x_test,y_test))


y_pred = rf.predict(x_test)
y_true = y_test
# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)


#  cm visualization
import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.show()


# # CONCLUSION
# - Except of logistic regression, all classification algorithms  have  accuracy of 1.0 meaning all test data 
