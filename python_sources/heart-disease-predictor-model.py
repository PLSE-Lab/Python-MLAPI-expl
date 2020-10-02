#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import accuracy_score,precision_recall_curve,precision_score,recall_score,confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss


# In[ ]:


dataset = pd.read_csv('../input/heart.csv')


# In[ ]:


columns = dataset.columns[0:-1]

print(columns)


# In[ ]:


sns.countplot(dataset.target)
plt.show()


# It is hard to call this data set imbalanced. But I have not so much data. So I just want to use an over-sampling technique which is called SMOTE.

# In[ ]:


smote = SMOTE()
dataset_x,dataset_y = smote.fit_sample(dataset.drop('target',axis=1),dataset[['target']])


# In[ ]:


dataset_x = pd.DataFrame(dataset_x)
dataset_x.columns = columns


# In[ ]:


dataset_y = pd.DataFrame(dataset_y)
dataset_y.columns = ['target']
dataset_y.head()


# In[ ]:


sns.countplot(dataset_y.target)


# Now number of 0s equals to number of 1s

# In[ ]:


dataset = dataset_x.join(dataset_y,how='right')
dataset.head()


# In[ ]:


normal = dataset[dataset.target == 0]
disease = dataset[dataset.target == 1]


# I have seperated data set into two sets which are for normal and disease. So I will plot for both and check distributions. 

# In[ ]:


for i in columns:
    sns.distplot(normal[i],color='g')
    sns.distplot(disease[i],color='r')
    plt.show()


# In[ ]:


sns.heatmap(dataset.corr())


# In[ ]:


dataset = dataset[['cp','thalach','oldpeak','target']]
dataset.head()


# Distributions are really bad. But I take least bad ones.

# In[ ]:


train_x, test_x, train_y, test_y = train_test_split(dataset.drop('target',axis=1),dataset[['target']],test_size = 0.20,random_state = 42)


# In[ ]:


scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)


# After this point, I check each metrics for each methods. Some methods are missing, I know..

# In[ ]:


cl = LogisticRegression()
cl.fit(train_x,train_y)
pred = cl.predict(test_x)
cm = confusion_matrix(test_y,pred)
print("Confusion Matrix")
print(cm)
print("Accuracy Score: ",accuracy_score(test_y,pred))
print("Precision Score: ",precision_score(test_y,pred))
print("Recall Score: ",recall_score(test_y,pred))


# In[ ]:


cl = KNeighborsClassifier(n_neighbors=5)
cl.fit(train_x,train_y)
pred = cl.predict(test_x)
cm = confusion_matrix(test_y,pred)
print("Confusion Matrix")
print(cm)
print("Accuracy Score: ",accuracy_score(test_y,pred))
print("Precision Score: ",precision_score(test_y,pred))
print("Recall Score: ",recall_score(test_y,pred))


# In[ ]:


cl = RandomForestClassifier()
cl.fit(train_x,train_y)
pred = cl.predict(test_x)
cm = confusion_matrix(test_y,pred)
print("Confusion Matrix")
print(cm)
print("Accuracy Score: ",accuracy_score(test_y,pred))
print("Precision Score: ",precision_score(test_y,pred))
print("Recall Score: ",recall_score(test_y,pred))


# In[ ]:


cl = GaussianNB()
cl.fit(train_x,train_y)
pred = cl.predict(test_x)
cm = confusion_matrix(test_y,pred)
print("Confusion Matrix")
print(cm)
print("Accuracy Score: ",accuracy_score(test_y,pred))
print("Precision Score: ",precision_score(test_y,pred))
print("Recall Score: ",recall_score(test_y,pred))


# In[ ]:


cl = SVC()
cl.fit(train_x,train_y)
pred = cl.predict(test_x)
cm = confusion_matrix(test_y,pred)
print("Confusion Matrix")
print(cm)
print("Accuracy Score: ",accuracy_score(test_y,pred))
print("Precision Score: ",precision_score(test_y,pred))
print("Recall Score: ",recall_score(test_y,pred))


# In[ ]:


cl = MLPClassifier(hidden_layer_sizes = 3,activation='relu',solver='adam',warm_start = False,max_iter=500)
cl.fit(train_x,train_y)
pred = cl.predict(test_x)
cm = confusion_matrix(test_y,pred)
print("Confusion Matrix")
print(cm)
print("Accuracy Score: ",accuracy_score(test_y,pred))
print("Precision Score: ",precision_score(test_y,pred))
print("Recall Score: ",recall_score(test_y,pred))


# **SUMMARY**
# 
# I assume that I have a scenario and in this scenario a person comes to doctor. Doctor checks her heart disease probability by using model. If model says she has heart disease, doctor wants check-up. For this scenario, if you call a normal person as patient, you do not loss anything. So here recall is more important. I can call normal one as patient but not patient as normal. So I choose to use Naive Bayes method for this data.
