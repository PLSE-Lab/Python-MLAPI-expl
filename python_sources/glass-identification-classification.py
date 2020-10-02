#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# In[2]:


glass_data = pd.read_csv('../input/glass.csv')


# In[3]:


glass_data.shape


# In[4]:


glass_data.head()


# In[5]:


glass_data.describe()


# In[6]:


glass_data.info()


# In[7]:


glass_data.corr()


# In[8]:


# To check whether the data is balanced or imbalanced
glass_data['Type'].value_counts()


# In[9]:


n_features = glass_data.shape[1]
x_train, x_test, y_train, y_test = train_test_split(glass_data.iloc[:,0:n_features-1], glass_data.iloc[:,n_features-1], test_size = 0.33)


# In[10]:


x_train = x_train.values
x_test = x_test.values

y_train = y_train.values
y_test = y_test.values


# In[11]:


def predict(x_train, y_train, x_test, k):
    predictions = []
    for x in x_test:
        pred = predict_item(x_train, y_train, x, k)
        predictions.append(pred)
    return predictions

def predict_item(x_train, y_train, x_test, k):
    distances = []
    for i in range(len(x_train)):
        distance = ((x_train[i] - x_test)**2).sum()
        distances.append([distance,i])
    distances = sorted(distances)
    targets = []
    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index])
    return Counter(targets).most_common(1)[0][0]


# In[12]:


neighbors = 15
accuracies = []
for k in range(1, neighbors):
    y_pred = predict(x_train, y_train, x_test, k)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print("K-Neighbours are {} and Accuracy is {}".format(k,accuracy))


# In[13]:


print(confusion_matrix(y_test, y_pred))


# In[14]:


neighbors = range(1,neighbors)


# In[15]:


plt.plot(neighbors, accuracies, label='Accuracy on Test Data')
plt.xlabel('Neighbors')
plt.ylabel('Accuracy')
plt.legend()


# In[18]:


param_grid = {'n_neighbors':list(range(1,15)), 'metric': ['euclidean']} 
clf_g = GridSearchCV(KNeighborsClassifier(), param_grid)
clf_g.fit(x_train,y_train)
print('Best Parameters are \n',clf_g.best_params_)
print('Best Estimator are \n',clf_g.best_estimator_)


# In[19]:


# Confusion Matrix on using sklearn library
clf = KNeighborsClassifier(n_neighbors = 8)
clf.fit(x_train,y_train)
y_pred_KNN = clf.predict(x_test)
print(confusion_matrix(y_test, y_pred_KNN))


# In[20]:


for i in range(1,20):
    clf = RandomForestClassifier(max_depth = i)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Accuracy is: {} at depth {}'.format(accuracy_score(y_test, y_pred), i))


# In[21]:


sc = StandardScaler()
data = sc.fit_transform(glass_data.iloc[:,0:n_features-1])


# In[22]:


x_train, x_test, y_train, y_test = train_test_split(data, glass_data.iloc[:,n_features-1], test_size = 0.33)


# In[23]:


y_train = y_train.values
y_test = y_test.values


# In[24]:


neighbors = 15
accuracies = []
for k in range(1, neighbors):
    y_pred = predict(x_train, y_train, x_test, k)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print("K-Neighbours are {} and Accuracy is {}".format(k,accuracy))


# In[ ]:




