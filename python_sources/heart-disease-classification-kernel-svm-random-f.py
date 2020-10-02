#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


ds = pd.read_csv('../input/heart.csv')


# In[5]:


ds.info()


# In[6]:


sns.heatmap(ds.isnull(), yticklabels = False, cbar=False, cmap='viridis')


# In[7]:


ds.head(5)


# In[8]:


sns.set_style('whitegrid')


# In[9]:


sns.countplot(x='target', hue='sex', data=ds, palette='RdBu_r')


# In[10]:


sns.countplot(x='target', hue='cp', data=ds)


# In[11]:


ds['age'].plot.hist(bins=35)


# In[12]:


ds.head(5)


# In[13]:


ds['trestbps'].hist(bins=40, figsize=(10, 4))


# In[14]:


plt.figure(figsize=(10, 8))
sns.boxplot(x='ca', y='age', data=ds)


# In[ ]:





# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# In[16]:


ds.head(5)


# In[17]:


y = ds.iloc[:, 13].values.reshape(-1, 1)
X = ds.iloc[:, 0:13].values


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[19]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[20]:


# Logistic Regression


# In[21]:


model_forScaledFeatures = LogisticRegression()
model_forScaledFeatures.fit(X_train, y_train)
y_pred_forscaledfeatures = model_forScaledFeatures.predict(X_test)
print(classification_report(y_test, y_pred_forscaledfeatures))
print('\n')
print(confusion_matrix(y_test, y_pred_forscaledfeatures))


# In[22]:


# K-Nearest Neighbors


# In[23]:


from sklearn.neighbors import KNeighborsClassifier


# In[24]:


#Elbow method
error_rate = []

for i in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=i, metric='minkowski', p=2)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[25]:


plt.figure(figsize=(12, 8))
plt.plot(range(1, 30), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate v/s K value')
plt.xlabel('K value')
plt.ylabel('Error rate')
plt.show()


# In[26]:


classifier_knn = KNeighborsClassifier(n_neighbors=11, metric='minkowski', p=2)


# In[27]:


classifier_knn.fit(X_train, y_train)


# In[28]:


ypred_from_knn = classifier_knn.predict(X_test)


# In[29]:


print(classification_report(y_test, ypred_from_knn))
print('\n')
print(confusion_matrix(y_test, ypred_from_knn))


# In[30]:


#SVM 


# In[31]:


from sklearn.svm import SVC


# In[32]:


model_SVC = SVC(kernel='linear', random_state=0)
model_SVC.fit(X_train, y_train)
ypred_from_svc = model_SVC.predict(X_test)
print(classification_report(y_test, ypred_from_svc))
print('\n')
print(confusion_matrix( y_test, ypred_from_svc))


# In[33]:


#Kernel SVM (Gaussian Kernel)


# In[34]:


model_SVM_Kernel = SVC(kernel='rbf', random_state=0)
model_SVM_Kernel.fit(X_train, y_train)
ypred_from_SVMKernel = model_SVM_Kernel.predict(X_test)
print(classification_report(y_test, ypred_from_SVMKernel))
print('\n')
print(confusion_matrix( y_test, ypred_from_SVMKernel))


# In[35]:


#Decision-Tree


# In[36]:


from sklearn.tree import DecisionTreeClassifier


# In[37]:


model_tree = DecisionTreeClassifier()
model_tree.fit(X_train, y_train)
ypred_from_tree = model_tree.predict(X_test)
print(classification_report(y_test, ypred_from_tree))
print('\n')
print(confusion_matrix( y_test, ypred_from_tree))


# In[38]:


#Random-Forest


# In[39]:


from sklearn.ensemble import RandomForestClassifier


# In[40]:


model_randomtree = RandomForestClassifier(n_estimators=200)
model_randomtree.fit(X_train, y_train)
ypred_from_randomtree = model_randomtree.predict(X_test)
print(classification_report(y_test, ypred_from_randomtree))
print('\n')
print(confusion_matrix( y_test, ypred_from_randomtree))


# In[ ]:




