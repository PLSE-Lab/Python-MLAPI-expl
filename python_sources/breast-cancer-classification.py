#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


# In[2]:


breast_cancer_data = pd.read_csv('../input/data.csv')


# In[3]:


breast_cancer_data.shape


# In[4]:


breast_cancer_data.head()


# In[5]:


breast_cancer_data.describe()


# In[6]:


breast_cancer_data.info()


# In[7]:


breast_cancer_data = breast_cancer_data.drop(['id','Unnamed: 32'], axis=1)


# In[8]:


breast_cancer_data.describe()


# In[9]:


# Correlation between all the features
breast_cancer_data.corr()


# In[10]:


# Remove the highly co-related features

breast_cancer_data = breast_cancer_data.drop(['radius_mean','perimeter_mean','concave points_mean','radius_se','radius_worst','perimeter_worst','perimeter_se','concave points_worst','compactness_worst','texture_worst','area_worst','concavity_worst'], axis=1)


# In[11]:


# Now we have the feauters which are least related to each-other
breast_cancer_data.corr()


# In[12]:


breast_cancer_data.shape
n_features = breast_cancer_data.shape[1]


# In[13]:


breast_cancer_data.head()
target = breast_cancer_data['diagnosis']
breast_cancer_data = breast_cancer_data.drop(['diagnosis'], axis=1)


# In[14]:


# Split train and test data
x_train, x_test, y_train, y_test = train_test_split(breast_cancer_data, target)


# In[15]:


# Find the depth which gives high accuracy
for depth in range(1,20):
    clf = RandomForestClassifier(max_depth = depth)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Accuracy is: {} at depth {}'.format(accuracy_score(y_test, y_pred), depth))
    #print('Score is: ',clf.score(x_test,y_test))


# In[16]:


# Classification
clf = RandomForestClassifier(max_depth = 5)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print('Accuracy is: {}'.format(accuracy_score(y_test, y_pred)))


# In[17]:


# Confusion Matrix
confusion_matrix = confusion_matrix(y_test,y_pred)
confusion_matrix


# In[18]:


clf_dt = tree.DecisionTreeClassifier(max_depth = 5)
clf_dt.fit(x_train,y_train)
y_pred_dt = clf_dt.predict(x_test)
print('Accuracy is: {}'.format(accuracy_score(y_test, y_pred_dt)))


# In[ ]:




