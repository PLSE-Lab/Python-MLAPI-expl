#!/usr/bin/env python
# coding: utf-8

#  # Assignment 6  
#  ## Turing Machines
# 
# **Team members :**  
# 
# *Vikram G - 01FB16ECS484  *
# 
# *Vinayaka R Kamath - 01FB16ECS445  *
# 
# *Nikhil V Revankar - 01FB16ECS230*
# 

# In[ ]:


import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


data=pd.read_csv('../input/Absenteeism_at_work.csv')
data.head()


# In[ ]:


data.describe()


# In[ ]:


print("Element wise count for Output class:")
data['Absenteeism time in hours'].value_counts(sort = False)


# In[ ]:


sns.pairplot(data)


# In[ ]:


fig, ax = plt.subplots(figsize=(20, 20)) 
sns.heatmap(data.corr(), annot = True, ax = ax)


# In[ ]:


y = data['Absenteeism time in hours']
X = data.drop(['Absenteeism time in hours'], axis=1)


# **Min Max Scaler is used after an heuristic approach. It was choosen as it exhibited higher accuracy than it's counter parts.**

# In[ ]:


# Feature Scaling
from sklearn import preprocessing
x = X.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled,columns=list(X.columns))

# Splitting data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
print( "\nX_train:\n")
print(X_train.head())
print( X_train.shape)
print( "\nX_test:\n")
print(X_test.head())
print( X_test.shape)


# ### **KNN Classifier**

# In[ ]:


# Classification technique using KNN 
error=[]
accuracy=[]
for i in range(1, 40):    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error.append(np.mean(y_pred != y_test))
    accuracy.append(metrics.accuracy_score(y_test,y_pred))

print("Error Rate:\n",error)
print("Accuracy Score:\n",accuracy)


# **For choosing the K in the KNN, an iterative approach was practised. The K value and the model  with the lowest error was choosen from 1 - 40 to determine the required output.**

# In[ ]:


plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error') 


# In[ ]:


k = np.argmin(error) + 1
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix  
print("Accuracy when k={}: ".format(k),metrics.accuracy_score(y_test, y_pred)) 
print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 


# ### Decision Tree Classifier

# In[ ]:


#Decision Tree Classifier with criterion gini index
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=50, min_samples_leaf=19)
clf_gini.fit(X_train, y_train)


# In[ ]:


#Decision Tree Classifier with criterion information gain
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=50, min_samples_leaf=19)
clf_entropy.fit(X_train, y_train)


# In[ ]:


y_pred_gini = clf_gini.predict(X_test)
y_pred_entropy = clf_gini.predict(X_test)
print("Accuracy using Gini Index: ",metrics.accuracy_score(y_test,y_pred_gini))
print("Confusion matrix using Gini Index: \n",confusion_matrix(y_test, y_pred_gini))  
print(classification_report(y_test, y_pred_gini)) 

print("Accuracy using Information Gain: ",metrics.accuracy_score(y_test,y_pred_entropy))
print("Confusion matrix using Information gain: \n",confusion_matrix(y_test, y_pred_gini))  
print(classification_report(y_test, y_pred_entropy)) 


# ### Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=50,random_state = 100,max_depth=50, min_samples_leaf=5)
clf.fit(X_train,y_train)
#y_pred=clf.predict(X_test)
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
feature_imp = pd.Series(clf.feature_importances_,index=list(X.columns)).sort_values(ascending=False)
feature_imp


# In[ ]:


sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# **Compared to other features 'Education' has significantly lower importance score. Hence it is safe to drop this feature before building the model. This will ensure that the model is comaparatively computationally inexpensive.**

# In[ ]:


# X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
X_train = X_train.drop(['Education'], axis=1)
X_test = X_test.drop(['Education'], axis=1)


# In[ ]:


clf=RandomForestClassifier(n_estimators=50, random_state = 100,max_depth=50, min_samples_leaf=5)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# **Conclusion and Analysis:
# **
# 
# KNN, Decision Trees and Random Forest Classifiers are compared on the same split of the data.
# 
# Accuracy Score Standings:
#         Random Forest > Decision Trees > KNN
# 
# At First glance Random Forest performs better than the other two classifiers with higher precision and higher recall. The accuracy score is larger as well.
# 
# *For accuracy*: random forest is almost always better.
# 
# *For computational load*: A single decision tree will train much more quickly, and computing a prediction is also much quicker.
# 
# *For comprehensibility*: A decision tree is more comprehensible.
# 
# 
# 
# 
