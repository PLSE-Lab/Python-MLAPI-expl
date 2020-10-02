#!/usr/bin/env python
# coding: utf-8

# Hello, everyone! I am new here. 

# <h1>Data Analysis of Heart Disease </h1>

# <h2>Import Required Modules</h2>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings("ignore")


# <h2>Read Data</h2>

# In[ ]:


df=pd.read_csv("../input/heart.csv")
df.head()


# In[ ]:


#Count the number of rows and columns in the daha set
df.shape


# In[ ]:


#count the number of missing values in each columns
df.isna().sum()


# In[ ]:


#get a count of the number of target(1) or not(0)
df.target.value_counts()


# In[ ]:


#visualize the count
sns.countplot(df.target,label="count")
plt.show()


# In[ ]:


#create a pair plot
sns.pairplot(df,hue="target")
plt.show()


# In[ ]:


df.corr()


# In[ ]:


#visualize the correlation
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), annot=True,fmt=".0%")
plt.show()


# In[ ]:


#Split the data set into independent(x) and dependent (y) data sets
x=df.iloc[:,0:13].values
y=df.iloc[:,-1].values


# In[ ]:


#split the data set into 75% training and 25% testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[ ]:


#scale the data(feature scaling)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[ ]:


#create a function for the models
def models(x_train,y_train):
  #Logistic Regression Model
  from sklearn.linear_model import LogisticRegression
  log=LogisticRegression(random_state=0)
  log.fit(x_train,y_train)
  
  #Decision Tree
  from sklearn.tree import DecisionTreeClassifier
  tree=DecisionTreeClassifier(criterion='entropy',random_state=0)
  tree.fit(x_train,y_train)
  
  #Random Forest Classifier
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=0)
  forest.fit(x_train,y_train)

  #Print the models accuracy on the training data
  print("[0]Logistic Regression Training Accuracy:",log.score(x_train,y_train))
  print("[1]Decision Tree Classifier Training Accuracy:",tree.score(x_train,y_train))
  print("[2]Random Forest Classifier Training Accuracy:",forest.score(x_train,y_train))
  
  return log,tree,forest


# In[ ]:


#Getting all of the models
model = models(x_train,y_train)


# In[ ]:


#test model accuracy on confusion matrix
from sklearn.metrics import confusion_matrix

for i in range(len(model)):
  print("Model ", i)
  cm =confusion_matrix(y_test,model[i].predict(x_test))

  TP=cm[0][0]
  TN=cm[1][1]
  FN=cm[1][0]
  FP=cm[0][1]

  print(cm)
  print("Testing Accuracy = ", (TP+TN) / (TP+TN+FN+FP))
  print()


# In[ ]:


#show another way to get metrics of the models
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

for i in range(len(model) ):
  print("Model ",i)
  print( classification_report(y_test,model[i].predict(x_test)))
  print( accuracy_score(y_test,model[i].predict(x_test)))
  print()


# In[ ]:


#print the prediction of random forest classifier model
pred=model[2].predict(x_test)
print(pred)
print()
print(y_test)


# References:
# 
# https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners
# 
# https://seaborn.pydata.org/tutorial/distributions.html
# 
# https://www.kaggle.com/kanncaa1/machine-learning-tutorial-for-beginners
# 
# https://www.youtube.com/watch?v=NSSOyhJBmWY
# 
# Thank you!
