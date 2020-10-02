#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
@author: Patel Shrey (xXShreyXx)
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.cross_validation import train_test_split #to split the dataset for training and testing
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accugracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
import warnings 
warnings.filterwarnings("ignore")
import os
for dirname, _, filenames in os.walk('/shrey821/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


iris = pd.read_csv("../input/Iris.csv")


# In[ ]:


iris.info()


# In[ ]:


iris.drop([0])


# In[ ]:


fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# In[ ]:


iris.shape


# In[ ]:


plt.figure(figsize=(7,4)) 
sns.heatmap(iris.corr(),annot=True,cmap='cubehelix_r')
plt.show()


# In[ ]:


train, test = train_test_split(iris, test_size = 0.2)
print(train.shape)
print(test.shape)


# In[ ]:


train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
train_y=train.Species
test_X= test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
test_y =test.Species


# In[ ]:


train_X.head(2)


# In[ ]:


test_X.head(2)


# In[ ]:


train_y.head()


# In[ ]:


model=DecisionTreeClassifier()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,test_y))


# In[ ]:


cm_tree = metrics.confusion_matrix(test_y,prediction)
cr_tree = metrics.classification_report(test_y,prediction)
print("confusion matrix : \n",cm_tree)
print("classification report : \n",cr_tree)
plt.figure(figsize = (10,8))
sns.heatmap(cm_tree,annot = True,xticklabels = np.arange(1,8),yticklabels = np.arange(1,8),cmap = "Greens")
plt.show()

