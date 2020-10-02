#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/iris/Iris.csv', index_col=0)


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


train.describe()


# How many species we have?

# In[ ]:


print(train['Species'].nunique())
print(train['Species'].unique())


# Let's see the distribution of the data in each of the columns

# In[ ]:


fig, ax = plt.subplots(2,2, figsize=(12,8))
ax[0,0] = sns.distplot(train['SepalLengthCm'], ax=ax[0,0])
ax[0,1] = sns.distplot(train['SepalWidthCm'], ax=ax[0,1])
ax[1,0] = sns.distplot(train['PetalLengthCm'], ax=ax[1,0])
ax[1,1] = sns.distplot(train['PetalWidthCm'], ax=ax[1,1])


# We have a mean of each column by specie

# In[ ]:


species = train.groupby('Species').mean()
species


# Now, boxplot of each column, and  we can see the difference between them
# 
# 
# **What is boxplot?**
# 
# In descriptive statistics, a box plot or boxplot is a method for graphically depicting groups of numerical data through their quartiles.
# Box plots may also have lines extending from the boxes (whiskers) indicating variability outside the upper and lower quartiles, hence the terms box-and-whisker plot and box-and-whisker diagram.

# In[ ]:


fig,ax = plt.subplots(2,2,figsize=(16,11))
ax[0,0] = sns.boxplot(x=train['Species'],y=train['SepalLengthCm'], ax=ax[0,0])
ax[0,0].set_title('SepalLengthCm')

ax[0,1] = sns.boxplot(x=train['Species'],y=train['SepalWidthCm'], ax=ax[0,1])
ax[0,1].set_title('SepalWidthCm')

ax[1,0] = sns.boxplot(x=train['Species'],y=train['PetalLengthCm'], ax=ax[1,0])
ax[1,0].set_title('PetalLengthCm')

ax[1,1] = sns.boxplot(x=train['Species'],y=train['PetalWidthCm'], ax=ax[1,1])
ax[1,1].set_title('PetalWidthCm')


# Looking at this heat map correlation, we can see high correlation between PetalLengthCm and PetalWidthCm and low correlation between SepalWidhtCm and PetalLengthCm

# In[ ]:


plt.figure(figsize=(7,7))
sns.heatmap(train.corr(),annot = True)


# In[ ]:


sns.pairplot(train,hue='Species')


# Sccaterplot of **PetalLengthCm x SepalWidhtCm** and **PetalWidthCm x PetalLengthCm**
# 
# 
# **What is scatterplot?**
# 
# A scatter plot is a type of plot or mathematical diagram using Cartesian coordinates to display values for typically two variables for a set of data. If the points are coded (color/shape/size), one additional variable can be displayed. The data are displayed as a collection of points, each having the value of one variable determining the position on the horizontal axis and the value of the other variable determining the position on the vertical axis

# In[ ]:


fig, ax= plt.subplots(1,2,figsize=(12,5))
ax[0] = sns.scatterplot(x=train['PetalLengthCm'],y=train['SepalWidthCm'],hue=train['Species'],ax=ax[0])
ax[0].set_title('PetalLengthCm x SepalWidhtCm')
ax[1] = sns.scatterplot(x=train['PetalWidthCm'],y=train['PetalLengthCm'],hue=train['Species'],ax=ax[1])
ax[1].set_title('PetalWidthCm x PetalLengthCm')


# Start with model and split train/test. Test size will be 30% of the dataset, and random_state = 0)

# In[ ]:


y = train['Species']
x = train.drop('Species',axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[ ]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=0,test_size=0.3)


# **First Model:** C-Support Vector Classification.

# In[ ]:


from sklearn import svm
model_svm = svm.SVC()
model_svm.fit(xtrain,ytrain)
svm_predict = model_svm.predict(xtest)
model_svm_acc = accuracy_score(ytest,svm_predict)
print(f'Accuracy score: {model_svm_acc}')


# **DecisionTreeClassifier**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=1)
model_tree = model.fit(xtrain,ytrain)
model_tree_predict = model.predict(xtest)
model_tree_acc = accuracy_score(ytest,model_tree_predict)
model_tree_acc


# **RandomForestClassifier**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
for i in [100,200,300,400,500]:    
    model_random_forest = RandomForestClassifier(n_estimators=i,random_state = 42)
    model_random_forest.fit(xtrain,ytrain)
    model_forest_predict = model.predict(xtest)
    model_forest_accuaracy = accuracy_score(ytest,model_forest_predict)
    print(model_forest_accuaracy)


# **LogisticRegression:**

# In[ ]:


from sklearn.linear_model import LogisticRegression
logist_regression = LogisticRegression(max_iter=200,random_state=42)
logist_regression.fit(xtrain,ytrain)
predict = logist_regression.predict(xtest)
logist_regression = accuracy_score(ytest,predict)
logist_regression


# **KNeighborsClassifier**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
model_KN=KNeighborsClassifier(n_neighbors=5) 
model_KN.fit(xtrain,ytrain)
KN_predict =model_KN.predict(xtest)
KN_accuracy = accuracy_score(ytest,KN_predict)
KN_accuracy


# **Evaluate a score by cross-validation**

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


# In[ ]:


index = np.arange(1,6)


# In[ ]:


my_pipeline = Pipeline(steps=[('model', LogisticRegression(max_iter=200))])
scores =cross_val_score(my_pipeline, x, y,
                              cv=len(index),scoring='accuracy')


# In[ ]:


plt.figure(figsize=(7,7))
sns.lineplot(x=index,y=scores)
plt.ylabel('Accuracy')
plt.xlabel('Cv')
plt.title('Accuracy with LogisticRegression')


# In[ ]:


print(scores.mean())


# In[ ]:




