#!/usr/bin/env python
# coding: utf-8

# # Iris dataset 
#  some EDA techniques on iris datset which can help you to get a better visualization on dataset 
#  ### please upvote you find it helpful

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

get_ipython().run_line_magic('matplotlib', 'inline')


# ### loading iris dataset

# In[ ]:


df=pd.read_csv('../input/iris/Iris.csv')


# In[ ]:


df.head()


# ### Images of  3 different speicies of iris datset

# In[ ]:


image=plt.imread('../input/irisimage/iris-machinelearning.png')
plt.figure(figsize=(15,10))
plt.imshow(image)
plt.axis('off')


# In[ ]:


df=df.drop('Id',axis=1)


# In[ ]:


df.describe()


# ## Visualizing features of Dataset

# In[ ]:


from pandas.tools.plotting import parallel_coordinates
plt.figure(figsize=(12,8))
parallel_coordinates(df, 'Species', colormap=plt.get_cmap("Set3"))
plt.xlabel("Features of data set")
plt.ylabel("cm")
plt.show()


# ## Pair plot 

# In[ ]:


sns.pairplot(df,hue='Species')


# ## Visualizing length and width of different species using swarm plot

# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.swarmplot(x='Species',y='PetalLengthCm',data=df)
plt.subplot(2,2,2)
sns.swarmplot(x='Species',y='PetalWidthCm',data=df)
plt.subplot(2,2,3)
sns.swarmplot(x='Species',y='SepalLengthCm',data=df)
plt.subplot(2,2,4)
sns.swarmplot(x='Species',y='SepalWidthCm',data=df)


# ## Some more plots for better EDA

# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=df)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=df)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=df)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=df)


# ### Box plots
# Box plots are important for EDA because they can handle a summary of a large amount of data. A box plot consists of the median, which is the midpoint of the range of data; the upper and lower quartiles, which represent the numbers above and below the highest and lower quarters of the data and the minimum and maximum data values

# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.boxplot(x='Species',y='PetalLengthCm',data=df)
plt.subplot(2,2,2)
sns.boxplot(x='Species',y='PetalWidthCm',data=df)
plt.subplot(2,2,3)
sns.boxplot(x='Species',y='SepalLengthCm',data=df)
plt.subplot(2,2,4)
sns.boxplot(x='Species',y='SepalWidthCm',data=df)


# ### Splitting the data into traning set and test set

# In[ ]:


x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

x=(x-x.mean(0))/x.std(0)
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.33, random_state=42)


# ## Scaling and encoding Feature Data

# In[ ]:


from sklearn.preprocessing import StandardScaler,LabelEncoder

sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

lb=LabelEncoder()
y_train=lb.fit_transform(y_train)
y_test=lb.fit_transform(y_test)


# ## Logistic Regression

# In[ ]:


model = LogisticRegression()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
print('Accuracy on iris dataset using Logistic Regression is',metrics.accuracy_score(prediction,y_test))


# ## Support Vector Machine

# In[ ]:


model = SVC() 
model.fit(x_train,y_train) 
prediction=model.predict(x_test)
print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,y_test))


# ## Decision Tree

# In[ ]:


model=DecisionTreeClassifier()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,y_test))


# ## K-Nearest Neighbours

# In[ ]:


# finding the value of k when the  accuray of knn would be highest
x_range=[]
y_range=[]
for i in range(1,20):
    model=KNeighborsClassifier(n_neighbors=i) 
    model.fit(x_train,y_train)
    prediction=model.predict(x_test)
    y_range.append((metrics.accuracy_score(prediction,y_test)))
    x_range.append(i)
plt.figure(figsize=(8,5))    
plt.plot(x_range, y_range)
plt.xlabel('values of n_neighbors')
plt.ylabel('Accuracy')
plt.show()


# From above plot we can see that the Accuracy is higest at value k=5 

# In[ ]:


model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
prediction=model.predict(x_test)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction,y_test))


# I hope my notebook will give you  better understanding on iris dataset 
# **Please upvote ** it if you find it usefull
