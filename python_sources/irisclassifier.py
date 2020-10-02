#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# From https://www.kaggle.com/ash316/ml-from-scratch-with-iris

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = pd.read_csv('../input/Iris.csv')

iris.describe()
iris.head()
iris.info()

iris.drop('Id',axis=1,inplace=True)
iris.head()


# In[ ]:





# In[ ]:


fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='green', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='red', label='Versicolor', ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='blue', label='Virginica', ax=fig)
fig.set_xlabel('SepalLengthCm')
fig.set_ylabel('SepalWidthCm')
fig.set_title('Sepal Width v/s Length')

fig=plt.gcf()
fig.set_size_inches(10, 6)

plt.show()


# In[ ]:


fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='green', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='red', label='Versicolor', ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='blue', label='Virginica', ax=fig)
fig.set_xlabel('PetalLengthCm')
fig.set_ylabel('PetalWidthCm')
fig.set_title('Petal Width v/s Length')

fig=plt.gcf()
fig.set_size_inches(10, 6)

plt.show()


# In[ ]:


iris.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12, 6)
plt.show()


# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species', y='PetalLengthCm',data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='Species', y='PetalWidthCm',data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='Species', y='SepalLengthCm',data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='Species', y='SepalWidthCm',data=iris)


# In[ ]:


from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


iris.shape

plt.figure(figsize=(7,4))

sns.heatmap(iris.corr(), annot=True, cmap='cubehelix_r')
plt.show()


# In[ ]:


train, test = train_test_split(iris, test_size=0.3)

print(train.shape)
print(test.shape)


# In[ ]:


feats = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

trainX = train[feats]
trainy = train.Species
testX = test[feats]
testy = test.Species

print(trainX.shape, testX.shape, trainy.shape, testy.shape)


# In[45]:


model = svm.SVC()

model.fit(trainX, trainy)

pred = model.predict(testX)

print('accuracy is: ', metrics.accuracy_score(pred, testy))


# In[ ]:


model = LogisticRegression()

model.fit(trainX, trainy)

pred = model.predict(testX)

print('accuracy is: ', metrics.accuracy_score(pred, testy))


# In[ ]:


model = DecisionTreeClassifier()

model.fit(trainX, trainy)

pred = model.predict(testX)

print('accuracy is: ', metrics.accuracy_score(pred, testy))


# In[ ]:


model = KNeighborsClassifier(n_neighbors=8)

model.fit(trainX, trainy)

pred = model.predict(testX)

print('accuracy is: ', metrics.accuracy_score(pred, testy))


# In[ ]:


nbrs = list(range(1,11))
a = pd.Series()

for i in nbrs:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(trainX, trainy)
    pred = model.predict(testX)
    a=a.append(pd.Series(metrics.accuracy_score(pred, testy)))

plt.plot(nbrs, a)
plt.show()


# In[37]:


trainXP = train[['PetalLengthCm', 'PetalWidthCm']]
trainXS = train[['SepalLengthCm', 'SepalWidthCm']]
trainY = train.Species

testXP = test[['PetalLengthCm', 'PetalWidthCm']]
testXS = test[['SepalLengthCm', 'SepalWidthCm']]
testY = test.Species


# In[39]:


model = svm.SVC()

model.fit(trainXP, trainY)
pred = model.predict(testXP)
print('accuracy for petals: ', metrics.accuracy_score(pred, testY))

model.fit(trainXS, trainY)
pred = model.predict(testXS)
print('accuracy for sepals: ', metrics.accuracy_score(pred, testY))


# In[41]:


model = LogisticRegression()

model.fit(trainXP, trainY)
pred = model.predict(testXP)
print('accuracy for petals: ', metrics.accuracy_score(pred, testY))

model.fit(trainXS, trainY)
pred = model.predict(testXS)
print('accuracy for sepals: ', metrics.accuracy_score(pred, testY))


# In[43]:


model = DecisionTreeClassifier()

model.fit(trainXP, trainY)
pred = model.predict(testXP)
print('accuracy for petals: ', metrics.accuracy_score(pred, testY))

model.fit(trainXS, trainY)
pred = model.predict(testXS)
print('accuracy for sepals: ', metrics.accuracy_score(pred, testY))


# In[44]:


model = KNeighborsClassifier(n_neighbors = 3)

model.fit(trainXP, trainY)
pred = model.predict(testXP)
print('accuracy for petals: ', metrics.accuracy_score(pred, testY))

model.fit(trainXS, trainY)
pred = model.predict(testXS)
print('accuracy for sepals: ', metrics.accuracy_score(pred, testY))

