#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='white', color_codes=True)
from sklearn.mixture import GaussianMixture
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
   for filename in filenames:
       print(os.path.join(dirname, filename))


# In[ ]:


import pandas as pd
iris=pd.read_csv("../input/iris/Iris.csv")
iris.head()


# In[ ]:


iris.describe()


# In[ ]:


iris['Species'].value_counts()


# In[ ]:


iris.plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm')


# In[ ]:


sns.jointplot(x='SepalLengthCm',y='SepalWidthCm', data=iris, size=5)


# In[ ]:


sns.FacetGrid(iris, hue='Species', size=5).map(plt.scatter, 'SepalLengthCm','SepalWidthCm').add_legend()


# In[ ]:


sns.boxplot(x='Species',y='SepalLengthCm', data=iris)


# In[ ]:


ax=sns.boxplot(x='Species',y='SepalLengthCm',data=iris)
ax=sns.stripplot(x='Species',y='SepalLengthCm', data=iris, jitter=True, edgecolor='gray')


# In[ ]:


sns.violinplot(x='Species',y='SepalLengthCm',data=iris,size=6)


# In[ ]:


sns.FacetGrid(iris, hue='Species',size=6).map(sns.kdeplot, 'PetalLengthCm').add_legend()


# In[ ]:


sns.pairplot(iris.drop('Id', axis=1), hue='Species', size=3)


# In[ ]:


sns.pairplot(iris.drop('Id', axis=1), hue='Species', size=3, diag_kind='kde')


# In[ ]:


iris.drop('Id', axis=1).boxplot(by='Species', figsize=(12,6))


# In[ ]:


iris.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()


# In[ ]:


iris.shape


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


iris.drop('Id', axis=1, inplace=True)


# In[ ]:


plt.figure(figsize=(7,4))
sns.heatmap(iris.corr(), annot=True, cmap='cubehelix_r')
plt.show()


# In[ ]:


train, test=train_test_split(iris, test_size=0.3)
print(train.shape)
print(test.shape)


# In[ ]:


train_X=train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
train_y=train.Species
test_X=test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
test_y=test.Species


# In[ ]:


train_X.head()


# In[ ]:


test_X.head()


# In[ ]:


train_y.head()


# In[ ]:


test_y.head()


# In[ ]:


model=svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print(classification_report(test_y, prediction))
print(confusion_matrix(test_y, prediction))
print('The Accuracy of SVM is:', metrics.accuracy_score(prediction,test_y))


# In[ ]:


model=LogisticRegression()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print(classification_report(test_y, prediction))
print(confusion_matrix(test_y, prediction))
print('The Accuracy of Logistic Regression is:', metrics.accuracy_score(prediction,test_y))


# In[ ]:


model=DecisionTreeClassifier()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print(classification_report(test_y, prediction))
print(confusion_matrix(test_y, prediction))
print('The Accuracy of DT is:', metrics.accuracy_score(prediction,test_y))


# In[ ]:


model=KNeighborsClassifier(n_neighbors=3)
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print(classification_report(test_y, prediction))
print(confusion_matrix(test_y, prediction))
print('The Accuracy of KNN is:', metrics.accuracy_score(prediction,test_y))


# In[ ]:


a_index=list(range(1,11))
a=pd.Series()
x=[1,2,3,4,5,6,7,8,9,10]
for i in list(range(1,11)):
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(train_X,train_y)
    prediction=model.predict(test_X)
    a=a.append(pd.Series(metrics.accuracy_score(prediction,test_y)))
plt.plot(a_index, a)
plt.xticks(x)


# In[ ]:


petal=iris[['PetalLengthCm','PetalWidthCm','Species']]
sepal=iris[['SepalLengthCm','SepalWidthCm','Species']]


# In[ ]:


train_p,test_p=train_test_split(petal,test_size=0.3,random_state=0)
train_x_p=train_p[['PetalLengthCm','PetalWidthCm']]
train_y_p=train_p.Species
test_x_p=test_p[['PetalLengthCm','PetalWidthCm']]
test_y_p=test_p.Species

train_s,test_s=train_test_split(sepal,test_size=0.3,random_state=0)
train_x_s=train_s[['SepalLengthCm','SepalWidthCm']]
train_y_s=train_s.Species
test_x_s=test_s[['SepalLengthCm','SepalWidthCm']]
test_y_s=test_s.Species


# In[ ]:


model=svm.SVC()
model.fit(train_x_p,train_y_p)
prediction=model.predict(test_x_p)
print(classification_report(test_y_p, prediction))
print(confusion_matrix(test_y_p, prediction))
print('The Accuracy of SVM is:', metrics.accuracy_score(prediction,test_y_p))


model=svm.SVC()
model.fit(train_x_s,train_y_s)
prediction=model.predict(test_x_s)
print(classification_report(test_y_s, prediction))
print(confusion_matrix(test_y_s, prediction))
print('The Accuracy of SVM is:', metrics.accuracy_score(prediction,test_y_s))


# In[ ]:


model=LogisticRegression()
model.fit(train_x_p,train_y_p)
prediction=model.predict(test_x_p)
print(classification_report(test_y_p, prediction))
print(confusion_matrix(test_y_p, prediction))
print('The Accuracy of LR using P is:', metrics.accuracy_score(prediction,test_y_p))

#model=LogisticRegression()
model.fit(train_x_s,train_y_s)
prediction=model.predict(test_x_s)
print(classification_report(test_y_s, prediction))
print(confusion_matrix(test_y_s, prediction))
print('The Accuracy of LR using S is:', metrics.accuracy_score(prediction,test_y_s))


# In[ ]:


model=DecisionTreeClassifier()
model.fit(train_x_p,train_y_p)
prediction=model.predict(test_x_p)
print(classification_report(test_y_p, prediction))
print(confusion_matrix(test_y_p, prediction))
print('The Accuracy of DT is:', metrics.accuracy_score(prediction, test_y_p))

model.fit(train_x_s,train_y_s)
prediction=model.predict(test_x_s)
print(classification_report(test_y_s, prediction))
print(confusion_matrix(test_y_s, prediction))
print('The Accuracy of DT is:', metrics.accuracy_score(prediction, test_y_s))


# In[ ]:


model=KNeighborsClassifier()
model.fit(train_x_p,train_y_p)
prediction=model.predict(test_x_p)
print(classification_report(test_y_p, prediction))
print(confusion_matrix(test_y_p, prediction))
print('The Accuracy of KNN is:', metrics.accuracy_score(prediction, test_y_p))

model.fit(train_x_s,train_y_s)
prediction=model.predict(test_x_s)
print(classification_report(test_y_s, prediction))
print(confusion_matrix(test_y_s, prediction))
print('The Accuracy of KNN is:', metrics.accuracy_score(prediction, test_y_s))


# In[ ]:


def get_outliers(log_prob, epsilon):
    outliners=np.where(log_prob <= epsilon, 1, 0)
    return outliners


# In[ ]:


features=['PetalLengthCm','PetalWidthCm']


# In[ ]:


def make_density_plot(iris, features, model, outliers):
    x=np.linspace(iris[features[0]].min(), iris[features[0]].max())
    y=np.linspace(iris[features[1]].min(), iris[features[1]].max())
    X, Y=np.meshgrid(x, y)
    
    XX=np.array([X.ravel(), Y.ravel()]).T
    Z=model.score_samples(XX)
    Z=Z.reshape(X.shape)
    
    levels=MaxNLocator(nbins=100).tick_values(Z.min(), Z.max())
    cmap=plt.get_cmap('BuGn')
    
    plt.figure(figsize=(10,10))
    plt.contourf(X, Y, Z.reshape(X.shape), cmap=cmap, levels=levels)
    plt.scatter(model.means_[:,0], model.means_[:,1], color='Blue')
    g1=plt.scatter(iris[iris.Outlier==0][features[0]].values, iris[iris.Outlier==0][features[1]].values, label='Normal',s=4.0,c='Pink')
    g2=plt.scatter(iris[iris.Outlier==1][features[0]].values, iris[iris.Outlier==1][features[1]].values, label='Abormal',s=4.5,c='Black')
    plt.legend(handles=[g1,g2])
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    return plt


# In[ ]:


X_train=iris[features].values
model=GaussianMixture(n_components=3, covariance_type='full')
model.fit(X_train)
log_prob=model.score_samples(X_train)
outliers=get_outliers(log_prob, 0.15)
iris['Outlier']=outliers


# In[ ]:


plt.figure(figsize=(20,5))
sns.distplot(log_prob, kde=False, bins=50, color='Green')
g1=plt.axvline(np.quantile(log_prob, 0.25), color='Red', label='Q_25')
g2=plt.axvline(np.quantile(log_prob, 0.5), color='Blue', label='Q_50-Median')
g3=plt.axvline(np.quantile(log_prob, 0.75), color='Violet', label='Q_75')
g4=plt.axvline(np.quantile(log_prob, 0.05), color='Brown', label='Q_5')

handles=[g1, g2, g3, g4]
plt.xlabel('log-probabilities of the data spots')
plt.ylabel('frequency')
plt.legend(handles)


# In[ ]:


epsilon=np.quantile(log_prob, 0.05)
print('epsilon: %f' % epsilon)


# In[ ]:


outliers=get_outliers(log_prob, epsilon)
iris['Outlier']=outliers


# In[ ]:


make_density_plot(iris, features, model, outliers)


# In[ ]:


iris.head()
iris.Outlier.value_counts()


# In[ ]:


import xgboost as xgb
model=xgb.XGBClassifier()
model.fit(train_x_p,train_y_p)
prediction=model.predict(test_x_p)
print(classification_report(test_y_p, prediction))
print(confusion_matrix(test_y_p, prediction))
print('The Accuracy is:', metrics.accuracy_score(prediction, test_y_p))

model.fit(train_x_s,train_y_s)
prediction=model.predict(test_x_s)
print(classification_report(test_y_s, prediction))
print(confusion_matrix(test_y_s, prediction))
print('The Accuracy is:', metrics.accuracy_score(prediction, test_y_s))


# In[ ]:


a=sns.lmplot(x='SepalLengthCm',y='SepalWidthCm',hue='Species', data=iris)


# In[ ]:


a=sns.lmplot(x='PetalLengthCm',y='PetalWidthCm',hue='Species', data=iris)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(train_x_p,train_y_p)
prediction=model.predict(test_x_p)
print(classification_report(test_y_p, prediction))
print(confusion_matrix(test_y_p, prediction))
print('The Accuracy is:', metrics.accuracy_score(prediction, test_y_p))

model.fit(train_x_s,train_y_s)
prediction=model.predict(test_x_s)
print(classification_report(test_y_s, prediction))
print(confusion_matrix(test_y_s, prediction))
print('The Accuracy is:', metrics.accuracy_score(prediction, test_y_s))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(train_x_p, train_y_p)
prediction=model.predict(test_x_p)
print(classification_report(test_y_p, prediction))
print(confusion_matrix(test_y_p, prediction))
print('The Accuracy is:', metrics.accuracy_score(prediction, test_y_p))

model.fit(train_x_s,train_y_s)
prediction=model.predict(test_x_s)
print(classification_report(test_y_s, prediction))
print(confusion_matrix(test_y_s, prediction))
print('The Accuracy is:', metrics.accuracy_score(prediction, test_y_s))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(train_x_p,train_y_p)
prediction=model.predict(test_x_p)
print(classification_report(test_y_p, prediction))
print(confusion_matrix(test_y_p, prediction))
print('The Accuracy is:', metrics.accuracy_score(prediction, test_y_p))

model.fit(train_x_s,train_y_s)
prediction=model.predict(test_x_s)
print(classification_report(test_y_s, prediction))
print(confusion_matrix(test_y_s, prediction))
print('The Accuracy is:', metrics.accuracy_score(prediction, test_y_s))


# In[ ]:


from lightgbm import LGBMClassifier
model=LGBMClassifier()
model.fit(train_x_p,train_y_p)
prediction=model.predict(test_x_p)
print(classification_report(test_y_p, prediction))
print(confusion_matrix(test_y_p, prediction))
print('The Accuracy is:', metrics.accuracy_score(prediction, test_y_p))

model.fit(train_x_s,train_y_s)
prediction=model.predict(test_x_s)
print(classification_report(test_y_s, prediction))
print(confusion_matrix(test_y_s, prediction))
print('The Accuracy is:', metrics.accuracy_score(prediction, test_y_s))


# In[ ]:




