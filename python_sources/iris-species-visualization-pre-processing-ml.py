#!/usr/bin/env python
# coding: utf-8

# # Import Liabraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# # Import Dataset

# In[ ]:


dataset=pd.read_csv('../input/Iris.csv')


# In[ ]:


dataset.head()


# In[ ]:


x=dataset.iloc[:,:-1]


# In[ ]:


x.head()


# In[ ]:


y=dataset.iloc[:,-1]


# In[ ]:


y.head()


# In[ ]:


dataset.describe()


# In[ ]:


dataset.info()


# In[ ]:


dataset['Species'].value_counts()


# In[ ]:


x.isnull().sum()


# In[ ]:


y.isnull().sum()


# # EDA and Visualization

# Scatter Plot 

# In[ ]:


plt.figure(2,figsize=(15,5))
plt.subplot(1,2,1)
plt.scatter(x=x.iloc[:,1],y=x.iloc[:,2] ,edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.subplot(1,2,2)
plt.scatter(x=x.iloc[:,3],y=x.iloc[:,4] ,edgecolor='r')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()


# # Feature Scaling

# Dropping the axis column as we find it futile for evaluation of models

# In[ ]:


data=dataset.drop('Id',axis=1)


# Pair-Plot is being plot so as visualize the relation among each targeted labels(categorical variable)

# In[ ]:


sns.pairplot(data,hue='Species',markers='o',diag_kind='kde',palette='husl')


# Plotting violin plot using box plot the inner depiction of variance of data 

# In[ ]:


g = sns.violinplot(y='Species', x='SepalLengthCm', data=data, inner='box')
plt.show()
g = sns.violinplot(y='Species', x='SepalWidthCm', data=data, inner='box')
plt.show()
g = sns.violinplot(y='Species', x='PetalLengthCm', data=data, inner='box')
plt.show()
g = sns.violinplot(y='Species', x='PetalWidthCm', data=data, inner='box')
plt.show()


# # Machine Learning

# Using Label encoder to deal with the categorical data

# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)


# In[ ]:


x=x.drop('Id',axis=1)


# Splitting the dataset into train and test sets

# In[ ]:


from sklearn.cross_validation import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)


# Standardizing the scalability of the data

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit_transform(x_train,y_train)
sc.fit(x_test,y_test)


# In[ ]:


print(x_test)
print('*******************************')
print(x_train)
print('*******************************')
print(y_test)
print('*******************************')
print(y_train)
print('*******************************')


# # KNN Model

# In[ ]:


scores=[]
lrange=list(range(1,26))
for k in lrange:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))

plt.plot(lrange, scores,ls='dashed')
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()
    


# # Logistic Regression

# In[ ]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
print('Logistic Regression Score:',metrics.accuracy_score(y_test, y_pred))


# # Applying PCA

# In[ ]:



from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(x_train)
X_test = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_
explained_variance


# In[ ]:


principalDf = pd.DataFrame(data =X_train
             , columns = ['principal component 1', 'principal component 2'])


# In[ ]:


finalDf = pd.concat([principalDf, dataset[['Species']]], axis = 1)


# In[ ]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for targets, colors in zip(targets,colors):
    indicesToKeep = finalDf['Species'] == targets
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = colors
               , s = 50)
ax.legend(targets)
ax.grid()


# # Applying Support Vector Classifier

# In[ ]:


from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)


# In[ ]:


svc_predictions = svc.predict(x_test)
svc_predictions


# In[ ]:


svc_accuracy = svc.score(x_test,y_test)
svc_accuracy


# # Applying Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=10)
rf.fit(x_train,y_train)


# In[ ]:


rf_prediction=rf.predict(x_test)


# In[ ]:


rf_accuracy=rf.score(x_test,y_test)
rf_accuracy

