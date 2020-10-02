#!/usr/bin/env python
# coding: utf-8

# ### importing our libraries 

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix


# ### Reading Data .. 

# In[ ]:


df= pd.read_csv('../input/weather-dataset-rattle-package/weatherAUS.csv')


# In[ ]:


df.head()


# ### Let's explore our data

# In[ ]:


# this to show the data type , columns names ,sum of data that not a nans and memory usage . 
df.info()


# ### it's  time to dealing with  missing values ...

# In[ ]:


# show the number of missing values in each column.
df.isna().sum().sort_values(ascending=False)


# In[ ]:


# Removing some columns that have > 28% nan values
max_nans=len(df)*0.28
df=df.loc[:, (df.isnull().sum(axis=0) <= max_nans)]


# In[ ]:


# show the number of the missing values 
df.isna().sum().sort_values(ascending=False)


# In[ ]:


#filling the missing values by the next value ('bfill') because the temperatures are nearly the same for the next day
df.fillna(method='bfill',inplace=True)


# ### let's working on the important features and drop some ...

# In[ ]:


df.drop(columns=['RISK_MM','Date','Location'],inplace=True)


# In[ ]:


# here we change the data type for these two column
df['RainTomorrow']=df['RainTomorrow'].astype('category')
df['RainToday']=df['RainToday'].astype('category')


# In[ ]:


# here we transform them to numeric values.
df['RainTomorrow']=df['RainTomorrow'].cat.codes
df['RainToday']=df['RainToday'].cat.codes


# In[ ]:


# to know the number of row and columns .
df.shape


# ### let's make some visualisation...

# In[ ]:


sns.countplot(df['RainToday'])


# In[ ]:


sns.countplot(df['RainTomorrow'])


# In[ ]:


g = sns.FacetGrid(df,hue='RainTomorrow',aspect=4)
g.map(plt.hist,'MinTemp',alpha=0.6,bins=5)
plt.legend()


# In[ ]:


g = sns.FacetGrid(df,hue='RainTomorrow',height=6,aspect=2)
g.map(plt.hist,'Humidity3pm',alpha=0.6,bins=5)
plt.legend()


# In[ ]:


g = sns.FacetGrid(df,hue='RainTomorrow',height=6,aspect=2)
g.map(plt.hist,'WindGustSpeed',alpha=0.6,bins=5)
plt.legend()


# ### it's ML working time ...

# In[ ]:


#select the columns for the model.
X=df.iloc[:,:16]


# In[ ]:


X.head()


# In[ ]:


# select our target .
y=df['RainTomorrow']


# In[ ]:


# here we transform the non_numeric features to make the model dealing with it . 
X=pd.get_dummies(X,columns=['WindDir9am','WindDir3pm','WindGustDir'],drop_first=True)


# In[ ]:


#Next, we split 75% of the data to the training set while 25% of the data to test set using below code.
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.25, random_state= 0,stratify=y)


# In[ ]:


#we need to bring all features to the same level of magnitudes. This can be achieved by a method called feature scaling.
scaler = preprocessing.MinMaxScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
X.iloc[4:10]


# ### For k = 5 : 

# In[ ]:


#Our next step is to K-NN model and train it with the training data. Here n_neighbors is the value of factor K.
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train,y_train)


# In[ ]:


# Let's train our test data and check its accuracy.
y_pred = classifier.predict(X_test)
score = accuracy_score(y_test,y_pred)
print('Accuracy :',score)


# In[ ]:


# let's see the classification report .
y_test_pred = classifier.predict(X_test)
print(classification_report(y_test,y_test_pred))


# In[ ]:


#let's show the confusion matrix.
confusion_matrix(y_test,y_test_pred)


# ### for K = 8 :

# In[ ]:


classifier = KNeighborsClassifier(n_neighbors = 8, p = 2)
classifier.fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import accuracy_score
y_pred = classifier.predict(X_test)
score = accuracy_score(y_test,y_pred)
print('Accuracy :',score)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
y_test_pred = classifier.predict(X_test)
print(classification_report(y_test,y_test_pred))


# 
# ### So we tried more than 3 values on K and  as we can see that Accuracy is maximum that is 83.7 when k = 8  ss
# 
