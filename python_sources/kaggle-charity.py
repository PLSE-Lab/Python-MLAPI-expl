#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[ ]:


data_train = pd.read_csv('/kaggle/input/udacity-mlcharity-competition/census.csv')
data_test = pd.read_csv('/kaggle/input/udacity-mlcharity-competition/test_census.csv')
goal_test = pd.read_csv('/kaggle/input/udacity-mlcharity-competition/example_submission.csv')


# In[ ]:


data_train.head()


# In[ ]:


data_test.head()


# In[ ]:


goal_test


# In[ ]:


data_train.income.unique()


# In[ ]:


income=data_train.income.map({'<=50K': 0, '>50K':1})


# In[ ]:


features = pd.get_dummies(data_train.drop(['income'],1))


# In[ ]:


scaler = StandardScaler()
features = scaler.fit_transform(features)


# In[ ]:


x_train , x_test , y_train , y_test = train_test_split(features,income,test_size=0.2,random_state=0)
logistic= LogisticRegression(random_state=0)
logistic.fit(x_train,y_train)


# In[ ]:


print('Train score is: ',logistic.score(x_train,y_train))
print('Test score is:',logistic.score(x_test,y_test))


# In[ ]:


cm = confusion_matrix(y_test,logistic.predict(x_test))
print(logistic.score(x_test,y_test))
pd.DataFrame(cm)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)


# In[ ]:


print(classifier.score(x_train,y_train))
print(classifier.score(x_test,y_test))


# In[ ]:


cm = confusion_matrix(y_test, classifier.predict(x_test))
print(classifier.score(x_test, y_test))
pd.DataFrame(cm)


# In[ ]:


x_train , x_test , y_train , y_test = train_test_split(features,income,test_size=0.2,random_state=0)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=9)
classifier.fit(x_train,y_train)


# In[ ]:


print('Train score is:',classifier.score(x_train,y_train))
print('Test score is :', classifier.score(x_test,y_test))


# In[ ]:


#cm = confusion_matrix(y_test, classifier.predict(x_test))
#print(classifier.score(x_test, y_test))
#pd.DataFrame(cm)


# # We will use Logistic Regression

# In[ ]:


test = data_test.drop(['Unnamed: 0'] , axis=1)


# In[ ]:


test.head()


# In[ ]:


test.fillna(method='ffill', inplace=True)


# In[ ]:


test = pd.get_dummies(data_test)


# In[ ]:


test.head()


# In[ ]:


final_test = test.drop(['Unnamed: 0'] , axis=1)


# In[ ]:


final_test.fillna(method='ffill', inplace=True)


# In[ ]:


final_test.head()


# In[ ]:


scaler= StandardScaler()
final_test = scaler.fit_transform(final_test)


# In[ ]:


x_train , x_test , y_train , y_test = train_test_split(features,income,test_size=0.2,random_state=0)
logistic= LogisticRegression(random_state=0)
logistic.fit(x_train,y_train)
logistic.predict(x_test)
logistic.predict(final_test)


# In[ ]:


goal_test.head()


# In[ ]:


goal_test['id'] = goal_test.iloc[:,0] 
goal_test['income'] = logistic.predict(final_test)


# In[ ]:


goal_test.head()


# In[ ]:


goal_test.to_csv('example_submission.csv',index=False,header=1)

