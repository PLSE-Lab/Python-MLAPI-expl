#!/usr/bin/env python
# coding: utf-8

# Hello, I want to share my humble work regarding this dataset. I tried to make ML models to predict whether the job posting is real or not. Here you can see the performance difference between the ML algorithms (Decision Tree, KNN, SVM, ANN)

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 


# In[ ]:


df=pd.read_csv('/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')


# In[ ]:


df.info()
df
df.isnull().sum()
df.nunique().sum


# In[ ]:


df.department=df.department.isna()
df.company_profile=df.company_profile.isna()
df.requirements=df.requirements.isna()
df.benefits=df.benefits.isna()
df.industry=df.industry.isna()
df.function=df.function.isna()
df[['department','company_profile','requirements','benefits','industry','function']] = df[['department','company_profile','requirements','benefits','industry','function']].replace({True: 1, False: 0})


# In[ ]:


df = df.dropna(subset=['location','description'])


# In[ ]:


df.isnull().sum()


# In[ ]:


df['location'] = df['location'].str.split(',').str[0]


# In[ ]:


df['salary_range'] = df['salary_range'].str.split('-').str[-1]
df['salary_range'] = df['salary_range'].fillna(0)
df = df[df.salary_range != 'Nov']
df = df[df.salary_range != 'Oct']
df = df[df.salary_range != 'Dec']
df = df[df.salary_range != 'Apr']
df = df[df.salary_range != 'Jun']
df = df[df.salary_range != 'Sep']


# In[ ]:


df[['employment_type','required_experience','required_education']] = df[['employment_type','required_experience','required_education']].fillna('Other')


# In[ ]:


df=df.drop('title',axis=1)
df=df.drop('description',axis=1)
df=df.drop('job_id',axis=1)


# In[ ]:


dummies= pd.get_dummies(df[['location','employment_type','required_experience','required_education']],drop_first=True)
df=pd.concat([df.drop(['location','employment_type','required_experience','required_education'], axis=1), dummies],axis=1) 


# In[ ]:


df.isnull().sum()


# In[ ]:


df.nunique().sum


# In[ ]:


df


# In[ ]:


df.info
sns.countplot(x='fraudulent',data=df)
#The data is very skew


# In[ ]:


x = df.drop('fraudulent',axis=1).values
y = df['fraudulent'].values


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=101)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# **Decision Tree**

# In[ ]:


scores=[]
for i in range(1,100):
  tree=DecisionTreeClassifier(max_depth = i) 
  tree.fit(x_train, y_train) 
  scores.append(tree.score(x_test,y_test)) 
plt.plot(range(1,100),scores) 
plt.show()


# In[ ]:


tree=DecisionTreeClassifier(max_depth =18) 
tree.fit(x_train, y_train) 
tree.score(x_test,y_test)


# In[ ]:


predictions = tree.predict(x_test) 
from sklearn.metrics import classification_report,confusion_matrix 
print(classification_report(y_test,predictions)) 
print(confusion_matrix(y_test,predictions))


# **KNN**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


accuracies=[]
for k in range(1,101):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(x_train, y_train)
  accuracies.append(classifier.score(x_test, y_test)) 
  
k_list=list(range(1,101)) 
plt.plot(k_list,accuracies)
plt.show() 


# In[ ]:


classifier = KNeighborsClassifier(n_neighbors =10)
classifier.fit(x_train, y_train)
classifier.score(x_test, y_test) 


# In[ ]:


predictions = classifier.predict(x_test) 
from sklearn.metrics import classification_report,confusion_matrix 
print(classification_report(y_test,predictions)) 
print(confusion_matrix(y_test,predictions))


# **SVM**

# In[ ]:


from sklearn.svm import SVC
scores=[]
for i in (np.arange(0.01,1,0.02)):
  classifier = SVC(kernel = 'linear', C = i)
  classifier.fit(x_train,y_train) 
  scores.append(classifier.score(x_test,y_test)) 
plt.plot(np.arange(0.01,1,0.02),scores) 
plt.show()


# In[ ]:


scores=[]
for i in (np.arange(0.01,1,0.02)):
  classifier = SVC(kernel = 'rbf', gamma=i, C = i)
  classifier.fit(x_train,y_train) 
  scores.append(classifier.score(x_test,y_test)) 
plt.plot(np.arange(0.01,1,0.02),scores) 
plt.show()


# In[ ]:


classifier = SVC(kernel = 'rbf', gamma=0.96, C = 0.96)
classifier.fit(x_train,y_train) 
classifier.score(x_test,y_test)


# In[ ]:


predictions = classifier.predict(x_test) 
from sklearn.metrics import classification_report,confusion_matrix 
print(classification_report(y_test,predictions)) 
print(confusion_matrix(y_test,predictions))


# **ANN**

# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


from tensorflow.keras.layers import Dropout
model = Sequential()
model.add(Dense(units=100,activation='relu'))
model.add(Dropout(0.5))


model.add(Dense(units=1,activation='sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])


# In[ ]:


#early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

model.fit(x_train,y_train,epochs=65,validation_data=(x_test, y_test), verbose=1)


# In[ ]:


losses = pd.DataFrame(model.history.history)

losses[['accuracy','val_accuracy']].plot()
losses[['loss','val_loss']].plot()


# In[ ]:


print(model.metrics_names) 
print(model.evaluate(x_test,y_test,verbose=0))


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix

predictions = model.predict_classes(x_test)
print(classification_report(y_test,predictions))


# In[ ]:




