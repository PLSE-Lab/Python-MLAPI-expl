#!/usr/bin/env python
# coding: utf-8

# Hello, I want to share my humble work regarding this dataset. I tried to make ML models to predict whether the shoppers will be generated revenue or not based on customer behaviour. Here you can see the performance difference between the ML algorithms (decision tree, random forest, SVM, ANN).

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

df=pd.read_csv('/kaggle/input/online-shoppers-intention/online_shoppers_intention.csv')


# In[ ]:


df.info()
sns.countplot(x='Revenue',data=df)
#the data is skew!


# In[ ]:


df


# In[ ]:


df.Weekend = df.Weekend.replace({True: 1, False: 0})
df.Revenue = df.Revenue.replace({True: 1, False: 0})


# In[ ]:


dummies= pd.get_dummies(df['VisitorType'],drop_first=True) 
df=pd.concat([df.drop('VisitorType', axis=1), dummies],axis=1) 
df=df.drop('Other',axis=1)


# In[ ]:


df.Month.unique()


# In[ ]:


df['Month'] = df['Month'].map({'Feb':2,'Mar':3,'May':5,'Oct':10,'June':6,'Jul':7,'Aug':8,'Nov':11,'Sep':9,'Dec':12})


# In[ ]:


df.info()


# In[ ]:


df = df.dropna()


# In[ ]:


df.corr()
plt.figure(figsize=(12,7))
sns.heatmap(df.corr(),annot=True,cmap='viridis')
sns.distplot(df['Revenue'],kde=False,bins=40)


# In[ ]:


df.corr()['Revenue'].sort_values().plot(kind='bar')


# In[ ]:


x = df.drop('Revenue',axis=1).values
y = df['Revenue'].values


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
for i in range(1,25):
  tree=DecisionTreeClassifier(max_depth = i) 
  tree.fit(x_train, y_train) 
  scores.append(tree.score(x_test,y_test)) 
plt.plot(range(1,25),scores) 
plt.show()


# In[ ]:


tree=DecisionTreeClassifier(max_depth =5) 
tree.fit(x_train, y_train) 
tree.score(x_test,y_test)


# In[ ]:


predict=tree.predict([x_test[0]])
predict


# In[ ]:


y_test[0]


# In[ ]:


predictions = tree.predict(x_test) 
from sklearn.metrics import classification_report,confusion_matrix 
print(classification_report(y_test,predictions)) 
print(confusion_matrix(y_test,predictions))


# **Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


scores=[]
for i in (np.arange(100,2000,100)):
  classifier = RandomForestClassifier(n_estimators =i, max_depth=5, random_state =101) 
  classifier.fit(x_train, y_train) 
  scores.append(classifier.score(x_test,y_test)) 
plt.plot(np.arange(100,2000,100),scores) 
plt.show()


# In[ ]:


classifier = RandomForestClassifier(n_estimators =1000,max_depth=5, random_state =101)
classifier.fit(x_train, y_train)
print(classifier.score(x_test,y_test))


# In[ ]:


classifier = RandomForestClassifier(n_estimators =1000,max_depth=5, random_state =101)
classifier.fit(x_train, y_train)
print(classifier.score(x_test,y_test))


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


classifier = SVC(kernel = 'linear', C = 1)
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
model.add(Dense(units=35,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=35,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1,activation='sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])


# In[ ]:


early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

model.fit(x_train,y_train,epochs=50,validation_data=(x_test, y_test), verbose=1, callbacks=[early_stop])


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




