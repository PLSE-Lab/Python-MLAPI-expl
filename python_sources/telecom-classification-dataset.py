#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data = pd.read_csv("../input/telecom churn.csv")


# In[ ]:


data.head()


# In[ ]:


data.state.value_counts().plot(figsize=(10,10),kind = 'bar')


# In[ ]:


df = data.iloc[ : , 4:]


# In[ ]:


df.head()


# In[ ]:


df["churn"] = df["churn"].apply(lambda x : 1 if x == True else 0)


# In[ ]:


df.head()


# In[ ]:


sns.pairplot(df)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x= df.drop(["churn"], axis = 1)


# In[ ]:


x.head()


# In[ ]:


y = df['churn']


# In[ ]:


x["voice mail plan"] = x["voice mail plan"].map({'no' : 0, 'yes' : 1})


# In[ ]:


x["international plan"] = x["international plan"].apply(lambda x : 0 if x == 'no' else 1)


# In[ ]:


x.isnull().sum()


# In[ ]:


x = x.fillna(method = 'ffill')


# In[ ]:


x.isnull().sum()


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[ ]:


x = scaler.fit_transform(x)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 3)
print(x_train.shape)
print(y_train.shape)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


tree = DecisionTreeClassifier()


# In[ ]:


tree.fit(x_train,y_train)


# In[ ]:


tree.score(x_test,y_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


forrest = RandomForestClassifier()


# In[ ]:


forrest.fit(x_train,y_train)


# In[ ]:


forrest.score(x_test,y_test)


# In[ ]:


y_pred  = forrest.predict(x_test)


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


cm = confusion_matrix(y_test,y_pred)


# In[ ]:


cm


# In[ ]:


sns.heatmap(cm, annot = True)


# In[ ]:


x_c= df.drop(["churn"], axis = 1)


# In[ ]:


y_c = df['churn']


# In[ ]:


x_c = x_c.fillna(method = 'ffill')


# In[ ]:


x_c.head()


# In[ ]:


x_c.shape


# In[ ]:


x_c['international plan'] = x_c['international plan'].apply(lambda x:0 if x == 'no' else 1)


# In[ ]:


x_c['voice mail plan'] = x_c['voice mail plan'].apply(lambda x: 0 if x == "no" else 1)


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


classifier = Sequential()


# In[ ]:


classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))


# In[ ]:


classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))


# In[ ]:


classifier.add(Dense(units = 1, kernel_initializer = 'uniform',activation = 'sigmoid'))


# In[ ]:


x_ctrain,x_ctest,y_ctrain,y_ctest = train_test_split(x_c,y_c, test_size = .25, random_state = 3)


# In[ ]:


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


classifier.fit(x_ctrain,y_ctrain, batch_size = 10, epochs = 100)


# In[ ]:


y_predict =  classifier.predict(x_ctest)


# In[ ]:


y_predict = (y_predict > 0.5)


# In[ ]:




