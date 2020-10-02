#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[51]:


df = pd.read_csv('../input/Iris.csv')
df.info()


# In[52]:


sns.countplot(df['Species'])


# In[53]:


sns.scatterplot(x='SepalLengthCm',y='SepalWidthCm',hue='Species',data=df)


# In[54]:


df = df.replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})


# In[55]:


df.head()


# In[56]:


x_features = ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
y_features = ["Species"]
x_df = df[x_features]
y_df = df[y_features]


# In[57]:


x_df.head()


# In[58]:


X_train, X_test, Y_train, Y_test = tts(x_df, y_df, test_size = 0.3, random_state = 5)


# In[59]:


X_train = (X_train - np.mean(X_train))/np.std(X_train)
X_test = (X_test - np.mean(X_test))/np.std(X_test)


# In[60]:


X_train = X_train.assign(b=1)
X_test = X_test.assign(b=1)


# In[61]:


X_train.head()


# In[62]:


Y_train.head()


# In[63]:


n = len(X_train.columns)
weights1 = np.zeros((n,))
weights2 = np.zeros((n,))
weights3 = np.zeros((n,))


# In[64]:


def multiclassperceptron(X_train, Y_train, weights1, weights2, weights3, epochs):
    for i in range(epochs):
        for index, row in X_train.iterrows():
            f1 = np.dot(weights1, row)
            f2 = np.dot(weights2, row)
            f3 = np.dot(weights3, row)
            if(f1 >= f2 and f2 >= f3):
                if(Y_train.loc[index].values[0] == 0):
                    continue
                if(Y_train.loc[index].values[0] == 1):
                    weights1 = weights1 + row
                    weights2 = weights2 - row
                if(Y_train.loc[index].values[0] == 2):
                    weights1 = weights1 + row
                    weights3 = weights3 - row
            if(f2 >= f3 and f3 >= f1):
                if(Y_train.loc[index].values[0] == 0):
                    weights1 = weights1 - row
                    weights2 = weights2 + row
                if(Y_train.loc[index].values[0] == 1):
                    continue
                if(Y_train.loc[index].values[0] == 2):
                    weights2 = weights2 + row
                    weights3 = weights3 - row

            if(f3 >= f2 and f2 >= f1):
                if(Y_train.loc[index].values[0] == 0):
                    weights1 = weights1 - row
                    weights3 = weights3 + row
                if(Y_train.loc[index].values[0] == 1):
                    weights2 = weights2 - row
                    weights3 = weights3 + row
                if(Y_train.loc[index].values[0] == 2):
                    continue
            
    return(weights1,weights2,weights3)


    


# In[65]:


weights = multiclassperceptron(X_train, Y_train, weights1, weights2, weights3, 100)


# In[66]:


def predict(weights, x_row):
    f1 = np.dot(weights[0],x_row)
    f2 = np.dot(weights[1],x_row)
    f3 = np.dot(weights[2],x_row)
    print(f1,f2,f3)
    k = [f1,f2,f3]
    return np.argmax(k)


# In[92]:


Y_predicted = [predict(weights, x) for x in X_test.values]
cm = confusion_matrix(Y_test, Y_predicted)
print("Confusion Matrix",cm)
ax = sns.heatmap(confusion_matrix(Y_test, Y_predicted))


# In[68]:


Y_test.iloc[23]


# In[69]:


X_train.iloc[23]


# In[70]:


weights[0] - X_train.iloc[23]

