#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir(".././input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


submission = pd.read_csv('.././input/sample_submission.csv')
submission.head()


# In[ ]:


train = pd.read_json('.././input/train.json')
test = pd.read_json('.././input/test.json')
# train = train[0:3000]
train.head()


# In[ ]:


train.describe()


# In[ ]:


train.columns


# In[ ]:


train.isnull().sum(axis=0)


# In[ ]:


del train['id']


# In[ ]:


cuisines = set()
for cuisine in train['cuisine']:
    cuisines.add(cuisine)
cuisines = list(cuisines)
print(cuisines)
print(len(cuisines))


# In[ ]:


ingredients = set()
for ingredient in train['ingredients']:
    for i in ingredient:
        ingredients.add(i)
for ingredient in test['ingredients']:
    for i in ingredient:
        ingredients.add(i)
ingredients = list(ingredients)


# In[ ]:


print(len(ingredients))


# In[ ]:


train.shape


# In[ ]:


test = pd.read_json('.././input/test.json')
print(test.shape)
test.head()


# 

# In[ ]:


ingredients
def one_hot_vector(unique, data):
    X_data = [0]*len(data)
    print(len(data))
    for i in range(len(data)):
        x = np.zeros(len(unique))
        for ing in data[i]:
            x[unique.index(ing)] = 1
        X_data[i] = x
    return X_data        


# In[ ]:


x_data = one_hot_vector(ingredients, train['ingredients'])


# In[ ]:


def one_hot_vector_y(unique, data):
    X_data = [0]*len(data)
    print(len(data))
    for i in range(len(data)):
        x = np.zeros(len(unique))
        x[unique.index(data[i])] = 1
        X_data[i] = x
    return X_data    

y_data = one_hot_vector_y(cuisines, train['cuisine'])
print(y_data[0])


# In[ ]:


assert(len(x_data) == len(train['ingredients']))
assert (len(x_data[0]) == len(ingredients))
assert(len(y_data) == len(train['ingredients']))
assert(len(y_data[0]) == len(cuisines))


# In[ ]:


x_test_data = one_hot_vector(ingredients, test['ingredients'])
print(x_test_data[0])
print(len(x_test_data))


# In[ ]:


assert(len(x_test_data) == len(test['ingredients']))
assert (len(x_test_data[0]) == len(ingredients))
print("y_data:", y_data[0])


# In[ ]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_transformed = encoder.fit_transform(train.cuisine)
print(y_transformed)


# In[ ]:


from sklearn.model_selection import train_test_split
x_data = np.array(x_data)
y_data = np.array(y_data)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_transformed, test_size=0.20, random_state=5)
print(x_train.shape, y_train.shape)
print(type(y_train[0]))


# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(dual=False,solver = 'lbfgs', max_iter=7888888)
logreg.fit(x_train, y_train)


# In[ ]:


train['ingredients'][0]
x_data[0]


# In[ ]:


score = logreg.score(x_test, y_test)
print(score)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time
training_accuracy = []
test_accuracy = []
neighbors_settings = [i for i in range(1, 12)]
for n_neighbors in range(1, 11):
    tt = time.time()
    print("n_neighbors: ", n_neighbors)
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    # record training set accuracy
    y_predict = knn.predict(x_test)
    score = accuracy_score(y_predict, y_test)
#     score = knn.score(np.array(x_test), np.array(y_test))
    print(score)
    training_accuracy.append(score)
print("time taken: ", time.time() - tt)


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(range(1, 11), training_accuracy, label="training accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# In[ ]:


from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear',gamma='auto', C = 1).fit(x_train, y_train) 
y_predict = svm_model_linear.predict(x_test) 
accuracy_score(y_predict, y_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)
classifier.fit(x_train, y_train)
y_predict = classifier.predict(x_test) 
accuracy_score(y_predict, y_test)


# In[ ]:


# neural networks
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(patience=5)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=5)
model = Sequential()
model.add(Dense(12, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(20, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model.fit(x_data, y_data, validation_split=0.2, epochs=1000, callbacks=[early_stopping_monitor], batch_size=10)


# In[ ]:





# In[ ]:




