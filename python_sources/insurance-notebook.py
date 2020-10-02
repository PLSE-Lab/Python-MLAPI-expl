#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import re
import keras


# # Importing dataset

# In[ ]:


dataset = pd.read_csv(r"C:\Users\Sunshine\Downloads\Compressed\insurance.csv")


# In[ ]:


X =  dataset.iloc[:, 0:6].values


# In[ ]:


y = dataset.iloc[:, 6].values


# In[ ]:


X


# In[ ]:


y


# # Categorical data

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


label_X = LabelEncoder()
X[:, -1] = label_X.fit_transform(X[:, -1])
X[:, 1] = label_X.fit_transform(X[:, 1])
X[:, 4] = label_X.fit_transform(X[:, 4])
label_y = LabelEncoder()
y = label_y.fit_transform(y)


# In[ ]:



X


# In[ ]:


y


# # Feature selection

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier


# In[ ]:


clf = ExtraTreesClassifier()
clf.fit(X, y)


# In[ ]:


print(clf.feature_importances_)


# In[ ]:


X = dataset.iloc[:, [0,2]].values


# In[ ]:


X


# In[ ]:


y = dataset.iloc[:, [6]].values


# In[ ]:


y


# # Spilitting of Dataset

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# # training of model

# In[ ]:


from keras import regularizers
model = keras.Sequential([
    keras.layers.Dense(units = 64, activation = 'relu', input_shape = [2]),
    keras.layers.Dense(units = 128, activation = 'relu'),
    keras.layers.Dense(units = 256, activation = 'relu'),
    keras.layers.Dense(units =1)
])

model.compile(optimizer = 'Adamax', loss = 'mean_absolute_percentage_error')
history = model.fit(X_train, y_train, epochs = 200)


# In[ ]:


model.summary()


# In[ ]:


y_pred_train = model.predict(X_train)


# In[ ]:


y_pred_test = model.predict(X_test)


# In[ ]:


y_pred_train


# In[ ]:


y_pred_test


# In[ ]:


print(history.history.keys())


# In[ ]:


X_train[0]


# In[ ]:


y_pred_train[0]


# In[ ]:





# In[ ]:





# In[ ]:


print("Train Accuracy : " , np.mean((y_pred_train / y_train) * 100), "%")


# In[ ]:


print("Test Accuarcy : " , np.mean((y_pred_test / y_test) * 100), "%")


# In[ ]:


print("Loss value : " , model.evaluate(X_train, y_train))


# In[ ]:





# In[ ]:




