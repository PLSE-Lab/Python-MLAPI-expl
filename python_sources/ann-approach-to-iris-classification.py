#!/usr/bin/env python
# coding: utf-8

# Import the dataset.

# In[ ]:


import pandas as pd

dataset = pd.read_csv('../input/Iris.csv')


# In[ ]:


dataset.head()


# Split dataset into Dependent Variables, y, and Independent Varaibles, X.

# In[ ]:


X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values


# Label encode y and then one hot encode it.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)
y = to_categorical(y)


# Feature Scale X using the Standard Scaler.

# In[ ]:


from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(X)


# Split X and y into the training and testing (validation) datasets.

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)


# Develop the ANN model.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(32, input_dim = 4, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()


# Train the model

# In[ ]:


model.fit(X_train, y_train, batch_size = 64, epochs = 50, verbose = 0)


# Evaluate the model using the testing dataset

# In[ ]:


model.evaluate(X_test, y_test)


# Does overfitting occur when you train for 100 epochs?

# In[ ]:


model.fit(X_train, y_train, batch_size = 64, epochs = 100, verbose = 0)


# In[ ]:


model.evaluate(X_test, y_test)


# The accuracy remains the same but the loss increases.
