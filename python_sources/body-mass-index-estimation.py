#!/usr/bin/env python
# coding: utf-8

# The purpose of this project is to classify people who we have height and weight knowledge with machine learning.
# 
# **!!! Attention Please !!!**
# 
# **The calculation method and classification table specified are not absolute.**
# 
# **How To Calculate Body Mass Index Value**
# 
# > Index Value = Weight / ( Height * Height)
# 
# **Index Classification Table** (Iv = Index Value)
# 
# >**0** - Extremely Weak **Iv < 16**
# 
# >**1** - Weak ** 16 <= Iv < 18.5**
# 
# >**2** - Normal ** 18.5 <= Iv < 25**
# 
# >**3** - Overweight ** 25 <= Iv < 30**
# 
# >**4** - Obesity ** 30 <= Iv < 40**
# 
# >**5** - Extreme Obesity ** 40 <= Iv**

# In[ ]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
dataset = pd.read_csv('../input/500_Person_Gender_Height_Weight_Index.csv')
dataset.head()


# We will not use the **Gender** field because it is not in the classification formula.

# In[ ]:


attribute_array = dataset.iloc[:,1:3].values
index_array = dataset.iloc[:,3].values


# In[ ]:


encoder = LabelEncoder()
index_array = encoder.fit_transform(index_array)
index_array = np_utils.to_categorical(index_array)


# Learning and test data spliting...

# In[ ]:


attribute_array_train, attribute_array_test, index_array_train, index_array_test = train_test_split(attribute_array, index_array, test_size=0.05, random_state=1)


# In[ ]:


scaler = StandardScaler()
attribute_array_train = scaler.fit_transform(attribute_array_train)


# In[ ]:


classifier = Sequential()
classifier.add(Dense(kernel_initializer = 'uniform', input_dim = 2, units = 2, activation = 'relu'))
classifier.add(Dense(kernel_initializer = 'uniform', units = 6,  activation = 'softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.fit(attribute_array_train, index_array_train, batch_size=1, epochs=100, verbose=0)


# > **TIP : ** classifier.fit( .... verbose = 0) no output for console

# In[ ]:


ratio = classifier.evaluate(x=attribute_array_train,y=index_array_train)
print('Learning Ratio : ' + str(ratio[1]))


# In[ ]:


attribute_array_test_pred = classifier.predict(attribute_array_test)
print('Accuracy Score : ' + str(accuracy_score(index_array_test, attribute_array_test_pred.round(), normalize=True)))

