#!/usr/bin/env python
# coding: utf-8

# <h2>A simple **Deep Neural Network** approach to get 94% accuracy on **customer attrition**</h2>

# In[ ]:


import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


telcom = pd.read_csv(r"../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
telcom.head()


# <h3>Checking the nature of the data</h3>
# 17 out of 21 columns seem **categorical** <br><br>
# 4 columns seem **non-categorical**: 
# * customerID
# * tenure
# * MonthlyCharges
# * TotalCharges
# 
# (As seen from the Unique Values below)

# In[ ]:


print ("Rows     : " ,telcom.shape[0])
print ("Columns  : " ,telcom.shape[1])
print ("\nMissing values :  ", telcom.isnull().sum().sum())
print ("\nUnique values :  \n",telcom.nunique())


# Going to change these columns to categorical by **dividing** them and **rounding down**.<br><br>
# Example: <br>
# [150, 200, 250, 370]**/100** = [1.5, 2.0, 2.5, 3.7] <br>
# **math.floor**[1.5, 2.0, 2.5, 2.7] = [1, 2, 2, 3]

# In[ ]:


import math
telcom['MonthlyCharges'] = telcom['MonthlyCharges'].apply(lambda x: math.floor(x/20))
telcom['tenure'] = telcom['tenure'].apply(lambda x: math.floor(x/10))

telcom['TotalCharges'] = pd.to_numeric(telcom['TotalCharges'], errors='coerce')
telcom['TotalCharges'] = telcom['TotalCharges'].fillna(np.mean(telcom['TotalCharges']))
telcom['TotalCharges'] = telcom['TotalCharges'].apply(lambda x: math.floor(x/1000))


# In[ ]:


telcom.head()


# Plotting a frequency distribution graph of the 3 columns - **tenure**, **MonthlyCharges** and **TotalCharges**
# <br><br>
# The reason of doing so is to check if the new "categories" are divided evenly *enough*

# In[ ]:


import matplotlib.pyplot as plt
x = np.arange(1,7)
y = telcom.groupby('MonthlyCharges')['customerID'].nunique()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y)
plt.show()


# In[ ]:


x = np.arange(1,9)
y = telcom.groupby('tenure')['customerID'].nunique()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y)
plt.show()


# In[ ]:


x = np.arange(telcom.TotalCharges.nunique())
y = telcom.groupby('TotalCharges')['customerID'].nunique()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y)
plt.show()


# Used a **label encoder** to change all the categories into **numerical** categories. <br><br>
# 
# example:<br>
# [yes, no] becomes [1, 0] <br>
# [Electronic, Mail, Bank] becomes [2, 1, 0] <br><br>
# 
# Numerical Categories are easier for the machines to compute and train algorithms. <br><br>

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

telcom = telcom.apply(lambda col: le.fit_transform(col))
telcom.head(3)


# In[ ]:


telcom = telcom.drop("customerID", axis=1)
y = telcom['Churn']
telcom = telcom.drop("Churn", axis=1)


# However, there's a problem with simple label encoding. The algorithm can assign a ranking to these numericals.<br><br>
# example:<br>
# 2 > 1 > 0 <br><br>
# 
# This would imply Electronic > Mail > Bank, which is not the case. There is not supposed to be a sense of hierarchy in the data. <br><br>
# 
# To counter this, we use **One Hot Encoding**.<br>
# This would divide Electronic, Mail and Bank into 3 separate columns, with a simple [1,0] to indicate true or false.<br>
# This would eliminate the possibility of hierarchy.
# 

# In[ ]:


onehotencoder = OneHotEncoder(categories = 'auto')
telcom = onehotencoder.fit_transform(telcom).toarray()
X = pd.DataFrame(telcom)
X.head(3)


# Build a **Deep Neural Network** Model. <br><br>
# After using **One Hot Encoding**, we have **66 columns**<br>
# Therefore, we set the input_dim to 66, and feed the data into our Deep Neural Network.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
model.add(Dense(64, input_dim=66, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, batch_size=10)

