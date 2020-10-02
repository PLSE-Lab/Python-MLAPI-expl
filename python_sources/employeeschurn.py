#!/usr/bin/env python
# coding: utf-8

# ## EmployeesChurn ##
# An Algorithm (A Very Deep Neural Network) written in python with Tensorflow and Keras which can classify if the employees will leave the Corporation or Not based on their Data which contains their Salary, the Job, etc. The algorithm was trained and tested over a Dataset Provided by Kaggle. The algorithm can classify that with a ~96% Accuracy. I built it for fun, As I'm a Deep-Learning Student but a Full Stack Node.js Developer.
# 
# Dependencies
# ------------
# 
#  1. Pandas (for Dataframe)
#  2.Sci Kit Learn (for Data Pre-Processing)
#  3. Tensorflow (for running Keras with TF Backend)
#  4. Keras (Deep Learning Library)
#  5. Conda (for Managing ENVs)
# 
# How to Use?
# -----------
# 
# 1. Clone or Download the Project from the Fancy Little Clone Button above.
# 2. Change Directory and enter the EmployeeChurn folder.
# 3. Install the Dependencies with PIP or Conda, your Preference. (Please Use Python 3.5.x)
# 4. Execute the Program with python clf.py or Run It Directly with Jupyter Notebook
# 
# Dataset
# -------
# 
# The Dataset was Provided by Kaggle, You can find it here - https://www.kaggle.com/nitishaadhikari/hrdata

# In[ ]:


import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


#data preprocessing
df = pd.read_csv('../input/HR_comma_sep.csv')
X = df.ix[:, df.columns != 'left']
X = X.iloc[:, 0:9].values
y = df.iloc[:, -4].values
# label encoding
le = LabelEncoder()
X[:, -1] = le.fit_transform(X[:, -1])
X[:, -2] = le.fit_transform(X[:, -2])
# One hot encoding
ohe = OneHotEncoder(categorical_features=[1])
X = ohe.fit_transform(X).toarray()
# Creating sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=1)
# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


# creating the model
clf = Sequential([
    Dense(units=11, kernel_initializer='uniform', activation='relu', input_dim=10),
    Dense(units=11, kernel_initializer='uniform', activation='relu'), # units are based on my creativity :3
    Dense(1, kernel_initializer='uniform', activation='sigmoid') #output
])


# In[ ]:


# compiling the model
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


clf.fit(X_train, Y_train, batch_size=9, epochs=10) # less training due to slow kaggle servers


# In[ ]:


score = clf.evaluate(X_test, Y_test, batch_size=128)
print(score[1]*100, '%') # 96.9333337148% or 0.96133

