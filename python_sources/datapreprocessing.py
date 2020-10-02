#!/usr/bin/env python
# coding: utf-8

# importing library
# *  NumPy is a package in Python used for Scientific Computing. The ndarray (NumPy Array) is a multidimensional array used to store values of same datatype
# * matplotlib.pyplot for data visualization 
# * pandas for data manipulation  
# 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# importing dataset to x and y

# In[ ]:


dataset=pd.read_csv('../input/Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values


# fixing missing data

# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])


#  LabelEncoder,OneHotEncoder used for convert categorical data, or text data, into numbers, which our predictive models can better understand.
# *  note:
#   FutureWarning: The handling of integer data will change in version 0.22.
#   thats why alternative method
#   from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# ct = ColumnTransformer(
#     [('one_hot_encoder', OneHotEncoder(), [0])],    
#     remainder='passthrough'                         # Leave the rest of the columns untouched
# )
# x = np.array(ct.fit_transform(x), dtype=np.float)

# In[ ]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
x=onehotencoder.fit_transform(x).toarray()
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)


# Splitting the dataset into the Training set and Test set
# * note: early its was in sklearn.cross_validation for new version it will be in sklearn.model_selection

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)


#  Feature Scaling is done by StandardScaler

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

