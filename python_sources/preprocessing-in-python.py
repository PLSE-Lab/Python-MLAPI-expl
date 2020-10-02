#!/usr/bin/env python
# coding: utf-8

# # This is a simple tutorial for pre processing data in Python.Tutorial will have following steps
# 
# 1.Importing Python Modules 
# 
# 2.Importing data 
# 
# 3.Displaying data 
# 
# 4.Creating the Independent and Dependent variables
# 
# 5.Replacing missing value with meaningful value 
# 
# 6.Encoding catogerical data
# 
# 7.Splitting the data into training and test set
# 
# 8.Doing feature scaling on data 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Step1: Importing Python Modlues

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# > # Step2: Importing data 

# In[ ]:


data=pd.read_csv('../input/Data_1.csv')


# # Step3: Displaying the data

# In[ ]:


data.head(9)


# Step4: Displaying data information

# In[ ]:


data.info()


# There are missing values in the data in the Age and Salary columns

# # Step5: Creating array of independent variables

# In[ ]:


X=data.iloc[:,:-1].values
X


# # Step6: Creating array of dependent variables

# In[ ]:


y=data.iloc[:,-1].values # or we can use y=data.iloc[:,3].values
y


# # Step7: Replacing missing values with the mean values of the columns

# In[ ]:


from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0) 
imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
X


# You can see salary data for Nepal and age data for Nepal are added as the mean of the columns

# # Step8:Encoding the Catogerical data

# In[ ]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0]) #Encoding the values of column Country
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
X


# In[ ]:


labelencoder_y=LabelEncoder()
y = labelencoder_y.fit_transform(y)
y


# # Step9: Splitting the data into training and test data

# In[ ]:


from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# # Step10: Doing a feature scaling on data

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

