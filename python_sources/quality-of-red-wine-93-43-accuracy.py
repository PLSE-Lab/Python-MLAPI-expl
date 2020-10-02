#!/usr/bin/env python
# coding: utf-8

# ***Welcome !!***
# 
# 1. The Probelm Is solved using Radom Forest Regression with 93.43% accuracy.
# 2. The code is pretty intutive and easy to follow.
# 
# 

# **Importing Necessary Packages**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder  
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score


# **Importing the dataset**

# In[ ]:


# Importing the dataset
wine = pd.read_csv('../input/winequality-red.csv')


# **Viewing the data set**

# In[ ]:


wine.head(10)


# **As given in the problem statement, quality above or equal to the rating of 7 is said to be Good Wine.
# So, Let's make a classification between the good and the bad wine.**

# In[ ]:


for i in range(len(wine)):
    if wine.iloc[i,11]>=7:
        wine.iloc[i,11]='good'
    else:
        wine.iloc[i,11]='bad'


# In[ ]:


wine.head(10)


# **Now let's label encode them as 1 for good and 0 for bad.**

# In[ ]:


labelencdoer=LabelEncoder()
wine['quality']=labelencdoer.fit_transform(wine['quality'])
wine.head(10)


# **Split the dataset into dependent and independent variables . Scale the X value and split it into test and training set.**

# In[ ]:


X = wine.iloc[:, :-1].values
y = wine.iloc[:, 11].values

#scaling the Xvalue
sc=StandardScaler()
X=sc.fit_transform(X[:,:])

#Splitting into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# **Fit the Training set into the RandomForestRegressor module. Predict the final value using test set.**

# In[ ]:


#n_estimators represents nos of trees to be used in the model
rfr=RandomForestRegressor(n_estimators=40,random_state=0)
rfr.fit(X_train,y_train)

#Final Prediction
y_pred=np.matrix.round(rfr.predict(X_test))


# **Calculate the Accuracy Score !**

# In[ ]:


acc=accuracy_score(y_test, y_pred)
print("accuracy: ",acc*100,'%' )


# **Thank You
# Please upvote if you find this code useful!**
