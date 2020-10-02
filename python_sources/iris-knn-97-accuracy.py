#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()


# In[ ]:


import pandas as pd
data = pd.read_csv("../input/iris/Iris.csv")


# In[ ]:


data.head()


# In[ ]:


#changing column names accordingly
data.columns=["id" , 
             "SepalLengthCm",
             "SepalWidthCm" ,
             "PetalLengthCm" ,
              "PetalWidthCm" ,
              "Species"]
data.head()                    


# In[ ]:


# X= given 
# y= to be founded
X = data.drop(["Species"] , axis = 1)   #axis=1 removes Species column
y = data["Species"]
print(X.shape,y.shape)


# In[ ]:


print(X,y)


# In[ ]:



#Data split into train and test

X_train ,X_test , y_train , y_test =train_test_split(X,y,test_size=.25,random_state=100)
print ("Shapes of X_train, X_test , y_train , y_test are :" )
print("       " , X_train.shape , X_test.shape , y_train.shape , y_test.shape)


# In[ ]:


X_test.head()


# In[ ]:


print(model)


# In[ ]:


model.fit(X_train , y_train)


# **PREDICTION OF OUTPUT**

# In[ ]:


till_row = 20
temp = X_test[:till_row]
temp["Species"] = y_test[ :till_row]
temp["predicted"] = model.predict(X_test[ : till_row])
temp


# **CALCULATING ACCURACY**

# In[ ]:


acc = model.score(X_test,y_test)
print(acc)
print(acc*100)


# In[ ]:


predicted = model.predict(X_test)
original = y_test.values
wrong = 0

for i in range (len(predicted)):
  if predicted[i]!=original[i]:
    wrong = wrong + 1
print(wrong)

