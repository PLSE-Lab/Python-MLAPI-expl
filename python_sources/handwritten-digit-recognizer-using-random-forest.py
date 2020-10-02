#!/usr/bin/env python
# coding: utf-8

# First import all the required libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[ ]:


#Read train data
train_data=pd.read_csv('../input/digit-recognizer/train.csv')
train_data.head()


# In[ ]:


#No.of Train data's row and column
train_data.shape


# In[ ]:


#Read test data
test_data=pd.read_csv('../input/digit-recognizer/test.csv')
test_data.head()


# In[ ]:


#NO. of Test data's row and column
test_data.shape


# **Viewing a random data from train_data and check with its label**

# In[ ]:


#viewing the 4th row of train_data
a=train_data.iloc[3,1:].values
a=a.reshape(28,28)
plt.imshow(a)


# In[ ]:


#Label of 4th row
train_data.iloc[3,0]


# Using Random Classifier to build a model to predict the output of test_data.

# In[ ]:


x=train_data.iloc[:,1:]
y=train_data.iloc[:,0]


# In[ ]:


#Creating training and testing samples from train_data with a ratio of 7:3(train:test)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=42)


# In[ ]:


model=RandomForestClassifier(n_estimators=200,max_samples=0.5)
model.fit(x_train,y_train)


# In[ ]:


#Predicting the testing sample of train_data
pred=model.predict(x_test)


# In[ ]:


#accuracy of training sample of train_data
model.score(x_train,y_train)


# In[ ]:


#accuracy of testing sample of train_data
model.score(x_test,y_test)


# In[ ]:


confusion_matrix(pred,y_test)


# In[ ]:


print(classification_report(pred,y_test))


# From Classification report we can see that the model has an accuracy of 96%.

# Now we will crosscheck 5 values from the testing sample of train_data with predicted data

# In[ ]:


#first 5 values of testing sample of train_data
y_test[0:5]


# In[ ]:


#first 5 values of predicting samples
pred[0:5]


# Thus we can see that the model is predicting accurately all the five data.

# Now we can predict the output for the test_data using the above model

# In[ ]:


prediction=model.predict(test_data)
prediction


# In[ ]:


prediction.shape


# **Now we will first visualize some rows of test_data using matplot library and then check that image with the value of our predicted data**

# In[ ]:


#Visualizing the 3rd row of test_data
b=test_data.iloc[2,0:].values
b=b.reshape(28,28)
plt.imshow(b)


# In[ ]:


#The 3rd value of prediction data
prediction[3]


# In[ ]:


#Visualizing the 1st row of test_data
b1=test_data.iloc[0,0:].values
b1=b1.reshape(28,28)
plt.imshow(b1)


# In[ ]:


#1st value of prediction data
prediction[0]


# Here both the data are predicted correctly. Hence we can say that the model is pretty much useful in predicting handwritten digit.

# Now we will see the wrongly predicted values and visualize the input digit. 

# In[ ]:


print("Predicted "+ str(y_test.iloc[np.where(y_test!=pred)[0][3]]) + " as "+str(pred[np.where(y_test!=pred)[0][3]]) )
plt.imshow(np.array(x_test.iloc[np.where(y_test!=pred)[0][3]]).reshape(28,28))


# Here 5 is assumed to be 3 by the model as the actual 5 learned by the model can be seen below: 

# In[ ]:


np.where(train_data['label']==5)


# In[ ]:


b=train_data.iloc[51,1:].values
b=b.reshape(28,28)
plt.imshow(b)

