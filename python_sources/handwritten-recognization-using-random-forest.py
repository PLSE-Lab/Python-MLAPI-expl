#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[ ]:


#Read train data
data=pd.read_csv('../input/mnist train.csv')
data.head()


# In[ ]:


data.shape


# In[ ]:


#Read test data
data1=pd.read_csv('../input/mnist test.csv')
data1.head()


# In[ ]:


data1.shape


# In[ ]:


#viewing the 2th row of train_data
a=data.iloc[1,1:].values
a=a.reshape(28,28)
plt.imshow(a)


# In[ ]:


data.iloc[1,0]


# In[ ]:


x=data.iloc[:,1:]
y=data.iloc[:,0]


# In[ ]:


#Creating training and testing samples from data with a ratio of 7:3(train:test)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=42)


# In[ ]:


model=RandomForestClassifier(n_estimators=200,max_samples=0.5)
model.fit(x_train,y_train)


# In[ ]:


#Predicting the testing sample of data
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


# In[ ]:


#first 5 values of testing sample of train_data
y_test[0:5]


# In[ ]:


#first 5 values of predicting samples
pred[0:5]


# In[ ]:


prediction=model.predict(data1)
prediction


# In[ ]:


prediction.shape


# In[ ]:


#Visualizing the 1rd row of test_data
b=data1.iloc[0,0:].values
b=b.reshape(28,28)
plt.imshow(b)


# In[ ]:


#The 1 value of prediction data
prediction[0]


# In[ ]:


#Visualizing the 3rd row of test_data
b=data1.iloc[2,0:].values
b=b.reshape(28,28)
plt.imshow(b)


# In[ ]:


#The 3rd value of prediction data
prediction[3]


# In[ ]:


print("Predicted "+ str(y_test.iloc[np.where(y_test!=pred)[0][3]]) + " as "+str(pred[np.where(y_test!=pred)[0][3]]) )
plt.imshow(np.array(x_test.iloc[np.where(y_test!=pred)[0][3]]).reshape(28,28))


# In[ ]:


np.where(data['label']==5)


# In[ ]:


b=data.iloc[51,1:].values
b=b.reshape(28,28)
plt.imshow(b)


# In[ ]:




