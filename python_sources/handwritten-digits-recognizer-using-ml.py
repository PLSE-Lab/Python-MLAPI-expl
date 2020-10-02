#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[ ]:


#read train data
train_data=pd.read_csv('../input/digit-recognizer/train.csv')
train_data.head()


# In[ ]:


train_data.shape


# In[ ]:


train_data.info()


# In[ ]:


#read test data
test_data=pd.read_csv('../input/digit-recognizer/test.csv')
test_data.head()


# In[ ]:


test_data.shape


# In[ ]:


test_data.info()


# * ****Check the label of  random row in train data ****

# In[ ]:


label=train_data.iloc[2,1:].values
label=label.reshape(28,28)
plt.imshow(label)


# In[ ]:


#Label of 4th row
train_data.iloc[2,0]


# # # using random forest classifier to build model by using train data******

# In[ ]:


X=train_data.iloc[:,1:]
Y=train_data.iloc[:,0]


# In[ ]:


#Creating training and testing samples from train_data with a ratio of 7:3(train:test)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.7,random_state=42)


# In[ ]:


model=RandomForestClassifier(n_estimators=200,max_samples=0.5)
model.fit(x_train,y_train)


# In[ ]:


pred=model.predict(x_test)
pred


# In[ ]:


pred.shape


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


# Classification report shows that model accuracy is 96% ****
# 
# 

# lets check 5 values from the testing data of train_data with predicted ****data
# 
# 

# In[ ]:


#random values of testing sample of train_data
y_test[5:10]


# In[ ]:


pred[5:10]


# from the above two cells we can see that there is 100% accuracy in the prediction hence we trained a very good model and now we can use the model to predict the test data 

# In[ ]:


test_data_prediction=model.predict(test_data)
test_data_prediction


# In[ ]:


test_data_prediction.shape


#  lets crosscheck the test_data digits and the test_data_predicted digits ****

# In[ ]:


#Visualizing the 39th row of test_data
b=test_data.iloc[39,0:].values
b=b.reshape(28,28)
plt.imshow(b)


# In[ ]:


#The 39th value of prediction data
test_data_prediction[39]


# digits are predicted correctly. Hence we can say that the model is pretty much useful in predicting handwritten digit.

# checking the wrongly predicted values 

# In[ ]:


print("Predicted "+ str(y_test.iloc[np.where(y_test!=pred)[0][8]]) + " as "+str(pred[np.where(y_test!=pred)[0][8]]) )
plt.imshow(np.array(x_test.iloc[np.where(y_test!=pred)[0][8]]).reshape(28,28))


# In[ ]:


np.where(train_data['label']==9)


# In[ ]:


b=train_data.iloc[28,1:].values
b=b.reshape(28,28)
plt.imshow(b)

