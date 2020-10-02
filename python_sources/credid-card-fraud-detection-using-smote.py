#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imblearn.combine

data=pd.read_csv('../input/creditcard.csv')


# In[ ]:


data.head()


# In[2]:


from sklearn.preprocessing import StandardScaler


# In[3]:


sc=StandardScaler()


# In[4]:


data['scaled_amount']=sc.fit_transform(data['Amount'].values.reshape(-1,1))
data['scaled_time']=sc.fit_transform(data['Time'].values.reshape(-1,1))


# In[5]:


print(data.columns)


# In[6]:


data = data.drop(['Time','Amount'],axis=1)


# In[7]:


x = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']


# In[8]:


y.columns


# In[9]:


x.columns


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=1)


# In[ ]:


print(y_train)


# In[ ]:


type(y_train)


# In[ ]:


y_train=pd.DataFrame(y_train)


# In[13]:


from imblearn.combine import SMOTEENN


# In[14]:


os_us=SMOTEENN(ratio=.5,random_state=1)
x_train,y_train=os_us.fit_sample(x_train,y_train.values.ravel())


# In[15]:


x_train=pd.DataFrame(x_train)


# In[16]:


x_train.head()


# In[17]:


y_train=pd.DataFrame(y_train)


# In[18]:


x_test.columns


# In[ ]:


x_test.head()


# In[19]:


x_test.columns=['0 ', '1 ', '2 ', '3 ', '4 ', '5 ', '6 ', '7 ', '8 ', '9 ', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']


# In[20]:


from xgboost import XGBClassifier


# In[21]:


from sklearn.metrics import accuracy_score


# In[23]:


model = XGBClassifier()
model.fit(x_train, y_train.values.ravel())


# In[24]:


y_pred = model.predict(x_test)


# In[25]:


predictions = [round(value) for value in y_pred]


# In[26]:


accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[28]:


from sklearn.metrics import confusion_matrix


# In[29]:


cm=confusion_matrix(y_test,y_pred)


# In[30]:


print(cm)


# In[31]:


from sklearn.metrics import classification_report


# In[32]:


print(classification_report(y_test,y_pred))


# In[33]:


from imblearn.under_sampling import RandomUnderSampler


# In[34]:


x_train2,x_test2,y_train2,y_test2=train_test_split(x,y,test_size=.25,random_state=1)


# In[35]:


us=RandomUnderSampler(ratio=.5,random_state=2)
x_train2,y_train2=us.fit_sample(x_train2,y_train2.values.ravel())


# In[36]:


x_train2=pd.DataFrame(x_train2)
y_train2=pd.DataFrame(y_train2)


# In[37]:


x_test2.columns=['0 ', '1 ', '2 ', '3 ', '4 ', '5 ', '6 ', '7 ', '8 ', '9 ', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']


# In[38]:


model2 = XGBClassifier()
model2.fit(x_train2, y_train2.values.ravel())


# In[39]:


y_pred2 = model2.predict(x_test2)


# In[40]:


cm2=confusion_matrix(y_test2,y_pred2)


# In[41]:


print(cm2)


# In[42]:


print(classification_report(y_test2,y_pred2))

