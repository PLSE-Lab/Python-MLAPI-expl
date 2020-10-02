#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


df=pd.read_csv('../input/datathon19/train.csv')
df.head()


# In[ ]:


df.describe()


# In[ ]:


df1=pd.read_csv('../input/datathon19/test.csv')
df1.head()


# In[ ]:


df1.describe()


# In[ ]:


df.shape


# In[ ]:


df1.shape


# In[ ]:


x_train=df.iloc[:,:9]
y_train=df['class']
x_train


# In[ ]:


x_test=df1.iloc[:,:9]
x_test


# In[ ]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


# In[ ]:


x_train['top_left_square'] = encoder.fit_transform(x_train['top_left_square'])
x_train['top_middle_square'] = encoder.fit_transform(x_train['top_middle_square'])
x_train['top_right_square'] = encoder.fit_transform(x_train['top_right_square'])
x_train['middle_left_square'] = encoder.fit_transform(x_train['middle_left_square'])
x_train['middle_middle_square'] = encoder.fit_transform(x_train['middle_middle_square'])
x_train['middle_right_square'] = encoder.fit_transform(x_train['middle_right_square'])
x_train['bottom_left_square'] = encoder.fit_transform(x_train['bottom_left_square'])
x_train['bottom_middle_square'] = encoder.fit_transform(x_train['bottom_middle_square'])
x_train['bottom_right_square'] = encoder.fit_transform(x_train['bottom_right_square'])

x_test['top_left_square'] = encoder.fit_transform(x_test['top_left_square'])
x_test['top_middle_square'] = encoder.fit_transform(x_test['top_middle_square'])
x_test['top_right_square'] = encoder.fit_transform(x_test['top_right_square'])
x_test['middle_left_square'] = encoder.fit_transform(x_test['middle_left_square'])
x_test['middle_middle_square'] = encoder.fit_transform(x_test['middle_middle_square'])
x_test['middle_right_square'] = encoder.fit_transform(x_test['middle_right_square'])
x_test['bottom_left_square'] = encoder.fit_transform(x_test['bottom_left_square'])
x_test['bottom_middle_square'] = encoder.fit_transform(x_test['bottom_middle_square'])
x_test['bottom_right_square'] = encoder.fit_transform(x_test['bottom_right_square'])


# In[ ]:


x_test.head()


# In[ ]:


x_train.head()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model=model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred


# In[ ]:


id=df1['Id']


# In[ ]:


np.random.seed(123)
pred=pd.DataFrame(y_pred,columns=['predictions'])
pred


# In[ ]:


pred = np.vstack((id,y_pred)).T
pred


# In[ ]:


np.savetxt('Ayush-Pratyay-datathon19.csv', pred, delimiter=',', fmt="%i")


# In[ ]:


import csv
with open('Ayush-Pratyay-datathon19.csv',newline='') as f:
    r = csv.reader(f)
    data = [line for line in r]
with open('Ayush-Pratyay-datathon19.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['Id','class'])
    w.writerows(data)


# In[ ]:




