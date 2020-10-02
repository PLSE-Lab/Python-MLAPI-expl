#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


train_df = pd.read_csv("C:/Users/Dayanand Baligar/train.csv")
train_df.head()


# In[ ]:


test_df = pd.read_csv("C:/Users/Dayanand Baligar/test.csv")
test_df.head()


# In[ ]:


train_df.describe()


# In[ ]:


test_df.describe()


# In[ ]:


train_df.shape


# In[ ]:


test_df.shape


# In[ ]:


x_train = train_df.iloc[:,0:9]
y_train = train_df['class']
x_test = test_df.iloc[:,0:9]
x_test.head()


# In[ ]:


x_train.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

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


classifier = DecisionTreeClassifier()
classifier = classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)


# In[ ]:


y_pred


# In[ ]:


id = test_df['Id']


# In[ ]:


np.random.seed(123)
e = np.random.normal(size=10)  
pred=pd.DataFrame(y_pred, columns=['prediction']) 
print (pred)


# In[ ]:


pred = np.vstack((id,y_pred)).T
pred


# In[ ]:


np.savetxt('daya_datathon.csv', pred, delimiter=',', fmt="%i")


# In[ ]:


import csv
with open('daya_datathon.csv',newline='') as f:
    r = csv.reader(f)
    data = [line for line in r]
with open('daya_datathon.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['Id','class'])
    w.writerows(data)


# In[ ]:




