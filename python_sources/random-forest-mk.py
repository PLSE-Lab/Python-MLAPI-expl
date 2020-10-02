
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd 
# data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.


# In[2]:




train = pd.read_csv("C:/Users/MINA K/Downloads/train.csv")
test = pd.read_csv("C:/Users/MINA K/Downloads/test.csv")

print(train.info())
print(test.info())



# In[3]:


from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing


# In[4]:


df1_train = train['TARGET_5Yrs']
#df1_train = pd.DataFrame(df1_train).fillna(df1_train.mean()) 


# In[5]:


df2_train = train.loc[ : ,'GP':'TOV']
df2_train = pd.DataFrame(df2_train).fillna(df2_train.mean())


# In[6]:


df2_test = test.loc[ : ,'GP':'TOV']
df2_test = pd.DataFrame(df2_test).fillna(df2_test.mean())


# In[7]:


# In[8]:


X_train = df2_train.values
Y_train = df1_train.values
X_test = df2_test.values


# In[31]:


from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier(max_depth=9, random_state=0)
from sklearn import preprocessing


# In[33]:


from sklearn.preprocessing import StandardScaler
#standardizing
std_scale = preprocessing.StandardScaler().fit(X_train)
train_std = std_scale.transform(X_train)
test_std = std_scale.transform(X_test)


# In[34]:


clf2.fit(train_std, Y_train)


# In[35]:


y_predicted_train = clf2.predict(train_std)
y_predicted_test = clf2.predict(test_std)


# In[36]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_train,y_predicted_train)


# In[38]:


cols = { 'PlayerID': [i+901 for i in range(440)] , 'TARGET_5Yrs': y_predicted_test }
submission = pd.DataFrame(cols)
print(submission)

submission.to_csv("submission4.csv", index=False)

