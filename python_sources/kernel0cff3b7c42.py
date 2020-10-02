#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


# In[ ]:


data1 = pd.read_csv("/home/kapish/Documents/3-2/DM/Assignment3/opcode_frequency_benign.csv", sep=",");
data2 = pd.read_csv("/home/kapish/Documents/3-2/DM/Assignment3/opcode_frequency_malware.csv", sep=",");


# In[ ]:


data1 = data1.assign(value = 0)
data2 = data2.assign(value = 1)


# In[ ]:


data = pd.concat([data1,data2])


# In[ ]:


data.sample(5)


# In[ ]:


print(data.FileName[1969])


# In[ ]:


data=data.dropna(axis=0)


# In[ ]:


zero_cols = [ col for col, is_zero in ((data == 0).sum() == data.shape[0]).items() if is_zero ]
data.drop(zero_cols, axis=1, inplace=True)
data.sample(5)


# In[ ]:


data.info()


# In[ ]:


data.duplicated().sum()


# In[ ]:


corr = data.corr()
colname = set()
for i in range(len(corr.columns)):
    #print('-a')
    for j in range(i):
        #print('-b')
        if (corr.iloc[i, j] >= 0.75):
           # print('-c')
            colname.add(corr.columns[i])
            print(colname)
            #data = data.drop([corr.columns[i]], axis=1)            


# In[ ]:


data = data.drop(colname, axis=1)


# In[ ]:


y=data['value']
x=data.drop(['value'],axis=1)
x.sample(5)


# In[ ]:


min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(x)
x = pd.DataFrame(np_scaled)
x.head()


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.4, random_state = 40)


# In[ ]:


rf = RandomForestClassifier()
rf.fit(X_train, Y_train)


# In[ ]:


print("Accuracy: ",rf.score(X_test, Y_test))


# In[ ]:


test = pd.read_csv("/home/kapish/Documents/3-2/DM/Assignment3/Test_data.csv", sep=",");
test.sample(5)


# In[ ]:


test['Unnamed: 1809'] = 0


# In[ ]:


test.drop(zero_cols, axis=1, inplace=True)
test.sample(5)


# In[ ]:


test.duplicated().sum()


# In[ ]:


test = test.drop(colname, axis=1)
test = test.drop(['Unnamed: 1809'], axis=1)
test.sample(5)


# In[ ]:


min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(test)
test = pd.DataFrame(np_scaled)
test.head()


# In[ ]:


prediction = rf.predict(test)

np.savetxt('/home/kapish/Documents/3-2/DM/Assignment3/assignment3.csv',prediction ,delimiter=',')


# In[ ]:




