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


data = pd.read_csv("/input/train.csv", sep=",");


# In[ ]:


Basic_data = data
data.head(5)


# In[ ]:


data.info()


# In[ ]:


#data = data.drop('Weaks', axis=1)


# In[ ]:


data.duplicated().sum()


# In[ ]:


data['Class'].unique()


# In[ ]:


data.head()


# In[ ]:


import seaborn as sns
f, ax = plt.subplots(figsize=(44, 42))
corr = data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, annot = True);
plt.show()


# In[ ]:


colname = 'ID'
for i in range(len(corr.columns)):
    #print('-a')
    for j in range(i):
        #print('-b')
        if (corr.iloc[i, j] >= 0.75):
           # print('-c')
            colname = corr.columns[i]
            print(colname)
            #data = data.drop([corr.columns[i]], axis=1)            


# In[ ]:


data = data.drop(colname, axis=1)


# In[ ]:


data = data.drop('ID', axis=1)


# In[ ]:


data.head()


# In[ ]:


data = data.drop(['Enrolled', 'MLU', 'Reason', 'Area', 'State', 'Fill'], axis=1)


# In[ ]:


data = data.drop('PREV', axis=1)


# In[ ]:


y=data['Class']
x=data.drop(['Class'],axis=1)
x.head()


# In[ ]:


def convert(data):
    col = data.columns.values
    for i in col:
        null = {}
        def int_con(string):
            return null[string]
        if data[i].dtype != np.int64 and data[i].dtype != np.float64:
            val = set(data[i].values.tolist())
            a = 0
            for j in val:
                if j not in null:
                    null[j] = a
                    a = a+1
            data[i] = list(map(int_con,data[i]))
    return data


# In[ ]:


x = convert(x)
x.head()


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


test = pd.read_csv("/input/test_1.csv", sep=",");


# In[ ]:


test = test.drop(['Enrolled', 'MLU', 'Reason', 'Area', 'State', 'Fill'], axis=1)


# In[ ]:


test = test.drop(['PREV'], axis=1)


# In[ ]:


ID_test = test


# In[ ]:


test = test.drop(['ID'], axis=1)


# In[ ]:


test = test.drop(colname, axis=1)


# In[ ]:


test = convert(test)

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(test)
test = pd.DataFrame(np_scaled)
test.head()


# In[ ]:


print(Basic_data['ID'])


# In[ ]:


ID = set()
prediction = rf.predict(test)

np.savetxt('/home/kapish/Documents/3-2/DM/Assignment2/assignment2.csv',prediction ,delimiter=',')


# In[ ]:




