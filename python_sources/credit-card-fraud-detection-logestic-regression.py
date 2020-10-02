#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


#Read and check the data
data =pd.read_csv('../input/creditcardfraud/creditcard.csv')
print(data.shape)
data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


#check missing data
data.isnull().sum()


# In[ ]:


#calculate how many are fraud and not fraud data? here 0 is not fraud and 1 is fraud
data.Class.value_counts()


# In[ ]:


# visualization of fraud and not fraud
LABELS=['Normal','Fraud']
sns.countplot(x = 'Class',data=data)
plt.xticks(range(2), LABELS)


# In[ ]:


amount_0 = data[data['Class']==0]['Amount']
amount_1 = data[data['Class']==1]['Amount']


# In[ ]:


time_0=data[data['Class']==0]['Time']
time_1 = data[data['Class']==1]['Time']


# In[ ]:


#Transactions in time
sns.kdeplot(time_0,label='Not_fraud')
sns.kdeplot(time_1,label='Fraud')
plt.title('Credit Card Transactions Time Density Plot',size = 20)


# In[ ]:


fig,(ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize = (8,9))
fig.suptitle('Time of transaction vs Amount by class')
ax1.scatter(time_0, amount_0)
ax1.set_title('Not Fraud')
ax2.scatter(time_1, amount_1)
ax2.set_title('Fraud')
plt.xlabel('Time[s]')
plt.ylabel('Amount')
plt.show()


# In[ ]:


#Feature correlation
corre = data.corr()
top_fea = corre.index
plt.figure(figsize=(20,20))
sns.heatmap(data[top_fea].corr(),annot = True,cmap="RdYlGn")


# In[ ]:


df1 = data[['Amount','Time']]


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled= scaler.fit_transform(df1)


# In[ ]:


data1 = pd.DataFrame(scaled ,columns= ['Amount_Scale','Time_Scale'])
data.drop(['Amount','Time'],axis= 1,inplace = True)
data = pd.concat([data1,data],axis=1)


# In[ ]:


data.head()


# In[ ]:


data = pd.concat([data1,data],axis=1)


# In[ ]:


X = data.drop('Class',axis = 1)
Y = data['Class']


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size =0.7,test_size=0.3,random_state=1)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
confusion_matrix(y_test,y_pred)


# In[ ]:


from sklearn import metrics
metrics.accuracy_score(y_test,y_pred)

