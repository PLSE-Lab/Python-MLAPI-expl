#!/usr/bin/env python
# coding: utf-8

# ### Author : Sanjoy Biswas
# ### Project : Credit Card Fraud Detection
# ### Email : sanjoy.eee32@gmail.com

# In this notebook I will try to predict fraud transactions from a given data set. Given that the data is imbalanced, standard metrics for evaluating classification algorithm (such as accuracy) are invalid. I will focus on the following metrics: Sensitivity (true positive rate) and Specificity (true negative rate). Of course, they are dependent on each other, so we want to find optimal trade-off between them. Such trade-off usually depends on the application of the algorithm, and in case of fraud detection I would prefer to see high sensitivity (e.g. given that a transaction is fraud, I want to be able to detect it with high probability).

# **IMPORTING LIBRARIES:**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import warnings
warnings.filterwarnings('ignore')


# **READING DATASET :**

# In[ ]:


data=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')


# In[ ]:


data.head()


# **NULL VALUES:**

# In[ ]:


data.isnull().sum()


# **Thus there are no null values in the dataset.**

# **INFORMATION**

# In[ ]:


data.info()


# **DESCRIPTIVE STATISTICS**

# In[ ]:


data.describe().T.head()


# In[ ]:


data.shape


# **Thus there are 284807 rows and 31 columns.**

# In[ ]:


data.columns


# **FRAUD CASES AND GENUINE CASES**

# In[ ]:


fraud_cases=len(data[data['Class']==1])


# In[ ]:


print(' Number of Fraud Cases:',fraud_cases)


# In[ ]:


non_fraud_cases=len(data[data['Class']==0])


# In[ ]:


print('Number of Non Fraud Cases:',non_fraud_cases)


# In[ ]:


fraud=data[data['Class']==1]


# In[ ]:


genuine=data[data['Class']==0]


# In[ ]:


fraud.Amount.describe()


# In[ ]:


genuine.Amount.describe()


# **EDA**

# In[ ]:


data.hist(figsize=(20,20),color='lime')
plt.show()


# In[ ]:


rcParams['figure.figsize'] = 16, 8
f,(ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(genuine.Time, genuine.Amount)
ax2.set_title('Genuine')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()


# **CORRELATION**

# In[ ]:


plt.figure(figsize=(10,8))
corr=data.corr()
sns.heatmap(corr,cmap='BuPu')


# **Let us build our models:**

# In[ ]:


from sklearn.model_selection import train_test_split


# **Model 1:**

# In[ ]:


X=data.drop(['Class'],axis=1)


# In[ ]:


y=data['Class']


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=123)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc=RandomForestClassifier()


# In[ ]:


model=rfc.fit(X_train,y_train)


# In[ ]:


prediction=model.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test,prediction)


# **Model 2:**

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


X1=data.drop(['Class'],axis=1)


# In[ ]:


y1=data['Class']


# In[ ]:


X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.3,random_state=123)


# In[ ]:


lr=LogisticRegression()


# In[ ]:


model2=lr.fit(X1_train,y1_train)


# In[ ]:


prediction2=model2.predict(X1_test)


# In[ ]:


accuracy_score(y1_test,prediction2)


# **Model 3:**

# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


X2=data.drop(['Class'],axis=1)


# In[ ]:


y2=data['Class']


# In[ ]:


dt=DecisionTreeRegressor()


# In[ ]:


X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.3,random_state=123)


# In[ ]:


model3=dt.fit(X2_train,y2_train)


# In[ ]:


prediction3=model3.predict(X2_test)


# In[ ]:


accuracy_score(y2_test,prediction3)


# **All of our models performed with a very high accuracy.**

# In[ ]:




