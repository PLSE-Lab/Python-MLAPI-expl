#!/usr/bin/env python
# coding: utf-8

# ### Importing the dataset

# In[ ]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

data = pd.read_csv('../input/paysim1/PS_20174392719_1491204439457_log.csv')

data.head()


# In[ ]:


#checking for null values
data.isnull().values.any()
count_type = data.type.value_counts()
count_type


# ### A bit of data visulalisation and analysis

# In[ ]:


# Finding the number of rows of the dataset 
r_len = len(data.index)

# count the number of rows with value 1 in isFlaggedFraud vs isFraud
count_fraud = data['isFraud'].value_counts()
count_detection = data['isFlaggedFraud'].value_counts()


#Visualisation
labels = ['isFraud','isFlaggedFraud']
plt.bar([1],count_fraud[1], width = 0.4)
plt.bar([2],count_detection[1],width = 0.4)
plt.ylabel('Count')
plt.title('Fraudulent transactions count')
plt.xticks([1,2], labels)
plt.show()
    


# In[ ]:


#checking fradulent transaction for each type of payments

count_payment = data.loc[(data.isFraud == 1) & (data.type == 'PAYMENT')]

count_transfer = data.loc[(data.isFraud == 1) & (data.type == 'TRANSFER')]

count_cashout = data.loc[(data.isFraud == 1) & (data.type == 'CASH_OUT')]

count_debit = data.loc[(data.isFraud == 1) & (data.type == 'DEBIT')]

count_cashin = data.loc[(data.isFraud == 1) & (data.type == 'CASH_IN')]


# In[ ]:


#Visualisation of all types of payments for number of frauds

labels = ['PAYMENT','TRANSFER','CASH_OUT','DEBIT','CASH_IN']
plt.bar([1,2,3,4,5],[len(count_payment),len(count_transfer), len(count_cashout), len(count_debit), len(count_cashin)])
plt.ylabel('Count of fraudulent transactions')
plt.title('Count of fraudlent transacitons for each type of payment')
plt.xticks([1,2,3,4,5], labels)
plt.show()


# As we can see only transfer and cashout type of payments are mostly responsible for the fraudlent transactions

# Checking for mutiple customer transactions to find the fradulent ones

# In[ ]:


count_fraud = data.loc[(data['isFraud'] == 1) ]
count_fraud.head()


# In[ ]:



count_nameOrig = count_fraud['nameOrig'].value_counts()
count_nameOrig.sort_values(ascending = False)
count_nameOrig.head()


# Checking the despostied ones for repeated fradulent transactions 

# In[ ]:


count_nameDest = count_fraud['nameDest'].value_counts()
count_nameDest.sort_values(ascending = False)
count_nameDest.head()


# From the above observations we can see that there are some repeat offenders in the depositor side 

# ### Data Cleaning

# As we can see from the above analysis the fraudulent transactions only occur in the TRANSFER and CASH_OUT side of the dataset. So we can assume that none of the other transactions contribute to the transaction fraud.

# In[ ]:


X = data.loc[(data['type'] == 'TRANSFER') | (data['type'] == 'CASH_OUT')]

Y = X['isFraud']
# Also we can drop the isFlaggedFraud coulmn as it has no siginificant impact on the dataset as observerd above.
# Also the names of the accounts can also be dropped as they are also irrelevant in this case.
X = X.drop(['nameOrig', 'isFlaggedFraud', 'nameDest', 'isFraud'], axis = 1)

X.head()


# In[ ]:


Y.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
onc =  LabelEncoder()
X['type'] = onc.fit_transform(X['type'])


# Now we have to convert the above data into train, test and cross validation sets.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(  X, Y, test_size=0.33, random_state=42)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
predictions = rfc.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_test,predictions)


# Thus we can see the score for this model is 0.9992 which is quite accurate for the test set that we created. 
