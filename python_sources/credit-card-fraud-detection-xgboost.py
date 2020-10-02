#!/usr/bin/env python
# coding: utf-8

# <h1>CREDIT CARD FRAUD DETECTION</h1>
# <p>The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# 
# It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.</p>

# In[ ]:


# Importing necessary libraries
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# <h1>Data Preprocessing</h1>

# In[ ]:


# Loading data
data = pd.read_csv('../input/creditcard.csv')


# In[ ]:


# View DataFrame
data.head()


# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


data.info()


# <p>Kudos! We don't have any missing values in our data</p>
# <p>30 columns in our data is float, and 1 is int</p>

# In[ ]:


data.describe()  #statistical inference


# In[ ]:


# Visualising every feature
data.hist(figsize=(20,20))
plt.show()


# In[ ]:


# Determine number of fraud cases in dataset
Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud)/(len(Valid))
print(outlier_fraction)

print('Fraud Cases : {}'.format(len(Fraud)))
print('Valid Cases : {}'.format(len(Valid)))


# In[ ]:


# Correlation
corr = data.corr()
figure = plt.figure(figsize=(12,10))
sns.heatmap(corr)


# In[ ]:


# Splitting data
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
xtr,xtest,ytr,ytest = train_test_split(x,y,test_size=0.3,random_state=0)


# In[ ]:


xtr.shape,ytr.shape


# In[ ]:


xtest.shape,ytest.shape


# <h1>XGBoost</h1>
# <p>XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.</p>

# <h3>Initialising and fitting data to the model</h3>

# In[ ]:


from xgboost import XGBClassifier
xg = XGBClassifier(random_state=0)
xg.fit(xtr,ytr)
xg.score(xtr,ytr)


# <h3>Validating on test data</h3>

# In[ ]:


pred = xg.predict(xtest)


# <h3>Checking Accuracy</h3>

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(pred,ytest)
cm


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(pred,ytest)

