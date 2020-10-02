#!/usr/bin/env python
# coding: utf-8

# ### Dataset Overview
# The dataset contains transactions made by credit cards in September of 2013 by European cardholders. This dataset presents transactions that occurred within a two day period. A link to the dataset can be found below:
# 
# https://www.kaggle.com/mlg-ulb/creditcardfraud
# 
# The dataset contains **284807 observations across 31 columns**. All columns, with the exception of 'Time', 'Amount', and 'Class' have been censored due to confidentiality issues.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Read the data in and get a feel for its statistical properties

# In[ ]:


fraud = pd.read_csv('../input/creditcard.csv')


# In[ ]:


fraud.shape


# In[ ]:


fraud.columns


# In[ ]:


fraud.describe()


# In[ ]:


#Determine missing values across dataframe
fraud.info()


# ### Exploratory Analysis

# In[ ]:


#First look at Time
sns.distplot(fraud.Time)
plt.title('Distribution of Time')
plt.show()


# In[ ]:


#Now look at Amount
sns.boxplot(x=fraud['Amount'])
plt.title('Distribution of Amount')
plt.show()


# In[ ]:


fraud_total = fraud['Class'].sum()
print('Percent Fraud: ' + str(round((fraud_total/fraud.shape[0])*100, 2)) + '%')


# ## Look at distribution of time greater than 100,000 seconds

# In[ ]:


plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
plt.title("Time < 100,000")
fraud[fraud['Time']<100000]['Time'].hist()


plt.subplot(1,2,2)
plt.title("Time >= 100,000")
fraud[fraud['Time']>=100000]['Time'].hist()

plt.tight_layout()
plt.show()


# In[ ]:


fraud['10k_time'] = np.where(fraud.Time<100000, 1,0)


# In[ ]:


features = fraud.drop(['Time'], 1)


# In[ ]:


# look at distribution below $5.00
np.log10(features[features.Amount < 5]['Amount']+1).hist()


# In[ ]:


# how many frauds are actually 0 dollars?
print("Non-Fraud Zero dollar Transactions:")
display(features[(features.Amount == 0) & (features.Class == 0)]['Class'].count())
print("Fraudulent Zero dollar Transactions:")
display(features[(features.Amount == 0) & (features.Class == 1)]['Class'].count())


# >## Is it safe to drop from the data?
#      -maybe?

# In[ ]:


features = fraud[fraud.Amount > 0]


# In[ ]:


features.head()


# In[ ]:


features.Amount.quantile(.99)


# >## What percentage of fraud is less than the 99th percentile for purchase amount?

# In[ ]:


display(features[(features.Amount < 1000) & (features.Class == 1)]['Class'].count()/features[features.Class==1].shape[0])


# >## Approximately 98% of all fraudulent transactions are less than $1000

# In[ ]:


# set features equal to purchases less than $1000

# features = features[features.Amount<1000]


# In[ ]:


sns.distplot(np.log10(features['Amount'][features.Amount>=2]))


# In[ ]:


# create feature for > $2.00 transaction
features['dollar2'] = np.where(features.Amount > 2, 1, 0)
# features['dollar_1000'] = np.where(features.Amount > 1000, 1, 0)


# In[ ]:


features = features.drop(['Amount'], 1)


# In[ ]:


features.head()


# ## Set baseline accuracy with Logistic Regression

# In[ ]:


# Set X and y for model
X = features.drop(['Class'], 1)
y = features['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[ ]:


# try logistic regression model

lr = LogisticRegression(penalty='l2', solver='liblinear')

# Fit the model.
lr.fit(X_train, y_train)


# In[ ]:


print(lr.score(X_test, y_test))
# display(cross_val_score(lr, X, y, cv=5))


# In[ ]:


y_pred = lr.predict(X)
print(classification_report(y, y_pred))
display(pd.crosstab(y_pred, y))


# >## Try using SMOTE to improve precision and recall

# In[ ]:


# Create smote object
sm = SMOTE(random_state=0)

# Create resampled data
X_res, y_res = sm.fit_resample(X, y)


# In[ ]:


# create new training and test set
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=0)


# In[ ]:


# fit model
lr.fit(X_train, y_train)


# In[ ]:


print(lr.score(X_test, y_test))
# display(cross_val_score(lr, X_res, y_res, cv=5))

y_pred = lr.predict(X)
print(pd.crosstab(y_pred, y))

print(classification_report(y, y_pred))


# ## We can see a great improvement in recall after oversampling, which may be more important to this model because false positives aren't as important as true negatives.
# but can we do better?
# - yes

# In[ ]:


#Set up function to run our model with different trees, criterion, max features and max depth
rfc = ensemble.RandomForestClassifier(random_state=0, n_estimators=100, n_jobs=-1)
get_ipython().run_line_magic('time', 'rfc.fit(X_train, y_train)')
print('\n Percentage accuracy for Random Forest Classifier')
print(rfc.score(X_test, y_test)*100, '%')
# print(cross_val_score(rfc, X, Y, cv=5))

# display(cross_val_score(rfc, X_res, y_res, cv=5))


# In[ ]:


y_pred = rfc.predict(X)
print(pd.crosstab(y_pred, y))

print(classification_report(y, y_pred))


# # Here we have 97% precision and 100% recall on the original data, using a model trained on the oversampled data.
# 
#     Final overall accuracy is 99.99%

# In[ ]:





# In[ ]:




