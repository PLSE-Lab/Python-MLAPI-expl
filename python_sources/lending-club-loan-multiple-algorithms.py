#!/usr/bin/env python
# coding: utf-8

# # Lending Club Dataset:

# ![](https://theme.zdassets.com/theme_assets/680652/3abc1fe11ed0a385b1298f0a1e44a7d7d5f78fc1.png)

# 
# For this project we'll be using publicly available data from LendingClub.com. Lending Club connects people who need money (borrowers) with people who have money (investors). Hopefully, as an investor you would want to invest in people who showed a profile of having a high probability of paying you back. We will try to create a model that will help predict this.

# In[ ]:


#Importing all the necessary library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings 
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input/dataset/"))


# In[ ]:


#Let is convert or dataset into a DataFrame
loans=pd.read_csv('../input/dataset/loan_borowwer_data.csv')
loans.head()


# In[ ]:


loans.shape


# **Observation:** We have 14 columns and 9578 rows. Lets check the datatype of each columns

# In[ ]:


loans.info()


# **Observation:** Only the purpose column is categorical. 

# In[ ]:


loans.describe()


# In[ ]:


plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')


# In[ ]:


plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')


# In[ ]:


purpose=list(loans['purpose'].unique())
purpose


# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(loans['purpose'])


# In[ ]:


loans['purpose'].value_counts()


# In[ ]:


sns.countplot(loans['not.fully.paid'])


# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x=loans['purpose'],hue=loans['not.fully.paid'],palette='Set1')


# In[ ]:


corrmat=loans.corr()
plt.subplots(figsize=(10, 9))
sns.heatmap(corrmat, vmax=.8, square=True,cbar=True, annot=True, fmt='.2f', annot_kws={'size': 10});


# In[ ]:


loans[['int.rate','inq.last.6mths','revol.util']].describe()


# In[ ]:


sns.jointplot(x='fico',y='int.rate',data=loans,color='green')


# In[ ]:


plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Accent_r')


# In[ ]:


loans.info()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[ ]:


label_encoder=LabelEncoder()
loans['purpose']=label_encoder.fit_transform(loans['purpose'])


# In[ ]:


loans['purpose'].dtype


# ## Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = loans.drop('not.fully.paid',axis=1)
y = loans['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# ## Training a Random Forest Model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)


# In[ ]:


predict_rfc = rfc.predict(X_test)


# ## Training a Logistic Regession Model

# In[ ]:


#Importing Logistic Regression
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(X_train,y_train)


# In[ ]:


predict_Log=log.predict(X_test)


# In[ ]:


#importing a Support Vector machine
from sklearn.svm import SVC
svm=SVC(gamma='auto')
svm.fit(X_train,y_train)


# In[ ]:


predict_svm=svm.predict(X_test)


# ## Predictions and Evaluation of All Models

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[ ]:


print('\t\t CLASSIFICATION REPORT- Random Forest')
print(classification_report(y_test,predict_rfc))
print('\t  CLASSIFICATION REPORT- Logistic Regression')
print(classification_report(y_test,predict_Log))


# In[ ]:


sns.heatmap(confusion_matrix(y_test,predict_rfc),annot=True,fmt='')
plt.title('Confusion Matrix-Random Forest')


# In[ ]:


sns.heatmap(confusion_matrix(y_test,predict_Log),annot=True,fmt='')
plt.title('Confusion Matrix-Logistic Regression')


# In[ ]:


print('Accuracy Score of Random Forest Classifier: {:.2f}'.format(accuracy_score(y_test,predict_rfc)))
print('Accuracy Score of Logistic Regression Classifier: {:.2f}'.format(accuracy_score(y_test,predict_Log)))


# **Conclusion:** As you can see the Logistic regression worked slightly better in this situation but thats not enougnh. We need to do a lot more feature engineering and try out differenet models in order to understand which is better. I hope you liked this notebook, if yes kindly leave an upvote. 
# I'll be updating this notebook with time.
