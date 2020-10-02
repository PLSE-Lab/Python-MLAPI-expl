#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as  plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/HR-Employee-Attrition.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.isna().sum()


# In[ ]:


df.notnull().sum()


# In[ ]:


df.describe()


# From the above descirbe table we can take decision of removing the column "Over18" because all the above emplyees are 18+. Since the column Over18 becomes meaning less.
# 
# Also Employee count, Employee number can be omitted. Employee count is one for every employees and employee number wont create any impact on attrition rate. 

# In[ ]:


df[['EmployeeCount','EmployeeNumber']]


# In[ ]:


df.head()


# In[ ]:


df['StandardHours'].value_counts()
df['StandardHours'].unique()


# In[ ]:





# In[ ]:


df.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'], axis = 1, inplace = True)


# In[ ]:


df.head()


# In[ ]:


df.dtypes


# 

# **Converting object data in to integer data in every column. In order to do that going for one hot encoding **
# 

# In[ ]:


numericalColumns = df.select_dtypes(include=np.number).columns
categoricalColumns = df.select_dtypes(exclude=np.number).columns


# In[ ]:


print(numericalColumns)
print(categoricalColumns)


# In[ ]:


ls = list(df[categoricalColumns])
for i in ls:
    print(i,":\n",df[i].unique(),'value Counts :',df[i].value_counts())
    print('-----------------------------------------------------------')


# 

# Replace Yes and No with 1 and 0

# In[ ]:


df.Attrition.replace({"Yes":1,"No":0},inplace = True)


# In[ ]:


df.head()


# In[ ]:


# df.Attrition.replace({"Yes":1,"No":0},inplace=True)
# df.Gender.replace({"Male":1,"Female":0},inplace=True)
# df.MaritalStatus.replace({"Single":1,"Married":0},inplace=True)


# In[ ]:


encodedCatCol = pd.get_dummies(df[categoricalColumns.drop(["Attrition"])])
encodedCatCol.head()


# In[ ]:


df_encoded_onehot = pd.concat([df[numericalColumns],encodedCatCol], axis = 1)
df_encoded_onehot.head()


# In[ ]:


df_final =pd.concat([df_encoded_onehot,df["Attrition"]],axis=1)


# In[ ]:


df_final.head()


# In[ ]:


df_final.corr()


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 


# In[ ]:


X = df_final.drop(columns='Attrition')
Y = df_final[['Attrition']]


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)


# In[ ]:


train_pred = logreg.predict(X_train)


# Confusion Matrix

# In[ ]:


from sklearn import metrics


# In[ ]:


metrics.confusion_matrix(Y_train,train_pred)


# Specificity****

# In[ ]:


tn, fp, fn, tp = metrics.confusion_matrix(Y_train,train_pred).ravel()
specificity = tn / (tn+fp)

print(specificity)


# In[ ]:


metrics.accuracy_score(Y_train,train_pred)


# In[ ]:


test_pred = logreg.predict(X_test)


# In[ ]:


metrics.confusion_matrix(Y_test, test_pred)


# In[ ]:


metrics.accuracy_score(Y_test, test_pred)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(Y_test,test_pred))


# In[ ]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
ROC_auc = roc_auc_score(Y_test,test_pred)
fpr, tpr, thresholds = roc_curve(Y_test,logreg.predict_proba(X_test)[:,1]) 
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % ROC_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

print(ROC_auc)
#print(thresholds)
#print(fpr)


# In[ ]:




