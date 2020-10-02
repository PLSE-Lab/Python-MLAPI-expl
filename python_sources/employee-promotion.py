#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('/kaggle/input/wns-inno/train_LZdllcl.csv')
df1=pd.read_csv('/kaggle/input/wns-inno/test_2umaH9m.csv')


# In[ ]:


df.head(3)


# In[ ]:


df1.head(3)


# **Train Data Analysis**

# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


df.is_promoted.value_counts()


# In[ ]:


df.is_promoted.value_counts(normalize=True)


# In[ ]:


df.isnull().sum()/len(df)


# In[ ]:


df.isnull().sum().sum()/len(df)


# In the data set,total missing values is around 12% 

# **Treatment of Missing Values**

# In[ ]:


a=df[df['previous_year_rating'].isnull()]
a.head()


# In[ ]:


a['length_of_service'].value_counts()


# Since the length of service is 1 for all the employees with previous year rating as null.,which means they are the new recruits with 1 year experience. So they may not be having the previous year rating.We impute 0 for the null values.

# In[ ]:


df['previous_year_rating'].fillna(value=0,inplace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


for i in df[df['education'].isnull()]['KPIs_met >80%']==0:
    df['education'].fillna(value="Bachelor's",inplace=True)


# In[ ]:


for i in df[df['education'].isnull()]['KPIs_met >80%']==1:
    df['education'].fillna(value="Master's & above",inplace=True)


# In[ ]:


df['education'].isnull().sum()


# In[ ]:


df.isnull().sum()


# We do not need the 'employee_id' column.,so dropping it

# In[ ]:


df.drop(['employee_id'],axis=1,inplace=True)


# In[ ]:


df.shape


# In[ ]:


plt.figure(figsize=(15,8))
sns.heatmap(df.corr(),annot=True,linewidths=0.8)
plt.show()


# Here previous_year_rating,KPIs_met>80%,awards_won? and avg_training_score are highly correlated with the target variable i.e is_promoted.
# 
# Length of the service and age are highly correlated with each other

# In[ ]:


df_new=pd.get_dummies(df,['department','region','education','gender','recruitment_channel'],drop_first=True)


# In[ ]:


df_new.head(2)


# **Model Building**

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score


# In[ ]:


X=df_new.drop('is_promoted',axis=1)
y=df_new['is_promoted']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# **Logistic Regression**

# In[ ]:


lr=LogisticRegression(max_iter=10000)


# In[ ]:


lr.fit(X_train,y_train)


# In[ ]:


pred=lr.predict(X_test)


# In[ ]:


print("The Train score of Logistic Regression is :",lr.score(X_train,y_train))
print("The Test score of Logistic Regression is :",lr.score(X_test,y_test))
print("The accuracy of Logistic Regression is ",accuracy_score(y_test,pred))
print("The f1 score of Logistic Regression is ",f1_score(y_test,pred))
print("The Confusion Matrix of Logistic Regression is \n \n",confusion_matrix(y_test,pred))
print("\n")
print("The Classification Report of Logistic Regression is \n \n",classification_report(y_test,pred))


# Here we got 1039 False Negatives which can be further reduced in the next model

# **Random Forest**

# In[ ]:


rf = RandomForestClassifier(random_state=101)
rf.fit(X_train,y_train)


# In[ ]:


pred_rf=rf.predict(X_test)


# In[ ]:


print("The Train score of Random Forest Classifier is :",rf.score(X_train,y_train))
print("The Test score of Random Forest Classifier is :",rf.score(X_test,y_test))
print("The accuracy of Random Forest Classifier is ",accuracy_score(y_test,pred_rf))
print("The f1 score of Random Forest Classifier is ",f1_score(y_test,pred_rf))
print("The Confusion Matrix of Random Forest Classifier is \n \n",confusion_matrix(y_test,pred_rf))
print("\n")
print("The Classification Report of Random Forest Classifier is \n \n",classification_report(y_test,pred_rf))


# Here the number of False Negatives has been reduced from 1039 to 992

# **XGBoost**

# In[ ]:


xgbc=xgb.XGBClassifier()
xgbc.fit(X_train,y_train)


# In[ ]:


pred_xgbc=xgbc.predict(X_test)


# In[ ]:


print("The Train score of XGBoosting is :",xgbc.score(X_train,y_train))
print("The Test score of XGBoosting is :",xgbc.score(X_test,y_test))
print("The accuracy of XGBoosting is ",accuracy_score(y_test,pred_xgbc))
print("The f1 score of XGBoosting is ",f1_score(y_test,pred_xgbc))
print("The Confusion Matrix of XGBoosting is \n \n",confusion_matrix(y_test,pred_xgbc))
print("\n")
print("The Classification Report of XGBoosting is \n \n",classification_report(y_test,pred_xgbc))


# Here the False Negatives has been further reduced from 992 to 905 
# Since the number of False Negatives are less for this model.,considering this as the final model which will be used on the test data.

# **Test Data Analysis**

# In[ ]:


df1.head()


# In[ ]:


df1.info()


# In[ ]:


df1.isnull().sum()


# In[ ]:


df1.isnull().sum().sum()/len(df1)


# In the data set,total missing values is around 12%

# **Treatment of Missing Values**

# In[ ]:


b=df1[df1['previous_year_rating'].isnull()]
b.head()


# In[ ]:


b['length_of_service'].value_counts()


# In[ ]:


df1['previous_year_rating'].fillna(value=0,inplace=True)


# In[ ]:


df1.isnull().sum()


# In[ ]:


for i in df1[df1['education'].isnull()]['KPIs_met >80%']==0:
    df1['education'].fillna(value="Bachelor's",inplace=True)


# In[ ]:


for i in df1[df1['education'].isnull()]['KPIs_met >80%']==1:
    df1['education'].fillna(value="Master's & above",inplace=True)


# In[ ]:


df1['education'].isnull().sum()


# In[ ]:


df1.isnull().sum()


# In[ ]:


df_test=pd.get_dummies(df1,['department','region','education','gender','recruitment_channel'],drop_first=True)


# In[ ]:


df_test.head(2)


# In[ ]:


emp_id=df_test['employee_id']


# In[ ]:


df_test.drop(['employee_id'],axis=1,inplace=True)


# In[ ]:


X_test1=df_test
test_pred=xgbc.predict(X_test1)


# In[ ]:


final=pd.DataFrame()
final['employee_id']=pd.Series(emp_id)
final['is_promoted']=pd.Series(test_pred)
final

