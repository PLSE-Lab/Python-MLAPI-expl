#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


dataset=pd.read_csv('/kaggle/input/hackerearth/train.csv',index_col='ID')
test1=pd.read_csv('/kaggle/input/hackerearth/train.csv')


# In[ ]:


dataset.columns[dataset.isnull().any()]


# In[ ]:


alldata=dataset.append(test1)


# In[ ]:


dataset.shape


# In[ ]:


alldata.shape


# In[ ]:


dataset.head()


# In[ ]:


test1.isnull().any()
test1=test1.fillna(method='ffill')


# In[ ]:


dataset.describe()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(dataset['Result'],label='Count')


# In[ ]:


dataset.isnull().any()
dataset = dataset.fillna(method='ffill')


# In[ ]:


dataset.hist(bins=10,figsize=(20,15))
plt.show()


# In[ ]:


plt.figure(figsize=(50,25))
sns.heatmap(data=dataset.corr(),annot=True)
plt.title('Co-Relation Mattrix')
plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix,classification_report,r2_score,accuracy_score


# In[ ]:


X=dataset.drop('Result',axis=1)
Y=dataset['Result']
model1=DecisionTreeClassifier()


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1)


# In[ ]:


model1.fit(X_train,y_train)


# In[ ]:


pred=model1.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,pred))


# In[ ]:


print(accuracy_score(y_test,pred))


# In[ ]:


print(classification_report(y_test,pred))


# In[ ]:


r2_score(y_test,pred)


# In[ ]:


df=pd.DataFrame({'Actual Pred':y_test,'Predicted ':pred})
df1=df.head(25)
print(df1)


# In[ ]:



df1.plot(kind='bar',figsize=(20,5))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[ ]:


print('Mean absolute Error',metrics.mean_absolute_error(y_test,pred))
print('Mean squared Error',metrics.mean_squared_error(y_test,pred))
print('Mean squared Error',np.sqrt(metrics.mean_absolute_error(y_test,pred)))


# In[ ]:





# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


model2=LogisticRegression()


# In[ ]:


model2.fit(X_train,y_train)


# In[ ]:


model2_pred=model2.predict(X_test)


# In[ ]:


print(classification_report(y_test,model2_pred))


# In[ ]:


print(accuracy_score(y_test,model2_pred))


# In[ ]:


r2_score(y_test,model2_pred)


# In[ ]:


print(confusion_matrix(y_test,model2_pred))


# In[ ]:


from sklearn.svm import LinearSVC


# In[ ]:


model3=LinearSVC(C=1000)


# In[ ]:


model3.fit(X_train,y_train)


# In[ ]:


model3_pred=model3.predict(X_test)


# In[ ]:


print(accuracy_score(y_test,model3_pred))


# In[ ]:


print(classification_report(y_test,model3_pred))


# In[ ]:


print(confusion_matrix(y_test,model3_pred))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
KNNclassifier = KNeighborsClassifier(n_neighbors=5)
KNNclassifier.fit(X_train, y_train)


# In[ ]:


KNN_pred = KNNclassifier.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test, KNN_pred))


# In[ ]:


print(accuracy_score(y_test,KNN_pred))


# In[ ]:


print(classification_report(y_test,KNN_pred))


# In[ ]:



error = []
# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))


# In[ ]:



plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='black', linestyle='dashed', marker='.',
         markerfacecolor='black', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


# In[ ]:


from xgboost import XGBRegressor


# In[ ]:


model4=XGBRegressor(n_estimators=1000,learning_rate=0.05)
model4.fit(X_train,y_train,early_stopping_rounds=50,eval_set=[(X_test,y_test)],verbose=False)


# In[ ]:


model4_pred=model4.predict(X_test)


# In[ ]:


print(r2_score(y_test,model4_pred))


# In[ ]:


df4=pd.DataFrame({'Actual prediction ': y_test, 'Model Prediction': model4_pred})
df5=df.head(25)
print(df5)


# In[ ]:




